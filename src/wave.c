/* Copyright 2015. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * Copyright 2018. Massachusetts Institute of Technology.
 *
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2015 Berkin Bilgic <berkin@nmr.mgh.harvard.edu>
 * 2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2018 Siddharth Iyer <ssi@mit.edu>
 *
 * B Bilgic, BA Gagoski, SF Cauley, AP Fan, JR Polimeni, PE Grant,
 * LL Wald, and K Setsompop, Wave-CAIPI for highly accelerated 3D
 * imaging. Magn Reson Med (2014) doi: 10.1002/mrm.25347
 */

#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/iovec.h"
#include "num/ops.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "iter/iter.h"
#include "iter/lsqr.h"
#include "iter/misc.h"

#include "linops/linop.h"
#include "linops/fmac.h"
#include "linops/someops.h"
#include "linops/realval.h"

#include "sense/optcom.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/utils.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "wavelet/wavthresh.h"

static const char usage_str[] = "<maps> <wave> <kspace> <output>";
static const char help_str[]  = 
	"Perform a wave-caipi reconstruction.\n\n"
	"Conventions:\n"
	"  * (sx, sy, sz) - Spatial dimensions.\n"
	"  * wx           - Extended FOV in READ_DIM due to\n"
	"                   wave's voxel spreading.\n"
	"  * (nc, md)     - Number of channels and ESPIRiT's \n"
	"                   extended-SENSE model operator\n"
	"                   dimensions (or # of maps).\n"
	"Expected dimensions:\n"
	"  * maps    - (   sx, sy, sz, nc, md)\n"
	"  * wave    - (   wx, sy, sz,  1,  1)\n"
	"  * phi     - (    1,  1,  1,  1,  1)\n"
	"  * output  - (   sx, sy, sz,  1, md)\n"
	"  * reorder - (    n,  3,  1,  1,  1)\n"
	"  * table   - (   wx, nc,  n,  1,  1)";

/* Helper function to print out operator dimensions. */
static void print_opdims(const struct linop_s* op) 
{
	const struct iovec_s* domain   = linop_domain(op);
	const struct iovec_s* codomain = linop_codomain(op);
	debug_printf(DP_INFO, "\tDomain:   ");
	debug_print_dims(DP_INFO, domain->N, domain->dims);
	debug_printf(DP_INFO, "\tCodomain: ");
	debug_print_dims(DP_INFO, codomain->N, codomain->dims);
}

/* ESPIRiT operator. */
static const struct linop_s* linop_espirit_create(long sx, long sy, long sz, long nc, long md, complex float* maps)
{
	long max_dims[] = { [0 ... DIMS - 1] = 1};
	max_dims[0] = sx;
	max_dims[1] = sy;
	max_dims[2] = sz;
	max_dims[3] = nc;
	max_dims[4] = md;
	const struct linop_s* E = linop_fmac_create(DIMS, max_dims, MAPS_FLAG, COIL_FLAG, ~(FFT_FLAGS|MAPS_FLAG|COIL_FLAG), maps);
	return E;
}

/* Resize operator. */
static const struct linop_s* linop_reshape_create(long wx, long sx, long sy, long sz, long nc)
{
	long input_dims[] = { [0 ... DIMS - 1] = 1};
	input_dims[0] = sx;
	input_dims[1] = sy;
	input_dims[2] = sz;
	input_dims[3] = nc;
	long output_dims[DIMS];
	md_copy_dims(DIMS, output_dims, input_dims);
	output_dims[0] = wx;
	struct linop_s* R = linop_resize_create(DIMS, output_dims, input_dims);
	return R;
}

/* Fx operator. */
static const struct linop_s* linop_fx_create(long wx, long sy, long sz, long nc)
{
	long dims[] = { [0 ... DIMS - 1] = 1};
	dims[0] = wx;
	dims[1] = sy;
	dims[2] = sz;
	dims[3] = nc;
	struct linop_s* Fx = linop_fft_create(DIMS, dims, READ_FLAG);
	return Fx;
}

/* Wave operator. */
static const struct linop_s* linop_wave_create(long wx, long sy, long sz, long nc, complex float* psf)
{
	long dims[] = { [0 ... DIMS - 1] = 1};
	dims[0] = wx;
	dims[1] = sy;
	dims[2] = sz;
	dims[3] = nc;
	struct linop_s* W = linop_cdiag_create(DIMS, dims, FFT_FLAGS, psf);
	return W;
}

/* Fyz operator. */
static const struct linop_s* linop_fyz_create(long wx, long sy, long sz, long nc)
{
	long dims[] = { [0 ... DIMS - 1] = 1};
	dims[0] = wx;
	dims[1] = sy;
	dims[2] = sz;
	dims[3] = nc;
	struct linop_s* Fyz = linop_fft_create(DIMS, dims, PHS1_FLAG|PHS2_FLAG);
	return Fyz;
}

/* Sampling operator. */
static const struct linop_s* linop_samp_create(long wx, long sy, long sz, long nc, complex float* mask)
{
	long dims[] = { [0 ... DIMS - 1] = 1};
	dims[0] = wx;
	dims[1] = sy;
	dims[2] = sz;
	dims[3] = nc;
	struct linop_s* M = linop_cdiag_create(DIMS, dims, FFT_FLAGS, mask);
	return M;
}

enum algo_t { CG, IST, FISTA };

int main_wave(int argc, char* argv[])
{
	double start_time = timestamp();

	float lambda    = 1E-5;
	int   maxiter   = 300;
	float step      = 0.5;
	float tol       = 1.E-3;
	bool  wav       = false;
	bool  fista     = false;
	bool  hgwld     = false;
	float cont      = 1;
	float eval      = -1;
	int   gpun      = -1;
	bool  rvc       = false;

	const struct opt_s opts[] = {
		OPT_FLOAT( 'r', &lambda,  "lambda", "Soft threshold lambda for wavelet or locally low rank."),
		OPT_INT(   'i', &maxiter, "mxiter", "Maximum number of iterations."),
		OPT_FLOAT( 's', &step,    "stepsz", "Step size for iterative method."),
		OPT_FLOAT( 'c', &cont,    "cntnu",  "Continuation value for IST/FISTA."),
		OPT_FLOAT( 't', &tol,     "toler",  "Tolerance convergence condition for iterative method."),
		OPT_FLOAT( 'e', &eval,    "eigvl",  "Maximum eigenvalue of normal operator, if known."),
		OPT_INT(   'g', &gpun,    "gpunm",  "GPU device number."),
		OPT_SET(   'f', &fista,             "Reconstruct using FISTA instead of IST."),
		OPT_SET(   'H', &hgwld,             "Use hogwild in IST/FISTA."),
		OPT_SET(   'w', &wav,               "Use wavelet."),
		OPT_SET(   'v', &rvc,               "Apply real valued constraint on coefficients."),
	};

	cmdline(&argc, argv, 4, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);

	debug_printf(DP_INFO, "Loading data... ");

	long maps_dims[DIMS];
	complex float* maps = load_cfl(argv[1], DIMS, maps_dims);

	long wave_dims[DIMS];
	complex float* wave = load_cfl(argv[2], DIMS, wave_dims);

	long kspc_dims[DIMS];
	complex float* kspc = load_cfl(argv[3], DIMS, kspc_dims);

	debug_printf(DP_INFO, "Done.\n");

	if (gpun >= 0)
		num_init_gpu_device(gpun);
	else
		num_init();

	int wx = wave_dims[0];
	int sx = maps_dims[0];
	int sy = maps_dims[1];
	int sz = maps_dims[2];
	int nc = maps_dims[3];
	int md = maps_dims[4];

	long recon_dims[] = { [0 ... DIMS - 1] = 1 };
	recon_dims[0] = sx;
	recon_dims[1] = sy;
	recon_dims[2] = sz;
	recon_dims[4] = md;

	debug_printf(DP_INFO, "FFTMOD maps and kspc... ");
	fftmod(DIMS, kspc_dims, FFT_FLAGS, kspc, kspc);
	fftmod(DIMS, maps_dims, FFT_FLAGS, maps, maps);
	debug_printf(DP_INFO, "Done.\n");

	debug_printf(DP_INFO, "Estimating sampling mask... ");
	long mask_dims[] = { [0 ... DIMS - 1] = 1 };
	mask_dims[0] = wx;
	mask_dims[1] = sy;
	mask_dims[2] = sz;
	complex float* mask = md_calloc(DIMS, mask_dims, CFL_SIZE);
	estimate_pattern(DIMS, kspc_dims, ~FFT_FLAGS, mask, kspc);
	debug_printf(DP_INFO, "Done.\n");

	debug_printf(DP_INFO, "Creating linear operators... ");
	const struct linop_s* E   = linop_espirit_create(sx, sy, sz, nc, md, maps);
	const struct linop_s* R   = linop_reshape_create(wx, sx, sy, sz, nc);
	const struct linop_s* Fx  = linop_fx_create(wx, sy, sz, nc);
	const struct linop_s* W   = linop_wave_create(wx, sy, sz, nc, wave);
	const struct linop_s* Fyz = linop_fyz_create(wx, sy, sz, nc);
	const struct linop_s* M   = linop_samp_create(wx, sy, sz, nc, mask);
	debug_printf(DP_INFO, "Done.\n");

	debug_printf(DP_INFO, "Forward linear operator information:\n");
	struct linop_s* A = linop_chain(linop_chain(linop_chain(linop_chain(linop_chain(
		E, R), Fx), W), Fyz), M);

	if (rvc == true) {
		debug_printf(DP_INFO, "\tDomain is restricted to real numbers.\n");
		struct linop_s* tmp = A;
		struct linop_s* rvcop = linop_realval_create(DIMS, linop_domain(A)->dims);

		A = linop_chain(rvcop, tmp);

		linop_free(rvcop);
		linop_free(tmp);
	}

	linop_free(E);
	linop_free(R);
	linop_free(Fx);
	linop_free(W);
	linop_free(Fyz);
	linop_free(M);

	print_opdims(A);

	if (eval < 0)	
#ifdef USE_CUDA
		eval = (gpun >= 0) ? estimate_maxeigenval_gpu(A->normal) : estimate_maxeigenval(A->normal);
#else
		eval = estimate_maxeigenval(A->normal);
#endif
	debug_printf(DP_INFO, "\tMax eval: %.2e\n", eval);
	step /= eval;

	debug_printf(DP_INFO, "Normalizing kspace... ");
	float norm = md_znorm(DIMS, kspc_dims, kspc);
	md_zsmul(DIMS, kspc_dims, kspc, kspc, 1. / norm);
	debug_printf(DP_INFO, "Done.\n");

	const struct operator_p_s* T = NULL;
	long minsize[] = { [0 ... DIMS - 1] = 1 };
	minsize[0] = MIN(sx, 16);
	minsize[1] = MIN(sy, 16);
	minsize[2] = MIN(sz, 16);
	unsigned int WAVFLAG = (sx > 1) * READ_FLAG | (sy > 1) * PHS1_FLAG | (sz > 2) * PHS2_FLAG;

	enum algo_t algo = CG;
	if (wav == true) {
		algo = (fista) ? FISTA : IST;
		debug_printf(DP_INFO, "Creating wavelet threshold operator... ");
		T = prox_wavelet_thresh_create(DIMS, recon_dims, WAVFLAG, 0u, minsize, lambda, true);
		debug_printf(DP_INFO, "Done.\n");
	}

	italgo_fun2_t italgo = iter2_call_iter;
	struct iter_call_s iter2_data;
	SET_TYPEID(iter_call_s, &iter2_data);
	iter_conf* iconf = CAST_UP(&iter2_data);

	struct iter_conjgrad_conf cgconf = iter_conjgrad_defaults;
	struct iter_fista_conf    fsconf = iter_fista_defaults;
	struct iter_ist_conf      isconf = iter_ist_defaults;

	switch(algo) {

		case IST:

			debug_printf(DP_INFO, "Using IST.\n");
			debug_printf(DP_INFO, "\tLambda:             %0.2e\n", lambda);
			debug_printf(DP_INFO, "\tMaximum iterations: %d\n", maxiter);
			debug_printf(DP_INFO, "\tStep size:          %0.2e\n", step);
			debug_printf(DP_INFO, "\tHogwild:            %d\n", (int) hgwld);
			debug_printf(DP_INFO, "\tTolerance:          %0.2e\n", tol);
			debug_printf(DP_INFO, "\tContinuation:       %0.2e\n", cont);

			isconf              = iter_ist_defaults;
			isconf.step         = step;
			isconf.maxiter      = maxiter;
			isconf.tol          = tol;
			isconf.continuation = cont;
			isconf.hogwild      = hgwld;

			iter2_data.fun   = iter_ist;
			iter2_data._conf = CAST_UP(&isconf);

			break;

		case FISTA:

			debug_printf(DP_INFO, "Using FISTA.\n");
			debug_printf(DP_INFO, "\tLambda:             %0.2e\n", lambda);
			debug_printf(DP_INFO, "\tMaximum iterations: %d\n", maxiter);
			debug_printf(DP_INFO, "\tStep size:          %0.2e\n", step);
			debug_printf(DP_INFO, "\tHogwild:            %d\n", (int) hgwld);
			debug_printf(DP_INFO, "\tTolerance:          %0.2e\n", tol);
			debug_printf(DP_INFO, "\tContinuation:       %0.2e\n", cont);

			fsconf              = iter_fista_defaults;
			fsconf.maxiter      = maxiter;
			fsconf.step         = step;
			fsconf.hogwild      = hgwld;
			fsconf.tol          = tol;
			fsconf.continuation = cont;

			iter2_data.fun   = iter_fista;
			iter2_data._conf = CAST_UP(&fsconf);

			break;

		default:
		case CG:

			debug_printf(DP_INFO, "Using CG.\n");
			debug_printf(DP_INFO, "\tMaximum iterations: %d\n", maxiter);
			debug_printf(DP_INFO, "\tTolerance:          %0.2e\n", tol);

			cgconf          = iter_conjgrad_defaults;
			cgconf.maxiter  = maxiter;
			cgconf.l2lambda = 0;
			cgconf.tol      = tol;

			iter2_data.fun   = iter_conjgrad;
			iter2_data._conf = CAST_UP(&cgconf);

			break;

	}

	complex float* tmp = create_cfl("../../tmp", DIMS, mask_dims);
  md_copy(DIMS, mask_dims, tmp, mask, CFL_SIZE);
  unmap_cfl(DIMS, mask_dims, tmp);

	debug_printf(DP_INFO, "Reconstruction... ");
	complex float* recon = create_cfl(argv[4], DIMS, recon_dims);
	struct lsqr_conf lsqr_conf = { 0., gpun >= 0 };
	double recon_start = timestamp();
	const struct operator_s* J = lsqr2_create(&lsqr_conf, italgo, iconf, NULL, A, NULL, 1, &T, NULL, NULL);
	operator_apply(J, DIMS, recon_dims, recon, DIMS, kspc_dims, kspc);
	double recon_end = timestamp();
	debug_printf(DP_INFO, "Done.\nReconstruction time: %f seconds.\n", recon_end - recon_start);

	debug_printf(DP_INFO, "Cleaning up and saving result... ");
	operator_free(J);
	linop_free(A);
	md_free(mask);
	unmap_cfl(DIMS, maps_dims, maps);
	unmap_cfl(DIMS, wave_dims, wave);
	unmap_cfl(DIMS, kspc_dims, kspc);
	unmap_cfl(DIMS, recon_dims, recon);
	debug_printf(DP_INFO, "Done.\n");

	double end_time = timestamp();
	debug_printf(DP_INFO, "Total time: %f seconds.\n", end_time - start_time);

	return 0;
}
