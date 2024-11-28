/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "iter/misc.h"
#include "iter/iter.h"
#include "iter/itop.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/opts.h"

#include "grecon/optreg.h"
#include "grecon/italgo.h"


static const char help_str[] = "Perform image denoising with regularization.\n";

int main_denoise(int argc, char* argv[argc])
{
	const char* img_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &img_file, "image"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	int shift_mode = 0;
	bool randshift = true;
	bool overlapping_blocks = false;
	int maxiter = 100;
	float step = -1.;

	// Start time count

	double start_time = timestamp();

	bool scale_im = false;
	bool eigen = false;
	float scaling = 0.;

	int llr_blk = 8;
	const char* wtype_str = "dau2";

	struct admm_conf admm = { false, false, false, .1, iter_admm_defaults.maxitercg, false };
	struct fista_conf fista = { { -1., -1., -1. }, false };
	struct pridu_conf pridu = { 1., false };

	enum algo_t algo = ALGO_DEFAULT;

	bool hogwild = false;

	bool gpu = false;

	struct opt_reg_s ropts;
	opt_reg_init(&ropts);

	const struct opt_s opts[] = {

		{ 'l', NULL, true, OPT_SPECIAL, opt_reg, &ropts, "\b1/-l2", "  toggle l1-wavelet or l2 regularization." },
		OPT_FLOAT('r', &ropts.lambda, "lambda", "regularization parameter"),
		{ 'R', NULL, true, OPT_SPECIAL, opt_reg, &ropts, "<T>:A:B:C", "generalized regularization options (-Rh for help)" },
		OPT_FLOAT('s', &step, "step", "iteration stepsize"),
		OPT_PINT('i', &maxiter, "iter", "max. number of iterations"),
		OPT_CLEAR('n', &randshift, "disable random wavelet cycle spinning"),
		OPT_SET('N', &overlapping_blocks, "do fully overlapping LLR blocks"),
		OPT_SET('g', &bart_use_gpu, "use GPU"),
		OPT_PINT('b', &llr_blk, "blk", "Lowrank block size"),
		OPT_SET('e', &eigen, "Scale stepsize based on max. eigenvalue"),
		OPTL_SET(0, "adaptive-stepsize", &(pridu.adaptive_stepsize), "PRIDU adaptive step size"),
		OPTL_SET(0, "asl", &(ropts.asl), "ASL reconstruction"),
		OPTL_SET(0, "teasl", &(ropts.teasl), "Time-encoded ASL reconstruction"),
		OPTL_FLVEC2(0, "theta", &ropts.theta, "theta1:theta2", "PWI weight for ASL reconstruction"),
		OPTL_FLVECN(0, "tvscales", ropts.tvscales, "Scaling of derivatives in TV or TGV regularization"),
		OPTL_FLVECN(0, "tvscales2", ropts.tvscales2, "Scaling of secondary derivatives in ICTV reconstruction"),
		OPTL_FLVEC2(0, "alpha", &ropts.alpha, "alpha1:alpha0", "regularization parameter for TGV and ICTGV reconstruction"),
		OPTL_FLVEC2(0, "gamma", &ropts.gamma, "gamma1:gamma2", "regularization parameter for ICTV and ICTGV reconstruction"),
		OPT_SET('H', &hogwild, "(hogwild)"),
		OPT_SET('D', &admm.dynamic_rho, "(ADMM dynamic step size)"),
		OPT_SET('F', &admm.fast, "(fast)"),
		OPT_SET('J', &admm.relative_norm, "(ADMM residual balancing)"),
		OPT_INT('d', &debug_level, "level", "Debug level"),
		OPT_FLOAT('u', &admm.rho, "rho", "ADMM rho"),
		OPT_PINT('C', &admm.maxitercg, "iter", "ADMM max. CG iterations"),
		OPTL_SELECT('I', "ist", enum algo_t, &algo, ALGO_IST, "select IST"),
		OPTL_SELECT(0, "fista", enum algo_t, &algo, ALGO_FISTA, "select FISTA"),
		OPTL_SELECT(0, "eulermaruyama", enum algo_t, &algo, ALGO_EULERMARUYAMA, "select Euler Maruyama"),
		OPTL_SELECT('m', "admm", enum algo_t, &algo, ALGO_ADMM, "select ADMM"),
		OPTL_SELECT('a', "pridu", enum algo_t, &algo, ALGO_PRIDU, "select Primal Dual"),
		OPT_FLOAT('w', &scaling, "", "inverse scaling of the data"),
		OPT_SET('S', &scale_im, "re-scale the image after denoising"),
		OPTL_STRING(0, "wavelet", &wtype_str, "name", "wavelet type (haar,dau2,cdf44)"),
		OPTL_FLVEC3(0, "fista_pqr", &fista.params, "p:q:r", "parameters for FISTA acceleration"),
		OPTL_SET(0, "fista_last", &fista.last, "end iteration with call to data consistency")
	};


	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	long img_dims[DIMS];

	// load image data and get dimensions

	complex float* img = load_cfl(img_file, DIMS, img_dims);

	if (ropts.asl && ropts.teasl)
		error("Use either TE-ASL or ASL denoising.\n");

	if (ropts.asl && 2 != img_dims[ITER_DIM])
		error("ASL denoising requires two slices (label and control) along ITER_DIM.\n");

	complex float* img_p = img;

	gpu = bart_use_gpu;

	num_init_gpu_support();

	// print options

	if (gpu)
		debug_printf(DP_INFO, "GPU denoising\n");

	if (hogwild)
		debug_printf(DP_INFO, "Hogwild stepsize\n");

	if (admm.dynamic_rho)
		debug_printf(DP_INFO, "ADMM Dynamic stepsize\n");

	if (admm.relative_norm)
		debug_printf(DP_INFO, "ADMM residual balancing\n");

	if (randshift)
		shift_mode = 1;

	if (overlapping_blocks) {

		if (randshift)
			debug_printf(DP_WARN, "Turning off random shifts\n");

		shift_mode = 2;

		debug_printf(DP_INFO, "Fully overlapping LLR blocks\n");
	}

	// initialize forward_op

	const struct linop_s* forward_op = linop_identity_create(DIMS, img_dims);

	// apply scaling

	if (0. == scaling) {

		scaling = 1.;

	} else {

		debug_printf(DP_DEBUG1, "Inverse scaling of the data: %f\n", scaling);
		md_zsmul(DIMS, img_dims, img_p, img_p, 1. / scaling);

		pridu.sigma_tau_ratio = scaling;
	}

	if (ropts.teasl) {

		const struct linop_s* hadamard_op = linop_hadamard_create(DIMS, img_dims, ITER_DIM);
		forward_op = linop_chain_FF(hadamard_op, forward_op);
	}

	complex float* image = create_cfl(out_file, DIMS, img_dims);
	md_clear(DIMS, img_dims, image, CFL_SIZE);

	// initialize prox functions

	const struct operator_p_s* thresh_ops[NUM_REGS] = { NULL };
	const struct linop_s* trafos[NUM_REGS] = { NULL };


	opt_reg_configure(DIMS, img_dims, &ropts, thresh_ops, trafos, NULL, llr_blk, shift_mode, wtype_str, gpu, ITER_DIM);

	int nr_penalties = ropts.r + ropts.sr;

	debug_printf(DP_INFO, "Regularization terms: %d, Supporting variables: %ld\n", nr_penalties, ropts.svars);

	// choose algorithm

	if (ALGO_DEFAULT == algo)
		algo = italgo_choose(ropts.r, ropts.regs);

	// choose step size

	if ((ALGO_IST == algo) || (ALGO_FISTA == algo) || (ALGO_PRIDU == algo)) {

		if ((-1. == step) && !eigen)
			debug_printf(DP_WARN, "No step size specified.\n");

		if (-1. == step)
			step = 0.95;
	}

	if ((ALGO_CG == algo) || (ALGO_ADMM == algo))
		if (-1. != step)
			debug_printf(DP_INFO, "Stepsize ignored.\n");

	// initialize algorithm
	struct iter it = italgo_config(algo, nr_penalties, ropts.regs, maxiter, step, eigen ? 30 : 0, hogwild, admm, fista, pridu, false);

	if (ALGO_CG == algo)
		nr_penalties = 0;

	bool trafos_cond = (   (ALGO_PRIDU == algo)
			    || (ALGO_ADMM == algo)
			    || (   (ALGO_NIHT == algo)
				&& (ropts.regs[0].xform == NIHTWAV)));

	if (0 < ropts.svars) {

		const struct linop_s* extract = linop_extract_create(1, MD_DIMS(0), MD_DIMS(md_calc_size(DIMS, img_dims)), MD_DIMS(md_calc_size(DIMS, img_dims) + ropts.svars));
		extract = linop_reshape_out_F(extract, DIMS, img_dims);
		forward_op = linop_chain_FF(extract, forward_op);
	}

	const struct operator_s* normaleq_op = operator_ref(forward_op->normal);
	const struct operator_s* adjoint = operator_ref(forward_op->adjoint);

	const struct operator_s* itop_op = itop_create(it.italgo, it.iconf, false, NULL, normaleq_op, nr_penalties, thresh_ops, trafos_cond ? trafos : NULL, NULL, NULL);

	if (gpu) {

		debug_printf(DP_DEBUG1, "itop: add GPU wrapper\n");
		auto tmp = operator_gpu_wrapper(itop_op);
		operator_free(itop_op);
		itop_op = tmp;
	}

	const struct operator_s* temp_op = operator_chain(adjoint, itop_op);
	operator_free(itop_op);
	itop_op = temp_op;

	operator_free(normaleq_op);
	operator_free(adjoint);
	linop_free(forward_op);

	if (0 < ropts.svars) {

		const struct linop_s* extract = linop_extract_create(1, MD_DIMS(0), MD_DIMS(md_calc_size(DIMS, img_dims)), MD_DIMS(md_calc_size(DIMS, img_dims) + ropts.svars));
		extract = linop_reshape_out_F(extract, DIMS, img_dims);

		auto op2 = operator_chain(itop_op, extract->forward);

		operator_free(itop_op);
		itop_op = op2;

		linop_free(extract);
	}

	operator_apply(itop_op, DIMS, img_dims, image, DIMS, img_dims, img_p);

	operator_free(itop_op);

	opt_reg_free(&ropts, thresh_ops, trafos);

	italgo_config_free(it);

	if (scale_im)
		md_zsmul(DIMS, img_dims, image, image, scaling);

	// clean up

	unmap_cfl(DIMS, img_dims, image);

	if (img_p != img)
		md_free(img_p);

	unmap_cfl(DIMS, img_dims, img);

	double end_time = timestamp();

	debug_printf(DP_INFO, "Total Time: %f\n", end_time - start_time);

	return 0;
}

