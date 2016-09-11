/* Copyright 2015. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2015 Berkin Bilgic <berkin@nmr.mgh.harvard.edu>
 * 2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
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

#include "iter/iter.h"
#include "iter/lsqr.h"

#include "linops/linop.h"
#include "linops/sampling.h"
#include "linops/someops.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/utils.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "sense/model.h"
#include "sense/optcom.h"

#include "wavelet3/wavthresh.h"


// create wavecaipi operator

static struct linop_s* wavecaipi_create(const long dims[DIMS], long img_read, const complex float* wave)
{
	// Wave-CAIPI linear operator created by chaining zero-padding, readout fft, 
	// psf multiplication, ky-kz fft
    
	long img_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|COIL_FLAG, img_dims, dims);
	img_dims[READ_DIM] = img_read;
    
	struct linop_s* fft_read = linop_fft_create(DIMS, dims, READ_FLAG);
	struct linop_s* fft_yz = linop_fft_create(DIMS, dims, PHS1_FLAG|PHS2_FLAG);
	struct linop_s* resize = linop_resize_create(DIMS, dims, img_dims);
	struct linop_s* wavemod = linop_cdiag_create(DIMS, dims, FFT_FLAGS, wave);

	struct linop_s* wc_op = linop_chain(linop_chain(linop_chain(resize, fft_read), wavemod), fft_yz);

	linop_free(fft_read);
	linop_free(fft_yz);
	linop_free(resize);
	linop_free(wavemod);

	return wc_op;
}



static const char usage_str[] = "<kspace> <sensitivities> <wave> <output>";
static const char help_str[] = "Perform iterative wavecaipi reconstruction.";



int main_wave(int argc, char* argv[])
{
	float lambda = 0.;
	float step = 0.95;

	bool l1wav = false;
	bool hogwild = false;
	bool randshift = true;
	bool adjoint = false;
	int maxiter = 50;

	const struct opt_s opts[] = {

		OPT_SET('l',  &l1wav, "use L1 penalty"),
		OPT_SET('a', &adjoint, "adjoint"),
		OPT_INT('i', &maxiter, "iter", "max. iterations"),
		OPT_FLOAT('r', &lambda, "lambda", "regularization parameter"),
	};

	cmdline(&argc, argv, 4, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);


	long map_dims[DIMS];
	long pat_dims[DIMS];
	long img_dims[DIMS];
	long ksp_dims[DIMS];
	long max_dims[DIMS];
	long wav_dims[DIMS];

	// load kspace and maps and get dimensions

	complex float* kspace = load_cfl(argv[1], DIMS, ksp_dims);
	complex float* maps = load_cfl(argv[2], DIMS, map_dims);
	complex float* wave = load_cfl(argv[3], DIMS, wav_dims);


	md_copy_dims(DIMS, max_dims, ksp_dims);
	max_dims[MAPS_DIM] = map_dims[MAPS_DIM];

	md_select_dims(DIMS, ~COIL_FLAG, pat_dims, ksp_dims);
	md_select_dims(DIMS, ~(COIL_FLAG|READ_FLAG), img_dims, max_dims);
	img_dims[READ_DIM] = map_dims[READ_DIM];


	for (int i = 1; i < 4; i++)	// sizes2[4] may be > 1
		if (ksp_dims[i] != map_dims[i])
			error("Dimensions of kspace and sensitivities do not match!\n");

	// FIXME: add more sanity checking of dimensions

	assert(1 == ksp_dims[MAPS_DIM]);

	num_init();

	// initialize sampling pattern

	complex float* pattern = md_alloc(DIMS, pat_dims, CFL_SIZE);
	estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pattern, kspace);

	// print some statistics

	size_t T = md_calc_size(DIMS, pat_dims);
	long samples = (long)pow(md_znorm(DIMS, pat_dims, pattern), 2.);
	debug_printf(DP_INFO, "Size: %ld Samples: %ld Acc: %.2f\n", T, samples, (float)T / (float)samples); 


	// apply scaling

	float scaling = estimate_scaling(ksp_dims, NULL, kspace);

	if (scaling != 0.)
		md_zsmul(DIMS, ksp_dims, kspace, kspace, 1. / scaling);


	const struct operator_p_s* thresh_op = NULL;

	// wavelet operator
	if (l1wav) {

		long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
		minsize[0] = MIN(img_dims[0], 16);
		minsize[1] = MIN(img_dims[1], 16);
		minsize[2] = MIN(img_dims[2], 16);

		thresh_op = prox_wavelet3_thresh_create(DIMS, img_dims, FFT_FLAGS, minsize, lambda, randshift);
	}
    

	complex float* image = create_cfl(argv[4], DIMS, img_dims);
	md_clear(DIMS, img_dims, image, CFL_SIZE);


	fftmod(DIMS, ksp_dims, FFT_FLAGS, kspace, kspace);
	fftmod(DIMS, map_dims, FFT_FLAGS, maps, maps);

//	initialize iterative algorithm
    
	italgo_fun_t italgo = NULL;
	iter_conf* iconf = NULL;

	struct iter_conjgrad_conf cgconf;
	struct iter_fista_conf fsconf;

	if (!l1wav) {

		// configuration for CG recon 
        	cgconf = iter_conjgrad_defaults;
		cgconf.maxiter = maxiter;        // max no of iterations
		cgconf.l2lambda = lambda;   // regularization parameter
		cgconf.tol = 1.E-3;         // cg tolerance     

		italgo = iter_conjgrad;
		iconf = CAST_UP(&cgconf);

	} else {

		// use FISTA for wavelet regularization
		fsconf = iter_fista_defaults;
		fsconf.maxiter = maxiter;
		fsconf.step = step;
		fsconf.hogwild = hogwild;

		italgo = iter_fista;
		iconf = CAST_UP(&fsconf);
	}


	// create sense maps operator
//	struct linop_s* mapsop = maps_create(map_dims, FFT_FLAGS|COIL_FLAG|MAPS_FLAG, maps, false);

	md_zsmul(DIMS, map_dims, maps, maps, 1. / sqrt((double)(ksp_dims[0] * ksp_dims[1] * ksp_dims[2])));
	struct linop_s* mapsop = maps2_create(map_dims, map_dims, img_dims, maps);
    
	// create wave caipi operator
	struct linop_s* waveop = wavecaipi_create(ksp_dims, img_dims[READ_DIM], wave);

	// create sense operator by chaining coil sens and wave operators    
	struct linop_s* sense_op = linop_chain(mapsop, waveop);

	// create forward operator by adding sampling mask to sense operator
	struct linop_s* forward = linop_chain(sense_op, linop_sampling_create(ksp_dims, pat_dims, pattern));

	struct lsqr_conf lsqr_conf = { 0., false };

	// reconstruction with LSQR

	if (adjoint)
		linop_adjoint(forward, DIMS, img_dims, image, DIMS, ksp_dims, kspace);
	else
	        lsqr(DIMS, &lsqr_conf, italgo, iconf, forward, thresh_op, img_dims, image, ksp_dims, kspace, NULL);
        
	unmap_cfl(DIMS, map_dims, maps);
	unmap_cfl(DIMS, wav_dims, wave);
	unmap_cfl(DIMS, ksp_dims, kspace);
	unmap_cfl(DIMS, img_dims, image);

	exit(0);
}


