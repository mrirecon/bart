/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2015-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014 Frank Ong <frankong@berkeley.edu>
 * 2014-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"
#include "misc/opts.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"
#include "num/ops.h"

#include "linops/linop.h"

#include "iter/iter.h"
#include "iter/lsqr.h"

#include "noncart/nufft.h"
#include "noncart/nudft.h"
#include "noncart/precond.h"


static const char help_str[] = "Perform non-uniform Fast Fourier Transform.";





int main_nufft(int argc, char* argv[argc])
{
	const char* traj_file = NULL;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &traj_file, "traj"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	bool adjoint = false;
	bool inverse = false;
	bool precond = false;
	bool dft = false;
	bool use_gpu = false;

	const char* basis_file = NULL;
	const char* pattern_file = NULL;

	struct nufft_conf_s conf = nufft_conf_defaults;
	struct iter_conjgrad_conf cgconf = iter_conjgrad_defaults;

	long coilim_vec[3] = { 0 };

	float lambda = 0.;
	bool precomp = true;

	const struct opt_s opts[] = {

		OPT_SET('a', &adjoint, "adjoint"),
		OPT_SET('i', &inverse, "inverse"),
		OPT_VEC3('x', &coilim_vec, "x:y:z", "dimensions"),
		OPT_VEC3('d', &coilim_vec, "", "(dimensions, deprecated)"),
		OPT_VEC3('D', &coilim_vec, "", "(dimensions, long deprecated)"),
		OPT_SET('t', &conf.toeplitz, "Toeplitz embedding for inverse NUFFT"),
		OPT_CLEAR('r', &conf.toeplitz, "turn-off Toeplitz embedding for inverse NUFFT"),
		OPT_SET('c', &precond, "Preconditioning for inverse NUFFT"),
		OPT_FLOAT('l', &lambda, "lambda", "l2 regularization"),
		OPT_UINT('m', &cgconf.maxiter, "iter", "max. number of iterations (inverse only)"),
		OPT_SET('P', &conf.periodic, "periodic k-space"),
		OPT_SET('s', &dft, "DFT"),
		OPT_SET('g', &use_gpu, "GPU (only inverse)"),
		OPT_CLEAR('1', &conf.decomp, "use/return oversampled grid"),
		OPTL_SET(0, "lowmem", &conf.lowmem, "Use low-mem mode of the nuFFT"),
		OPTL_CLEAR(0, "no-precomp", &precomp, "Use low-low-mem mode of the nuFFT"),
		OPT_INFILE('B', &basis_file, "file", "temporal (or other) basis"),
		OPT_INFILE('p', &pattern_file, "file", "weighting of nufft"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if (!precomp) {

		conf.lowmem = true;
		conf.precomp_linphase = false;
		conf.precomp_fftmod = false;
		conf.precomp_roll = false;
	}

	// avoid computing PSF if not necessary
	if (!inverse)
		conf.toeplitz = false;

	long coilim_dims[DIMS] = { 0 };
	md_copy_dims(3, coilim_dims, coilim_vec);

	// Read trajectory
	long traj_dims[DIMS];
	complex float* traj = load_cfl(traj_file, DIMS, traj_dims);

	assert(3 == traj_dims[0]);

	long coilest_dims[DIMS];

	estimate_im_dims(DIMS, FFT_FLAGS, coilest_dims, traj_dims, traj);

	if (8 >= md_calc_size(3, coilest_dims)) {

		debug_printf(DP_WARN,	"\tThe estimated image size %ldx%ldx%ld is very small.\n"
					"\tDid you scale your trajectory correctly?\n"
					"\tThe unit of measurement is pixel_size / FOV.\n",
					coilest_dims[0], coilest_dims[1], coilest_dims[2]);
	}

	(use_gpu ? num_init_gpu : num_init)();

	long basis_dims[DIMS];
	complex float* basis = NULL;

	if (NULL != basis_file) {

		basis = load_cfl(basis_file, DIMS, basis_dims);
		assert(!md_check_dimensions(DIMS, basis_dims, COEFF_FLAG | TE_FLAG));
	}

	long pattern_dims[DIMS];
	complex float* pattern = NULL;

	if (NULL != pattern_file)
		pattern = load_cfl(pattern_file, DIMS, pattern_dims);


	if (inverse || adjoint) {

		long ksp_dims[DIMS];
		const complex float* ksp = load_cfl(in_file, DIMS, ksp_dims);

		assert(1 == ksp_dims[0]);
		assert(md_check_compat(DIMS, ~(PHS1_FLAG|PHS2_FLAG), ksp_dims, traj_dims));

		if (0 == md_calc_size(3, coilim_dims)) {

			md_copy_dims(DIMS, coilim_dims, coilest_dims);
			debug_printf(DP_INFO, "Est. image size: %ldx%ldx%ld\n", coilim_dims[0], coilim_dims[1], coilim_dims[2]);

			if (!conf.decomp) {

				for (int i = 0; i < DIMS; i++)
					if (MD_IS_SET(FFT_FLAGS, i) && (1 < coilim_dims[i]))
						coilim_dims[i] *= 2;
			}
		}

		md_copy_dims(DIMS - 3, coilim_dims + 3, ksp_dims + 3);

		if (NULL != basis) {

			coilim_dims[COEFF_DIM] = basis_dims[COEFF_DIM];
			coilim_dims[TE_DIM] = 1;
		}

		complex float* img = create_cfl(out_file, DIMS, coilim_dims);

		md_clear(DIMS, coilim_dims, img, CFL_SIZE);

		const struct linop_s* nufft_op;

		if (!dft) {
#ifdef USE_CUDA
			if (use_gpu && !precond && !dft) {

				complex float* traj_gpu = md_gpu_move(DIMS, traj_dims, traj, CFL_SIZE);

				auto tmp = nufft_create2(DIMS, ksp_dims, coilim_dims, traj_dims, traj_gpu, pattern_dims, pattern, basis_dims, basis, conf);
				nufft_op = linop_gpu_wrapper((struct linop_s*)tmp);
				linop_free(tmp);

				md_free(traj_gpu);
			} else 
#endif
				nufft_op = nufft_create2(DIMS, ksp_dims, coilim_dims, traj_dims, traj, pattern_dims, pattern, basis_dims, basis, conf);
		} else
			nufft_op = nudft_create(DIMS, FFT_FLAGS, ksp_dims, coilim_dims, traj_dims, traj);


		if (inverse) {

			const struct operator_s* precond_op = NULL;

			if (conf.toeplitz && precond)
				precond_op = nufft_precond_create(nufft_op);

			struct lsqr_conf lsqr_conf = lsqr_defaults;
			lsqr_conf.lambda = lambda;
			lsqr_conf.it_gpu = use_gpu;

			lsqr(DIMS, &lsqr_conf, iter_conjgrad, CAST_UP(&cgconf),
			     nufft_op, NULL, coilim_dims, img, ksp_dims, ksp, precond_op);

			if (conf.toeplitz && precond)
				operator_free(precond_op);

		} else {

			linop_adjoint(nufft_op, DIMS, coilim_dims, img, DIMS, ksp_dims, ksp);
		}

		linop_free(nufft_op);
		unmap_cfl(DIMS, ksp_dims, ksp);
		unmap_cfl(DIMS, coilim_dims, img);

	} else {

		// Read image data
		const complex float* img = load_cfl(in_file, DIMS, coilim_dims);

		// Initialize kspace data
		long ksp_dims[DIMS];
		md_select_dims(DIMS, PHS1_FLAG|PHS2_FLAG, ksp_dims, traj_dims);
		md_copy_dims(DIMS - 3, ksp_dims + 3, coilim_dims + 3);

		if (NULL != basis) {

			ksp_dims[TE_DIM] = basis_dims[TE_DIM];
			ksp_dims[COEFF_DIM] = 1;
		}

		complex float* ksp = create_cfl(out_file, DIMS, ksp_dims);

		const struct linop_s* nufft_op;

		if (!dft)
			nufft_op = nufft_create2(DIMS, ksp_dims, coilim_dims, traj_dims, traj, pattern_dims, pattern, basis_dims, basis, conf);
		else
			nufft_op = nudft_create(DIMS, FFT_FLAGS, ksp_dims, coilim_dims, traj_dims, traj);

		if (use_gpu) {

			auto tmp = nufft_op;
			nufft_op = linop_gpu_wrapper((struct linop_s*)tmp);
			linop_free(tmp);
		}

		// nufft
		linop_forward(nufft_op, DIMS, ksp_dims, ksp, DIMS, coilim_dims, img);

		linop_free(nufft_op);

		unmap_cfl(DIMS, coilim_dims, img);
		unmap_cfl(DIMS, ksp_dims, ksp);
	}

	unmap_cfl(DIMS, traj_dims, traj);

	if (NULL != basis)
		unmap_cfl(DIMS, basis_dims, basis);

	if (NULL != pattern)
		unmap_cfl(DIMS, pattern_dims, pattern);

	debug_printf(DP_INFO, "Done.\n");

	return 0;
}


