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


static const char usage_str[] = "<traj> <input> <output>";
static const char help_str[] = "Perform non-uniform Fast Fourier Transform.";





int main_nufft(int argc, char* argv[])
{
	bool adjoint = false;
	bool inverse = false;
	bool precond = false;
	bool dft = false;
	bool gpu = false;

	struct nufft_conf_s conf = nufft_conf_defaults;
	struct iter_conjgrad_conf cgconf = iter_conjgrad_defaults;

	long coilim_vec[3] = { 0 };

	float lambda = 0.;

	const struct opt_s opts[] = {

		OPT_SET('a', &adjoint, "adjoint"),
		OPT_SET('i', &inverse, "inverse"),
		OPT_VEC3('d', &coilim_vec, "x:y:z", "dimensions"),
		OPT_VEC3('D', &coilim_vec, "", "()"),
		OPT_SET('t', &conf.toeplitz, "Toeplitz embedding for inverse NUFFT"),
		OPT_CLEAR('r', &conf.toeplitz, "turn-off Toeplitz embedding for inverse NUFFT"),
		OPT_SET('c', &precond, "Preconditioning for inverse NUFFT"),
		OPT_FLOAT('l', &lambda, "lambda", "l2 regularization"),
		OPT_UINT('m', &cgconf.maxiter, "", "()"),
		OPT_SET('P', &conf.periodic, "periodic k-space"),
		OPT_SET('s', &dft, "DFT"),
		OPT_SET('g', &gpu, "GPU (only inverse)"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	// avoid computing PSF if not necessary
	if (!inverse)
		conf.toeplitz = false;


	long coilim_dims[DIMS] = { 0 };
	md_copy_dims(3, coilim_dims, coilim_vec);

	// Read trajectory
	long traj_dims[DIMS];
	complex float* traj = load_cfl(argv[1], DIMS, traj_dims);

	assert(3 == traj_dims[0]);


	(gpu ? num_init_gpu : num_init)();

	if (inverse || adjoint) {

		long ksp_dims[DIMS];
		const complex float* ksp = load_cfl(argv[2], DIMS, ksp_dims);

		assert(1 == ksp_dims[0]);
		assert(md_check_compat(DIMS, ~(PHS1_FLAG|PHS2_FLAG), ksp_dims, traj_dims));

		if (0 == md_calc_size(3, coilim_dims)) {

			estimate_im_dims(DIMS, FFT_FLAGS, coilim_dims, traj_dims, traj);
			debug_printf(DP_INFO, "Est. image size: %ld %ld %ld\n", coilim_dims[0], coilim_dims[1], coilim_dims[2]);
		}

		md_copy_dims(DIMS - 3, coilim_dims + 3, ksp_dims + 3);

		complex float* img = create_cfl(argv[3], DIMS, coilim_dims);

		md_clear(DIMS, coilim_dims, img, CFL_SIZE);

		const struct linop_s* nufft_op;

		if (!dft)
			nufft_op = nufft_create(DIMS, ksp_dims, coilim_dims, traj_dims, traj, NULL, conf);
		else
			nufft_op = nudft_create(DIMS, FFT_FLAGS, ksp_dims, coilim_dims, traj_dims, traj);


		if (inverse) {

			const struct operator_s* precond_op = NULL;

			if (conf.toeplitz && precond)
				precond_op = nufft_precond_create(nufft_op);

			lsqr(DIMS, &(struct lsqr_conf){ lambda, gpu }, iter_conjgrad, CAST_UP(&cgconf),
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
		const complex float* img = load_cfl(argv[2], DIMS, coilim_dims);

		// Initialize kspace data
		long ksp_dims[DIMS];
		md_select_dims(DIMS, PHS1_FLAG|PHS2_FLAG, ksp_dims, traj_dims);
		md_copy_dims(DIMS - 3, ksp_dims + 3, coilim_dims + 3);

		complex float* ksp = create_cfl(argv[3], DIMS, ksp_dims);

		const struct linop_s* nufft_op;

		if (!dft)
			nufft_op = nufft_create(DIMS, ksp_dims, coilim_dims, traj_dims, traj, NULL, conf);
		else
			nufft_op = nudft_create(DIMS, FFT_FLAGS, ksp_dims, coilim_dims, traj_dims, traj);

		// nufft
		linop_forward(nufft_op, DIMS, ksp_dims, ksp, DIMS, coilim_dims, img);

		linop_free(nufft_op);
		unmap_cfl(DIMS, coilim_dims, img);
		unmap_cfl(DIMS, ksp_dims, ksp);
	}

	unmap_cfl(DIMS, traj_dims, traj);

	debug_printf(DP_INFO, "Done.\n");
	return 0;
}


