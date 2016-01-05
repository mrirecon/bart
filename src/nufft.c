/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014 Frank Ong <frankong@berkeley.edu>
 * 2014-2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
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

#include "linops/linop.h"

#include "iter/iter.h"
#include "iter/lsqr.h"

#include "noncart/nufft.h"




static const char* usage_str = "<traj> <input> <output>";
static const char* help_str = "Perform non-uniform Fast Fourier Transform.";





int main_nufft(int argc, char* argv[])
{
	bool adjoint = false;
	bool inverse = false;
	bool use_gpu = false;

	struct nufft_conf_s conf = nufft_conf_defaults;
	struct iter_conjgrad_conf cgconf = iter_conjgrad_defaults;

	long coilim_dims[DIMS] = { 0 };

	float lambda = 0.;

	const struct opt_s opts[] = {

		{ 'a', false, opt_set, &adjoint, "\tadjoint" },
		{ 'i', false, opt_set, &inverse, "\tinverse" },
		{ 'd', true, opt_vec3, &coilim_dims, " x:y:z\tdimensions" },
		{ 'D', true, opt_vec3, &coilim_dims, NULL },
		{ 't', false, opt_set, &conf.toeplitz, "\ttoeplitz" },
		{ 'l', true, opt_float, &lambda, " lambda\tl2 regularization" },
		{ 'm', true, opt_int, &cgconf.maxiter, NULL },
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);


	// Read trajectory
	long traj_dims[DIMS];
	complex float* traj = load_cfl(argv[1], DIMS, traj_dims);

	assert(3 == traj_dims[0]);


	num_init();

	if (inverse || adjoint) {

		long ksp_dims[DIMS];
		const complex float* ksp = load_cfl(argv[2], DIMS, ksp_dims);

		assert(1 == ksp_dims[0]);
		assert(md_check_compat(DIMS, ~(PHS1_FLAG|PHS2_FLAG), ksp_dims, traj_dims));

		md_copy_dims(DIMS - 3, coilim_dims + 3, ksp_dims + 3);

		if (0 == md_calc_size(DIMS, coilim_dims)) {

			estimate_im_dims(DIMS, coilim_dims, traj_dims, traj);
			debug_printf(DP_INFO, "Est. image size: %ld %ld %ld\n", coilim_dims[0], coilim_dims[1], coilim_dims[2]);
		}

		complex float* img = create_cfl(argv[3], DIMS, coilim_dims);

		md_clear(DIMS, coilim_dims, img, CFL_SIZE);

		const struct linop_s* nufft_op = nufft_create(DIMS, ksp_dims, coilim_dims, traj_dims, traj, NULL, conf, use_gpu);

		if (inverse) {

			lsqr(DIMS, &(struct lsqr_conf){ lambda }, iter_conjgrad, &cgconf,
				nufft_op, NULL, coilim_dims, img, ksp_dims, ksp);

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

		const struct linop_s* nufft_op = nufft_create(DIMS, ksp_dims, coilim_dims, traj_dims, traj, NULL, conf, use_gpu);

		// nufft
		linop_forward(nufft_op, DIMS, ksp_dims, ksp, DIMS, coilim_dims, img);

		linop_free(nufft_op);
		unmap_cfl(DIMS, coilim_dims, img);
		unmap_cfl(DIMS, ksp_dims, ksp);
	}

	unmap_cfl(DIMS, traj_dims, traj);

	debug_printf(DP_INFO, "Done.\n");
	exit(0);
}


