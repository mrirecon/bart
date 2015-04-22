/* Copyright 2014-2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014 Frank Ong <frankong@berkeley.edu> 
 * 2014-2015 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>
#include <getopt.h>
#include <unistd.h>
#include <string.h>

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "linops/linop.h"

#include "iter/iter.h"
#include "iter/lsqr.h"

#include "noncart/nufft.h"




static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage: %s <traj> <input> <output>\n", name);
}


static void help(void)
{
	printf( "\n"
		"Perform non-uniform Fast Fourier Transform.\n"
		"\n"
		"-a\tadjoint\n"
		"-i\tinverse\n"
		"-d x:y:z \tdimensions\n"
		"-t\ttoeplitz\n"
		"-l lambda\tl2 regularization\n"
		"-h\thelp\n");
}




int main_nufft(int argc, char* argv[])
{
	int c;
	bool adjoint = false;
	bool inverse = false;
	bool use_gpu = false;
	bool sizeinit = false;

	struct nufft_conf_s conf = nufft_conf_defaults;
	struct iter_conjgrad_conf cgconf = iter_conjgrad_defaults;

	long coilim_dims[DIMS];
	md_singleton_dims(DIMS, coilim_dims);

	float lambda = 0.;

	while (-1 != (c = getopt(argc, argv, "d:m:l:aiht"))) {

		switch (c) {

		case 'i':
			inverse = true;
			break;

		case 'a':
			adjoint = true;
			break;

		case 'd':
			sscanf(optarg, "%ld:%ld:%ld", &coilim_dims[0], &coilim_dims[1], &coilim_dims[2]);
			sizeinit = true;
			break;

		case 'm':
			cgconf.maxiter = atoi(optarg);
			break;

		case 'l':
			lambda = atof(optarg);
			break;

		case 't':
			conf.toeplitz = true;
			break;

		case 'h':
			usage(argv[0], stdout);
			help();
			exit(0);

		default:
			usage(argv[0], stderr);
			exit(1);
		}
	}

	if (argc - optind != 3) {

		usage(argv[0], stderr);
		exit(1);
	}

	// Read trajectory
	long traj_dims[DIMS];
	complex float* traj = load_cfl(argv[optind + 0], DIMS, traj_dims);

	assert(3 == traj_dims[0]);


	num_init();

	if (inverse || adjoint) {

		long ksp_dims[DIMS];
		const complex float* ksp = load_cfl(argv[optind + 1], DIMS, ksp_dims);

		assert(1 == ksp_dims[0]);
		assert(md_check_compat(DIMS, ~(PHS1_FLAG|PHS2_FLAG), ksp_dims, traj_dims));

		md_copy_dims(DIMS - 3, coilim_dims + 3, ksp_dims + 3);

		if (!sizeinit) {

			estimate_im_dims(DIMS, coilim_dims, traj_dims, traj);
			debug_printf(DP_INFO, "Est. image size: %ld %ld %ld\n", coilim_dims[0], coilim_dims[1], coilim_dims[2]);
		}

		complex float* img = create_cfl(argv[optind + 2], DIMS, coilim_dims);

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
		const complex float* img = load_cfl(argv[optind + 1], DIMS, coilim_dims);
 
		// Initialize kspace data
		long ksp_dims[DIMS];
		md_select_dims(DIMS, PHS1_FLAG|PHS2_FLAG, ksp_dims, traj_dims);
		md_copy_dims(DIMS - 3, ksp_dims + 3, coilim_dims + 3);

		complex float* ksp = create_cfl(argv[optind + 2], DIMS, ksp_dims);

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


