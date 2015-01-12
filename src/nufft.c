/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014 Frank Ong <frankong@berkeley.edu> 
 * 2014 Martin Uecker <uecker@eecs.berkeley.edu>
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

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "linops/linop.h"

#include "iter/iter.h"

#include "noncart/nufft.h"
#include "num/fft.h"

#ifdef BERKELEY_SVN
#include "noncart/nufft2.h"
#endif



static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage: %s <traj> <input> <output>\n", name);
}


static void help(void)
{
	printf( "\n"
		"Performs non-uniform Fast Fourier Transform.\n"
		"\n"
		"-a\tadjoint\n"
		"-i\tinverse\n"
		"-c x:y:z \tmake grid calibration region\n"
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
	bool toeplitz = false;
	bool precond = false;
	bool use_gpu = false;
	bool two = false;
	bool calib = false;
	bool sizeinit = false;
	bool stoch = false;

	long coilim_dims[DIMS];
	md_singleton_dims(DIMS, coilim_dims);

	int maxiter = 50;
	float lambda = 0.00;

	const char* pat_str = NULL;

	while (-1 != (c = getopt(argc, argv, "d:m:l:p:aihCto:w:2:c:S"))) {

		switch (c) {

		case '2':
			two = true;
			break;

		case 'i':
			inverse = true;
			break;

		case 'a':
			adjoint = true;
			break;

		case 'C':
			precond = true;
			break;

		case 'S':
			stoch = true;
			break;

		case 'c':
			calib = true;
			inverse = true;
		case 'd':
			sscanf(optarg, "%ld:%ld:%ld", &coilim_dims[0], &coilim_dims[1], &coilim_dims[2]);
			sizeinit = true;
			break;

		case 'm':
			maxiter = atoi(optarg);
			break;

		case 'p':
			pat_str = strdup(optarg);
			break;

		case 'l':
			lambda = atof(optarg);
			break;

		case 't':
			toeplitz = true;
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
	long traj_dims[2];
	complex float* traj = load_cfl(argv[optind + 0], 2, traj_dims);

	assert(3 == traj_dims[0]);

	if (!sizeinit)
		estimate_im_dims(coilim_dims, traj_dims, traj);

	num_init();

	// Load pattern / density compensation (if any)
	complex float* pat = NULL;
	long pat_dims[2];

	if (pat_str) {

		pat = load_cfl(pat_str, 2, pat_dims);
		assert(pat_dims[0] == 1);
		assert(pat_dims[1] == traj_dims[1]);
	}

	if (inverse || adjoint) {

		long ksp_dims[DIMS];
		const complex float* ksp = load_cfl(argv[optind + 1], DIMS, ksp_dims);

		coilim_dims[COIL_DIM] = ksp_dims[COIL_DIM];

		long out_dims[DIMS];

		if (calib) {

			md_singleton_dims(DIMS, out_dims);
			estimate_im_dims(out_dims, traj_dims, traj);
			out_dims[COIL_DIM] = ksp_dims[COIL_DIM];

		} else {

			md_copy_dims(DIMS, out_dims, coilim_dims);
		}

		complex float* out = create_cfl(argv[optind + 2], DIMS, out_dims);
		complex float* img = out;

		if (calib)
			img = md_alloc(DIMS, coilim_dims, CFL_SIZE);

		md_clear(DIMS, coilim_dims, img, CFL_SIZE);

		struct iter_conjgrad_conf cgconf = iter_conjgrad_defaults;
		cgconf.maxiter = maxiter;
		cgconf.l2lambda = 0.;
		cgconf.tol = 0;

		const struct linop_s* nufft_op;

		// Get nufft_op
		if (two)
#ifdef BERKELEY_SVN
			nufft_op = nufft2_create(ksp_dims, coilim_dims, traj, pat, toeplitz, precond, &cgconf, use_gpu);
#else
			assert(!two);
#endif
		else
			nufft_op = nufft_create(ksp_dims, coilim_dims, traj, pat, toeplitz, precond, stoch, &cgconf, use_gpu);

		if (inverse) {

			linop_pseudo_inv(nufft_op, lambda, DIMS, coilim_dims, img, DIMS, ksp_dims, ksp);

		} else {

			linop_adjoint(nufft_op, DIMS, coilim_dims, img, DIMS, ksp_dims, ksp);
		}

		if (calib) {

			fftc(DIMS, coilim_dims, FFT_FLAGS, img, img);
			md_resizec(DIMS, out_dims, out, coilim_dims, img, CFL_SIZE);
			md_free(img);
		}

		linop_free(nufft_op);
		unmap_cfl(DIMS, ksp_dims, ksp);
		unmap_cfl(DIMS, out_dims, out);

	} else {

		// Read image data
		const complex float* img = load_cfl(argv[optind + 1], DIMS, coilim_dims);
 
		// Initialize kspace data
		long ksp_dims[DIMS];
		md_select_dims(DIMS, 2, ksp_dims, traj_dims);
		ksp_dims[COIL_DIM] = coilim_dims[COIL_DIM];
		complex float* ksp = create_cfl(argv[optind + 2], DIMS, ksp_dims);

		const struct linop_s* nufft_op;

		// Get nufft_op
		if (two)
#ifdef BERKELEY_SVN
			nufft_op = nufft2_create(ksp_dims, coilim_dims, traj, pat, toeplitz, precond, NULL, use_gpu);
#else
			assert(!two);
#endif
		else
			nufft_op = nufft_create(ksp_dims, coilim_dims, traj, pat, toeplitz, precond, stoch, NULL, use_gpu);

		// nufft
		linop_forward(nufft_op, DIMS, ksp_dims, ksp, DIMS, coilim_dims, img);

		linop_free(nufft_op);
		unmap_cfl(DIMS, coilim_dims, img);
		unmap_cfl(DIMS, ksp_dims, ksp);
	}


	unmap_cfl(2, traj_dims, traj);

	printf("Done.\n");
	exit(0);
}


