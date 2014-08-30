/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014 Frank Ong <frankong@berkeley.edu> 
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
#include "num/linop.h"

#include "iter/iter.h"

#include "noncart/nufft.h"



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
		"-i\titerative gridding\n"
		"-t\ttoeplitz\n"
		"-d x:y:z \tdimensions\n"
		"-l lambda\tl2 regularization\n"
		"-h\thelp\n");
}




int main(int argc, char* argv[])
{
	int c;
	_Bool adjoint = false;
	_Bool inverse = false;
	_Bool toeplitz = false;
	_Bool precond = false;
	_Bool use_gpu = false;

	long coilim_dims[DIMS];
	md_singleton_dims(DIMS, coilim_dims);

	int maxiter = 50;
	float lambda = 0.00001;

	const char* pat_str = NULL;

	while (-1 != (c = getopt(argc, argv, "d:m:l:p:aihct"))) {

		switch (c) {

		case 'i':
			inverse = true;
			break;

		case 'a':
			adjoint = true;
			break;

		case 'c':
			precond = true;
			break;

		case 'd':
			sscanf(optarg, "%ld:%ld:%ld", &coilim_dims[0], &coilim_dims[1], &coilim_dims[2]);
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

	// Load pattern / density compensation (if any)
	complex float* pat = NULL;
	long pat_dims[2];
	if (pat_str)
	{
		pat = load_cfl( pat_str, 2, pat_dims );
		assert( pat_dims[0] == 0 );
		assert( pat_dims[1] == traj_dims[1] );
	}



	complex float* dst = NULL;
	complex float* src = NULL;
	long ksp_dims[DIMS];
	const struct linop_s* nufft_op;

	if (adjoint)
	{
		// Read kspace data
		src = load_cfl(argv[optind + 1], DIMS, ksp_dims);

		// Initialize image data
		if ( coilim_dims[0] == 1 && coilim_dims[1] == 1 && coilim_dims[2] == 1)
			estimate_im_dims( coilim_dims, traj_dims, traj );
		coilim_dims[COIL_DIM] = ksp_dims[COIL_DIM];

		dst = create_cfl(argv[optind + 2], DIMS, coilim_dims);
		
		// Get nufft_op
		nufft_op = nufft_create( ksp_dims, coilim_dims, traj, pat, toeplitz, precond, NULL, use_gpu);

		// nufftH
		linop_adjoint (nufft_op, DIMS, coilim_dims, dst, ksp_dims, src);

		linop_free(nufft_op);
		unmap_cfl(DIMS, ksp_dims, src);
		unmap_cfl(DIMS, coilim_dims, dst);

	} else if (inverse)
	{
		// Read kspace data
		src = load_cfl(argv[optind + 1], DIMS, ksp_dims);

		// Initialize image data
		if ( coilim_dims[0] == 1 && coilim_dims[1] == 1 && coilim_dims[2] == 1)
			estimate_im_dims( coilim_dims, traj_dims, traj );
		coilim_dims[COIL_DIM] = ksp_dims[COIL_DIM];

		dst = create_cfl(argv[optind + 2], DIMS, coilim_dims);
		md_clear( DIMS, coilim_dims, dst, CFL_SIZE );

		struct iter_conjgrad_conf cgconf = {.maxiter = maxiter, .l2lambda = 0 };

		// Get nufft_op
		nufft_op = nufft_create( ksp_dims, coilim_dims, traj, pat, toeplitz, precond, &cgconf, use_gpu);

		complex float* adj = md_alloc( DIMS, coilim_dims, CFL_SIZE );
		linop_adjoint( nufft_op, DIMS, coilim_dims, adj, ksp_dims, src );

		// nuifft
		linop_pinverse_unchecked ( nufft_op, lambda, dst, adj );


		md_free( adj );
		linop_free( nufft_op );
		unmap_cfl(DIMS, ksp_dims, src);
		unmap_cfl(DIMS, coilim_dims, dst);

	} else
	{
		// Read image data
		src = load_cfl(argv[optind + 1], DIMS, coilim_dims);

		// Initialize kspace data
		md_select_dims(DIMS, 2 , ksp_dims, traj_dims);
		ksp_dims[COIL_DIM] = coilim_dims[COIL_DIM];
		dst = create_cfl(argv[optind + 2], DIMS, ksp_dims);

		// Get nufft_op
		nufft_op = nufft_create( ksp_dims, coilim_dims, traj, pat, toeplitz, precond, NULL, use_gpu);

		// nufft
		linop_forward(nufft_op, DIMS, ksp_dims, dst, coilim_dims, src);

		linop_free(nufft_op);

		unmap_cfl(DIMS, coilim_dims, src);
		unmap_cfl(DIMS, ksp_dims, dst);
	}

	printf("Done.\n");

	unmap_cfl(2, traj_dims, traj);

	exit(0);
}


