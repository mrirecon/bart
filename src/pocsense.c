/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012, 2014 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>
#include <getopt.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/ops.h"
#include "num/iovec.h"
#include "num/someops.h"
#include "num/linop.h"
#include "num/thresh.h"

#include "iter/iter.h"
#include "iter/prox.h"

#include "sense/pocs.h"
#include "sense/optcom.h"

#include "wavelet2/wavelet.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/debug.h"





static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage: %s [-l1/-l2] [-r lambda] <kspace> <sensitivities> <output>\n", name);
}

static void help(void)
{
	printf( "\n"
		"Perform POCSENSE reconstruction.\n"
		"-l1/-l2\ttoggle l1-wavelet or l2 regularization.\n"
		"-r alpha\tregularization parameter\n");
}

	

int main(int argc, char* argv[])
{
	int c;
	float alpha = 0.;
	int maxiter = 50;
	bool l1wav = false;
	float lambda = -1.;
	bool use_gpu = false;
	bool use_admm = false;
	float admm_rho = 0.1;

	while (-1 != (c = getopt(argc, argv, "m:ghi:r:o:l:"))) {
		switch (c) {

		case 'i':
			maxiter = atoi(optarg);
			break;

		case 'r':
			alpha = atof(optarg);
			break;

		case 'l':
			if (1 == atoi(optarg))
				l1wav = true;
			else
			if (2 == atoi(optarg))
				l1wav = false;
			else {
				usage(argv[0], stderr);
				exit(1);
			}
			break;

		case 'g':
			use_gpu = true;
			break;

		case 'o':
			lambda = atof(optarg);
			break;

		case 'm':
			use_admm = true;
			admm_rho = atof(optarg);
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
	
	unsigned int N = DIMS;

	long dims[N];
	long ksp_dims[N];

	complex float* kspace_data = load_cfl(argv[optind + 0], N, ksp_dims);
	complex float* sens_maps = load_cfl(argv[optind + 1], N, dims);


	for (int i = 0; i < 4; i++) {	// sizes2[4] may be > 1
		if (ksp_dims[i] != dims[i]) {
		
			fprintf(stderr, "Dimensions of kspace and sensitivities do not match!\n");
			exit(1);
		}
	}

	assert(1 == ksp_dims[MAPS_DIM]);

	num_init();


	
	long dims1[N];
	
	md_select_dims(N, ~(COIL_FLAG|MAPS_FLAG), dims1, dims);


	// -----------------------------------------------------------
	// memory allocation
	
	complex float* result = create_cfl(argv[optind + 2], N, ksp_dims);
	complex float* pattern = md_alloc(N, dims1, CFL_SIZE);


	// -----------------------------------------------------------
	// pre-process data
	
	float scaling = estimate_scaling(ksp_dims, NULL, kspace_data);
	md_zsmul(N, ksp_dims, kspace_data, kspace_data, 1. / scaling);

	estimate_pattern(N, ksp_dims, COIL_DIM, pattern, kspace_data);


	// -----------------------------------------------------------
	// l1-norm threshold operator
	
	const struct operator_p_s* thresh_op = NULL;
	const struct linop_s* wave_op = NULL;

	if (l1wav) {

		long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
		minsize[0] = MIN(ksp_dims[0], 16);
		minsize[1] = MIN(ksp_dims[1], 16);
		minsize[2] = MIN(ksp_dims[2], 16);

		wave_op = wavelet_create(DIMS, ksp_dims, FFT_FLAGS, minsize, true, use_gpu);
		thresh_op = prox_unithresh_create(DIMS, wave_op, alpha, COIL_FLAG, use_gpu);

	}
#if 0
	else {
		thresh_op = prox_leastsquares_create(DIMS, ksp_dims, alpha, NULL);
	}
#endif

	// -----------------------------------------------------------
	// italgo interface
	
	italgo_fun2_t italgo = NULL;
	void* iconf = NULL;

	struct iter_pocs_conf pconf;
	memcpy(&pconf, &iter_pocs_defaults, sizeof(struct iter_pocs_conf));
	pconf.maxiter = maxiter;

	struct iter_admm_conf mmconf;
	memcpy(&mmconf, &iter_admm_defaults, sizeof(struct iter_admm_conf));
	mmconf.maxiter = maxiter;
	mmconf.rho = admm_rho;

	struct linop_s* eye = linop_identity_create(DIMS, ksp_dims);
	struct linop_s* ops[3] = { eye, eye, eye };
	struct linop_s** ops2 = NULL;

	if (use_admm) {

		italgo = iter2_admm;
		iconf = &mmconf;
		ops2 = ops;

	} else {

		italgo = iter2_pocs;
		iconf = &pconf;
	}


	// -----------------------------------------------------------
	// pocsense recon

	debug_printf(DP_INFO, "Reconstruction...\n");
	
	fftmod(N, ksp_dims, FFT_FLAGS, kspace_data, kspace_data);

	if (use_gpu)
#ifdef USE_CUDA
		pocs_recon_gpu2(italgo, iconf, (const struct linop_s**)ops2, dims, thresh_op, alpha, lambda, result, sens_maps, pattern, kspace_data);
#else
		assert(0);
#endif
	else
		pocs_recon2(italgo, iconf, (const struct linop_s**)ops2, dims, thresh_op, alpha, lambda, result, sens_maps, pattern, kspace_data);

	fftmod(N, ksp_dims, FFT_FLAGS, result, result);


	debug_printf(DP_INFO, "Done.\n");

	md_zsmul(N, ksp_dims, result, result, scaling);

	linop_free(eye);

	md_free(pattern);
	
	if (NULL != thresh_op)
		operator_p_free(thresh_op);

	if (NULL != wave_op)
		linop_free(wave_op);

	unmap_cfl(N, ksp_dims, result);
	unmap_cfl(N, ksp_dims, kspace_data);
	unmap_cfl(N, dims, sens_maps);

	exit(0);
}


