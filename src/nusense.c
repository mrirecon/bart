/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014 Frank Ong <uecker@eecs.berkeley.edu>
 * 2012 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/ops.h"

#include "iter/lsqr.h"

#include "linops/linop.h"

#include "noncart/nufft.h"

#include "sense/model.h"
#include "sense/optcom.h"

#include "wavelet2/wavelet.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"



static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage: %s [-l1/-l2] [-r lambda]  <traj> <kspace> <sensitivities> <output>\n", name);
}

static void help(void)
{
	printf( "\n"
		"Perform non-Cartesian iterative SENSE/ESPIRiT reconstruction.\n"
		"\n"
		"-l1/-l2\t\ttoggle l1-wavelet or l2 regularization.\n"
		"-r lambda\tregularization parameter\n"
#ifdef BERKELEY_SVN
		"-s step\t\titeration stepsize\n"
		"-i maxiter\tnumber of iterations\n"
		"-n \t\tdisable random wavelet cycle spinning\n"
		"-g \t\tuse GPU\n"
		"-p pat\t\tpattern or weights\n"
#endif
	);
}



int main_nusense(int argc, char* argv[])
{
	// Initialize default parameters

	bool l1wav = false;
	bool randshift = true;
	float lambda = 0.;
	unsigned int maxiter = 50;
	float step = 0.9;

	// Start time count

	double start_time = timestamp();
	debug_printf(DP_DEBUG3, "Start Time: %f\n", start_time);

	// Read input options

	bool use_gpu = false;
	bool precond = false;
	const char* pat_file = NULL;
	const char* dcf_file = NULL;
	bool hogwild = false;
	bool toeplitz = true;
	bool stoch = false;
	bool ist = false;

	int c;
	while (-1 != (c = getopt(argc, argv, "Ir:i:l:u:t:p:nhHs:Sd:c"))) {
		switch(c) {

		case 'I':
			ist = true;
			break;


		case 'H':
			hogwild = true;
			break;

		case 's':
			step = atof(optarg);
			break;

		case 'S':
			stoch = true;
			break;


		case 'c':
			precond = true;
			break;

		case 'r':
			lambda = atof(optarg);
			break;

		case 'i':
			maxiter = atoi(optarg);
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

		case 'p':
			pat_file = strdup(optarg);
			break;

		case 'd':
			dcf_file = strdup(optarg);
			break;

		case 'n':
			randshift = false;
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

	if (argc - optind != 4) {

		usage(argv[0], stderr);
		exit(1);
	}

	long map_dims[DIMS];
	long pat_dims[DIMS];
	long img_dims[DIMS];
	long coilim_dims[DIMS];
	long ksp_dims[DIMS];
	long traj_dims[2];


	// load kspace and maps and get dimensions

	complex float* traj = load_cfl(argv[optind + 0], 2, traj_dims);
	complex float* kspace = load_cfl(argv[optind + 1], DIMS, ksp_dims);
	complex float* maps = load_cfl(argv[optind + 2], DIMS, map_dims);

	md_select_dims(DIMS, ~COIL_FLAG, pat_dims, ksp_dims);
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, map_dims);
	md_select_dims(DIMS, ~(MAPS_FLAG), coilim_dims, map_dims);
	assert(1 == ksp_dims[MAPS_DIM]);


	(use_gpu ? num_init_gpu : num_init)();

	// print options

	if (use_gpu)
		debug_printf(DP_INFO, "GPU reconstruction\n");

	if (map_dims[MAPS_DIM] > 1) 
		debug_printf(DP_INFO, "%ld maps.\nESPIRiT reconstruction.\n", map_dims[MAPS_DIM]);

	if (l1wav)
		debug_printf(DP_INFO, "l1-wavelet regularization\n");

	if (hogwild)
		debug_printf(DP_INFO, "Hogwild stepsize\n");

	if (precond)
		debug_printf(DP_INFO, "Circular Preconditioned\n");



	// initialize sampling pattern

	complex float* pattern = NULL;

	if (NULL != pat_file) {

		pattern = load_cfl(pat_file, DIMS, pat_dims);

	} else {

		pattern = md_alloc(DIMS, pat_dims, CFL_SIZE);
		estimate_pattern(DIMS, ksp_dims, COIL_DIM, pattern, kspace);
	}

	// print some statistics

	size_t T = md_calc_size(DIMS, pat_dims);
	long samples = (long)pow(md_znorm(DIMS, pat_dims, pattern), 2.);

	debug_printf(DP_INFO, "Size: %ld Samples: %ld Acc: %.2f\n", T, samples, (float)T / (float)samples); 


	// create image

	complex float* image = create_cfl(argv[optind + 3], DIMS, img_dims);
	md_clear(DIMS, img_dims, image, CFL_SIZE);

	// initialize algorithm

	italgo_fun_t italgo = NULL;
	void* iconf = NULL;

	struct iter_conjgrad_conf cgconf;
	struct iter_fista_conf fsconf;
	struct iter_ist_conf isconf;

	if (!l1wav) {

		memcpy(&cgconf, &iter_conjgrad_defaults, sizeof(struct iter_conjgrad_conf));
		cgconf.maxiter = maxiter;
		cgconf.l2lambda = lambda;

		italgo = iter_conjgrad;
		iconf = &cgconf;

	} else if (ist) {

		memcpy(&isconf, &iter_ist_defaults, sizeof(struct iter_ist_conf));
		isconf.maxiter = maxiter;
		isconf.step = step;
		isconf.hogwild = hogwild;

		italgo = iter_ist;
		iconf = &isconf;
	} else {

		memcpy(&fsconf, &iter_fista_defaults, sizeof(struct iter_fista_conf));
		fsconf.maxiter = maxiter;
		fsconf.step = step;
		fsconf.hogwild = hogwild;

		italgo = iter_fista;
		iconf = &fsconf;
	}

	// initialize fft_op
	const struct linop_s* fft_op
		= nufft_create(ksp_dims, coilim_dims, traj, pattern, toeplitz, precond, stoch, NULL, use_gpu);
	
	// initialize maps_op
	const struct linop_s* maps_op
		= maps2_create(coilim_dims, map_dims, img_dims, maps, use_gpu);

	// initialize forward_op
	const struct linop_s* forward_op = linop_chain(maps_op, fft_op);

	// initialize thresh_op
	const struct operator_p_s* thresh_op = NULL;

	if (l1wav) {

		long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
		minsize[0] = MIN(img_dims[0], 16);
		minsize[1] = MIN(img_dims[1], 16);
		minsize[2] = MIN(img_dims[2], 16);
		thresh_op = prox_wavethresh_create(DIMS, img_dims, 3, minsize, lambda, randshift, use_gpu);
	}

	// apply scaling

	complex float* adj = md_alloc(DIMS, coilim_dims, CFL_SIZE);

	linop_adjoint(fft_op, DIMS, coilim_dims, adj, DIMS, ksp_dims, kspace);
	fftuc(DIMS, coilim_dims, FFT_FLAGS, adj, adj);
		
	float scaling = estimate_scaling(coilim_dims, NULL, adj);

	md_free(adj);

	// initialize image with dcf

	if (NULL != dcf_file)
	{
		long dcf_dims[DIMS];
		long dcf_strs[DIMS];
		complex float* dcf = load_cfl(dcf_file, DIMS, dcf_dims);
		md_calc_strides( DIMS, dcf_strs, dcf_dims, CFL_SIZE );


		long ksp_strs[DIMS];
		md_calc_strides( DIMS, ksp_strs, ksp_dims, CFL_SIZE );
		complex float* ksp_dcf = md_alloc(DIMS, ksp_dims, CFL_SIZE );

		md_zmul2( DIMS, ksp_dims, ksp_strs, ksp_dcf, ksp_strs, kspace, dcf_strs, dcf );

		linop_adjoint( forward_op, DIMS, img_dims, image, DIMS, ksp_dims, ksp_dcf );

		float s = md_znorm(DIMS, ksp_dims, kspace) / md_znorm(DIMS, ksp_dims, ksp_dcf);

		md_zsmul( DIMS, img_dims, image, image, s );

		debug_printf(DP_INFO, "Scaling2: %.2f\n", s); 


		md_free( ksp_dcf );
		unmap_cfl(DIMS, dcf_dims, dcf);
	}

	if (scaling != 0.)
		md_zsmul(DIMS, ksp_dims, kspace, kspace, 1. / scaling);


	struct lsqr_conf lsqr_conf = { 0. };

	// do recon

	if (use_gpu) 
#ifdef USE_CUDA	
		lsqr_gpu(DIMS, &lsqr_conf, italgo, iconf, forward_op, thresh_op, img_dims, image, ksp_dims, kspace);
#else
		assert(0);
#endif
	else
		lsqr(DIMS, &lsqr_conf, italgo, iconf, forward_op, thresh_op, img_dims, image, ksp_dims, kspace);


	// clean up

	if (NULL != pat_file)
		unmap_cfl(DIMS, pat_dims, pattern);
	else
		md_free(pattern);

	unmap_cfl(DIMS, map_dims, maps);
	unmap_cfl(DIMS, ksp_dims, kspace);
	unmap_cfl(DIMS, img_dims, image);
	unmap_cfl(2, traj_dims, traj);

	double end_time = timestamp();

	debug_printf(DP_INFO, "Total Time: %f\n", end_time - start_time);
	exit(0);
}


