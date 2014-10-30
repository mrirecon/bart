/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2014 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2014 Frank Ong <frankong@berkeley.edu>
 * 2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
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

#include "iter/iter.h"

#include "sense/recon.h"
#include "sense/optcom.h"

// #define W3
#ifndef W3
#include "wavelet2/wavelet.h"
#else
#include "wavelet3/wavthresh.h"
#endif

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/utils.h"
#include "misc/mmio.h"
#include "misc/misc.h"



static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage: %s [-l1/-l2] [-r lambda]  <kspace> <sensitivities> <output>\n", name);
}

static void help(void)
{
	printf( "\n"
		"Perform iterative SENSE/ESPIRiT reconstruction.\n"
		"\n"
		"-l1/-l2\ttoggle l1-wavelet or l2 regularization.\n"
		"-r lambda\tregularization parameter\n"
		"-c\treal-value constraint\n");
}






int main_sense(int argc, char* argv[])
{
	struct sense_conf conf;
	memcpy(&conf, &sense_defaults, sizeof(struct sense_conf));

	double start_time = timestamp();

	bool admm = false;
	bool ist = false;
	bool use_gpu = false;
	bool l1wav = false;
	bool randshift = true;
	int maxiter = 30;
	float step = 0.95;
	float lambda = 0.;

	float restrict_fov = -1.;
	const char* pat_file = NULL;
	const char* traj_file = NULL;
	const char* image_truth_file = NULL;
	bool im_truth = false;
	bool scale_im = false;

	bool hogwild = false;
	bool fast = false;
	float admm_rho = iter_admm_defaults.rho;

	int c;
	while (-1 != (c = getopt(argc, argv, "Fq:l:r:s:i:u:o:O:f:t:cT:Imghp:Sd:H"))) {
		switch(c) {

		case 'H':
			hogwild = true;
			break;

		case 'F':
			fast = true;
			break;

		case 'I':
			ist = true;
			break;

		case 'T':
			im_truth = true;
			image_truth_file = strdup(optarg);
			assert(NULL != image_truth_file);
			break;

		case 'd':
			debug_level = atoi(optarg);
			break;

		case 'r':
			lambda = atof(optarg);
			break;

		case 'O':
			conf.rwiter = atoi(optarg);
			break;

		case 'o':
			conf.gamma = atof(optarg);
			break;

		case 's':
			step = atof(optarg);
			break;

		case 'i':
			maxiter = atoi(optarg);
			break;

		case 'l':
			if (1 == atoi(optarg)) {

				l1wav = true;

			} else
			if (2 == atoi(optarg)) {

				l1wav = false;

			} else {

				usage(argv[0], stderr);
				exit(1);
			}
			break;

		case 'q':
			conf.cclambda = atof(optarg);
			break;

		case 'c':
			conf.rvc = true;
			break;

		case 'f':
			restrict_fov = atof(optarg);
			break;

		case 'm':
			admm = true;
			break;

		case 'u':
			admm_rho = atof(optarg);
			break;

		case 'g':
			use_gpu = true;
			break;

		case 'p':
			pat_file = strdup(optarg);
			break;

		case 't':
			assert(0);
			break;

		case 'S':
			scale_im = true;
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

	long map_dims[DIMS];
	long pat_dims[DIMS];
	long img_dims[DIMS];
	long ksp_dims[DIMS];
	long max_dims[DIMS];


	// load kspace and maps and get dimensions

	complex float* kspace = load_cfl(argv[optind + 0], DIMS, ksp_dims);
	complex float* maps = load_cfl(argv[optind + 1], DIMS, map_dims);

	md_copy_dims(DIMS, max_dims, ksp_dims);
	max_dims[MAPS_DIM] = map_dims[MAPS_DIM];

	md_select_dims(DIMS, ~COIL_FLAG, pat_dims, ksp_dims);
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, max_dims);

	for (int i = 0; i < 4; i++) {	// sizes2[4] may be > 1
		if (ksp_dims[i] != map_dims[i]) {
		
			fprintf(stderr, "Dimensions of kspace and sensitivities do not match!\n");
			exit(1);
		}
	}


	assert(1 == ksp_dims[MAPS_DIM]);

	(use_gpu ? num_init_gpu : num_init)();

	// print options

	if (use_gpu)
		debug_printf(DP_INFO, "GPU reconstruction\n");

	if (map_dims[MAPS_DIM] > 1) 
		debug_printf(DP_INFO, "%ld maps.\nESPIRiT reconstruction.\n", map_dims[MAPS_DIM]);

	if (l1wav)
		debug_printf(DP_INFO, "l1-wavelet regularization\n");

	if (ist)
		debug_printf(DP_INFO, "Use IST\n");

	if (im_truth)
		debug_printf(DP_INFO, "Compare to truth\n");



	// initialize sampling pattern

	complex float* pattern = NULL;
	long pat_dims2[DIMS];

	if (NULL != pat_file) {

		pattern = load_cfl(pat_file, DIMS, pat_dims2);

		// FIXME: check compatibility
	} else {

		pattern = md_alloc(DIMS, pat_dims, CFL_SIZE);
		estimate_pattern(DIMS, ksp_dims, COIL_DIM, pattern, kspace);
	}


	
	// print some statistics

	size_t T = md_calc_size(DIMS, pat_dims);
	long samples = (long)pow(md_znorm(DIMS, pat_dims, pattern), 2.);
	debug_printf(DP_INFO, "Size: %ld Samples: %ld Acc: %.2f\n", T, samples, (float)T / (float)samples); 

	fftmod(DIMS, ksp_dims, FFT_FLAGS, kspace, kspace);
	fftmod(DIMS, map_dims, FFT_FLAGS, maps, maps);


	// apply fov mask to sensitivities

	if (-1. != restrict_fov) {

		float restrict_dims[DIMS] = { [0 ... DIMS - 1] = 1. };
		restrict_dims[0] = restrict_fov;
		restrict_dims[1] = restrict_fov;
		restrict_dims[2] = restrict_fov;

		apply_mask(DIMS, map_dims, maps, restrict_dims);
	}


	// apply scaling

	float scaling = 1.;

	if (NULL == traj_file) {

		scaling = estimate_scaling(ksp_dims, NULL, kspace);

		if (scaling != 0.)
			md_zsmul(DIMS, ksp_dims, kspace, kspace, 1. / scaling);
	}


	const struct operator_p_s* thresh_op = NULL;

	if (l1wav) {

		long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
		minsize[0] = MIN(img_dims[0], 16);
		minsize[1] = MIN(img_dims[1], 16);
		minsize[2] = MIN(img_dims[2], 16);
#ifndef W3
		thresh_op = prox_wavethresh_create(DIMS, img_dims, FFT_FLAGS, minsize, lambda, randshift, use_gpu);
#else
		unsigned int wflags = 0;
		for (unsigned int i = 0; i < 3; i++)
			if (1 < img_dims[i])
				wflags |= (1 << i);

		thresh_op = prox_wavelet3_thresh_create(DIMS, img_dims, wflags, minsize, lambda, randshift);
#endif
	}



	complex float* image = create_cfl(argv[optind + 2], DIMS, img_dims);
	md_clear(DIMS, img_dims, image, CFL_SIZE);


	long img_truth_dims[DIMS];
	complex float* image_truth = NULL;

	if (im_truth) {

		image_truth = load_cfl(image_truth_file, DIMS, img_truth_dims);
		//md_zsmul(DIMS, img_dims, image_truth, image_truth, 1. / scaling);
	}


	italgo_fun_t italgo = NULL;
	void* iconf = NULL;

	// FIXME: change named initializers to traditional?
	struct iter_conjgrad_conf cgconf;
	struct iter_fista_conf fsconf;
	struct iter_ist_conf isconf;
	struct iter_admm_conf mmconf;

	if (!l1wav) {

		memcpy(&cgconf, &iter_conjgrad_defaults, sizeof(struct iter_conjgrad_conf));
		cgconf.maxiter = maxiter;
		cgconf.l2lambda = lambda;

		italgo = iter_conjgrad;
		iconf = &cgconf;

	} else if (admm) {

		memcpy(&mmconf, &iter_admm_defaults, sizeof(struct iter_admm_conf));
		mmconf.maxiter = maxiter;
		mmconf.rho = admm_rho;
		mmconf.hogwild = hogwild;
		mmconf.fast = fast;

		italgo = iter_admm;
		iconf = &mmconf;

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


	if (use_gpu) 
#ifdef USE_CUDA
		sense_recon_gpu(&conf, max_dims, image, maps, pat_dims, pattern, italgo, iconf, thresh_op, ksp_dims, kspace, image_truth);
#else
		assert(0);
#endif
	else
		sense_recon(&conf, max_dims, image, maps, pat_dims, pattern, italgo, iconf, thresh_op, ksp_dims, kspace, image_truth);

	if (scale_im)
		md_zsmul(DIMS, img_dims, image, image, scaling);

	if (NULL != pat_file)
		unmap_cfl(DIMS, pat_dims2, pattern);
	else
		md_free(pattern);


	unmap_cfl(DIMS, map_dims, maps);
	unmap_cfl(DIMS, ksp_dims, kspace);
	unmap_cfl(DIMS, img_dims, image);

	if (im_truth) {

		free((void*)image_truth_file);
		unmap_cfl(DIMS, img_dims, image_truth);
	}


	double end_time = timestamp();
	debug_printf(DP_INFO, "Total Time: %f\n", end_time - start_time);
	exit(0);
}


