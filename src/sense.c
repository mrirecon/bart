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

#include "wavelet2/wavelet.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"



static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage: %s [-l1/-l2] [-r lambda] [-c] <kspace> <sensitivities> <output>\n", name);
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



static void apply_mask(const long dims[DIMS], complex float* sens_maps, float restrict_fov)
{
	long msk_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, msk_dims, dims);

	long msk_strs[DIMS];
	md_calc_strides(DIMS, msk_strs, msk_dims, CFL_SIZE);

	complex float* mask = compute_mask(DIMS, msk_dims, restrict_fov);

	long strs[DIMS];
	md_calc_strides(DIMS, strs, dims, CFL_SIZE);
	md_zmul2(DIMS, dims, strs, sens_maps, strs, sens_maps, msk_strs, mask);
	md_free(mask);
}




int main(int argc, char* argv[])
{
	struct sense_conf conf;
	memcpy(&conf, &sense_defaults, sizeof(struct sense_conf));

	double start_time = timestamp();
	debug_printf(DP_DEBUG3, "Start Time: %f\n", start_time);

	bool admm = false;
	bool ist = false;
	bool usegpu = false;

	float restrict_fov = -1.;
	const char* psf = NULL;
	char image_truth_fname[100];
	bool im_truth = false;
	bool scale_im = false;

	bool hogwild = false;
	bool fast = false;
	float admm_rho = iter_admm_defaults.rho;

	int c;
	while (-1 != (c = getopt(argc, argv, "Fq:l:r:s:i:u:o:O:f:t:cTImghp:Sd:H"))) {
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
			sprintf(image_truth_fname, "%s", optarg);
			break;

		case 'd':
			debug_level = atoi(optarg);
			break;


		case 'r':
			conf.lambda = atof(optarg);
			break;

		case 'O':
			conf.rwiter = atoi(optarg);
			break;

		case 'o':
			conf.gamma = atof(optarg);
			break;

		case 's':
			conf.step = atof(optarg);
			break;

		case 'i':
			conf.maxiter = atoi(optarg);
			break;

		case 'l':
			if (1 == atoi(optarg)) {

				conf.l1wav = true;

			} else if (2 == atoi(optarg)) {

				conf.l1wav = false;

			} else {

				usage(argv[0], stderr);
				exit(1);
			}
			break;

		case 'q':
			conf.ccrobust = true;
			conf.cclambda = atof(optarg);
			break;

		case 'c':
			conf.rvc = true;
			break;

		case 'f':
			restrict_fov = atof(optarg);
			break;

		case 'm':
			admm = 1;
			break;

		case 'u':
			admm_rho = atof(optarg);
			break;

		case 'g':
			usegpu = true;
			break;

		case 'p':
			psf = strdup(optarg);
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

	int N = DIMS;

	long dims[N];
	long pat_dims[N];
	long img_dims[N];
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


	(usegpu ? num_init_gpu : num_init)();

	if (dims[MAPS_DIM] > 1) 
		debug_printf(DP_INFO, "%ld maps.\nESPIRiT reconstruction.\n", dims[4]);

	if (conf.l1wav)
		debug_printf(DP_INFO, "l1-wavelet regularization\n");

	if (ist)
		debug_printf(DP_INFO, "Use IST\n");

	if (im_truth)
		debug_printf(DP_INFO, "Compare to truth\n");


	md_select_dims(N, ~(COIL_FLAG | MAPS_FLAG), pat_dims, ksp_dims);
	md_select_dims(N, ~COIL_FLAG, img_dims, dims);



	// initialize sampling pattern

	complex float* pattern = NULL;
	long pat_dims2[N];

	if (NULL != psf) {

		pattern = load_cfl(psf, N, pat_dims2);

		// FIXME: check compatibility
	} else {

		pattern = md_alloc(N, pat_dims, CFL_SIZE);
		estimate_pattern(N, ksp_dims, COIL_DIM, pattern, kspace_data);
	}


	
	// print some statistics

	size_t T = md_calc_size(N, pat_dims);
	long samples = (long)pow(md_znorm(N, pat_dims, pattern), 2.);
	debug_printf(DP_INFO, "Size: %ld Samples: %ld Acc: %.2f\n", T, samples, (float)T / (float)samples); 

	fftmod(N, ksp_dims, FFT_FLAGS, kspace_data, kspace_data);
	fftmod(N, dims, FFT_FLAGS, sens_maps, sens_maps);


	// apply scaling

	float scaling = estimate_scaling(ksp_dims, NULL, kspace_data);

	if (scaling != 0.)
		md_zsmul(N, ksp_dims, kspace_data, kspace_data, 1. / scaling);


	// apply fov mask to sensitivities

	if (-1. != restrict_fov)
		apply_mask(dims, sens_maps, restrict_fov);


	const struct operator_p_s* thresh_op = NULL;

	if (conf.l1wav) {

		long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
		minsize[0] = MIN(img_dims[0], 16);
		minsize[1] = MIN(img_dims[1], 16);
		minsize[2] = MIN(img_dims[2], 16);

		thresh_op = prox_wavethresh_create(DIMS, img_dims, FFT_FLAGS, minsize, conf.lambda, conf.randshift, usegpu);
	}





	complex float* image = create_cfl(argv[optind + 2], N, img_dims);
	md_clear(N, img_dims, image, CFL_SIZE);

	long img_truth_dims[DIMS];
	complex float* image_truth = NULL;

	if (im_truth) {

		image_truth = load_cfl(image_truth_fname, DIMS, img_truth_dims);
		//md_zsmul(DIMS, img_dims, image_truth, image_truth, 1. / scaling);
	}


	italgo_fun_t italgo = NULL;
	void* iconf = NULL;

	// FIXME: change named initializers to traditional?
	struct iter_conjgrad_conf cgconf;
	memcpy(&cgconf, &iter_conjgrad_defaults, sizeof(struct iter_conjgrad_conf) );
	cgconf.maxiter = conf.maxiter;
	cgconf.l2lambda = conf.lambda;

	struct iter_fista_conf fsconf;
	memcpy(&fsconf, &iter_fista_defaults, sizeof(struct iter_fista_conf) );
	fsconf.maxiter = conf.maxiter;
	fsconf.step = conf.step;
	fsconf.hogwild = hogwild;

	struct iter_ist_conf isconf;
	memcpy(&isconf, &iter_ist_defaults, sizeof(struct iter_ist_conf) );
	isconf.maxiter = conf.maxiter;
	isconf.step = conf.step;
	isconf.hogwild = hogwild;


	struct iter_admm_conf mmconf;
	memcpy(&mmconf, &iter_admm_defaults, sizeof(struct iter_admm_conf));
	mmconf.maxiter = conf.maxiter;
	mmconf.rho = admm_rho;
	mmconf.hogwild = hogwild;
	mmconf.fast = fast;

	if (!conf.l1wav) {

		italgo = iter_conjgrad;
		iconf = &cgconf;

	} else if (admm) {

		italgo = iter_admm;
		iconf = &mmconf;

	} else if (ist) {

		italgo = iter_ist;
		iconf = &isconf;

	} else {

		italgo = iter_fista;
		iconf = &fsconf;
	}


	if (usegpu) 
#ifdef USE_CUDA
		sense_recon_gpu(&conf, dims, image, sens_maps, pat_dims, pattern, italgo, iconf, thresh_op, ksp_dims, kspace_data, image_truth);
#else
		assert(0);
#endif
	else
		sense_recon(&conf, dims, image, sens_maps, pat_dims, pattern, italgo, iconf, thresh_op, ksp_dims, kspace_data, image_truth);

	if (scale_im)
		md_zsmul(N, img_dims, image, image, scaling);

	if (NULL != psf)
		unmap_cfl(N, pat_dims2, pattern);
	else
		md_free(pattern);


	unmap_cfl(N, dims, sens_maps);
	unmap_cfl(N, ksp_dims, kspace_data);
	unmap_cfl(N, img_dims, image);

	double end_time = timestamp();
	debug_printf(DP_INFO, "Total Time: %f\n", end_time - start_time);
	exit(0);
}


