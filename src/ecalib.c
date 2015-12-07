/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2013, Martin Uecker <uecker@eecs.berkeley.edu>
 * 2013 Dara Bahri <dbahri123@gmail.com>
 * 2015 Siddharth Iyer <sid8795@gmail.com>
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/fft.h"
#include "num/init.h"

#include "calib/calib.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif



static void usage(const char* name, FILE* fd)
{
	fprintf(fd, 	"Usage: %s [-n num. s.values] [-t eigenv. threshold] [-W soft-weight] [-c crop_value] [-k kernel_size] [-r cal_size] [-m maps]"
			" <kspace> <sensitivites> [<ev-maps>]\n", name);
}

static void help(void)
{
	printf( "\n"
		"Estimate coil sensitivities using ESPIRiT calibration.\n"
		"Optionally outputs the eigenvalue maps.\n"
		"\n"
		"-t threshold\tThis determined the size of the null-space.\n"
		"-c crop_value\tCrop the sensitivities if the eigenvalue is smaller than {crop_value}.\n"
		"-k ksize\tkernel size\n"
		"-r cal_size\tLimits the size of the calibration region.\n"
		"-m maps\t\tNumber of maps to compute.\n"
		"-S\t\tcreate maps with smooth transitions (Soft-SENSE).\n"
                "-W\t\tsoft-weighting of the singular vectors.\n"
		"-I\t\tintensity correction\n"
		"-1\t\tperform only first part of the calibration\n");
}



int main_ecalib(int argc, char* argv[])
{
	long calsize[3] = { 24, 24, 24 }; 
	int maps = 2;
	bool one = false;
	bool calcen = false;
	bool print_svals = false;

	struct ecalib_conf conf = ecalib_defaults;

	int c;
	while (-1 != (c = getopt(argc, argv, "OWS1CVIt:p:n:c:k:K:r:R:m:b:h"))) {

		switch (c) {

		case 'I':
			conf.intensity = true;
			break;

		case '1':
			one = true;
			break;

		case 'W':
			conf.numsv      = -1;
			conf.threshold  = 0;
			conf.orthiter   = false;
			conf.weighting  = true;
			break;

		case 'S':
			conf.softcrop = true;
			break;

		case 'O':
			conf.orthiter = false;
			break;

		case 't':
			conf.threshold = atof(optarg);
			break;

		case 'c':
			conf.crop = atof(optarg);
			break;

		case 'p':
			conf.percentsv = atof(optarg);
			conf.threshold = -1.;
			break;

		case 'b':
			conf.perturb = atof(optarg);
			break;

		case 'n':
			conf.numsv = atoi(optarg);
			conf.threshold = -1.;
			break;

		case 'V':
			print_svals = true;
			break;

		case 'k':

			conf.kdims[0] = atoi(optarg);
			conf.kdims[1] = atoi(optarg);
			conf.kdims[2] = atoi(optarg);
			break;

		case 'K':
			sscanf(optarg, "%ld:%ld:%ld", &conf.kdims[0], &conf.kdims[1], &conf.kdims[2]);
			break;

		case 'r':
			calsize[0] = atoi(optarg);
			calsize[1] = atoi(optarg);
			calsize[2] = atoi(optarg);
			break;

		case 'R':
			sscanf(optarg, "%ld:%ld:%ld", &calsize[0], &calsize[1], &calsize[2]);
			break;

		case 'C':
			calcen = true;
			break;

		case 'm':
			maps = atoi(optarg);
			break;

		case 'g':
			conf.usegpu = true;
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

	if ((argc - optind != 3) && (argc - optind != 2)) {

		usage(argv[0], stderr);
		exit(1);
	}

	int N = DIMS;
	long ksp_dims[N];

	complex float* in_data = load_cfl(argv[optind + 0], N, ksp_dims);

	
	// assert((kdims[0] < calsize_ro) && (kdims[1] < calsize_ro) && (kdims[2] < calsize_ro));
	// assert((ksp_dims[0] == 1) || (calsize_ro < ksp_dims[0]));
	assert(1 == ksp_dims[MAPS_DIM]);



	long cal_dims[N];
	complex float* cal_data = NULL;

	 if (!calcen) {

#ifdef USE_CC_EXTRACT_CALIB
		cal_data = cc_extract_calib(cal_dims, calsize, ksp_dims, in_data);
#else
		cal_data = extract_calib(cal_dims, calsize, ksp_dims, in_data, false);
#endif

	} else {
	
		for (int i = 0; i < 3; i++)
			cal_dims[i] = (calsize[i] < ksp_dims[i]) ? calsize[i] : ksp_dims[i];

		for (int i = 3; i < N; i++)
			cal_dims[i] = ksp_dims[i];

		cal_data = md_alloc(5, cal_dims, CFL_SIZE);
		md_resize_center(5, cal_dims, cal_data, ksp_dims, in_data, CFL_SIZE);
	 }



	 for (int i = 0; i < 3; i++)
		 if (1 == ksp_dims[i])
			 conf.kdims[i] = 1;


	 long channels = cal_dims[3];
	 unsigned int K = conf.kdims[0] * conf.kdims[1] * conf.kdims[2] * channels;
	 float svals[K];


	 for (unsigned int i = 0; i < 3; i++) {

		if ((1 == cal_dims[i]) && (1 != ksp_dims[i])) {

			fprintf(stderr, "Calibration region not found!\n");
			exit(1);
		}
	}


	// To reproduce old results turn off rotation of phase.
	// conf.rotphase = false;


	// FIXME: we should scale the data

	(conf.usegpu ? num_init_gpu : num_init)();



	if (one) {

#if 0
		long maps = out_dims[4];

		assert(caldims[3] == out_dims[3]);
		assert(maps <= channels);
#endif
		long cov_dims[4];

		calone_dims(&conf, cov_dims, channels);
		complex float* imgcov = md_alloc(4, cov_dims, CFL_SIZE);


		calone(&conf, cov_dims, imgcov, K, svals, cal_dims, cal_data);

		complex float* out = create_cfl(argv[optind + 1], 4, cov_dims);
		md_copy(4, cov_dims, out, imgcov, CFL_SIZE);
		unmap_cfl(4, cov_dims, out);

//		caltwo(crthr, out_dims, out_data, emaps, cov_dims, imgcov, NULL, NULL);

		md_free(imgcov);

	} else {

		long out_dims[N];
		long map_dims[N];

		for (int i = 0; i < N; i++) {

			out_dims[i] = 1;
			map_dims[i] = 1;

			if ((i < 3) && (1 < conf.kdims[i])) {

				out_dims[i] = ksp_dims[i];
				map_dims[i] = ksp_dims[i];
			}
		}


		assert(maps <= ksp_dims[COIL_DIM]);


		out_dims[COIL_DIM] = ksp_dims[COIL_DIM];
		out_dims[MAPS_DIM] = maps;	
		map_dims[COIL_DIM] = 1;
		map_dims[MAPS_DIM] = maps;

		const char* emaps_file = NULL;

		if (3 == argc - optind)
			emaps_file = argv[optind + 2];

		complex float* out_data = create_cfl(argv[optind + 1], N, out_dims);
		complex float* emaps = (emaps_file ? create_cfl : anon_cfl)(emaps_file, N, map_dims);

		calib(&conf, out_dims, out_data, emaps, K, svals, cal_dims, cal_data); 

		unmap_cfl(N, out_dims, out_data);
		unmap_cfl(N, map_dims, emaps);
	}


	if (print_svals) {

		for (unsigned int i = 0; i < K; i++)
			printf("SVALS %d %f\n", i, svals[i]);
	}

	printf("Done.\n");

	unmap_cfl(N, ksp_dims, in_data);
	md_free(cal_data);

	exit(0);
}


