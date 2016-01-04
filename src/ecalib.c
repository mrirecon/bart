/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2013 Dara Bahri <dbahri123@gmail.com>
 * 2015 Siddharth Iyer <sid8795@gmail.com>
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <stdio.h>

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/opts.h"

#include "num/multind.h"
#include "num/fft.h"
#include "num/init.h"

#include "calib/calib.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif



static const char* usage_str = "<kspace> <sensitivites> [<ev-maps>]";
static const char* help_str =
		"Estimate coil sensitivities using ESPIRiT calibration.\n"
		"Optionally outputs the eigenvalue maps.";





int main_ecalib(int argc, char* argv[])
{
	long calsize[3] = { 24, 24, 24 }; 
	int maps = 2;
	bool one = false;
	bool calcen = false;
	bool print_svals = false;

	struct ecalib_conf conf = ecalib_defaults;

	const struct opt_s opts[] = {

		{ 't', true, opt_float, &conf.threshold, " threshold\tThis determined the size of the null-space." },
		{ 'c', true, opt_float, &conf.crop, " crop_value\tCrop the sensitivities if the eigenvalue is smaller than {crop_value}." },
//		{ 'k', true, opt_vec3, &conf.kdims, " ksize\tkernel size" },
		OPT_VEC3('k', &conf.kdims, "ksize", "kernel size"),
		{ 'K', true, opt_vec3, &conf.kdims, NULL },
		{ 'r', true, opt_vec3, &calsize, " cal_size\tLimits the size of the calibration region." },
		{ 'R', true, opt_vec3, &calsize, NULL },
		{ 'm', true, opt_int, &maps, " maps\t\tNumber of maps to compute." },
		{ 'S', false, opt_set, &conf.softcrop, "\t\tcreate maps with smooth transitions (Soft-SENSE)." },
		{ 'W', false, opt_set, &conf.weighting, "\t\tsoft-weighting of the singular vectors." },
		{ 'I', false, opt_set, &conf.intensity, "\t\tintensity correction" },
		{ '1', false, opt_set, &one, "\t\tperform only first part of the calibration" },
		{ 'O', false, opt_clear, &conf.orthiter, NULL },
		{ 'b', true, opt_float, &conf.perturb, NULL },
		{ 'V', false, opt_set, &print_svals, NULL },
		{ 'C', false, opt_set, &calcen, NULL },
		{ 'm', true, opt_int, &maps, NULL },
		{ 'g', false, opt_set, &conf.usegpu, NULL },
		{ 'p', true, opt_float, &conf.percentsv, NULL },
		{ 'n', true, opt_int, &conf.numsv, NULL },
	};

	cmdline(&argc, argv, 2, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (conf.weighting) {

		conf.numsv      = -1;
		conf.threshold  = 0;
		conf.orthiter   = false;
	}

	if (-1. != conf.percentsv)
		conf.threshold = -1.;

	if (-1 != conf.numsv)
		conf.threshold = -1.;



	int N = DIMS;
	long ksp_dims[N];

	complex float* in_data = load_cfl(argv[1], N, ksp_dims);

	
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


	 for (unsigned int i = 0; i < 3; i++)
		if ((1 == cal_dims[i]) && (1 != ksp_dims[i]))
			error("Calibration region not found!\n");


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

		complex float* out = create_cfl(argv[2], 4, cov_dims);
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

		if (4 == argc)
			emaps_file = argv[3];

		complex float* out_data = create_cfl(argv[2], N, out_dims);
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


