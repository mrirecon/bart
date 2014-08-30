/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdbool.h>
#include <complex.h>
#include <assert.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/gpuops.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "parslices.h"


#ifdef USE_CUDA
#include <omp.h>
#endif



extern bool num_auto_parallelize; // FIXME



void parslices2(grecon_fun_t grecon, void* param, const long dims[DIMS],
	const long img_strs[DIMS], complex float* image,
	const long sens_dims[DIMS], const complex float* sens_maps,
	const long pat_dims[DIMS], const complex float* pattern,
	const long ksp_strs[DIMS], const complex float* kspace_data,	
	bool output_ksp, bool gpu)
{ 
	// dimensions and strides

	int N = DIMS;

	long ksp_dims[N];
	long img_dims[N];

	long strs[N];
	long sens_strs[N];
	long pat_strs[N];


	md_calc_strides(N, strs, dims, CFL_SIZE);
	md_calc_strides(N, sens_strs, sens_dims, CFL_SIZE);
	
	md_select_dims(N, output_ksp ? ~MAPS_FLAG : ~COIL_FLAG, img_dims, dims);
	md_select_dims(N, ~MAPS_FLAG, ksp_dims, dims);

	if (NULL != pattern)
		md_calc_strides(N, pat_strs, pat_dims, CFL_SIZE);

	// dimensions and strides for one slice

	long dims1[N];
	long ksp1_dims[N];
	long img1_dims[N];
	long sens1_dims[N];
	long pat1_dims[N];

	long ksp1_strs[N];
	long img1_strs[N];
	long strs1[N];
	long sens1_strs[N];
	long pat1_strs[N];

	md_select_dims(N, ~READ_FLAG, dims1, dims);
	md_calc_strides(N, strs1, dims1, CFL_SIZE);

	md_select_dims(N, ~READ_FLAG, sens1_dims, sens_dims);
	md_calc_strides(N, sens1_strs, sens1_dims, CFL_SIZE);

	md_select_dims(N, ~READ_FLAG, ksp1_dims, ksp_dims);
	md_calc_strides(N, ksp1_strs, ksp1_dims, CFL_SIZE);

	md_select_dims(N, ~READ_FLAG, img1_dims, img_dims);
	md_calc_strides(N, img1_strs, img1_dims, CFL_SIZE);

	if (NULL != pattern) {

		md_select_dims(N, ~READ_FLAG, pat1_dims, pat_dims);
		md_calc_strides(N, pat1_strs, pat1_dims, CFL_SIZE);
	}

	

//	estimate_pattern(ksp1_dims, 3, pattern, kspace_data + (ksp_dims[0] / 2) * ksp_strs[0]);	// extract pattern form center of readout

	bool ap_save = num_auto_parallelize;
	num_auto_parallelize = false;

	if (gpu) {

#ifdef USE_CUDA
		int nr_cuda_devices = cuda_devices();
		omp_set_num_threads(nr_cuda_devices * 2);
//		fft_set_num_threads(1);
#else
		assert(0);
#endif		

	} else {

		fft_set_num_threads(1);
	}

	int counter = 0;

	#pragma omp parallel for
	for (int i = 0; i < ksp_dims[READ_DIM]; i++) {

		complex float* image1 = md_alloc(N, img1_dims, CFL_SIZE);
		md_clear(N, img1_dims, image1, CFL_SIZE);

		complex float* kspace1 = md_alloc(N, ksp1_dims, CFL_SIZE);
		md_copy2(N, ksp1_dims, ksp1_strs, kspace1, ksp_strs, ((char*)kspace_data) + i * ksp_strs[0], CFL_SIZE);

		complex float* cov1 = md_alloc(N, sens1_dims, CFL_SIZE);
		md_copy2(N, sens1_dims, sens1_strs, cov1, sens_strs, ((char*)sens_maps) + i * sens_strs[0], CFL_SIZE);

		complex float* pattern1 = NULL;

		if (NULL != pattern) {

			pattern1 = md_alloc(N, pat1_dims, CFL_SIZE);
			md_copy2(N, pat1_dims, pat1_strs, pattern1, pat_strs, ((char*)pattern) + i * pat_strs[0], CFL_SIZE);
		}

//		if (i == 96)
		grecon(param, dims1, image1, sens1_dims, cov1, pat1_dims, pattern1, kspace1, gpu);

		md_copy2(N, img1_dims, img_strs, ((char*)image) + i * img_strs[0], img1_strs, image1, CFL_SIZE);

		if (NULL != pattern)
			md_free((void*)pattern1);

		md_free(image1);
		md_free(kspace1);
		md_free(cov1);

		#pragma omp critical
                { debug_printf(DP_DEBUG2, "%04d/%04ld    \r", ++counter, ksp_dims[0]); }
	}

	num_auto_parallelize = ap_save;

	debug_printf(DP_DEBUG2, "\n");
}




void parslices(grecon_fun_t grecon, void* param, const long dims[DIMS],
	complex float* image, const long sens_dims[DIMS], const complex float* sens_maps,
	const long pat_dims[DIMS], const complex float* pattern,
	const complex float* kspace_data, bool output_ksp, bool gpu)
{
	unsigned int N = DIMS;

	long ksp_dims[N];
	long img_dims[N];

	long ksp_strs[N];
	long img_strs[N];
	
	md_select_dims(N, ~MAPS_FLAG, ksp_dims, dims);
	md_calc_strides(N, ksp_strs, ksp_dims, CFL_SIZE);

	md_select_dims(N, output_ksp ? ~MAPS_FLAG : ~COIL_FLAG, img_dims, dims);
	md_calc_strides(N, img_strs, img_dims, CFL_SIZE);

	parslices2(grecon, param, dims, img_strs, image, sens_dims, sens_maps, pat_dims, pattern, ksp_strs, kspace_data, output_ksp, gpu);
}




