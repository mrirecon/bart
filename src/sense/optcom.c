/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 * 2012 Martin Uecker <uecker@eecs.berkeley.edu>
 */
 
#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include "sense/model.h"
#include "linops/linop.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "optcom.h"



/** 
 * Optimally combine coil images.
 *
 * Assumptions:
 * @param data fully sampled data
 * @param sens physical (unormalized) coil sensitivites
 * @param alpha the best estimator has alpha > 0.
 */ 
void optimal_combine(const long dims[DIMS], float alpha, complex float* image, const complex float* sens, const complex float* data)
{
	long dims_one[DIMS];
	long dims_img[DIMS];
	long dims_cim[DIMS];

	md_select_dims(DIMS, ~(COIL_FLAG|MAPS_FLAG), dims_one, dims);
	md_select_dims(DIMS, ~(COIL_FLAG), dims_img, dims);
	md_select_dims(DIMS, ~(MAPS_FLAG), dims_cim, dims);

	const struct linop_s* sense_data = sense_init(dims, FFT_FLAGS|COIL_FLAG|MAPS_FLAG, sens);
	linop_adjoint(sense_data, DIMS, dims_img, image, DIMS, dims_cim, data);
	linop_free(sense_data);

	complex float* norm = md_alloc(DIMS, dims_img, CFL_SIZE);
	md_zrss(DIMS, dims, COIL_FLAG, norm, sens);
	
	long imsize = md_calc_size(DIMS, dims_img);

	for (unsigned int i = 0; i < imsize; i++)
		image[i] /= (powf(cabsf(norm[i]), 2.) + alpha);

	md_free(norm);	
}


void rss_combine(const long dims[DIMS], complex float* image, const complex float* data)
{
	complex float* tmp = md_alloc_sameplace(DIMS, dims, CFL_SIZE, data);

	ifft(DIMS, dims, FFT_FLAGS, tmp, data);
	fftscale(DIMS, dims, FFT_FLAGS, tmp, tmp);
	md_zrss(DIMS, dims, COIL_FLAG, image, tmp);

	md_free(tmp);
}



static int compare_cmpl_magn(const void* a, const void* b)
{
	return (int)copysignf(1., (cabsf(*(complex float*)a) - cabsf(*(complex float*)b)));
}


float estimate_scaling_norm(float rescale, long imsize, complex float* tmpnorm, bool compat)
{
	qsort(tmpnorm, (size_t)imsize, sizeof(complex float), compare_cmpl_magn);

	float median = cabsf(tmpnorm[imsize / 2]) / rescale; //median
	float p90 = cabsf(tmpnorm[(int)trunc(imsize * 0.9)]) / rescale;
	float max = cabsf(tmpnorm[imsize - 1]) / rescale;

	float scale = ((max - p90) < 2 * (p90 - median)) ? p90 : max;

	if (compat)
		scale = median;

	debug_printf(DP_DEBUG1, "Scaling: %f%c (max = %f/p90 = %f/median = %f)\n", scale, (scale == max) ? '!' : ' ', max, p90, median);

	return scale;
}


extern float estimate_scaling_cal(const long dims[DIMS], const complex float* sens, const long cal_dims[DIMS], const complex float* cal_data, bool compat)
{
	long img_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, cal_dims);

	long imsize = md_calc_size(DIMS, img_dims);

	complex float* tmp1 = md_alloc(DIMS, img_dims, CFL_SIZE);

	float rescale = sqrtf((float)dims[0] / (float)cal_dims[0])
			* sqrtf((float)dims[1] / (float)cal_dims[1])
			* sqrtf((float)dims[2] / (float)cal_dims[2]);

	if (NULL == sens) {

		rss_combine(cal_dims, tmp1, cal_data);

	 } else {

		optimal_combine(cal_dims, 0., tmp1, sens, cal_data);
	}

	float scale = estimate_scaling_norm(rescale, imsize, tmp1, compat);

	md_free(tmp1);

	return scale;
}


static float estimate_scaling_internal(const long dims[DIMS], const complex float* sens, const long strs[DIMS], const complex float* data, bool compat)
{
	assert(1 == dims[MAPS_DIM]);

	long small_dims[DIMS];
	long cal_size[3] = { 32, 32, 32 };
	// maybe we should just extract a fixed-sized block here?
	complex float* tmp = extract_calib2(small_dims, cal_size, dims, strs, data, false);

	float scaling = estimate_scaling_cal(dims, sens, small_dims, tmp, compat);

	md_free(tmp);

	return scaling;

}



float estimate_scaling2(const long dims[DIMS], const complex float* sens, const long strs[DIMS], const complex float* data2)
{	
	return estimate_scaling_internal(dims, sens, strs, data2, false);
}

float estimate_scaling(const long dims[DIMS], const complex float* sens, const complex float* data2)
{
	long strs[DIMS];
	md_calc_strides(DIMS, strs, dims, CFL_SIZE);

	return estimate_scaling2(dims, sens, strs, data2);
}

float estimate_scaling_old2(const long dims[DIMS], const complex float* sens, const long strs[DIMS], const complex float* data)
{
	return estimate_scaling_internal(dims, sens, strs, data, true);
}





void fake_kspace(const long dims[DIMS], complex float* kspace, const complex float* sens, const complex float* image)
{
	long dims_one[DIMS];
	long dims_img[DIMS];
	long dims_ksp[DIMS];

	md_select_dims(DIMS, ~(COIL_FLAG | MAPS_FLAG), dims_one, dims);
	md_select_dims(DIMS, ~COIL_FLAG, dims_img, dims);
	md_select_dims(DIMS, ~MAPS_FLAG, dims_ksp, dims);

	const struct linop_s* sense_data = sense_init(dims, FFT_FLAGS|COIL_FLAG|MAPS_FLAG, sens);
	linop_forward(sense_data, DIMS, dims_ksp, kspace, DIMS, dims_img, image);
	linop_free(sense_data);
}




void replace_kspace(const long dims[DIMS], complex float* out, const complex float* kspace, const complex float* sens, const complex float* image)
{
	long dims_one[DIMS];
	long dims_img[DIMS];
	long dims_ksp[DIMS];

	md_select_dims(DIMS, ~(COIL_FLAG|MAPS_FLAG), dims_one, dims);
	md_select_dims(DIMS, ~(COIL_FLAG), dims_img, dims);
	md_select_dims(DIMS, ~(MAPS_FLAG), dims_ksp, dims);

	complex float* data = md_alloc(DIMS, dims_ksp, CFL_SIZE);

	fake_kspace(dims, data, sens, image);

	complex float* pattern = md_alloc(DIMS, dims_one, CFL_SIZE);

	estimate_pattern(DIMS, dims_ksp, COIL_FLAG, pattern, kspace);

	data_consistency(dims_ksp, out, pattern, kspace, data);

	md_free(pattern);
	md_free(data);
}



void replace_kspace2(const long dims[DIMS], complex float* out, const complex float* kspace, const complex float* sens, const complex float* image)
{
	long dims_ksp[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, dims_ksp, dims);

	complex float* data = md_alloc(DIMS, dims_ksp, CFL_SIZE);

	replace_kspace(dims, data, kspace, sens, image);

	rss_combine(dims, out, data);
//	optimal_combine(dims, 0.1, out, sens, data);

	md_free(data);	
}


