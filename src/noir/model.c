/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2011-2012 Martin Uecker <uecker@eecs.berkeley.edu>
 *
 *
 * Uecker M, Hohage T, Block KT, Frahm J. Image reconstruction by regularized nonlinear
 * inversion – Joint estimation of coil sensitivities and image content. 
 * Magn Reson Med 2008; 60:674-682.
 */


#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/fft.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/filter.h"

#include "model.h"



struct noir_data {

	long dims[DIMS];

	long sign_dims[DIMS];
	long sign_strs[DIMS];

	long data_dims[DIMS];
	long data_strs[DIMS];

	long coil_dims[DIMS];
	long coil_strs[DIMS];

	long imgs_dims[DIMS];
	long imgs_strs[DIMS];

	long mask_dims[DIMS];
	long mask_strs[DIMS];

	long ptrn_dims[DIMS];
	long ptrn_strs[DIMS];

	long wght_dims[DIMS];
	long wght_strs[DIMS];


	const complex float* pattern;
	const complex float* mask;
	const complex float* weights;

	complex float* sens;
	complex float* xn;

	complex float* tmp;

	bool rvc;
};



static void noir_calc_weights(const long dims[3], complex float* dst)
{
	unsigned int flags = 0;

	for (int i = 0; i < 3; i++)
		if (1 != dims[i])
			flags = MD_SET(flags, i);

	klaplace(3, dims, flags, dst);
	md_zsmul(3, dims, dst, dst, 220.);
	md_zsadd(3, dims, dst, dst, 1.);
	md_zspow(3, dims, dst, dst, -16.);	// 1 + 222. \Laplace^16
}


struct noir_data* noir_init(const long dims[DIMS], const complex float* mask, const complex float* psf, bool rvc, bool use_gpu)
{
#ifdef USE_CUDA
	md_alloc_fun_t my_alloc = use_gpu ? md_alloc_gpu : md_alloc;
#else
	assert(!use_gpu);
	md_alloc_fun_t my_alloc = md_alloc;
#endif

	PTR_ALLOC(struct noir_data, data);


	data->rvc = rvc;

	md_copy_dims(DIMS, data->dims, dims);

	md_select_dims(DIMS, FFT_FLAGS|COIL_FLAG|CSHIFT_FLAG, data->sign_dims, dims);
	md_calc_strides(DIMS, data->sign_strs, data->sign_dims, CFL_SIZE);

	md_select_dims(DIMS, FFT_FLAGS|COIL_FLAG|MAPS_FLAG, data->coil_dims, dims);
	md_calc_strides(DIMS, data->coil_strs, data->coil_dims, CFL_SIZE);

	md_select_dims(DIMS, FFT_FLAGS|MAPS_FLAG|CSHIFT_FLAG, data->imgs_dims, dims);
	md_calc_strides(DIMS, data->imgs_strs, data->imgs_dims, CFL_SIZE);

	md_select_dims(DIMS, FFT_FLAGS|COIL_FLAG, data->data_dims, dims);
	md_calc_strides(DIMS, data->data_strs, data->data_dims, CFL_SIZE);

	md_select_dims(DIMS, FFT_FLAGS, data->mask_dims, dims);
	md_calc_strides(DIMS, data->mask_strs, data->mask_dims, CFL_SIZE);

	md_select_dims(DIMS, FFT_FLAGS, data->wght_dims, dims);
	md_calc_strides(DIMS, data->wght_strs, data->wght_dims, CFL_SIZE);

	md_select_dims(DIMS, FFT_FLAGS|CSHIFT_FLAG, data->ptrn_dims, dims);
	md_calc_strides(DIMS, data->ptrn_strs, data->ptrn_dims, CFL_SIZE);


	complex float* weights = md_alloc(DIMS, data->wght_dims, CFL_SIZE);

	noir_calc_weights(dims, weights);
	fftmod(DIMS, data->wght_dims, FFT_FLAGS, weights, weights);
	fftscale(DIMS, data->wght_dims, FFT_FLAGS, weights, weights);

	data->weights = weights;

#ifdef USE_CUDA
	if (use_gpu) {

		data->weights = md_gpu_move(DIMS, data->wght_dims, weights, CFL_SIZE);
		md_free(weights);
	}
#endif


	complex float* ptr = my_alloc(DIMS, data->ptrn_dims, CFL_SIZE);

	md_copy(DIMS, data->ptrn_dims, ptr, psf, CFL_SIZE);
	fftmod(DIMS, data->ptrn_dims, FFT_FLAGS, ptr, ptr);

	data->pattern = ptr;

	complex float* msk = my_alloc(DIMS, data->mask_dims, CFL_SIZE);

	if (NULL == mask) {

		assert(!use_gpu);
		md_zfill(DIMS, data->mask_dims, msk, 1.);

	} else {

		md_copy(DIMS, data->mask_dims, msk, mask, CFL_SIZE);
	}

//	fftmod(DIMS, data->mask_dims, 7, msk, msk);
	fftscale(DIMS, data->mask_dims, FFT_FLAGS, msk, msk);

	data->mask = msk;

	data->sens = my_alloc(DIMS, data->coil_dims, CFL_SIZE);
	data->xn = my_alloc(DIMS, data->imgs_dims, CFL_SIZE);
	data->tmp = my_alloc(DIMS, data->sign_dims, CFL_SIZE);

	return PTR_PASS(data);
}


void noir_free(struct noir_data* data)
{
	md_free((void*)data->pattern);
	md_free((void*)data->mask);
	md_free((void*)data->xn);
	md_free((void*)data->sens);
	md_free((void*)data->weights);
	md_free((void*)data->tmp);
	free(data);
}


void noir_forw_coils(struct noir_data* data, complex float* dst, const complex float* src)
{
	md_zmul2(DIMS, data->coil_dims, data->coil_strs, dst, data->coil_strs, src, data->wght_strs, data->weights);
	ifft(DIMS, data->coil_dims, FFT_FLAGS, dst, dst);
//	fftmod(DIMS, data->coil_dims, 7, dst);
}


void noir_back_coils(struct noir_data* data, complex float* dst, const complex float* src)
{
//	fftmod(DIMS, data->coil_dims, 7, dst);
	fft(DIMS, data->coil_dims, FFT_FLAGS, dst, src);
	md_zmulc2(DIMS, data->coil_dims, data->coil_strs, dst, data->coil_strs, dst, data->wght_strs, data->weights);
}


void noir_fun(struct noir_data* data, complex float* dst, const complex float* src)
{	
	long split = md_calc_size(DIMS, data->imgs_dims);

	md_copy(DIMS, data->imgs_dims, data->xn, src, CFL_SIZE);
	noir_forw_coils(data, data->sens, src + split);

	md_clear(DIMS, data->sign_dims, data->tmp, CFL_SIZE);
	md_zfmac2(DIMS, data->sign_dims, data->sign_strs, data->tmp, data->imgs_strs, src, data->coil_strs, data->sens);

	// could be moved to the benning, but see comment below
	md_zmul2(DIMS, data->sign_dims, data->sign_strs, data->tmp, data->sign_strs, data->tmp, data->mask_strs, data->mask);

	fft(DIMS, data->sign_dims, FFT_FLAGS, data->tmp, data->tmp);

	md_clear(DIMS, data->data_dims, dst, CFL_SIZE);
	md_zfmac2(DIMS, data->sign_dims, data->data_strs, dst, data->sign_strs, data->tmp, data->ptrn_strs, data->pattern);
}


void noir_der(struct noir_data* data, complex float* dst, const complex float* src)
{
	long split = md_calc_size(DIMS, data->imgs_dims);

	md_clear(DIMS, data->sign_dims, data->tmp, CFL_SIZE);
	md_zfmac2(DIMS, data->sign_dims, data->sign_strs, data->tmp, data->imgs_strs, src, data->coil_strs, data->sens);

	complex float* delta_coils = md_alloc_sameplace(DIMS, data->coil_dims, CFL_SIZE, src);
	noir_forw_coils(data, delta_coils, src + split);
	md_zfmac2(DIMS, data->sign_dims, data->sign_strs, data->tmp, data->coil_strs, delta_coils, data->imgs_strs, data->xn);
	md_free(delta_coils);

	// could be moved to the benning, but see comment below
	md_zmul2(DIMS, data->sign_dims, data->sign_strs, data->tmp, data->sign_strs, data->tmp, data->mask_strs, data->mask);

	fft(DIMS, data->sign_dims, FFT_FLAGS, data->tmp, data->tmp);

	md_clear(DIMS, data->data_dims, dst, CFL_SIZE);
	md_zfmac2(DIMS, data->sign_dims, data->data_strs, dst, data->sign_strs, data->tmp, data->ptrn_strs, data->pattern);
}


void noir_adj(struct noir_data* data, complex float* dst, const complex float* src)
{
	long split = md_calc_size(DIMS, data->imgs_dims);

	md_zmulc2(DIMS, data->sign_dims, data->sign_strs, data->tmp, data->data_strs, src, data->ptrn_strs, data->pattern);

	ifft(DIMS, data->sign_dims, FFT_FLAGS, data->tmp, data->tmp);

	// we should move it to the end, but fft scaling is applied so this would be need to moved into data->xn or weights maybe?
	md_zmulc2(DIMS, data->sign_dims, data->sign_strs, data->tmp, data->sign_strs, data->tmp, data->mask_strs, data->mask);

	md_clear(DIMS, data->coil_dims, dst + split, CFL_SIZE);
	md_zfmacc2(DIMS, data->sign_dims, data->coil_strs, dst + split, data->sign_strs, data->tmp, data->imgs_strs, data->xn);

	noir_back_coils(data, dst + split, dst + split);

	md_clear(DIMS, data->imgs_dims, dst, CFL_SIZE);
	md_zfmacc2(DIMS, data->sign_dims, data->imgs_strs, dst, data->sign_strs, data->tmp, data->coil_strs, data->sens);

	if (data->rvc)
		md_zreal(DIMS, data->imgs_dims, dst, dst);
}



