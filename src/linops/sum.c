/* Copyright 2014. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014 Frank Ong <frankong@berkeley.edu>
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 *
 */

#include <string.h>
#include <complex.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/ops.h"
#include "num/iovec.h"

#include "linops/linop.h"

#include "misc/misc.h"
#include "misc/mri.h"

#include "sum.h"

#include "misc/debug.h"



/**
 * data structure
 */
struct sum_data {

	INTERFACE(linop_data_t);

	bool use_gpu;

	long imgd_dims[DIMS];
	long img_dims[DIMS];
	long levels;

	long imgd_strs[DIMS];
	long img_strs[DIMS];

	complex float* tmp;
};

DEF_TYPEID(sum_data);

static struct sum_data* sum_create_data(const long imgd_dims[DIMS], bool use_gpu);
static void sum_free_data(const linop_data_t* _data);
static void sum_apply(const linop_data_t* _data, complex float* _dst, const complex float* _src);
static void sum_apply_adjoint(const linop_data_t* _data, complex float* _dst, const complex float* _src);
static void sum_apply_normal(const linop_data_t* _data, complex float* _dst, const complex float* _src);
static void sum_apply_pinverse(const linop_data_t* _data, float lambda, complex float* _dst, const complex float* _src);


/**
 * Create sum operator
 */
const struct linop_s* linop_sum_create(const long imgd_dims[DIMS], bool use_gpu)
{
	struct sum_data* data = sum_create_data(imgd_dims, use_gpu);

	// create operator interface
	return linop_create(DIMS, data->img_dims, DIMS, data->imgd_dims,
			CAST_UP(data), sum_apply, sum_apply_adjoint, sum_apply_normal,
			sum_apply_pinverse, sum_free_data);
}


static struct sum_data* sum_create_data(const long imgd_dims[DIMS], bool use_gpu)
{
	PTR_ALLOC(struct sum_data, data);
	SET_TYPEID(sum_data, data);

	// decom dimensions
	md_copy_dims(DIMS, data->imgd_dims, imgd_dims);
	md_calc_strides(DIMS, data->imgd_strs, imgd_dims, CFL_SIZE);

	// image dimensions
	data->levels = imgd_dims[LEVEL_DIM];
	md_select_dims(DIMS, ~LEVEL_FLAG, data->img_dims, imgd_dims);
	md_calc_strides(DIMS, data->img_strs, data->img_dims, CFL_SIZE);

	data->tmp = NULL;

	data->use_gpu = use_gpu;

	return PTR_PASS(data);
}



void sum_free_data(const linop_data_t* _data)
{
        struct sum_data* data = CAST_DOWN(sum_data, _data);

	if (NULL != data->tmp)
		md_free(data->tmp);

	xfree(data);
}


void sum_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
        struct sum_data* data = CAST_DOWN(sum_data, _data);

	md_clear(DIMS, data->img_dims, dst, CFL_SIZE);

	md_zaxpy2(DIMS, data->imgd_dims, data->img_strs, dst, 1. / sqrtf(data->levels), data->imgd_strs, src);
}


void sum_apply_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
        struct sum_data* data = CAST_DOWN(sum_data, _data);

	md_clear(DIMS, data->imgd_dims, dst, CFL_SIZE);

	md_zaxpy2(DIMS, data->imgd_dims, data->imgd_strs, dst, 1. / sqrtf(data->levels), data->img_strs, src);
}


void sum_apply_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
        struct sum_data* data = CAST_DOWN(sum_data, _data);

	if (NULL == data->tmp) {

#ifdef USE_CUDA
		data->tmp = (data->use_gpu ? md_alloc_gpu : md_alloc)(DIMS, data->img_dims, CFL_SIZE);
#else
		data->tmp = md_alloc(DIMS, data->img_dims, CFL_SIZE);
#endif
	}

	sum_apply(_data, data->tmp, src);
	sum_apply_adjoint(_data, dst, data->tmp);
}



/**
 * 
 * x = (ATA + uI)^-1 b
 * 
 */
void sum_apply_pinverse(const linop_data_t* _data, float rho, complex float* dst, const complex float* src)
{
        struct sum_data* data = CAST_DOWN(sum_data, _data);

	if (NULL == data->tmp) {

#ifdef USE_CUDA
		data->tmp = (data->use_gpu ? md_alloc_gpu : md_alloc)(DIMS, data->img_dims, CFL_SIZE);
#else
		data->tmp = md_alloc(DIMS, data->img_dims, CFL_SIZE);
#endif
	}


	// get average
	md_clear(DIMS, data->img_dims, data->tmp, CFL_SIZE);

	md_zadd2(DIMS, data->imgd_dims, data->img_strs, data->tmp, data->img_strs, data->tmp , data->imgd_strs, src);
	md_zsmul(DIMS, data->img_dims, data->tmp, data->tmp, 1. / data->levels);


	// get non-average
	md_zsub2(DIMS, data->imgd_dims, data->imgd_strs, dst, data->imgd_strs, src, data->img_strs, data->tmp);

	// avg = avg / (1 + rho)
	md_zsmul(DIMS, data->img_dims, data->tmp, data->tmp, 1. / (1. + rho));

	// nonavg = nonavg / rho
	md_zsmul(DIMS, data->imgd_dims, dst, dst, 1. / rho);

	// dst = avg + nonavg
	md_zadd2(DIMS, data->imgd_dims, data->imgd_strs, dst, data->imgd_strs, dst, data->img_strs, data->tmp);
}

