/* Copyright 2014. The Regents of the University of California.
 * Copyright 2016-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014 Frank Ong <frankong@berkeley.edu>
 * 2016-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 *
 */

#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "linops/linop.h"

#include "misc/misc.h"
#include "misc/mri.h"

#include "sum.h"



struct sum_data {

	INTERFACE(linop_data_t);

	long imgd_dims[DIMS];
	long img_dims[DIMS];
	long levels;

	long imgd_strs[DIMS];
	long img_strs[DIMS];
};

static DEF_TYPEID(sum_data);



static struct sum_data* sum_create_data(const long imgd_dims[DIMS], unsigned long flags)
{
	PTR_ALLOC(struct sum_data, data);
	SET_TYPEID(sum_data, data);

	// decom dimensions
	md_copy_dims(DIMS, data->imgd_dims, imgd_dims);
	md_calc_strides(DIMS, data->imgd_strs, imgd_dims, CFL_SIZE);

	long level_dims[DIMS];
	md_select_dims(DIMS, flags, level_dims, imgd_dims);

	data->levels = md_calc_size(DIMS, level_dims);

	// image dimensions
	md_select_dims(DIMS, ~flags, data->img_dims, imgd_dims);
	md_calc_strides(DIMS, data->img_strs, data->img_dims, CFL_SIZE);

	return PTR_PASS(data);
}



static void sum_free_data(const linop_data_t* _data)
{
        auto data = CAST_DOWN(sum_data, _data);

	xfree(data);
}


static void sum_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
        auto data = CAST_DOWN(sum_data, _data);

	md_clear(DIMS, data->img_dims, dst, CFL_SIZE);
	md_zaxpy2(DIMS, data->imgd_dims, data->img_strs, dst, 1. / sqrtf(data->levels), data->imgd_strs, src);
}


static void sum_apply_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
        auto data = CAST_DOWN(sum_data, _data);

	md_clear(DIMS, data->imgd_dims, dst, CFL_SIZE);
	md_zaxpy2(DIMS, data->imgd_dims, data->imgd_strs, dst, 1. / sqrtf(data->levels), data->img_strs, src);
}


static void sum_apply_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
        auto data = CAST_DOWN(sum_data, _data);

	complex float* tmp = md_alloc_sameplace(DIMS, data->img_dims, CFL_SIZE, dst);

	sum_apply(_data, tmp, src);
	sum_apply_adjoint(_data, dst, tmp);

	md_free(tmp);
}



/**
 * 
 * x = (ATA + uI)^-1 b
 * 
 */
static void sum_apply_pinverse(const linop_data_t* _data, float rho, complex float* dst, const complex float* src)
{
        auto data = CAST_DOWN(sum_data, _data);

	complex float* tmp = md_alloc_sameplace(DIMS, data->img_dims, CFL_SIZE, dst);

	// get average
	md_clear(DIMS, data->img_dims, tmp, CFL_SIZE);

	md_zadd2(DIMS, data->imgd_dims, data->img_strs, tmp, data->img_strs, tmp , data->imgd_strs, src);
	md_zsmul(DIMS, data->img_dims, tmp, tmp, 1. / data->levels);


	// get non-average
	md_zsub2(DIMS, data->imgd_dims, data->imgd_strs, dst, data->imgd_strs, src, data->img_strs, tmp);

	// avg = avg / (1 + rho)
	md_zsmul(DIMS, data->img_dims, tmp, tmp, 1. / (1. + rho));

	// nonavg = nonavg / rho
	md_zsmul(DIMS, data->imgd_dims, dst, dst, 1. / rho);

	// dst = avg + nonavg
	md_zadd2(DIMS, data->imgd_dims, data->imgd_strs, dst, data->imgd_strs, dst, data->img_strs, tmp);

	md_free(tmp);
}


const struct linop_s* linop_sum_create(const long imgd_dims[DIMS], unsigned long flags)
{
	struct sum_data* data = sum_create_data(imgd_dims, flags);
	data->levels = 1;

	return linop_create(DIMS, data->img_dims, DIMS, data->imgd_dims,
			CAST_UP(data), sum_apply, sum_apply_adjoint, sum_apply_normal,
			sum_apply_pinverse, sum_free_data);
}

const struct linop_s* linop_avg_create(const long imgd_dims[DIMS], unsigned long flags)
{
	struct sum_data* data = sum_create_data(imgd_dims, flags);

	return linop_create(DIMS, data->img_dims, DIMS, data->imgd_dims,
			CAST_UP(data), sum_apply, sum_apply_adjoint, sum_apply_normal,
			sum_apply_pinverse, sum_free_data);
}


