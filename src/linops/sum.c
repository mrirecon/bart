/* Copyright 2014. The Regents of the University of California.
 * Copyright 2016-2021. Uecker Lab. University Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014 Frank Ong
 * 2016-2020 Martin Uecker
 * 2021 Moritz Blumenthal
 */

#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "linops/linop.h"

#include "misc/misc.h"

#include "sum.h"



struct sum_data {

	INTERFACE(linop_data_t);

	int N;
	long *imgd_dims;
	long *img_dims;
	long levels;

	long *imgd_strs;
	long *img_strs;

	unsigned long flags;
};

static DEF_TYPEID(sum_data);



static struct sum_data* sum_create_data(int N, const long imgd_dims[N], unsigned long flags)
{
	PTR_ALLOC(struct sum_data, data);
	SET_TYPEID(sum_data, data);

	data->N = N;
	data->imgd_dims = *TYPE_ALLOC(long[N]);
	data->imgd_strs = *TYPE_ALLOC(long[N]);
	data->img_dims = *TYPE_ALLOC(long[N]);
	data->img_strs = *TYPE_ALLOC(long[N]);

	// decom dimensions
	md_copy_dims(N, data->imgd_dims, imgd_dims);
	md_calc_strides(N, data->imgd_strs, imgd_dims, CFL_SIZE);

	long level_dims[N];
	md_select_dims(N, flags, level_dims, imgd_dims);

	data->flags = flags;
	data->levels = md_calc_size(N, level_dims);

	// image dimensions
	md_select_dims(N, ~flags, data->img_dims, imgd_dims);
	md_calc_strides(N, data->img_strs, data->img_dims, CFL_SIZE);

	return PTR_PASS(data);
}



static void sum_free_data(const linop_data_t* _data)
{
	auto data = CAST_DOWN(sum_data, _data);

	xfree(data->imgd_dims);
	xfree(data->imgd_strs);
	xfree(data->img_dims);
	xfree(data->img_strs);

	xfree(data);
}


static void sum_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(sum_data, _data);

	md_clear(data->N, data->img_dims, dst, CFL_SIZE);
	md_zsum(data->N, data->imgd_dims, data->flags, dst, src);
	md_zsmul(data->N, data->img_dims, dst, dst, 1. / sqrtf(data->levels));
}


static void sum_apply_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(sum_data, _data);

	md_clear(data->N, data->imgd_dims, dst, CFL_SIZE);
	md_copy2(data->N, data->imgd_dims, data->imgd_strs, dst, data->img_strs, src, CFL_SIZE);
	md_zsmul(data->N, data->imgd_dims, dst, dst, 1. / sqrtf(data->levels));
}


static void sum_apply_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(sum_data, _data);

	complex float* tmp = md_alloc_sameplace(data->N, data->img_dims, CFL_SIZE, dst);

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
	int N = data->N;

	complex float* tmp = md_alloc_sameplace(N, data->img_dims, CFL_SIZE, dst);

	// get average
	md_clear(N, data->img_dims, tmp, CFL_SIZE);

	md_zadd2(N, data->imgd_dims, data->img_strs, tmp, data->img_strs, tmp , data->imgd_strs, src);
	md_zsmul(N, data->img_dims, tmp, tmp, 1. / data->levels);


	// get non-average
	md_zsub2(N, data->imgd_dims, data->imgd_strs, dst, data->imgd_strs, src, data->img_strs, tmp);

	// avg = avg / (1 + rho)
	md_zsmul(N, data->img_dims, tmp, tmp, 1. / (1. + rho));

	// nonavg = nonavg / rho
	md_zsmul(N, data->imgd_dims, dst, dst, 1. / rho);

	// dst = avg + nonavg
	md_zadd2(N, data->imgd_dims, data->imgd_strs, dst, data->imgd_strs, dst, data->img_strs, tmp);

	md_free(tmp);
}


const struct linop_s* linop_sum_create(int N, const long imgd_dims[N], unsigned long flags)
{
	struct sum_data* data = sum_create_data(N, imgd_dims, flags);
	data->levels = 1;

	return linop_create(N, data->img_dims, N, data->imgd_dims,
			CAST_UP(data), sum_apply, sum_apply_adjoint, sum_apply_normal,
			sum_apply_pinverse, sum_free_data);
}

const struct linop_s* linop_scaled_sum_create(int N, const long imgd_dims[N], unsigned long flags)
{
	struct sum_data* data = sum_create_data(N, imgd_dims, flags);

	return linop_create(N, data->img_dims, N, data->imgd_dims,
			CAST_UP(data), sum_apply, sum_apply_adjoint, sum_apply_normal,
			sum_apply_pinverse, sum_free_data);
}

const struct linop_s* linop_avg_create(int N, const long imgd_dims[N], unsigned long flags)
{
	struct sum_data* data = sum_create_data(N, imgd_dims, flags);
	data->levels = data->levels * data->levels;

	return linop_create(N, data->img_dims, N, data->imgd_dims,
			CAST_UP(data), sum_apply, sum_apply_adjoint, sum_apply_normal,
			NULL, sum_free_data);
}


struct repmat_data {

	INTERFACE(linop_data_t);

	int N;
	long *odims;
	long *idims;
	long *ostrs;
	long *istrs;
};

static DEF_TYPEID(repmat_data);



static struct repmat_data* repmat_create_data(int N, const long odims[N], unsigned long flags)
{
	PTR_ALLOC(struct repmat_data, data);
	SET_TYPEID(repmat_data, data);

	data->N = N;
	data->odims = *TYPE_ALLOC(long[N]);
	data->ostrs = *TYPE_ALLOC(long[N]);
	data->idims = *TYPE_ALLOC(long[N]);
	data->istrs = *TYPE_ALLOC(long[N]);

	// decom dimensions
	md_copy_dims(N, data->odims, odims);
	md_calc_strides(N, data->ostrs, odims, CFL_SIZE);
	md_select_dims(N, ~flags, data->idims, odims);
	md_calc_strides(N, data->istrs, data->idims, CFL_SIZE);

	return PTR_PASS(data);
}



static void repmat_free_data(const linop_data_t* _data)
{
	auto data = CAST_DOWN(repmat_data, _data);

	xfree(data->odims);
	xfree(data->ostrs);
	xfree(data->idims);
	xfree(data->istrs);

	xfree(data);
}


static void repmat_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(repmat_data, _data);
	md_copy2(data->N, data->odims, data->ostrs, dst, data->istrs, src, CFL_SIZE);
}


static void repmat_apply_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(repmat_data, _data);

	md_clear(data->N, data->idims, dst, CFL_SIZE);
	md_zadd2(data->N, data->odims, data->istrs, dst, data->istrs, dst, data->ostrs, src);
}

const struct linop_s* linop_repmat_create(int N, const long odims[N], unsigned long flags)
{
	struct repmat_data* data = repmat_create_data(N, odims, flags);

	return linop_create(N, data->odims, N, data->idims,
			CAST_UP(data), repmat_apply, repmat_apply_adjoint, NULL, NULL, repmat_free_data);
}

