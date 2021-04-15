/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */


#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"

#include "zexp.h"


struct zexp_s {

	INTERFACE(nlop_data_t);

	int N;
	const long* dims;
	complex float* xn;
};

DEF_TYPEID(zexp_s);

static void zexp_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(zexp_s, _data);

	if (NULL == data->xn)
		data->xn = md_alloc_sameplace(data->N, data->dims, CFL_SIZE, dst);

	md_zexp(data->N, data->dims, data->xn, src);
	md_copy(data->N, data->dims, dst, data->xn, CFL_SIZE);
}

static void zexp_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zexp_s, _data);
	md_zmul(data->N, data->dims, dst, src, data->xn);
}

static void zexp_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zexp_s, _data);
	md_zmulc(data->N, data->dims, dst, src, data->xn);
}

static void zexp_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(zexp_s, _data);

	md_free(data->xn);
	xfree(data->dims);
	xfree(data);
}


struct nlop_s* nlop_zexp_create(int N, const long dims[N])
{
	PTR_ALLOC(struct zexp_s, data);
	SET_TYPEID(zexp_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);

	data->N = N;
	data->dims = *PTR_PASS(ndims);
	data->xn = NULL;

	return nlop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)),
		zexp_fun, zexp_der, zexp_adj, NULL, NULL, zexp_del);
}


struct zlog_s {

	INTERFACE(nlop_data_t);

	int N;
	const long* dims;
	complex float* xn;
};

DEF_TYPEID(zlog_s);

static void zlog_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(zlog_s, _data);
	if (NULL == data->xn)
		data->xn = md_alloc_sameplace(data->N, data->dims, CFL_SIZE, dst);
	md_zfill(data->N, data->dims,dst, 1);
	md_zdiv(data->N, data->dims, data->xn, dst, src);
	md_zlog(data->N, data->dims, dst, src);
}

static void zlog_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zlog_s, _data);
	md_zmul(data->N, data->dims, dst, src, data->xn);
}

static void zlog_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zlog_s, _data);
	md_zmulc(data->N, data->dims, dst, src, data->xn);
}

static void zlog_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(zlog_s, _data);

	md_free(data->xn);
	xfree(data->dims);
	xfree(data);
}


struct nlop_s* nlop_zlog_create(int N, const long dims[N])
{
	PTR_ALLOC(struct zlog_s, data);
	SET_TYPEID(zlog_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);

	data->N = N;
	data->dims = *PTR_PASS(ndims);
	data->xn = NULL;

	return nlop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)),
		zlog_fun, zlog_der, zlog_adj, NULL, NULL, zlog_del);
}