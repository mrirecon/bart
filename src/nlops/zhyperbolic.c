/* Copyright 2021. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * Moritz Blumenthal
 */


#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"

#include "zhyperbolic.h"


struct zsinh_s {

	INTERFACE(nlop_data_t);

	int N;
	const long* dims;
	complex float* xn;
};

DEF_TYPEID(zsinh_s);

static void zsinh_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(zsinh_s, _data);
	if (NULL == data->xn)
		data->xn = md_alloc_sameplace(data->N, data->dims, CFL_SIZE, dst);
	md_zsinh(data->N, data->dims, dst, src);
	md_zcosh(data->N, data->dims, data->xn, src);
}

static void zsinh_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zsinh_s, _data);
	md_zmul(data->N, data->dims, dst, src, data->xn);
}

static void zsinh_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zsinh_s, _data);
	md_zmulc(data->N, data->dims, dst, src, data->xn);
}

static void zsinh_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(zsinh_s, _data);

	md_free(data->xn);
	xfree(data->dims);
	xfree(data);
}

struct nlop_s* nlop_zsinh_create(int N, const long dims[N])
{
	PTR_ALLOC(struct zsinh_s, data);
	SET_TYPEID(zsinh_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);

	data->N = N;
	data->dims = *PTR_PASS(ndims);
	data->xn = NULL;


	return nlop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)),
		zsinh_fun, zsinh_der, zsinh_adj, NULL, NULL, zsinh_del);
}

struct zcosh_s {

	INTERFACE(nlop_data_t);

	int N;
	const long* dims;
	complex float* xn;
};

DEF_TYPEID(zcosh_s);

static void zcosh_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(zcosh_s, _data);
	if (NULL == data->xn)
		data->xn = md_alloc_sameplace(data->N, data->dims, CFL_SIZE, dst);
	md_zcosh(data->N, data->dims, dst, src);
	md_zsinh(data->N, data->dims, data->xn, src);
}

static void zcosh_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zcosh_s, _data);
	md_zmul(data->N, data->dims, dst, src, data->xn);
}

static void zcosh_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zcosh_s, _data);
	md_zmulc(data->N, data->dims, dst, src, data->xn);
}

static void zcosh_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(zcosh_s, _data);

	md_free(data->xn);
	xfree(data->dims);
	xfree(data);
}

struct nlop_s* nlop_zcosh_create(int N, const long dims[N])
{
	PTR_ALLOC(struct zcosh_s, data);
	SET_TYPEID(zcosh_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);

	data->N = N;
	data->dims = *PTR_PASS(ndims);
	data->xn = NULL;

	return nlop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)),
		zcosh_fun, zcosh_der, zcosh_adj, NULL, NULL, zcosh_del);
}

