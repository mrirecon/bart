/* Copyright 2020-2024. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2023-2024. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/rand.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/someops.h"
#include "linops/sum.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"
#include "nlops/const.h"
#include "nlops/someops.h"

#include "nn/layers.h"

#include "batchnorm.h"


struct stats_s {

	nlop_data_t super;

	unsigned long flags;
	const struct iovec_s* dom;
	const struct iovec_s* codom;

	complex float n;

	complex float* x;
};

DEF_TYPEID(stats_s);

static void stats_clear_der(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(stats_s, _data);

	md_free(data->x);
	data->x = NULL;
}


static void stats_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(stats_s, _data);

	complex float* mean = args[0];
	complex float* var = args[1];
	complex float* src = args[2];

	if (NULL == data->x)
		data->x = md_alloc_sameplace(data->dom->N, data->dom->dims, CFL_SIZE, args[0]);
#ifdef USE_CUDA
	assert((cuda_ondevice(mean) == cuda_ondevice(src)) && (cuda_ondevice(var) == cuda_ondevice(src)));
#endif

	md_zsum(data->dom->N, data->dom->dims, data->flags, mean, src);
	md_zsmul(data->dom->N, data->codom->dims, mean, mean, 1. / data->n);


	complex float* neg_mean = md_alloc_sameplace(data->codom->N, data->codom->dims, data->codom->size, mean);

	md_zsmul(data->codom->N, data->codom->dims, neg_mean, mean, -1);

	md_zadd2(data->dom->N, data->dom->dims, data->dom->strs, data->x, data->dom->strs, src, data->codom->strs, neg_mean);

	md_free(neg_mean);

#if 1
	complex float* tmp = md_alloc_sameplace(data->dom->N, data->dom->dims, data->dom->size, data->x);
	md_zmulc(data->dom->N, data->dom->dims, tmp, data->x, data->x);
	md_clear(data->codom->N, data->codom->dims, var, data->codom->size);
	md_zadd2(data->dom->N, data->dom->dims, data->codom->strs, var, data->codom->strs, var, data->dom->strs, tmp);
	md_free(tmp);
#else
	md_ztenmulc(data->dom->N, data->codom->dims, var, data->dom->dims, data->x, data->dom->dims, data->x);
#endif
	md_zsmul(data->codom->N, data->codom->dims, var, var, 1. / data->n);

	md_zreal(data->codom->N, data->codom->dims, var, var);

	bool der1 = nlop_der_requested(CAST_UP(data), 0, 0);
	bool der2 = nlop_der_requested(CAST_UP(data), 0, 1);

	if (!(der1 || der2)) {

		md_free(data->x);
		data->x = NULL;
	}
}

static void stats_der_mean(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(stats_s, _data);

	md_zsum(data->dom->N, data->dom->dims, data->flags, dst, src);
	md_zsmul(data->dom->N, data->codom->dims, dst, dst, 1. / data->n);
}

static void stats_adj_mean(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(stats_s, _data);

	complex float* tmp = md_alloc_sameplace(data->codom->N, data->codom->dims, data->codom->size, src);

	md_zsmul(data->codom->N, data->codom->dims, tmp, src, 1. / data->n);
	md_copy2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->codom->strs, tmp, data->dom->size);

	md_free(tmp);
}

static void stats_der_var(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(stats_s, _data);

	md_ztenmulc(data->dom->N, data->codom->dims, dst, data->dom->dims, src, data->dom->dims, data->x);
	md_zsmul(data->codom->N, data->codom->dims, dst, dst, (2. / data->n));
	md_zreal(data->codom->N, data->codom->dims, dst, dst);
}

static void stats_adj_var(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(stats_s, _data);

	complex float* tmp = md_alloc_sameplace(data->codom->N, data->codom->dims, data->codom->size, src);

	md_zreal(data->codom->N, data->codom->dims, tmp, src);
	md_zmul2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->codom->strs, tmp, data->dom->strs, data->x);
	md_zsmul(data->dom->N, data->dom->dims, dst, dst, (2. / data->n));

	md_free(tmp);
}


static void stats_del(const struct nlop_data_s* _data)
{
	const auto data = CAST_DOWN(stats_s, _data);

	md_free(data->x);

	iovec_free(data->dom);
	iovec_free(data->codom);

	xfree(data);
}


/**
 * Nlop to compute mean and variance of input
 *
 * @param dims dims of input tensor
 * @param flags dims to compute mean/var over, i.e. dimensions that do not stay
 *
 * In 0:	Input
 * Out 0:	Mean \mu = \sum_{i=1}^N x_i/N
 * Out 1: 	Variance \var = \sum_{i=1}^N |(x_i-\mu)|^2/N
 *
 * Note the difference of the definition compared to md_zvar which has factor 1/(N-1)
 **/
const struct nlop_s* nlop_stats_create(int N, const long dims[N], unsigned long flags)
{
	PTR_ALLOC(struct stats_s, data);
	SET_TYPEID(stats_s, data);

	// will be initialized later, to transparently support GPU
	data->x = NULL;

	long codims[N];
	md_select_dims(N, ~flags, codims, dims);

	data->dom = iovec_create(N, dims, CFL_SIZE);
	data->codom = iovec_create(N, codims, CFL_SIZE);
	data->flags = flags;

	data->n = (float)md_calc_size(N, dims) / md_calc_size(N, codims);

	long nl_odims[2][N];
	md_copy_dims(N, nl_odims[0], codims);
	md_copy_dims(N, nl_odims[1], codims);

	long nl_idims[1][N];
	md_copy_dims(N, nl_idims[0], dims);


	return nlop_generic_managed_create(2, N, nl_odims, 1, N, nl_idims, CAST_UP(PTR_PASS(data)),
		stats_fun, (nlop_der_fun_t[1][2]){ { stats_der_mean, stats_der_var } },
		(nlop_der_fun_t[1][2]){ { stats_adj_mean, stats_adj_var } }, NULL, NULL, stats_del, stats_clear_der, NULL);
}

struct normalize_s {

	nlop_data_t super;

	const struct iovec_s* dom;
	const struct iovec_s* statdom;

	unsigned long flags;

	complex float* tmp; // (src - mu)
	complex float* scale; // sqrt(var + epsilon)

	float epsilon;
};

DEF_TYPEID(normalize_s);


static void normalize_clear_der_var(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(normalize_s, _data);

	md_free(data->tmp);
	data->tmp = NULL;
}

static void normalize_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(normalize_s, _data);

	assert(4 == N);

	complex float* dst = args[0];
	complex float* src = args[1];
	complex float* mean = args[2];
	complex float* var = args[3];

	if (NULL == data->tmp)
		data->tmp = md_alloc_sameplace(data->dom->N, data->dom->dims, data->dom->size, dst);

	if (NULL == data->scale)
		data->scale = md_alloc_sameplace(data->statdom->N, data->statdom->dims, data->statdom->size, dst);

	md_zsadd(data->statdom->N, data->statdom->dims, data->scale, var, data->epsilon);
	md_zreal(data->statdom->N, data->statdom->dims, data->scale, data->scale); //assert that sigma is real
	md_zspow(data->statdom->N, data->statdom->dims, data->scale, data->scale, -0.5);

#ifdef USE_CUDA //FIXME: Optimize zsub2 for these strides
	if (cuda_ondevice(src)) {

		complex float* tmp = md_alloc_sameplace(data->dom->N, data->dom->dims, CFL_SIZE, src);

		md_copy2(data->dom->N, data->dom->dims, data->dom->strs, tmp, data->statdom->strs, mean, CFL_SIZE);
		md_zsub(data->dom->N, data->dom->dims, data->tmp, src, tmp);

		md_free(tmp);

	} else {
#else
	{
#endif
		md_zsub2(data->dom->N, data->dom->dims, data->dom->strs, data->tmp, data->dom->strs, src, data->statdom->strs, mean);
	}

	md_zmul2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->dom->strs, data->tmp, data->statdom->strs, data->scale);

	bool der3 = nlop_der_requested(_data, 2, 0);

	if (!der3)
		normalize_clear_der_var(_data);
}

static void normalize_deradj_src(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(normalize_s, _data);

	md_zmul2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->dom->strs, src, data->statdom->strs, data->scale);
}

static void normalize_der_mean(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(normalize_s, _data);
	md_zmul2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->statdom->strs, src, data->statdom->strs, data->scale);
	md_zsmul(data->dom->N, data->dom->dims, dst, dst, -1.);
}

static void normalize_adj_mean(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(normalize_s, _data);

	md_clear(data->statdom->N, data->statdom->dims, dst, data->statdom->size);
	md_zadd2(data->dom->N, data->dom->dims, data->statdom->strs, dst, data->statdom->strs, dst, data->dom->strs, src);

	md_zmul(data->statdom->N, data->statdom->dims, dst, dst, data->scale);
	md_zsmul(data->statdom->N, data->statdom->dims, dst, dst, -1.);
}

static void normalize_der_var(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(normalize_s, _data);

	assert(NULL != data->tmp);

	complex float* tmp = md_alloc_sameplace(data->statdom->N, data->statdom->dims, data->statdom->size, dst);

	md_zreal(data->statdom->N, data->statdom->dims, tmp, src);

	md_zmul(data->statdom->N, data->statdom->dims, tmp, tmp, data->scale);
	md_zmul(data->statdom->N, data->statdom->dims, tmp, tmp, data->scale);
	md_zmul(data->statdom->N, data->statdom->dims, tmp, tmp, data->scale);
	md_zsmul(data->statdom->N, data->statdom->dims, tmp, tmp, -.5);

	md_zmul2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->statdom->strs, tmp, data->dom->strs, data->tmp);

	md_free(tmp);
}

static void normalize_adj_var(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(normalize_s, _data);

	assert(NULL != data->tmp);

	md_clear(data->statdom->N, data->statdom->dims, dst, data->statdom->size);

#if 0
	//change when zfmacc is optimized for this case
	md_zfmac2(data->dom->N, data->dom->dims, data->statdom->strs, dst, data->dom->strs, src, data->dom->strs, tmp);
#else
	complex float* tmp = md_alloc_sameplace(data->dom->N, data->dom->dims, data->dom->size, dst);

	md_zmulc(data->dom->N, data->dom->dims, tmp, src, data->tmp);

	md_zadd2(data->dom->N, data->dom->dims, data->statdom->strs, dst, data->statdom->strs, dst, data->dom->strs, tmp);

	md_free(tmp);
#endif

	md_zmul(data->statdom->N, data->statdom->dims, dst, dst, data->scale);
	md_zmul(data->statdom->N, data->statdom->dims, dst, dst, data->scale);
	md_zmul(data->statdom->N, data->statdom->dims, dst, dst, data->scale);
	md_zsmul(data->statdom->N, data->statdom->dims, dst, dst, -.5);
	md_zreal(data->statdom->N, data->statdom->dims, dst, dst);
}

static void normalize_del(const struct nlop_data_s* _data)
{
	const auto data = CAST_DOWN(normalize_s, _data);

	md_free(data->scale);
	md_free(data->tmp);

	iovec_free(data->dom);
	iovec_free(data->statdom);

	xfree(data);
}

/**
 * Nlop to normalize input by given mean/variance
 *
 * @param dims dims of input tensor
 * @param flags dims to compute mean/var over, i.e. dimensions that are not present in mean/variance
 * @param epsilon to update the floating mean and varinace
 *
 * In 0:	Input
 * In 1:	Mean mu
 * In 2: 	Variance sigma^2

 * Out 0:	Normalized input (x - mu) / sqrt(sigma^2 + epsilon)
 *
 **/
const struct nlop_s* nlop_normalize_stats_create(int N, const long dims[N], unsigned long flags, float epsilon)
{
	PTR_ALLOC(struct normalize_s, data);
	SET_TYPEID(normalize_s, data);

	long statdims[N];
	md_select_dims(N, ~flags, statdims, dims);

	data->dom = iovec_create(N, dims, CFL_SIZE);
	data->statdom = iovec_create(N, statdims, CFL_SIZE);
	data->epsilon = epsilon;
	data->scale = NULL;
	data->tmp = NULL;
	data->flags = flags;

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], dims);

	long nl_idims[3][N];
	md_copy_dims(N, nl_idims[0], dims);
	md_copy_dims(N, nl_idims[1], statdims);
	md_copy_dims(N, nl_idims[2], statdims);

	return nlop_generic_managed_create(	1, N, nl_odims, 3, N, nl_idims, CAST_UP(PTR_PASS(data)), normalize_fun,
						(nlop_der_fun_t[3][1]){ { normalize_deradj_src}, { normalize_der_mean }, { normalize_der_var } },
						(nlop_der_fun_t[3][1]){ { normalize_deradj_src}, { normalize_adj_mean }, { normalize_adj_var } },
						NULL, NULL, normalize_del, normalize_clear_der_var, NULL);
}


struct bn_s {

	nlop_data_t super;

	unsigned long flags;
	const struct iovec_s* dom;
	const struct iovec_s* stat_dom;

	float mean_size;

	complex float* der_out;
	complex float* der_scale;

	complex float epsilon;
};

DEF_TYPEID(bn_s);

static void bn_clear_der(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(bn_s, _data);

	md_free(data->der_out);
	md_free(data->der_scale);

	data->der_out = NULL;
	data->der_scale = NULL;
}

static void bn_init(struct bn_s* data, const complex float* ref)
{
	bool der = nlop_der_requested(CAST_UP(data), 0, 0);

	if (der) {

		if (NULL == data->der_out)
			data->der_out = md_alloc_sameplace(data->dom->N, data->dom->dims, data->dom->size, ref);

		if (NULL == data->der_scale)
			data->der_scale = md_alloc_sameplace(data->stat_dom->N, data->stat_dom->dims, data->stat_dom->size, ref);

	} else {

		bn_clear_der(CAST_UP(data));
	}
}

static void bn_fun(const nlop_data_t* _data, int D, complex float* args[D])
{
	const auto data = CAST_DOWN(bn_s, _data);

	complex float* out = args[0];
	complex float* mean = args[1];
	complex float* var = args[2];

	complex float* src = args[3];
	assert(4 == D);


	int N = data->dom->N;

	long nstat_dims[N]; //dims that not stay
	long nstat_strs[N];
	md_select_dims(N, data->flags, nstat_dims, data->dom->dims);
	md_calc_strides(N, nstat_strs, nstat_dims, CFL_SIZE);

	complex float* tmp = md_alloc_sameplace(N, data->dom->dims, CFL_SIZE, args[0]);
	complex float* scale = md_alloc_sameplace(N, data->stat_dom->dims, CFL_SIZE, args[0]);

	//compute mean
	md_zsum(N, data->dom->dims, data->flags, mean, src);
	md_zsmul(N, data->stat_dom->dims, mean, mean, 1. / data->mean_size);

	//compute var
	md_copy2(N, data->dom->dims, data->dom->strs, tmp, data->stat_dom->strs, mean, CFL_SIZE);
	md_zsub(N, data->dom->dims, tmp, src, tmp);

	md_zmulc(N, data->dom->dims, out, tmp, tmp);
	md_zsum(N, data->dom->dims, data->flags, var, out);
	md_zsmul(N, data->stat_dom->dims, var, var, 1. / data->mean_size);
	md_zreal(N, data->stat_dom->dims, var, var);

	//compute scale (1/sqrt(var + epsilon))
	md_zsadd(N, data->stat_dom->dims, scale, var, data->epsilon);
	md_sqrt(N + 1, MD_REAL_DIMS(N, data->stat_dom->dims), (float*)scale, (float*)scale);

	complex float* ones_tmp = md_alloc_sameplace(N, data->stat_dom->dims, CFL_SIZE, scale);
	md_zfill(N, data->stat_dom->dims, ones_tmp, 1.);
	md_zdiv(N, data->stat_dom->dims, scale, ones_tmp, scale);
	md_free(ones_tmp);

	md_zmul2(N, data->dom->dims, data->dom->strs, out, data->dom->strs, tmp, data->stat_dom->strs, scale);

	//output unbiased variance
	md_zsmul(N, data->stat_dom->dims, var, var, data->mean_size / (data->mean_size - 1));

	bool der = nlop_der_requested(_data, 0, 0);

	if (der) {

		bn_init(data, out);

		md_copy(N, data->dom->dims, data->der_out, out, CFL_SIZE);
		md_copy(N, data->stat_dom->dims, data->der_scale, scale, CFL_SIZE);

	} else {

		bn_clear_der(_data);
	}

	md_free(tmp);
	md_free(scale);
}


static void bn_deradj_in(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(bn_s, _data);

	complex float* der_out = data->der_out;
	complex float* der_scale = data->der_scale;

	int N = data->dom->N;

	long nstat_dims[N]; //dims that not stay
	long nstat_strs[N];
	md_select_dims(N, data->flags, nstat_dims, data->dom->dims);
	md_calc_strides(N, nstat_strs, nstat_dims, CFL_SIZE);

	md_zmul2(N, data->dom->dims, data->dom->strs, dst, data->dom->strs, src, data->stat_dom->strs, der_scale);

	complex float* stat_tmp = md_alloc_sameplace(N, data->stat_dom->dims, CFL_SIZE, dst);
	complex float* tmp = md_alloc_sameplace(N, data->dom->dims, CFL_SIZE, dst);


	//derivative through sigma_b
	md_zmulc(N, data->dom->dims, tmp, dst, der_out);
	md_zsum(N, data->dom->dims, data->flags, stat_tmp, tmp); //FIXME: unify with zfmacc?
	md_zreal(N, data->stat_dom->dims, stat_tmp, stat_tmp);
	md_zsmul(N, data->stat_dom->dims, stat_tmp, stat_tmp, 1. / data->mean_size);

	md_zmul2(N, data->dom->dims, data->dom->strs, tmp, data->dom->strs, der_out, data->stat_dom->strs, stat_tmp);
	md_zsub(N, data->dom->dims, dst, dst, tmp);

	//derivative through mu_b
	md_zsum(N, data->dom->dims, data->flags, stat_tmp, src);
	md_zsmul(N, data->stat_dom->dims, stat_tmp, stat_tmp, -1. / data->mean_size);
	md_zmul(N, data->stat_dom->dims, stat_tmp, stat_tmp, der_scale);

	complex float* ones = md_alloc_sameplace(N, nstat_dims, CFL_SIZE, dst);
	md_zfill(N, nstat_dims, ones, 1.);
	md_zfmac2(N, data->dom->dims, data->dom->strs, dst, data->stat_dom->strs, stat_tmp, nstat_strs, ones);
	md_free(ones);

	md_free(tmp);
	md_free(stat_tmp);
}


static void bn_del(const struct nlop_data_s* _data)
{
	const auto data = CAST_DOWN(bn_s, _data);

	iovec_free(data->dom);
	iovec_free(data->stat_dom);

	md_free(data->der_out);
	md_free(data->der_scale);

	xfree(data);
}


/**
 * Nlop to compute mean and variance of input
 *
 * @param N number of dimension
 * @param dims dims of input tensor
 * @param flags dims to compute mean/var over, i.e. dimensions that do not stay
 * @param epsilon small number to stabilise division
 *
 * In 0:	Input
 * Out 0:	Normalized out
 * Out 1:	Mean \mu = \sum_{i=1}^N x_i/N
 * Out 2: 	Variance \var = \sum_{i=1}^N |(x_i-\mu)|^2/N
 *
 * Note the difference of the definition compared to md_zvar which has factor 1/(N-1)
 **/
static const struct nlop_s* nlop_bn_create(int N, const long dims[N], unsigned long flags, float epsilon)
{
	PTR_ALLOC(struct bn_s, data);
	SET_TYPEID(bn_s, data);

	// will be initialized later, to transparently support GPU
	data->flags = flags;
	data->dom = iovec_create(N, dims, CFL_SIZE);
	long stat_dims[N];
	md_select_dims(N, ~flags, stat_dims, dims);
	data->stat_dom = iovec_create(N, stat_dims, CFL_SIZE);

	data->mean_size = (float)md_calc_size(N, dims) / md_calc_size(N, stat_dims);
	data->epsilon = epsilon;

	data->der_out = NULL;
	data->der_scale = NULL;

	long nl_odims[3][N];
	md_copy_dims(N, nl_odims[0], dims);
	md_copy_dims(N, nl_odims[1], stat_dims);
	md_copy_dims(N, nl_odims[2], stat_dims);

	long nl_idims[1][N];
	md_copy_dims(N, nl_idims[0], dims);

	return nlop_generic_managed_create(3, N, nl_odims, 1, N, nl_idims, CAST_UP(PTR_PASS(data)), bn_fun,
						(nlop_der_fun_t[1][3]){ { bn_deradj_in, NULL, NULL } },
						(nlop_der_fun_t[1][3]){ { bn_deradj_in, NULL, NULL } },
						 NULL, NULL, bn_del, bn_clear_der, NULL);
}


/**
 * Nlop to batch normalize input
 *
 * @param dims dims of input tensor
 * @param flags dims to compute mean/var over, i.e. dimensions that are not present in Mean/Var
 * @param epsilon small factor for numerical stability
 *
 * In 0:	Input			dims: {n1, n2, ..., nN}
 * In 1:	Floating Mean/Var	dims: {n1, 1,  ..., nN | 2 (mean/var)}
 *
 * Out 0:	Normalized Input	dims: {n1, n2, ..., nN}
 * Out 1:	Mean/Var		dims: {n1, 1,  ..., nN | 2 (mean/var)}
 **/
const struct nlop_s* nlop_batchnorm_create(int N, const long dims[N], unsigned long flags, float epsilon, enum NETWORK_STATUS status)
{
	long stat_dims[N];
	md_select_dims(N, ~flags, stat_dims, dims);

	const struct nlop_s* result = NULL;
	const struct iovec_s* iov = NULL;

	switch (status) {

	case STAT_TRAIN:

		result = nlop_bn_create(N, dims, flags, epsilon);
		result = nlop_append_singleton_dim_out_F(result, 1);
		result = nlop_append_singleton_dim_out_F(result, 2);
		result = nlop_stack_outputs_F(result, 1, 2, N);
		iov = nlop_generic_codomain(result, 1);
		result = nlop_combine_FF(result, nlop_del_out_create(iov->N, iov->dims));
		break;

	case STAT_TEST:

		result = nlop_normalize_stats_create(N, dims, flags, epsilon);
		result = nlop_append_singleton_dim_in_F(result, 1);
		result = nlop_append_singleton_dim_in_F(result, 2);
		result = nlop_stack_inputs_F(result, 1, 2, N);
		iov = nlop_generic_domain(result, 1);
		result = nlop_combine_FF(result, nlop_from_linop_F(linop_identity_create(iov->N, iov->dims)));
		result = nlop_dup_F(result, 1, 2);
		break;

	default:
		assert(0);
	}

	return result;
}


/**
 * Nlop to normalize input
 *
 * @param dims dims of input tensor
 * @param flags dims to compute mean/var over, i.e. dimensions that are not present in Mean/Var
 * @param epsilon small factor for numerical stability
 *
 * In 0:	Input			dims: {n1, n2, ..., nN}
 *
 * Out 0:	Normalized Input	dims: {n1, n2, ..., nN}
 **/
const struct nlop_s* nlop_normalize_create(int N, const long dims[N], unsigned long flags, float epsilon)
{
	const struct nlop_s* result = NULL;

	result = nlop_combine_FF(nlop_normalize_stats_create(N, dims, flags, epsilon), nlop_stats_create(N, dims, flags));
	result = nlop_dup_F(result, 0, 3);
	result = nlop_link_F(result, 1, 1);
	result = nlop_link_F(result, 1, 1);

	return result;
}


struct norm_std_s {

	INTERFACE(nlop_data_t);

	unsigned long flags;
	const struct iovec_s* idom;
	const struct iovec_s* sdom;

	float scale;
	float epsilon;

	complex float* der_inp;
	complex float* der_scale;
};

DEF_TYPEID(norm_std_s);

static void norm_std_clear_der(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(norm_std_s, _data);

	md_free(data->der_inp);
	md_free(data->der_scale);

	data->der_inp = NULL;
	data->der_scale = NULL;
}

static void norm_std_init(struct norm_std_s* data, const complex float* ref)
{
	bool der = nlop_der_requested(CAST_UP(data), 0, 0);

	if (der) {

		if (NULL == data->der_inp)
			data->der_inp = md_alloc_sameplace(data->idom->N, data->idom->dims, data->idom->size, ref);

		if (NULL == data->der_scale)
			data->der_scale = md_alloc_sameplace(data->sdom->N, data->sdom->dims, data->sdom->size, ref);

	} else {

		norm_std_clear_der(CAST_UP(data));
	}
}

static void norm_std_fun(const nlop_data_t* _data, int D, complex float* args[D])
{
	const auto d = CAST_DOWN(norm_std_s, _data);

	complex float* out = args[0];
	complex float* var = args[1];

	complex float* src = args[2];
	assert(3 == D);

	//compute var
#if 0
	md_zss(d->idom->N, d->idom->dims, d->flags, var, src);
#else
	md_zmulc(d->idom->N, d->idom->dims, out, src, src);
	md_zsum(d->idom->N, d->idom->dims, d->flags, var, out);
	md_zreal(d->sdom->N, d->sdom->dims, var, var);
#endif
	md_zsmul(d->sdom->N, d->sdom->dims, var, var, d->scale);

	//compute scale (1/sqrt(var + epsilon))
	complex float* scale = md_alloc_sameplace(d->sdom->N, d->sdom->dims, d->sdom->size, var);
	md_zsadd(d->sdom->N, d->sdom->dims, scale, var, d->epsilon);
	md_zspow(d->sdom->N, d->sdom->dims, scale, scale, -0.5);

	md_ztenmul(d->idom->N, d->idom->dims, out, d->idom->dims, src, d->sdom->dims, scale);

	bool der = nlop_der_requested(_data, 0, 0);

	if (der) {

		norm_std_init(d, out);

		md_copy(d->idom->N, d->idom->dims, d->der_inp, src, CFL_SIZE);
		md_copy(d->sdom->N, d->sdom->dims, d->der_scale, scale, CFL_SIZE);
	} else {

		norm_std_clear_der(_data);
	}

	md_free(scale);
}

static void norm_std_der_var(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(norm_std_s, _data);

	complex float* tmp = md_alloc_sameplace(d->idom->N, d->idom->dims, d->idom->size, dst);

	md_zmulc(d->idom->N, d->idom->dims, tmp, src, d->der_inp);
	md_zsum(d->idom->N, d->idom->dims, d->flags, dst, tmp);
	md_zreal(d->sdom->N, d->sdom->dims, dst, dst);
	md_zsmul(d->sdom->N, d->sdom->dims, dst, dst, 2 * d->scale);

	md_free(tmp);
}

static void norm_std_adj_var(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(norm_std_s, _data);

	complex float* tmp = md_alloc_sameplace(d->sdom->N, d->sdom->dims, d->sdom->size, dst);

	md_zsmul(d->sdom->N, d->sdom->dims, tmp, src, 2 * d->scale);
	md_zreal(d->sdom->N, d->sdom->dims, tmp, tmp);
	md_ztenmul(d->idom->N, d->idom->dims, dst, d->idom->dims, d->der_inp, d->sdom->dims, tmp);

	md_free(tmp);
}


static void norm_std_deradj_in(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(norm_std_s, _data);

	complex float* dvar = md_alloc_sameplace(d->sdom->N, d->sdom->dims, d->sdom->size, dst);

	complex float* tmp = md_alloc_sameplace(d->idom->N, d->idom->dims, d->idom->size, dst);
	md_zmulc(d->idom->N, d->idom->dims, tmp, src, d->der_inp);
	md_zsum(d->idom->N, d->idom->dims, d->flags, dvar, tmp);

	md_zreal(d->sdom->N, d->sdom->dims, dvar, dvar);
	md_zsmul(d->sdom->N, d->sdom->dims, dvar, dvar, -d->scale);

	complex float* dscale3 = md_alloc_sameplace(d->sdom->N, d->sdom->dims, d->sdom->size, dst);
	md_zspow(d->sdom->N, d->sdom->dims, dscale3, d->der_scale, 3.);
	md_zmul(d->sdom->N, d->sdom->dims, dvar, dvar, dscale3);
	md_free(dscale3);

	md_ztenmul(d->idom->N, d->idom->dims, dst, d->idom->dims, d->der_inp, d->sdom->dims, dvar);
	md_free(dvar);

	md_ztenmul(d->idom->N, d->idom->dims, tmp, d->idom->dims, src, d->sdom->dims, d->der_scale);
	md_zadd(d->idom->N, d->idom->dims, dst, dst, tmp);
	md_free(tmp);
}


static void norm_std_del(const struct nlop_data_s* _data)
{
	const auto data = CAST_DOWN(norm_std_s, _data);

	iovec_free(data->idom);
	iovec_free(data->sdom);

	md_free(data->der_inp);
	md_free(data->der_scale);

	xfree(data);
}


/**
 * Nlop to normalize by standard deviation of input
 *
 * @param N number of dimension
 * @param dims dims of input tensor
 * @param flags dims to compute mean/var over, i.e. dimensions that do not stay
 * @param epsilon small number to stabilise division
 *
 * In 0:	Input
 * Out 0:	Normalized out
 * Out 2: 	Variance \var = \sum_{i=1}^N |(x_i-\mu)|^2/N
 *
 * Note the difference of the definition compared to md_zvar which has factor 1/(N-1)
 **/
const struct nlop_s* nlop_norm_std_create(int N, const long dims[N], unsigned long flags, float epsilon)
{
	PTR_ALLOC(struct norm_std_s, data);
	SET_TYPEID(norm_std_s, data);

	// will be initialized later, to transparently support GPU
	data->flags = flags;
	data->idom = iovec_create(N, dims, CFL_SIZE);
	long stat_dims[N];
	md_select_dims(N, ~flags, stat_dims, dims);
	data->sdom = iovec_create(N, stat_dims, CFL_SIZE);

	data->scale = (float)md_calc_size(N, stat_dims) / md_calc_size(N, dims);
	data->epsilon = epsilon;

	data->der_inp = NULL;
	data->der_scale = NULL;

	long nl_odims[3][N];
	md_copy_dims(N, nl_odims[0], dims);
	md_copy_dims(N, nl_odims[1], stat_dims);

	long nl_idims[1][N];
	md_copy_dims(N, nl_idims[0], dims);

	return nlop_generic_managed_create(2, N, nl_odims, 1, N, nl_idims, CAST_UP(PTR_PASS(data)), norm_std_fun,
						(nlop_der_fun_t[1][2]){ { norm_std_deradj_in, norm_std_der_var } },
						(nlop_der_fun_t[1][2]){ { norm_std_deradj_in, norm_std_adj_var } },
						 NULL, NULL, norm_std_del, norm_std_clear_der, NULL);
}


/**
 * Nlop to normalize by substracting average of input
 *
 * @param N number of dimension
 * @param dims dims of input tensor
 * @param flags dims to compute mean/var over, i.e. dimensions that do not stay
 *
 * In 0:	Input
 * Out 0:	Normalized out
 * Out 2: 	Average
 **/
const struct nlop_s* nlop_norm_avg_create(int N, const long dims[N], unsigned long flags)
{
	const struct nlop_s* ret = nlop_from_linop_F(linop_avg_create(N, dims, flags));
	ret = nlop_chain2_keep_FF(ret, 0, nlop_zaxpbz2_create(N, dims, ~0UL, 1., ~flags, -1.), 1);
	ret = nlop_dup_F(ret, 0, 1);

	return ret;
}







