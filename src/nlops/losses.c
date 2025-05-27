/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <assert.h>
#include <stdbool.h>
#include <math.h>

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"

#include "num/iovec.h"
#include "num/ops.h"
#include "num/multind.h"
#include "num/flpmath.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/sum.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/conv.h"
#include "nlops/tenmul.h"
#include "nlops/const.h"
#include "nlops/someops.h"

#include "losses.h"

struct znorm_s {

	nlop_data_t super;

	int N;
	const long* ridims;
	const long* rodims;

	float scale;
	float* tmp;
};

DEF_TYPEID(znorm_s);

static void znorm_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src)));
#endif
	const auto d = CAST_DOWN(znorm_s, _data);

	if (NULL == d->tmp)
		d->tmp = md_alloc_sameplace(d->N, d->ridims, FL_SIZE, dst);

	md_tenmul(d->N, d->rodims, d->tmp, d->ridims, (const float*)src, d->ridims, (const float*)src);
	md_smul(d->N, d->rodims, d->tmp, d->tmp, 1. / d->scale);
	md_zcmpl_real(d->N - 1, d->rodims + 1, dst, d->tmp);

	md_smul(d->N, d->ridims, d->tmp, (const float*)src, 2. / d->scale);
}

static void znorm_der(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(znorm_s, _data);
	assert(NULL != d->tmp);

	float* tmp = md_alloc_sameplace(d->N, d->rodims, FL_SIZE, dst);

	md_tenmul(d->N, d->rodims, tmp, d->ridims, d->tmp, d->ridims, (const float*)src);
	md_zcmpl_real(d->N - 1, d->rodims + 1, dst, tmp);

	md_free(tmp);
}

static void znorm_adj(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(znorm_s, _data);
	assert(NULL != d->tmp);

	float* tmp = md_alloc_sameplace(d->N, d->rodims, FL_SIZE, dst);

	md_real(d->N - 1, d->rodims + 1, tmp, src);
	md_tenmul(d->N, d->ridims, (float*)dst, d->ridims, d->tmp, d->rodims, tmp);

	md_free(tmp);
}

static void znorm_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(znorm_s, _data);

	md_free(data->tmp);

	xfree(data->rodims);
	xfree(data->ridims);
	xfree(data);
}

const struct nlop_s* nlop_znorm_create(int N, const long dims[N], unsigned long mean_dims)
{
	PTR_ALLOC(struct znorm_s, data);
	SET_TYPEID(znorm_s, data);

	PTR_ALLOC(long[N + 1], rodims);
	PTR_ALLOC(long[N + 1], ridims);
	(*ridims)[0] = 2;

	md_copy_dims(N, *ridims + 1, dims);
	md_singleton_dims(N + 1, *rodims);

	data->N = N + 1;
	data->rodims = *PTR_PASS(rodims);
	data->ridims = *PTR_PASS(ridims);
	data->tmp = NULL;

	long tdims[N];
	md_select_dims(N, mean_dims, tdims, dims);
	data->scale = (float)md_calc_size(N, tdims);

	return nlop_create(1, MD_DIMS(1), N, dims, CAST_UP(PTR_PASS(data)), znorm_fun, znorm_der, znorm_adj, NULL, NULL, znorm_del);
}

const struct nlop_s* nlop_mse_create(int N, const long dims[N], unsigned long mean_dims)
{
	return nlop_chain2_FF(nlop_zaxpbz_create(N, dims, 1, -1), 0, nlop_znorm_create(N, dims, mean_dims), 0);
}


// out: min_l 1/N ||l * x - y||^2 ; in : x, y
const struct nlop_s* nlop_mse_scaled_create(int N, const long dims[N], unsigned long mean_dims)
{
	long scl_dims[N];
	md_select_dims(N, mean_dims, scl_dims, dims);

	auto scl1 = nlop_tenmul_create(N, scl_dims, dims, dims);
	scl1 = nlop_prepend_FF(nlop_from_linop_F(linop_zconj_create(N, dims)), scl1, 1);
	
	auto scl2 = nlop_tenmul_create(N, scl_dims, dims, dims);
	scl2 = nlop_prepend_FF(nlop_from_linop_F(linop_zconj_create(N, dims)), scl2, 1);
	scl2 = nlop_dup_F(scl2, 0, 1);

	auto scl = nlop_zdiv_create(N, scl_dims);
	scl = nlop_chain2_FF(scl2, 0, scl, 1);
	scl = nlop_chain2_swap_FF(scl1, 0, scl, 0);
	scl = nlop_dup_F(scl, 1, 2);
	scl = nlop_append_FF(scl, 0, nlop_from_linop_F(linop_zreal_create(N, scl_dims)));	// out: (sum Re[y * conj(x)]) / (sum x * conj(x)); in: y, x

	scl = nlop_chain2_FF(scl, 0, nlop_tenmul_create(N, dims, dims, scl_dims), 1);
	scl = nlop_dup_F(scl, 0, 2);								// out: x * (sum Re[y * conj(x)]) / (sum x * conj(x)); in: x, y
			
	auto mse = nlop_chain2_FF(nlop_zaxpbz_create(N, dims, 1, -1), 0, nlop_znorm_create(N, dims, mean_dims), 0);	// out: 1 / N || x - y ||^2; in: x, y

	auto ret = nlop_chain2_swap_FF(scl, 0, mse, 0);		// out: min_l 1/N ||lx - y||^2 ; in : x, y, y
	ret = nlop_dup_F(ret, 1, 2);

	return ret;
}




const struct nlop_s* nlop_nmse_create(int N, const long dims[N], unsigned long batch_flags)
{
	long bat_dims[N];
	md_select_dims(N, batch_flags, bat_dims, dims);

	auto result = nlop_zaxpbz_create(N, dims, 1., -1.);
	result = nlop_chain2_FF(result, 0, nlop_zss_create(N, dims, ~batch_flags), 0);
	result = nlop_chain2_FF(result, 0, nlop_tenmul_create(N, MD_SINGLETON_DIMS(N), bat_dims, bat_dims), 0);
	result = nlop_chain2_FF(nlop_zinv_create(N, bat_dims),0 , result, 0);
	result = nlop_chain2_FF(nlop_zss_create(N, dims, ~batch_flags),0 , result, 2);
	result = nlop_dup_F(result, 1, 2);
	result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_scale_create(N, MD_SINGLETON_DIMS(N), 1. / md_calc_size(N, bat_dims))), 0);
	result = nlop_reshape_out_F(result, 0, 1, MD_SINGLETON_DIMS(1));

	return result;
}


const struct nlop_s* nlop_nrmse_create(int N, const long dims[N], unsigned long batch_flags)
{
	long bat_dims[N];
	md_select_dims(N, batch_flags, bat_dims, dims);

	auto result = nlop_zaxpbz_create(N, dims, 1., -1.);

	result = nlop_chain2_FF(result, 0, nlop_zrss_create(N, dims, ~batch_flags), 0);
	result = nlop_chain2_FF(nlop_tenmul_create(N, MD_SINGLETON_DIMS(N), bat_dims, bat_dims), 0, result, 0);
	result = nlop_chain2_FF(nlop_zinv_create(N, bat_dims),0 , result, 0);
	result = nlop_chain2_FF(nlop_zrss_create(N, dims, ~batch_flags),0 , result, 2);
	result = nlop_dup_F(result, 1, 2);
	result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_scale_create(N, MD_SINGLETON_DIMS(N), 1. / md_calc_size(N, bat_dims))), 0);
	result = nlop_reshape_out_F(result, 0, 1, MD_SINGLETON_DIMS(1));

	return result;
}


struct zasum_s {

	nlop_data_t super;

	int N;
	const long* rdims;
	float scaling;

	float* der;
};

DEF_TYPEID(zasum_s);

static void zasum_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(zasum_s, _data);

	if (NULL == data->der)
		data->der = md_alloc_sameplace(data->N, data->rdims, FL_SIZE, dst);

	complex float result = md_asum(data->N, data->rdims, (const float*)src) / data->scaling;

	md_sgreatequal(data->N, data->rdims, data->der, (const float*)src, 0);

	float* tmp = md_alloc_sameplace(data->N, data->rdims, FL_SIZE, dst);
	md_slessequal(data->N, data->rdims, tmp, (const float*)src, 0);
	md_sub(data->N, data->rdims, data->der, data->der, tmp);
	md_free(tmp);

	md_smul(data->N, data->rdims, data->der, data->der, 1. / data->scaling);

	md_copy(1, MAKE_ARRAY(1l), dst, &result, CFL_SIZE);
}


static void zasum_der(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(zasum_s, _data);
	assert(NULL != data->der);

	md_clear(1, MD_DIMS(1), dst, CFL_SIZE);
	md_tenmul(data->N, MD_SINGLETON_DIMS(data->N), (float*)dst, data->rdims, (float*)src, data->rdims, data->der);
}

static void zasum_adj(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(zasum_s, _data);
	assert(NULL != data->der);

	md_tenmul(data->N, data->rdims, (float*)dst, MD_SINGLETON_DIMS(data->N), (float*)src, data->rdims, data->der);
}

static void zasum_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(zasum_s, _data);

	md_free(data->der);
	xfree(data->rdims);
	xfree(data);
}

const struct nlop_s* nlop_zasum_create(int N, const long dims[N], unsigned long mean_dims)
{
	PTR_ALLOC(struct zasum_s, data);
	SET_TYPEID(zasum_s, data);

	PTR_ALLOC(long[N + 1], rdims);
	(*rdims)[0] = 2;
	md_copy_dims(N, *rdims + 1, dims);

	data->N = N + 1;
	data->rdims = *PTR_PASS(rdims);
	data->der = NULL;

	long tdims[N];
	md_select_dims(N, mean_dims, tdims, dims);
	data->scaling = (float)md_calc_size(N, tdims);

	return nlop_create(1, MD_SINGLETON_DIMS(1), N, dims, CAST_UP(PTR_PASS(data)), zasum_fun, zasum_der, zasum_adj, NULL, NULL, zasum_del);
}

const struct nlop_s* nlop_z1norm_create(int N, const long dims[N], unsigned long mean_dims)
{
	return nlop_chain_FF(nlop_smo_abs_create(N, dims, 0), nlop_zasum_create(N, dims, mean_dims));
}

const struct nlop_s* nlop_mad_create(int N, const long dims[N], unsigned long mean_dims)
{
	return nlop_chain2_FF(nlop_zaxpbz_create(N, dims, 1, -1), 0, nlop_zasum_create(N, dims, mean_dims), 0);
}

