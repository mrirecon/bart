/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <complex.h>
#include <math.h>

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
#include "nlops/stack.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/someops.h"
#include "linops/someops.h"
#include "nlops/cast.h"
#include "nn/layers.h"

#include "nn_ops.h"



const struct nlop_s* nlop_maxpool_create(int N, const long dims[N], const long pool_size[N])
{	
	long ndims[2 * N];
	long odims[2 * N];

	unsigned int perm[2 * N];

	

	for (int i = 0; i < N; i++) {
		
		assert(0 == dims[i] % pool_size[i]);

		odims[i] = dims[i] / pool_size[i];
		odims[i + N] = pool_size[i];

		ndims[2 * i] = pool_size[i];
		ndims[2 * i + 1] = odims[i];

		perm[i] = 2 * i + 1;
		perm[i + N] = 2 * i;
	}

	auto result = nlop_zmax_create(2 * N, odims, (MD_BIT(2 * N) - 1) & ~(MD_BIT(N) - 1));
	result = nlop_chain_FF(nlop_from_linop_F(linop_permute_create(2 * N, perm, ndims)), result);
	result = nlop_reshape_in_F(result, 0, N, dims);
	result = nlop_reshape_out_F(result, 0, N, odims);

	return result;
}


struct dropout_s {

	INTERFACE(nlop_data_t);

	int N;
	float p;

	const struct iovec_s* tmpdom;
	const struct iovec_s* dom;
	const struct iovec_s* codom;

	complex float* tmp;
};

DEF_TYPEID(dropout_s);


static void dropout_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(dropout_s, _data);

	if (NULL == data->tmp)
		data->tmp = md_alloc_sameplace(data->N, data->tmpdom->dims, CFL_SIZE, dst);
#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src)));
#endif
	if (NULL == data->tmp)
		data->tmp = md_alloc_sameplace(data->N, data->tmpdom->dims, CFL_SIZE, dst);

	md_rand_one(data->N, data->tmpdom->dims, data->tmp, (1. - data->p));

	md_ztenmul2(data->N, data->codom->dims, data->codom->strs, dst, data->tmpdom->strs, data->tmp, data->dom->strs, src);
}

static void dropout_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);
	const auto data = CAST_DOWN(dropout_s, _data);
	assert(NULL != data->tmp);

	md_ztenmul2(data->N, data->codom->dims, data->codom->strs, dst, data->tmpdom->strs, data->tmp, data->dom->strs, src);
}

static void dropout_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);
	const auto data = CAST_DOWN(dropout_s, _data);
	assert(NULL != data->tmp);

	md_ztenmul2(data->N, data->dom->dims, data->dom->strs, dst, data->tmpdom->strs, data->tmp, data->codom->strs, src);
}


static void dropout_del(const struct nlop_data_s* _data)
{
	const auto data = CAST_DOWN(dropout_s, _data);

	md_free(data->tmp);

	iovec_free(data->dom);
	iovec_free(data->codom);
	iovec_free(data->tmpdom);

	xfree(data);
}


const struct nlop_s* nlop_dropout_create(int N, const long dims[N], float p, unsigned int shared_dims_flag)
{
	PTR_ALLOC(struct dropout_s, data);
	SET_TYPEID(dropout_s, data);

	data->N = N;
	data->p = p;

	// will be initialized later, to transparently support GPU
	data->tmp = NULL;

	long tmpdims[N];
	md_select_dims(N, ~shared_dims_flag, tmpdims, dims);

	data->tmpdom = iovec_create(N, tmpdims, CFL_SIZE);
	data->dom = iovec_create(N, dims, CFL_SIZE);
	data->codom = iovec_create(N, dims, CFL_SIZE);

	return nlop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), dropout_fun, dropout_der, dropout_adj, NULL, NULL, dropout_del);
}

struct norm_max_abs_s {

	INTERFACE(nlop_data_t);

	unsigned long N;
	const long* dims;
	const long* sdims;
};

DEF_TYPEID(norm_max_abs_s);

static void norm_max_abs_fun(const nlop_data_t* _data, int D, complex float* args[D])
{
	assert(3 == D);
	complex float* dst = args[0];
	complex float* scale = args[1];
	complex float* src = args[2];

	const auto data = CAST_DOWN(norm_max_abs_s, _data);

	unsigned long N = data->N;
	const long* dims = data->dims;
	const long* sdims = data->sdims;

#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src)));
#endif

	complex float* tmp = md_alloc_sameplace(N, dims, CFL_SIZE, dst);
	md_zabs(N, dims, tmp, src);

	md_copy2(N, sdims, MD_STRIDES(N, sdims, CFL_SIZE), scale, MD_STRIDES(N, dims, CFL_SIZE), tmp, CFL_SIZE);
	md_zmax2(N, dims, MD_STRIDES(N, sdims, CFL_SIZE), scale, MD_STRIDES(N, sdims, CFL_SIZE), scale, MD_STRIDES(N, dims, CFL_SIZE), tmp);

	complex float* ones = md_alloc_sameplace(N, sdims, CFL_SIZE, dst);
	md_zfill(N, sdims, ones, 1.);
	md_zdiv(N, sdims, tmp, ones, scale);

	md_zmul2(N, dims,
		MD_STRIDES(N, dims, CFL_SIZE), dst,
		MD_STRIDES(N, dims, CFL_SIZE), src,
		MD_STRIDES(N, sdims, CFL_SIZE), tmp);

	md_free(ones);
	md_free(tmp);
}

static void norm_max_abs_del(const struct nlop_data_s* _data)
{
	const auto data = CAST_DOWN(norm_max_abs_s, _data);

	xfree(data->dims);
	xfree(data->sdims);

	xfree(data);
}

const struct nlop_s* nlop_norm_max_abs_create(int N, const long dims[N], unsigned long batch_flag)
{
	PTR_ALLOC(struct norm_max_abs_s, data);
	SET_TYPEID(norm_max_abs_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);
	PTR_ALLOC(long[N], sdims);
	md_select_dims(N, batch_flag, *sdims, dims);

	data->N = N;
	data->dims = *PTR_PASS(ndims);
	data->sdims = *PTR_PASS(sdims);

	long nl_odims[2][N];
	md_copy_dims(N, nl_odims[0], dims);
	md_copy_dims(N, nl_odims[1], data->sdims);

	long nl_idims[1][N];
	md_copy_dims(N, nl_idims[0], dims);

	return nlop_generic_create(2, N, nl_odims, 1, N, nl_idims, CAST_UP(PTR_PASS(data)), norm_max_abs_fun, NULL, NULL, NULL, NULL, norm_max_abs_del);

}