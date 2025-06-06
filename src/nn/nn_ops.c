/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <complex.h>
#include <assert.h>
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
#include "linops/fmac.h"

#include "nlops/nlop.h"
#include "nlops/stack.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/someops.h"
#include "nlops/const.h"
#include "nlops/tenmul.h"

#include "nn/layers.h"

#include "nn_ops.h"

static bool update_random_state = true;



const struct nlop_s* nlop_maxpool_create(int N, const long dims[N], const long pool_size[N])
{
	long ndims[2 * N];
	long odims[2 * N];

	int perm[2 * N];

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

struct rand_mask_s {

	nlop_data_t super;

	int N;
	float p;
	long* dims;
	complex float* state;
};

DEF_TYPEID(rand_mask_s);


static void rand_mask_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	assert(1 == N);
	const auto data = CAST_DOWN(rand_mask_s, _data);

	if (update_random_state)
		md_rand_one(data->N, data->dims, data->state, (1. - data->p));

	md_copy(data->N, data->dims, args[0], data->state, CFL_SIZE);
}

static void rand_mask_del(const struct nlop_data_s* _data)
{
	const auto data = CAST_DOWN(rand_mask_s, _data);

	md_free(data->state);
	xfree(data->dims);
	xfree(data);
}

//nlop creating random mask with (1. - p) ones and p. zeros
const struct nlop_s* nlop_rand_mask_create(int N, const long dims[N], float p)
{
	PTR_ALLOC(struct rand_mask_s, data);
	SET_TYPEID(rand_mask_s, data);

	data->N = N;
	data->p = p;
	data->dims = *TYPE_ALLOC(long[N]);
	md_copy_dims(N, data->dims, dims);
	data->state = md_alloc(N, dims, CFL_SIZE);

	long odims[1][N];
	md_copy_dims(N, odims[0], dims);

	return nlop_generic_create(1, N, odims, 0, 0, NULL, CAST_UP(PTR_PASS(data)), rand_mask_fun, NULL, NULL, NULL, NULL, rand_mask_del);
}

//input is multiplied with p ones to create first output
//input is multiplied with 1.-p ones to create second output
const struct nlop_s* nlop_rand_split_create(int N, const long dims[N], unsigned long shared_dims_flag, float p)
{
	long dims2[N];
	md_select_dims(N, ~shared_dims_flag, dims2, dims);

	complex float one = 1.;

	auto invert = nlop_zaxpbz_create(N, dims2, -1., 1.);
	invert = nlop_set_input_const_F2(invert, 1, N, dims2, MD_SINGLETON_STRS(N), true, &one);

	auto result = nlop_chain2_FF(invert, 0, nlop_tenmul_create(N, dims, dims, dims2), 1);
	result = nlop_combine_FF(nlop_tenmul_create(N, dims, dims, dims2), result);
	result = nlop_dup_F(result, 0, 2);
	result = nlop_dup_F(result, 1, 2);
	result = nlop_chain2_FF(nlop_rand_mask_create(N, dims2, 1. -p), 0, result, 1);

	return result;
}


struct rand_mask_fixed_s {

	nlop_data_t super;

	int N;
	long* dims;
	unsigned long bat_flags;
	float p;

	complex float* state;

	struct bart_rand_state* rand_state;
};

DEF_TYPEID(rand_mask_fixed_s);


static void rand_mask_fixed_fun(const nlop_data_t* _data, int D, complex float* args[D])
{
	assert(1 == D);
	const auto data = CAST_DOWN(rand_mask_fixed_s, _data);

	int N = data->N;

	if (!update_random_state) {
		
		md_copy(data->N, data->dims, args[0], data->state, CFL_SIZE);
		return;
	}

	md_clear(data->N, data->dims, data->state, CFL_SIZE);

	long pos[N];
	md_set_dims(N, pos, 0);

	long tdims[N];
	md_select_dims(N, ~data->bat_flags, tdims, data->dims);
	long NV = md_calc_size(N, tdims);

	assert( 1 >= fabsf(data->p));

	long strs[N];
	md_calc_strides(N, strs, data->dims, CFL_SIZE);

	do {
		long ones = data->p * NV;

		if (0 > ones) {

			ones *= -1;
			ones = rand_range_state(data->rand_state, ones + 1);
		}

		for (int i = 0; i < ones; i++) {

			md_unravel_index(N, pos, ~data->bat_flags, data->dims, 0);

			long idx = rand_range_state(data->rand_state, (NV - i));

			while ((0 < idx) || (1. == MD_ACCESS(N, strs, pos, data->state))) {

				if (0. == MD_ACCESS(N, strs, pos, data->state))
					idx--;
				
				md_next(N, data->dims, ~data->bat_flags, pos);
			}

			MD_ACCESS(N, strs, pos, data->state) = 1.;
		}

	} while (md_next(N, data->dims, data->bat_flags, pos));

	md_copy(data->N, data->dims, args[0], data->state, CFL_SIZE);
}

static void rand_mask_fixed_del(const struct nlop_data_s* _data)
{
	const auto data = CAST_DOWN(rand_mask_fixed_s, _data);

	md_free(data->state);

	xfree(data->dims);
	xfree(data->rand_state);

	xfree(data);
}

//nlop creating random mask with (1. - p) ones and p. zeros
const struct nlop_s* nlop_rand_mask_fixed_create(int N, const long dims[N], float p, unsigned long bat_flags)
{
	PTR_ALLOC(struct rand_mask_fixed_s, data);
	SET_TYPEID(rand_mask_fixed_s, data);

	data->N = N;
	data->p = p;
	data->dims = *TYPE_ALLOC(long[N]);
	md_copy_dims(N, data->dims, dims);

	data->state = md_calloc(data->N, data->dims, CFL_SIZE);

	data->rand_state = rand_state_create(123);
	data->bat_flags = bat_flags;

	long odims[1][N];
	md_copy_dims(N, odims[0], dims);

	return nlop_generic_create(1, N, odims, 0, 0, NULL, CAST_UP(PTR_PASS(data)), rand_mask_fixed_fun, NULL, NULL, NULL, NULL, rand_mask_fixed_del);
}

//input is multiplied with p ones to create first output
//input is multiplied with 1.-p ones and p times "leaky" to create second output
//if fix_first==1, output one is multiplied with 1 and output two is multiplied with zero
const struct nlop_s* nlop_rand_split_fixed_create(int N, const long dims[N], unsigned long shared_dims_flag, unsigned long bat_dims_flag, float p, unsigned long fix_flags, const complex float* _fix_first, float leaky_val)
{
	long fix_dims[N];
	md_select_dims(N, fix_flags, fix_dims, dims);

	complex float* fix_first = md_alloc(N, fix_dims, CFL_SIZE);		// 0 if must be in first output

	md_zfill(N, fix_dims, fix_first, 1);

	if (NULL != _fix_first)
		md_zsub(N, fix_dims, fix_first, fix_first, _fix_first);
	
	long dims2[N];
	md_select_dims(N, (~shared_dims_flag) | fix_flags, dims2, dims);

	complex float one = 1.;

	auto invert = nlop_zaxpbz_create(N, dims2, -1., 1.);
	invert = nlop_set_input_const_F2(invert, 1, N, dims2, MD_SINGLETON_STRS(N), true, &one);
	auto first = nlop_chain2_FF(invert, 0, nlop_tenmul_create(N, dims, dims, dims2), 1);

	auto leaky = nlop_zaxpbz_create(N, dims2, 1. - leaky_val, leaky_val);
	leaky = nlop_set_input_const_F2(leaky, 1, N, dims2, MD_SINGLETON_STRS(N), true, &one);
	auto second = nlop_chain2_FF(leaky, 0, nlop_tenmul_create(N, dims, dims, dims2), 1);

	auto result = nlop_combine_FF(first, second);
	result = nlop_dup_F(result, 0, 2);
	result = nlop_dup_F(result, 1, 2);												// out: (1 - mask) * in, mask * in ; in: in, mask 

	result = nlop_prepend_FF(nlop_from_linop_F(linop_fmac_create(N, dims2, 0, shared_dims_flag, ~fix_flags, fix_first)), result, 1);// out: (1 - mask') * in, mask' * in ; in: in, mask ; where mask' = (1 - fix_first) * mask 

	md_select_dims(N, ~shared_dims_flag, dims2, dims);
	result = nlop_chain2_FF(nlop_rand_mask_fixed_create(N, dims2, (0 < p ? 1. : -1) - p, bat_dims_flag), 0, result, 1);

	md_free(fix_first);

	return result;
}




const struct nlop_s* nlop_dropout_create(int N, const long dims[N], float p, unsigned long shared_dims_flag)
{
	long dims2[N];
	md_select_dims(N, ~shared_dims_flag, dims2, dims);

	return nlop_chain2_FF(nlop_rand_mask_create(N, dims2, p), 0, nlop_tenmul_create(N, dims, dims, dims2), 1);
}

struct noise_s {

	nlop_data_t super;

	int N;

	const long* noi_dims;
	const long* out_dims;

	float var;	//negative value means var is drawn from gaussian distribution with abs(var) (take magnitude)
	unsigned long shared_var_flag;
};

DEF_TYPEID(noise_s);


static void noise_fun(const nlop_data_t* _data, int Nargs, complex float* args[Nargs])
{
	const auto data = CAST_DOWN(noise_s, _data);

	assert(1 == Nargs);
	complex float* dst = args[0];

	complex float* tmp = md_alloc_sameplace(data->N, data->noi_dims, CFL_SIZE, dst);

	long noi_dims[data->N];
	md_select_dims(data->N, (data->shared_var_flag), noi_dims, data->noi_dims);

	long pos[data->N];
	md_singleton_strides(data->N, pos);

	do {
		complex float* tmp2 = md_alloc_sameplace(data->N, noi_dims, CFL_SIZE, dst);

		md_gaussian_rand(data->N, noi_dims, tmp2);

		float var = (0 > data->var) ? pow(fabs(crealf(gaussian_rand())) * sqrtf(-(data->var)), 2) : data->var;
		md_zsmul(data->N, noi_dims, tmp2, tmp2, sqrtf(var));

		md_copy2(data->N, noi_dims,
			MD_STRIDES(data->N, data->noi_dims, CFL_SIZE), &(MD_ACCESS(data->N, MD_STRIDES(data->N, data->noi_dims, CFL_SIZE), pos, tmp)),
			MD_STRIDES(data->N, noi_dims, CFL_SIZE), tmp2, CFL_SIZE);

		md_free(tmp2);

	} while (md_next(data->N, data->noi_dims, ~(data->shared_var_flag), pos));

	md_copy2(data->N, data->out_dims,
			MD_STRIDES(data->N, data->out_dims, CFL_SIZE), dst,
			MD_STRIDES(data->N, data->noi_dims, CFL_SIZE), tmp, CFL_SIZE);

	md_free(tmp);
}


static void noise_del(const struct nlop_data_s* _data)
{
	const auto data = CAST_DOWN(noise_s, _data);

	xfree(data->noi_dims);
	xfree(data->out_dims);

	xfree(data);
}


const struct nlop_s* nlop_noise_create(int N, const long dims[N], float var, unsigned long shared_dims_flag, unsigned long shared_var_flag)
{
	PTR_ALLOC(struct noise_s, data);
	SET_TYPEID(noise_s, data);

	data->N = N;

	long* out_dims = *TYPE_ALLOC(long[N]);
	long* noi_dims = *TYPE_ALLOC(long[N]);

	md_copy_dims(N, out_dims, dims);
	md_select_dims(N, ~shared_dims_flag, noi_dims, dims);

	data->out_dims = out_dims;
	data->noi_dims = noi_dims;
	data->var = var;
	data->shared_var_flag = shared_var_flag;

	long odims[1][N];
	md_copy_dims(N, odims[0], dims);

	return nlop_generic_create(1, N, odims, 0, 0, NULL, CAST_UP(PTR_PASS(data)), noise_fun, NULL, NULL, NULL, NULL, noise_del);
}

const struct nlop_s* nlop_add_noise_create(int N, const long dims[N], float var, unsigned long shared_dims_flag, unsigned long shared_var_flag)
{
	auto result = nlop_zaxpbz_create(N, dims, 1, 1);
	result = nlop_chain2_FF(nlop_noise_create(N, dims, var, shared_dims_flag, shared_var_flag), 0, result, 1);
	return result;
}

struct norm_max_abs_s {

	nlop_data_t super;

	int N;
	const long* dims;
	const long* sdims;

	complex float* inv_scale;
};

DEF_TYPEID(norm_max_abs_s);

static void norm_max_abs_fun(const nlop_data_t* _data, int D, complex float* args[D])
{
	assert(3 == D);
	complex float* dst = args[0];
	complex float* scale = args[1];
	complex float* src = args[2];

	const auto data = CAST_DOWN(norm_max_abs_s, _data);

	int N = data->N;
	const long* dims = data->dims;
	const long* sdims = data->sdims;

	if (NULL == data->inv_scale)
		data->inv_scale = md_alloc_sameplace(N, sdims, CFL_SIZE, dst);

#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src)));
#endif

	complex float* tmp = md_alloc_sameplace(N, dims, CFL_SIZE, dst);
	md_zabs(N, dims, tmp, src);

	md_copy2(N, sdims, MD_STRIDES(N, sdims, CFL_SIZE), scale, MD_STRIDES(N, dims, CFL_SIZE), tmp, CFL_SIZE);
	md_zmax2(N, dims, MD_STRIDES(N, sdims, CFL_SIZE), scale, MD_STRIDES(N, sdims, CFL_SIZE), scale, MD_STRIDES(N, dims, CFL_SIZE), tmp);

	md_free(tmp);

	complex float* ones = md_alloc_sameplace(N, sdims, CFL_SIZE, dst);

	md_zfill(N, sdims, ones, 1.);
	md_zdiv(N, sdims, data->inv_scale, ones, scale);

	md_zmul2(N, dims,
		MD_STRIDES(N, dims, CFL_SIZE), dst,
		MD_STRIDES(N, dims, CFL_SIZE), src,
		MD_STRIDES(N, sdims, CFL_SIZE), data->inv_scale);

	md_free(ones);
}

static void norm_max_abs_deradj(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(norm_max_abs_s, _data);

	int N = data->N;
	const long* dims = data->dims;
	const long* sdims = data->sdims;

	md_zmul2(N, dims,
		MD_STRIDES(N, dims, CFL_SIZE), dst,
		MD_STRIDES(N, dims, CFL_SIZE), src,
		MD_STRIDES(N, sdims, CFL_SIZE), data->inv_scale); //inv_scale is real -> selfadjoint
}

static void norm_max_abs_del(const struct nlop_data_s* _data)
{
	const auto data = CAST_DOWN(norm_max_abs_s, _data);

	xfree(data->dims);
	xfree(data->sdims);

	md_free(data->inv_scale);

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

	data->inv_scale = NULL;

	long nl_odims[2][N];
	md_copy_dims(N, nl_odims[0], dims);
	md_copy_dims(N, nl_odims[1], data->sdims);

	long nl_idims[1][N];
	md_copy_dims(N, nl_idims[0], dims);

	return nlop_generic_create(2, N, nl_odims, 1, N, nl_idims, CAST_UP(PTR_PASS(data)), norm_max_abs_fun, (nlop_der_fun_t[1][2]){ { norm_max_abs_deradj, NULL } }, (nlop_der_fun_t[1][2]){ { norm_max_abs_deradj, NULL } }, NULL, NULL, norm_max_abs_del);
}

struct norm_znorm_s {

	nlop_data_t super;

	int N;
	const long* dims;
	const long* sdims;

	complex float* inv_scale;
};

DEF_TYPEID(norm_znorm_s);

static void norm_znorm_fun(const nlop_data_t* _data, int D, complex float* args[D])
{
	assert(3 == D);
	complex float* dst = args[0];
	complex float* scale = args[1];
	complex float* src = args[2];

	const auto data = CAST_DOWN(norm_znorm_s, _data);

	int N = data->N;
	const long* dims = data->dims;
	const long* sdims = data->sdims;

	if (NULL == data->inv_scale)
		data->inv_scale = md_alloc_sameplace(N, sdims, CFL_SIZE, dst);

#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src)));
#endif
	md_ztenmulc(N, sdims, scale, dims, src, dims, src);
	md_zreal(N, sdims, scale, scale);
	md_sqrt(N + 1, MD_REAL_DIMS(N, sdims), (float*)scale, (float*)scale);

	//scale[0] = md_znorm(N, dims, src);

	md_zfill(N, sdims, data->inv_scale, 1);
	md_zdiv(N, sdims, data->inv_scale, data->inv_scale, scale);
	md_zmul2(N, dims, MD_STRIDES(N, dims, CFL_SIZE), dst, MD_STRIDES(N, dims, CFL_SIZE), src, MD_STRIDES(N, sdims, CFL_SIZE), data->inv_scale);
}

static void norm_znorm_deradj(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(norm_znorm_s, _data);

	int N = data->N;
	const long* dims = data->dims;
	const long* sdims = data->sdims;

	md_zmul2(N, dims,
		MD_STRIDES(N, dims, CFL_SIZE), dst,
		MD_STRIDES(N, dims, CFL_SIZE), src,
		MD_STRIDES(N, sdims, CFL_SIZE), data->inv_scale); //inv_scale is real -> selfadjoint
}

static void norm_znorm_del(const struct nlop_data_s* _data)
{
	const auto data = CAST_DOWN(norm_znorm_s, _data);

	xfree(data->dims);
	xfree(data->sdims);

	md_free(data->inv_scale);

	xfree(data);
}

const struct nlop_s* nlop_norm_znorm_create(int N, const long dims[N], unsigned long batch_flag)
{
	PTR_ALLOC(struct norm_znorm_s, data);
	SET_TYPEID(norm_znorm_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);

	PTR_ALLOC(long[N], sdims);
	md_select_dims(N, batch_flag, *sdims, dims);

	data->N = N;
	data->dims = *PTR_PASS(ndims);
	data->sdims = *PTR_PASS(sdims);

	data->inv_scale = NULL;

	long nl_odims[2][N];
	md_copy_dims(N, nl_odims[0], dims);
	md_copy_dims(N, nl_odims[1], data->sdims);

	long nl_idims[1][N];
	md_copy_dims(N, nl_idims[0], dims);

	return nlop_generic_create(2, N, nl_odims, 1, N, nl_idims, CAST_UP(PTR_PASS(data)), norm_znorm_fun, (nlop_der_fun_t[1][2]){ { norm_znorm_deradj, NULL } }, (nlop_der_fun_t[1][2]){ { norm_znorm_deradj, NULL } }, NULL, NULL, norm_znorm_del);

}


const struct nlop_s* nlop_norm_create(int N, const long dims[N], unsigned long batch_flag, enum norm norm, bool stop_grad)
{
	const struct nlop_s* result = NULL;

	switch (norm) {

	case NORM_MAX:
		result = nlop_norm_max_abs_create(N, dims, batch_flag);
		break;

	case NORM_L2:
		result = nlop_norm_znorm_create(N, dims, batch_flag);
		break;

	case NORM_NONE:
	default:
		error("No normalization selected!\n");
	}

	if (stop_grad)
		result = nlop_no_der_F(result, 1, 0);

	return result;
}

