/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */


#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops_p.h"
#include "num/ops.h"
#include "num/iovec.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "misc/misc.h"

#include "proj.h"



/**
 * We write projections to a set M as proximal functions,
 * i.e. f(x) = 0 if x in M and infinity else.
 *
 * Proximal function of f is defined as
 * (prox_f)(z) = arg min_x 0.5 || z - x ||_2^2 + f(x)
 *
 * (prox_{mu f})(z) = arg min_x 0.5 || z - x ||_2^2 + mu f(x)
 */



/**
 * Data for computing proj_pos_real_fun
 *
 * @param N number of dimensions
 * @param dims dimensions
 */

 struct proj_pos_real_s {

	INTERFACE(operator_data_t);
	long N;
	const long* dims;

	float min;
};

DEF_TYPEID(proj_pos_real_s);

static void proj_pos_real_fun(const operator_data_t* _data, unsigned int N, void* args[N])
{
	assert(2 == N);
	const auto data = CAST_DOWN(proj_pos_real_s, _data);

	complex float* dst = args[0];
	const complex float* src = args[1];

	md_zsmax(data->N, data->dims, dst, src, data->min);
}

static void proj_pos_real_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	UNUSED(mu);
	proj_pos_real_fun(_data, 2, MAKE_ARRAY((void*)dst, (void*)src));
}

static void proj_pos_real_del(const operator_data_t* _data)
{
	const auto data = CAST_DOWN(proj_pos_real_s, _data);
	xfree(data->dims);
	xfree(data);
}

/**
 * Create operator projecting inputs to positive real values
 *
 * @param N
 * @param dims
 */
const struct operator_p_s* operator_project_pos_real_create(long N, const long dims[N])
{
	PTR_ALLOC(struct proj_pos_real_s, data);
	SET_TYPEID(proj_pos_real_s, data);

	data->N = N;
	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);
	data->dims = *PTR_PASS(ndims);
	data->min = 0;

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), proj_pos_real_apply, proj_pos_real_del);
}

/**
 * Create operator projecting inputs to real values larger min
 *
 * @param N
 * @param dims
 * @param min
 */
const struct operator_p_s* operator_project_min_real_create(long N, const long dims[N], float min)
{
	PTR_ALLOC(struct proj_pos_real_s, data);
	SET_TYPEID(proj_pos_real_s, data);

	data->N = N;
	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);
	data->dims = *PTR_PASS(ndims);
	data->min = min;

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), proj_pos_real_apply, proj_pos_real_del);
}


/**
 * Data for computing proj_mean_free_fun
 *
 * @param N number of dimensions
 * @param dims dimensions
 * @param bflag independent dimensions
 */

struct proj_mean_free_s {

	INTERFACE(operator_data_t);
	long N;
	const long* dims;
	unsigned long bflag;
};

DEF_TYPEID(proj_mean_free_s);

static void proj_mean_free_fun(const operator_data_t* _data, unsigned int N, void* args[N])
{
	assert(2 == N);
	const auto data = CAST_DOWN(proj_mean_free_s, _data);

	complex float* dst = args[0];
	const complex float* src = args[1];

	long batch_dims[data->N];
	long mf_dims[data->N];
	md_select_dims(data->N, data->bflag, batch_dims, data->dims);
	md_select_dims(data->N, ~data->bflag, mf_dims, data->dims);

	complex float* tmp = md_alloc_sameplace(data->N, batch_dims, CFL_SIZE, dst);

	md_zsum(data->N, data->dims, ~data->bflag, tmp, src);
	md_zsmul(data->N, batch_dims, tmp, tmp, 1./(float)md_calc_size(data->N, mf_dims));
	md_zsub2(data->N, data->dims, MD_STRIDES(data->N, data->dims, CFL_SIZE), dst, MD_STRIDES(data->N, data->dims, CFL_SIZE), src, MD_STRIDES(data->N, batch_dims, CFL_SIZE), tmp);
	md_free(tmp);
}

static void proj_mean_free_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	UNUSED(mu);
	proj_mean_free_fun(_data, 2, MAKE_ARRAY((void*)dst, (void*)src));
}


static void proj_mean_free_del(const operator_data_t* _data)
{
	const auto data = CAST_DOWN(proj_mean_free_s, _data);
	xfree(data->dims);
	xfree(data);
}

/**
 * Create operator subtracting the mean value
 * Dimensions selected by bflag stay independent
 *
 * @param N
 * @param dims
 * @param bflag batch dims -> dimensions which stay independent
 */
const struct operator_p_s* operator_project_mean_free_create(long N, const long dims[N], unsigned long bflag)
{
	PTR_ALLOC(struct proj_mean_free_s, data);
	SET_TYPEID(proj_mean_free_s, data);

	data->N = N;
	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);
	data->dims = *PTR_PASS(ndims);
	data->bflag = bflag;

	return operator_p_create(N, dims, N , dims, CAST_UP(PTR_PASS(data)), proj_mean_free_apply, proj_mean_free_del);
}


/**
 * Data for computing proj_sphere_fun
 *
 * @param N number of dimensions
 * @param dims dimensions
 * @param bflag independent dimensions
 */

struct proj_sphere_s {

	INTERFACE(operator_data_t);
	long N;
	const long* dims;
	unsigned long bflag;
};

DEF_TYPEID(proj_sphere_s);

static void proj_sphere_real_fun(const struct operator_data_s* _data, unsigned int N, void* args[N])
{
	assert(2 == N);
	assert(args[0] != args[1]);
	const auto data = CAST_DOWN(proj_sphere_s, _data);

	complex float* dst = args[0];
	const complex float* src = args[1];

	long bdims[data->N];
	md_select_dims(data->N, data->bflag, bdims, data->dims);
	complex float* tmp = md_alloc_sameplace(data->N, bdims, CFL_SIZE, dst);

	md_zrmul(data->N, data->dims, dst, src, src);
	md_zsum(data->N, data->dims, ~data->bflag, tmp, dst);

	long rdims[data->N + 1];
	long brdims[data->N + 1];
	rdims[0] = 2;
	brdims[0] = 2;
	md_copy_dims(data->N, rdims + 1, data->dims);
	md_copy_dims(data->N, brdims + 1, bdims);

	md_sqrt(data->N + 1, brdims, (float*)tmp, (float*)tmp);
	md_copy2(data->N, data->dims, MD_STRIDES(data->N, data->dims, CFL_SIZE), dst, MD_STRIDES(data->N, bdims, CFL_SIZE), tmp, CFL_SIZE);
	md_div(data->N + 1, rdims, (float*)dst, (float*)src, (float*)dst);
	md_free(tmp);
}

static void proj_sphere_complex_fun(const struct operator_data_s* _data, unsigned int N, void* args[N])
{
	assert(2 == N);
	assert(args[0] != args[1]);
	const auto data = CAST_DOWN(proj_sphere_s, _data);

	complex float* dst = args[0];
	const complex float* src = args[1];

	long bdims[data->N];
	md_select_dims(data->N, data->bflag, bdims, data->dims);

	complex float* tmp = md_alloc_sameplace(data->N, bdims, CFL_SIZE, dst);
	md_zmulc(data->N, data->dims, dst, src, src);
	md_zsum(data->N, data->dims, ~data->bflag, tmp, dst);

	md_clear(data->N, bdims, dst, FL_SIZE);
	md_real(data->N, bdims, (float*)dst, tmp); // I don't trust zmulc to have vanishing imag on gpu
	md_sqrt(data->N, bdims, (float*)tmp, (float*)dst);//propably more efficient than md_zsqrt

	md_clear(data->N, data->dims, dst, CFL_SIZE);
	md_copy2(data->N, data->dims, MD_STRIDES(data->N, data->dims, CFL_SIZE), dst, MD_STRIDES(data->N, bdims, FL_SIZE), tmp, FL_SIZE);
	md_zdiv(data->N, data->dims, dst, src, dst);
	md_free(tmp);
}

static void proj_sphere_real_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	UNUSED(mu);
	proj_sphere_real_fun(_data, 2, MAKE_ARRAY((void*)dst, (void*)src));
}

static void proj_sphere_complex_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	UNUSED(mu);
	proj_sphere_complex_fun(_data, 2, MAKE_ARRAY((void*)dst, (void*)src));
}

static void proj_sphere_del(const operator_data_t* _data)
{
	const auto data = CAST_DOWN(proj_sphere_s, _data);
	xfree(data->dims);
	xfree(data);
}

/**
 * Create operator scaling inputs to unit sphere
 *
 * @param N
 * @param dims
 * @param bflag
 * @param real if true, real and imaginary part are handeled independently (as bflag is set for dimension real/imag)
 */
const struct operator_p_s* operator_project_sphere_create(long N, const long dims[N], unsigned long bflag, bool real)
{
	PTR_ALLOC(struct proj_sphere_s, data);
	SET_TYPEID(proj_sphere_s, data);

	data->N = N;
	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);
	data->dims = *PTR_PASS(ndims);
	data->bflag = bflag;

	return operator_p_create(N, dims, N , dims, CAST_UP(PTR_PASS(data)), (real ? proj_sphere_real_apply : proj_sphere_complex_apply), proj_sphere_del);
}

/**
 * Create operator projectiong to mean free unit sphere by first subtrct the mean and scale to unitsphere afterwards
 * Real and imaginary part are considered independently
 *
 * @param N
 * @param dims
 * @param bflag
 * @param real if real, real and imaginary part are handeled independently (as bflag is set for dimension real/imag)
 */

const struct operator_p_s* operator_project_mean_free_sphere_create(long N, const long dims[N], unsigned long bflag, bool real)
{
	auto op_p_mean_free = operator_project_mean_free_create(N, dims, bflag);
	auto op_p_sphere = operator_project_sphere_create(N, dims, bflag, real);
	auto op_sphere = operator_p_bind(op_p_sphere, 1.);
	auto result = operator_p_pst_chain(op_p_mean_free, op_sphere);

	operator_p_free(op_p_mean_free);
	operator_p_free(op_p_sphere);
	operator_free(op_sphere);

	return result;
}
