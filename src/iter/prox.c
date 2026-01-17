/* Copyright 2014-2017. The Regents of the University of California.
 * Copyright 2016-2021. Uecker. University Medical Center Göttingen.
 * Copyright 2022-2025. Intitute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2017	Jon Tamir
 * 2016-2019	Martin Uecker
 */

#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include "num/multind.h"
#include "num/multiplace.h"
#include "num/flpmath.h"
#include "num/ops_p.h"
#include "num/ops.h"
#include "num/iovec.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "iter/iter.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "prox.h"


/**
 * Proximal function of f is defined as
 * (prox_f)(z) = arg min_x 0.5 || z - x ||_2^2 + f(x)
 *
 * (prox_{mu f})(z) = arg min_x 0.5 || z - x ||_2^2 + mu f(x)
 */



/**
 * Data for computing prox_leastsquares_fun:
 * Proximal function for f(z) = lambda / 2 || W * (y - z) ||_2^2.
 *
 * @param y
 * @param lambda regularization
 * @param size size of z
 */
struct prox_weighted_leastsquares_data {

	operator_data_t super;

	int N;

	const long* dims;
	struct multiplace_array_s* y;

	const long* wdims;
	struct multiplace_array_s* W;

	float lambda;
};

static DEF_TYPEID(prox_weighted_leastsquares_data);


/**
 * Proximal function for f(z) = lambda / 2 || W (y - z) ||_2^2.
 * Solution is z =  (mu * lambda * y + x_plus_u) / (mu * lambda * |W|^2 + 1)
 *
 * @param prox_data should be of type prox_weighted_leastsquares_data
 * @param mu proximal penalty
 * @param z output
 * @param x_plus_u input
 */
static void prox_weighted_leastsquares_fun(const operator_data_t* prox_data, float mu, complex float* z, const complex float* x_plus_u)
{
	auto pdata = CAST_DOWN(prox_weighted_leastsquares_data, prox_data);

	int N = pdata->N;
	const long* dims = pdata->dims;
	const long* wdims = pdata->wdims;

	const complex float* y = multiplace_read(pdata->y, z);
	const complex float* W_sqr = multiplace_read(pdata->W, z);

	if (z != x_plus_u)
		md_copy(N, dims, z, x_plus_u, CFL_SIZE);

	if (0 != mu) {

		if (NULL == W_sqr) {

			if (NULL != y)
				md_zaxpy(N, dims, z, pdata->lambda * mu, multiplace_read(pdata->y, z));

			md_zsmul(N, dims, z, z, 1. / (mu * pdata->lambda + 1));

		} else {

			complex float* tmp = md_alloc_sameplace(N, wdims, CFL_SIZE, z);
			md_zsmul(N, wdims, tmp, W_sqr, mu * pdata->lambda);

			if (NULL != y)
				md_zfmac2(N, dims, MD_STRIDES(N, dims, CFL_SIZE), z, MD_STRIDES(N, dims, CFL_SIZE), y, MD_STRIDES(N, wdims, CFL_SIZE), tmp);

			complex float* ones = md_alloc_sameplace(N, wdims, CFL_SIZE, z);
			md_zfill(N, wdims, ones, 1);

			md_zadd(N, wdims, tmp, tmp, ones);
			md_zdiv(N, wdims, tmp, ones, tmp);

			md_free(ones);

			md_zmul2(N, dims, MD_STRIDES(N, dims, CFL_SIZE), z, MD_STRIDES(N, dims, CFL_SIZE), z, MD_STRIDES(N, wdims, CFL_SIZE), tmp);
			md_free(tmp);
		}
	}
}


static void prox_weighted_leastsquares_del(const operator_data_t* _data)
{
	auto data = CAST_DOWN(prox_weighted_leastsquares_data, _data);

	multiplace_free(data->y);
	multiplace_free(data->W);

	xfree(data->dims);
	xfree(data->wdims);

	xfree(data);
}

const struct operator_p_s* prox_weighted_leastsquares_create(int N, const long dims[N], float lambda, const complex float* y, unsigned long flags, const complex float* W)
{
	PTR_ALLOC(struct prox_weighted_leastsquares_data, pdata);
	SET_TYPEID(prox_weighted_leastsquares_data, pdata);

	pdata->N = N;
	pdata->dims = ARR_CLONE(long[N], dims);

	pdata->y = (NULL == y) ? NULL : multiplace_move(N, dims, CFL_SIZE, y);
	pdata->lambda = lambda;

	long wdims[N];
	md_select_dims(N, flags, wdims, dims);
	pdata->wdims = ARR_CLONE(long[N], wdims);

	if (NULL != W) {

		complex float* tmp = md_alloc_sameplace(N, wdims, CFL_SIZE, W);
		md_zmulc(N, wdims, tmp, W, W);

		pdata->W = multiplace_move_F(N, wdims, CFL_SIZE, tmp);

	} else {

		pdata->W = NULL;
	}

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(pdata)), prox_weighted_leastsquares_fun, prox_weighted_leastsquares_del);
}

const struct operator_p_s* prox_leastsquares_create(int N, const long dims[N], float lambda, const complex float* y)
{
	return prox_weighted_leastsquares_create(N, dims, lambda, y, 0, NULL);
}


/**
 * Data for computing prox_l2norm_fun:
 * Proximal function for f(z) = lambda || z ||_2.
 *
 * @param lambda regularization
 * @param size size of z
 */
struct prox_l2norm_data {

	operator_data_t super;

	float lambda;
	long size;
};

static DEF_TYPEID(prox_l2norm_data);


/**
 * Proximal function for f(z) = lambda  || z ||_2.
 * Solution is z =  ( 1 - lambda * mu / norm(z) )_+ * z,
 * i.e. block soft thresholding
 *
 * @param prox_data should be of type prox_l2norm_data
 * @param mu proximal penalty
 * @param z output
 * @param x_plus_u input
 */
static void prox_l2norm_fun(const operator_data_t* prox_data, float mu, float* z, const float* x_plus_u)
{
	auto pdata = CAST_DOWN(prox_l2norm_data, prox_data);

	md_clear(1, MD_DIMS(pdata->size), z, FL_SIZE);

	double q1 = md_norm(1, MD_DIMS(pdata->size), x_plus_u);

	if (q1 != 0) {

		double q2 = 1 - pdata->lambda * mu / q1;

		if (q2 > 0.)
			md_smul(1, MD_DIMS(pdata->size), z, x_plus_u, q2);
	}
}

static void prox_l2norm_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	prox_l2norm_fun(_data, mu, (float*)dst, (const float*)src);
}

static void prox_l2norm_del(const operator_data_t* _data)
{
	xfree(CAST_DOWN(prox_l2norm_data, _data));
}

const struct operator_p_s* prox_l2norm_create(int N, const long dims[N], float lambda)
{
	PTR_ALLOC(struct prox_l2norm_data, pdata);
	SET_TYPEID(prox_l2norm_data, pdata);

	pdata->lambda = lambda;
	pdata->size = md_calc_size(N, dims) * 2;

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(pdata)), prox_l2norm_apply, prox_l2norm_del);
}



/**
 * Data for computing prox_l2ball_fun:
 * Proximal function for f(z) = Ind{ || y - z ||_2 < eps }
 *
 * @param y y
 * @param eps
 * @param size size of z
 */
struct prox_l2ball_data {

	operator_data_t super;

	struct multiplace_array_s* y;
	float eps;

	int N;
	unsigned long flags;
	const long* dims;

};

static DEF_TYPEID(prox_l2ball_data);


/**
 * Proximal function for f(z) = Ind{ || y - z ||_2 < eps }
 * Solution is y + (x - y) * q, where q = eps / norm(x - y) if norm(x - y) > eps, 1 o.w.
 *
 * @param prox_data should be of type prox_l2ball_data
 * @param mu proximal penalty
 * @param z output
 * @param x_plus_u input
 */
static void prox_l2ball_fun(const operator_data_t* prox_data, float /*mu*/, float* z, const float* x_plus_u)
{
	auto d = CAST_DOWN(prox_l2ball_data, prox_data);

	if (NULL != d->y)
		md_zsub(d->N, d->dims, (complex float*)z, (const complex float*)x_plus_u, multiplace_read(d->y, z));
	else
		md_copy(d->N, d->dims, z, x_plus_u, CFL_SIZE);

	long rdims[d->N];
	md_select_dims(d->N, d->flags, rdims, d->dims);
	complex float* q1 = md_alloc_sameplace(d->N, rdims, CFL_SIZE, z);
	complex float* q2 = md_alloc_sameplace(d->N, rdims, CFL_SIZE, z);

	md_ztenmulc(d->N, rdims, q1, d->dims, (complex float*)z, d->dims, (complex float*)z);
	md_zreal(d->N, rdims, q1, q1);
	md_zsqrt(d->N, rdims, q1, q1);
	md_zreal(d->N, rdims, q1, q1);

	md_zfill(d->N, rdims, q2, d->eps);
	md_zdiv(d->N, rdims, q2, q2, q1);
	md_zsmin(d->N, rdims, q2, q2, 1.);
	md_zreal(d->N, rdims, q2, q2);

	md_zmul2(d->N, d->dims, MD_STRIDES(d->N, d->dims, CFL_SIZE), (complex float*)z, MD_STRIDES(d->N, d->dims, CFL_SIZE), (complex float*)z, MD_STRIDES(d->N, rdims, CFL_SIZE), q2);

	md_free(q1);
	md_free(q2);

	if (NULL != d->y)
		md_zadd(d->N, d->dims, (complex float*)z, (complex float*)z, multiplace_read(d->y, z));
}

static void prox_l2ball_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	prox_l2ball_fun(_data, mu, (float*)dst, (const float*)src);
}

static void prox_l2ball_del(const operator_data_t* _data)
{
	auto data = CAST_DOWN(prox_l2ball_data, _data);

	multiplace_free(data->y);
	xfree(data->dims);

	xfree(data);
}

const struct operator_p_s* prox_l2ball2_create(int N, unsigned long flags, const long dims[N], float eps, const complex float* y)
{
	PTR_ALLOC(struct prox_l2ball_data, pdata);
	SET_TYPEID(prox_l2ball_data, pdata);

	pdata->y = (NULL == y) ? NULL : multiplace_move(N, dims, CFL_SIZE, y);
	pdata->eps = eps;
	pdata->N = N;
	pdata->flags = flags;
	pdata->dims = ARR_CLONE(long[N], dims);


	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(pdata)), prox_l2ball_apply, prox_l2ball_del);
}

const struct operator_p_s* prox_l2ball_create(int N, const long dims[N], float eps, const complex float* y)
{
	return prox_l2ball2_create(N, 0, dims, eps, y);
}



#if 0
/**
 * Data for computing prox_thresh_fun:
 * Proximal function for f(z) = lambda || z ||_1
 *
 * @param thresh function to apply SoftThresh
 * @param data data used by thresh function
 * @param lambda regularization
 */
struct prox_thresh_data {

	void (*thresh)(void* _data, float lambda, float* _dst, const float* _src);
	void* data;
	float lambda;
};

/**
 * Proximal function for f(z) = lambda || z ||_1
 * Solution is z = SoftThresh(x_plus_u, lambda * mu)
 *
 * @param prox_data should be of type prox_thresh_data
 */
void prox_thresh_fun(void* prox_data, float mu, float* z, const float* x_plus_u)
{
	struct prox_thresh_data* pdata = (struct prox_thresh_data*)prox_data;
	pdata->thresh(pdata->data, pdata->lambda * mu, z, x_plus_u);
}

static void prox_thresh_apply(const void* _data, float mu, complex float* dst, const complex float* src)
{
	prox_thresh_fun((void*)_data, mu, (float*)dst, (const float*)src);
}

static void prox_thresh_del(const void* _data)
{
	xfree((void*)_data);
}

const struct operator_p_s* prox_thresh_create(int N, const long dims[N], float lambda,
		void (*thresh)(void* _data, float lambda, float* _dst, const float* _src),
		void* data)
{
	PTR_ALLOC(struct prox_thresh_data, pdata);

	pdata->thresh = thresh;
	pdata->lambda = lambda;
	pdata->data = data;

	return operator_p_create(N, dims, dims, PTR_PASS(pdata), prox_thresh_apply, prox_thresh_del);
}
#endif


/**
 * Data for computing prox_zero_fun:
 * Proximal function for f(z) = 0
 *
 * @param size size of z
 */
struct prox_zero_data {

	operator_data_t super;

	long size;
};

static DEF_TYPEID(prox_zero_data);


/**
 * Proximal function for f(z) = 0
 * Solution is z = x_plus_u
 *
 * @param prox_data should be of type prox_zero_data
 * @param mu proximal penalty
 * @param z output
 * @param x_plus_u input
 */
static void prox_zero_fun(const operator_data_t* prox_data, float /*mu*/, float* z, const float* x_plus_u)
{
	auto pdata = CAST_DOWN(prox_zero_data, prox_data);

	md_copy(1, MD_DIMS(pdata->size), z, x_plus_u, FL_SIZE);
}

static void prox_zero_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	prox_zero_fun(_data, mu, (float*)dst, (const float*)src);
}

static void prox_zero_del(const operator_data_t* _data)
{
	xfree(CAST_DOWN(prox_zero_data, _data));
}

const struct operator_p_s* prox_zero_create(int N, const long dims[N])
{
	PTR_ALLOC(struct prox_zero_data, pdata);
	SET_TYPEID(prox_zero_data, pdata);

	pdata->size = md_calc_size(N, dims) * 2;

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(pdata)), prox_zero_apply, prox_zero_del);
}




/**
 * Data for computing prox_ineq_fun:
 * Proximal function for f(z) = 1{ z <= b }
 *  and f(z) = 1{ z >= b }
 *
 * @param b b
 * @param size size of z
 */
struct prox_ineq_data {

	operator_data_t super;

	const float* b;
	float a;
	long size;
	bool positive;
};

static DEF_TYPEID(prox_ineq_data);

static void prox_ineq_fun(const operator_data_t* _data, float /*mu*/, float* dst, const float* src)
{
	auto pdata = CAST_DOWN(prox_ineq_data, _data);

	if (NULL == pdata->b) {

		if (0. == pdata->a) {

			(pdata->positive ? md_smax : md_smin)(1, MD_DIMS(pdata->size), dst, src, 0.);

		} else {

			(pdata->positive ? md_smax : md_smin)(1, MD_DIMS(pdata->size), dst, src, pdata->a);

			md_zreal(1, MD_DIMS(pdata->size/2), (complex float*)dst, (complex float*)dst);
		}

	} else {

		(pdata->positive ? md_max : md_min)(1, MD_DIMS(pdata->size), dst, src, pdata->b);
	}
}

static void prox_ineq_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	prox_ineq_fun(_data, mu, (float*)dst, (const float*)src);
}

static void prox_ineq_del(const operator_data_t* _data)
{
	xfree(CAST_DOWN(prox_ineq_data, _data));
}

static const struct operator_p_s* prox_ineq_create(int N, const long dims[N], const complex float* b, float a, bool positive)
{
	PTR_ALLOC(struct prox_ineq_data, pdata);
	SET_TYPEID(prox_ineq_data, pdata);

	pdata->size = md_calc_size(N, dims) * 2;
	pdata->b = (const float*)b;
	pdata->a = a;
	pdata->positive = positive;

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(pdata)), prox_ineq_apply, prox_ineq_del);
}


/*
 * Proximal function for less than or equal to:
 * f(z) = 1{z <= b}
 */
const struct operator_p_s* prox_lesseq_create(int N, const long dims[N], const complex float* b)
{
	return prox_ineq_create(N, dims, b, 0., false);
}

/*
 * Proximal function for greater than or equal to:
 * f(z) = 1{z >= b}
 */
const struct operator_p_s* prox_greq_create(int N, const long dims[N], const complex float* b)
{
	return prox_ineq_create(N, dims, b, 0., true);
}

/*
 * Proximal function for nonnegative orthant
 * f(z) = 1{z >= 0}
 */
const struct operator_p_s* prox_nonneg_create(int N, const long dims[N])
{
	return prox_ineq_create(N, dims, NULL, 0., true);
}

/*
 * Proximal function for greater than or equal to a scalar:
 * f(z) = 1{z >= a}
 */
const struct operator_p_s* prox_zsmax_create(int N, const long dims[N], float a)
{
	return prox_ineq_create(N, dims, NULL, a, true);
}


struct prox_rvc_data {

	operator_data_t super;

	long size;
};

static DEF_TYPEID(prox_rvc_data);


static void prox_rvc_apply(const operator_data_t* _data, float /*mu*/, complex float* dst, const complex float* src)
{
	auto pdata = CAST_DOWN(prox_rvc_data, _data);

	md_zreal(1, MD_DIMS(pdata->size), dst, src);
}

static void prox_rvc_del(const operator_data_t* _data)
{
	xfree(CAST_DOWN(prox_rvc_data, _data));
}

/*
 * Proximal function for real-value constraint
 */
const struct operator_p_s* prox_rvc_create(int N, const long dims[N])
{
	PTR_ALLOC(struct prox_rvc_data, pdata);
	SET_TYPEID(prox_rvc_data, pdata);

	pdata->size = md_calc_size(N, dims);
	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(pdata)), prox_rvc_apply, prox_rvc_del);
}



