/* * Copyright 2024. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <assert.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/ops.h"
#include "num/ops_p.h"
#include "num/iovec.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "prox3.h"

/**
 * Proximal function of f is defined as
 * (prox_f)(z) = arg min_x 0.5 || z - x ||_2^2 + f(x)
 *
 * (prox_{mu f})(z) = arg min_x 0.5 || z - x ||_2^2 + mu f(x)
 */



/**
 * Data for computing prox of convex conjugate with moreau identity:
 * Proximal function for f*(z)
 *
 * @param op prox of f
 */
struct prox_convex_conjugate_data {

	operator_data_t super;
	const struct operator_p_s* op;
};

static DEF_TYPEID(prox_convex_conjugate_data);

/**
 * Proximal function for f*(z)
 * Solution is prox_{mu f*}(z) =  z - mu prox_{1/mu f}(z / mu)
 *
 * @param prox_data should be of type prox_convex_conjugate_data
 * @param mu proximal penalty
 * @param z output
 * @param x_plus_u input
 */

static void prox_convex_conjugate_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(prox_convex_conjugate_data, _data);
	auto iov = operator_p_domain(data->op);

	complex float* tmp1 = md_alloc_sameplace(iov->N, iov->dims, iov->size, dst);
	complex float* tmp2 = md_alloc_sameplace(iov->N, iov->dims, iov->size, dst);

	md_zsmul(iov->N, iov->dims, tmp1, src, 1. / mu);
	operator_p_apply_unchecked(data->op, 1. / mu, tmp2, tmp1);
	md_zsmul(iov->N, iov->dims, tmp2, tmp2, mu);
	md_zsub(iov->N, iov->dims, dst, src, tmp2);

	md_free(tmp1);
	md_free(tmp2);
}

static void prox_convex_conjugate_del(const operator_data_t* _data)
{
	auto pdata = CAST_DOWN(prox_convex_conjugate_data, _data);

	operator_p_free(pdata->op);
	xfree(pdata);
}

const struct operator_p_s* prox_convex_conjugate(const struct operator_p_s* op)
{
	PTR_ALLOC(struct prox_convex_conjugate_data, pdata);
	SET_TYPEID(prox_convex_conjugate_data, pdata);

	pdata->op = operator_p_ref(op);
	auto iov = operator_p_domain(op);

	return operator_p_create(iov->N, iov->dims, iov->N, iov->dims, CAST_UP(PTR_PASS(pdata)), prox_convex_conjugate_apply, prox_convex_conjugate_del);
}

const struct operator_p_s* prox_convex_conjugate_F(const struct operator_p_s* op)
{
	const struct operator_p_s* tmp = prox_convex_conjugate(op);
	operator_p_free(op);
	return tmp;
}




