/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2016-2018. Martin Uecker.
 * Copyright 2017. University of Oxford.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013-2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2013-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2017 Sofia Dimoudi <sofia.dimoudi@cardiov.ox.ac.uk>
 */


#include <complex.h>
#include <stdbool.h>
#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops_p.h"
#include "num/ops.h"
#include "num/iovec.h"

#include "linops/linop.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "thresh.h"


/**
 * Contains parameters for soft threshold functions
 *
 * @param lambda threshold parameter
 * @param D number of dimensions
 * @param dim dimensions of input
 * @param str strides of input
 * @param flags bitmask for joint thresholding
 * @param unitary_op linear operator if using unitary soft thresholding
 */
struct thresh_s {

	INTERFACE(operator_data_t);

	float lambda; //for soft thresholding
        unsigned int k; // for hard thresholding

	int D;

	const long* dim;
	const long* str;

	const long* norm_dim;

	unsigned int flags;

	const struct linop_s* unitary_op;
};

static DEF_TYPEID(thresh_s);



static void softthresh_apply(const operator_data_t* _data, float mu, complex float* optr, const complex float* iptr)
{
	const auto data = CAST_DOWN(thresh_s, _data);

	if (0. == mu) {

		md_copy(data->D, data->dim, optr, iptr, CFL_SIZE);

	} else {

		complex float* tmp_norm = md_alloc_sameplace(data->D, data->norm_dim, CFL_SIZE, optr);
		md_zsoftthresh_core2(data->D, data->dim, data->lambda * mu, data->flags, tmp_norm, data->str, optr, data->str, iptr);

		md_free(tmp_norm);
	}
}


static void unisoftthresh_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(thresh_s, _data);

	if (0. == mu) {

		md_copy(data->D, data->dim, dst, src, CFL_SIZE);

	} else {

		const long* transform_dims = linop_codomain(data->unitary_op)->dims;
		const long* transform_strs = linop_codomain(data->unitary_op)->strs;

		complex float* tmp = md_alloc_sameplace(data->D, transform_dims, CFL_SIZE, dst);

		linop_forward(data->unitary_op, data->D, transform_dims, tmp, data->D, data->dim, src);

		complex float* tmp_norm = md_alloc_sameplace(data->D, data->norm_dim, CFL_SIZE, dst);
		md_zsoftthresh_core2(data->D, transform_dims, data->lambda * mu, data->flags, tmp_norm, transform_strs, tmp, transform_strs, tmp);
		md_free(tmp_norm);

		linop_adjoint(data->unitary_op, data->D, data->dim, dst, data->D, transform_dims, tmp);

		md_free(tmp);
	}
}

static void hardthresh_apply(const operator_data_t* _data,  float mu, complex float* optr, const complex float* iptr)
{
	UNUSED(mu);
	const auto data = CAST_DOWN(thresh_s, _data);

	complex float* tmp_norm = md_alloc_sameplace(data->D, data->norm_dim, CFL_SIZE, optr);
	//only producing the support mask
	md_zhardthresh_mask2(data->D, data->dim, data->k, data->flags, tmp_norm, data->str, optr, data->str, iptr);

	md_free(tmp_norm);
}


static void thresh_del(const operator_data_t* _data)
{
	const auto data = CAST_DOWN(thresh_s, _data);

	xfree(data->dim);
	xfree(data->str);
	xfree(data->norm_dim);

	xfree(data);
}


/**
 * Proximal operator for l1-norm: f(x) = lambda || x ||_1
 * y = ST(x, lambda)
 * 1) computes resid = MAX( (norm(x) - lambda)/norm(x), 0 )
 * 2) multiplies y = resid * x
 *
 * @param D number of dimensions
 * @param dim dimensions of x
 * @param lambda threshold parameter
 * @param flags bitmask for joint soft-thresholding
 */
const struct operator_p_s* prox_thresh_create(unsigned int D, const long dim[D], const float lambda, const unsigned long flags)
{
	PTR_ALLOC(struct thresh_s, data);
	SET_TYPEID(thresh_s, data);

	data->lambda = lambda;
	data->D = D;
	data->flags = flags;
	data->unitary_op = NULL;

	PTR_ALLOC(long[D], ndim);
	md_copy_dims(D, *ndim, dim);
	data->dim = *PTR_PASS(ndim);

	// norm dimensions are the flagged input dimensions
	PTR_ALLOC(long[D], norm_dim);
	md_select_dims(D, ~flags, *norm_dim, data->dim);
	data->norm_dim = *PTR_PASS(norm_dim);

	PTR_ALLOC(long[D], nstr);
	md_calc_strides(D, *nstr, data->dim, CFL_SIZE);
	data->str = *PTR_PASS(nstr);

	return operator_p_create(D, dim, D, dim, CAST_UP(PTR_PASS(data)), softthresh_apply, thresh_del);
}


/**
 * Proximal operator for l1-norm with unitary transform: f(x) = lambda || T x ||_1
 *
 * @param D number of dimensions
 * @param dim dimensions of x
 * @param lambda threshold parameter
 * @param unitary_op unitary linear operator
 * @param flags bitmask for joint soft-thresholding
 */
extern const struct operator_p_s* prox_unithresh_create(unsigned int D, const struct linop_s* unitary_op, const float lambda, const unsigned long flags)
{
	PTR_ALLOC(struct thresh_s, data);
	SET_TYPEID(thresh_s, data);

	data->lambda = lambda;
	data->D = D;
	data->flags = flags;
	data->unitary_op = unitary_op;

	const long* dims = linop_domain(unitary_op)->dims;

	PTR_ALLOC(long[D], ndim);
	md_copy_dims(D, *ndim, dims);
	data->dim = *PTR_PASS(ndim);

	PTR_ALLOC(long[D], nstr);
	md_calc_strides(D, *nstr, data->dim, CFL_SIZE);
	data->str = *PTR_PASS(nstr);

	// norm dimensions are the flagged transform dimensions
	// FIXME should use linop_codomain(unitary_op)->N 
	PTR_ALLOC(long[D], norm_dim);
	md_select_dims(D, ~flags, *norm_dim, linop_codomain(unitary_op)->dims);
	data->norm_dim = *PTR_PASS(norm_dim);

	return operator_p_create(D, dims, D, dims, CAST_UP(PTR_PASS(data)), unisoftthresh_apply, thresh_del);
}

/**
 * Thresholding operator for l0-norm: f(x) =  || x ||_0 <= k, as used in NIHT algorithm.
 * y = HT(x, k) (hard thresholding, ie keeping the k largest elements).
 *
 * @param D number of dimensions
 * @param dim dimensions of x
 * @param k threshold parameter (non-zero elements to keep)
 * @param flags bitmask for joint thresholding
 */
const struct operator_p_s* prox_niht_thresh_create(unsigned int D, const long dim[D], const unsigned int k, const unsigned long flags)
{
	PTR_ALLOC(struct thresh_s, data);
	SET_TYPEID(thresh_s, data);

	data->lambda = 0.;
	data->k = k;
	data->D = D;
	data->flags = flags;
	data->unitary_op = NULL;

	PTR_ALLOC(long[D], ndim);
	md_copy_dims(D, *ndim, dim);
	data->dim = *PTR_PASS(ndim);

	// norm dimensions are the flagged input dimensions
	PTR_ALLOC(long[D], norm_dim);
	md_select_dims(D, ~flags, *norm_dim, data->dim);
	data->norm_dim = *PTR_PASS(norm_dim);

	PTR_ALLOC(long[D], nstr);
	md_calc_strides(D, *nstr, data->dim, CFL_SIZE);
	data->str = *PTR_PASS(nstr);

	return operator_p_create(D, dim, D, dim, CAST_UP(PTR_PASS(data)), hardthresh_apply, thresh_del);
}

void thresh_free(const struct operator_p_s* o)
{
	operator_p_free(o);
}






/**
 * Change the threshold parameter of the soft threshold function
 *
 * @param o soft threshold prox operator
 * @param lambda new threshold parameter
 */
void set_thresh_lambda(const struct operator_p_s* o, const float lambda)
{
	auto data = CAST_DOWN(thresh_s, operator_p_get_data(o));
	data->lambda = lambda;
}

/**
 * Returns the regularization parameter of the soft threshold function
 *
 * @param o soft threshold prox operator
 */
float get_thresh_lambda(const struct operator_p_s* o)
{
	auto data = CAST_DOWN(thresh_s, operator_p_get_data(o));
	return data->lambda;
}

