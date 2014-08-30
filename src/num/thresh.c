/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013-2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2013      Martin Uecker <uecker@eecs.berkeley.edu>
 */


#include <complex.h>
#include <stdbool.h>
#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/linop.h"
#include "num/ops.h"
#include "num/iovec.h"

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
 * @param tmp_norm temporary storage for norm computation
 * @param flags bitmask for joint thresholding
 * @param unitary_op linear operator if using unitary soft thresholding
 */
struct thresh_s {

	float lambda;

	int D;

	long* dim;
	long* str;

	complex float* tmp_norm;

	unsigned int flags;

	const struct linop_s* unitary_op;
};



static void softthresh_apply(const void* _data, float mu, complex float* optr, const complex float* iptr)
{
	const struct thresh_s* data = _data;

	if (0. == mu)
		md_copy(data->D, data->dim, optr, iptr, CFL_SIZE);
	else
		md_zsoftthresh_core2(data->D, data->dim, data->lambda * mu, data->flags, data->tmp_norm, data->str, optr, data->str, iptr);
}


static void unisoftthresh_apply(const void* _data, float mu, complex float* dst, const complex float* src)
{
	const struct thresh_s* data = _data;

	if (0. == mu)
		md_copy(data->D, data->dim, dst, src, CFL_SIZE);
	else
	{
		const long* transform_dims = linop_codomain(data->unitary_op)->dims;
		const long* transform_strs = linop_codomain(data->unitary_op)->strs;

		complex float* tmp = md_alloc_sameplace(data->D, transform_dims, CFL_SIZE, dst);

		linop_forward(data->unitary_op, data->D, transform_dims, tmp, data->dim, src);

		md_zsoftthresh_core2(data->D, transform_dims, data->lambda * mu, data->flags, data->tmp_norm, transform_strs, tmp, transform_strs, tmp);

		linop_adjoint(data->unitary_op, data->D, data->dim, dst, transform_dims, tmp);

		md_free(tmp);
	}
}

static void thresh_del(const void* _data)
{
	const struct thresh_s* data = _data;
	free(data->dim);
	free(data->str);
	md_free(data->tmp_norm);
	free((void*)data);
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
 * @param gpu true if using gpu, false if using cpu
 */
const struct operator_p_s* prox_thresh_create(unsigned int D, const long dim[D], const float lambda, const unsigned long flags, bool gpu)
{
	struct thresh_s* data = xmalloc(sizeof(struct thresh_s));

	data->lambda = lambda;
	data->D = D;
	data->flags = flags;
	data->unitary_op = NULL;

	data->dim = xmalloc(D * sizeof(long));
	md_copy_dims(D, data->dim, dim);

	// norm dimensions are the flagged input dimensions
	long norm_dim[D];
	md_select_dims(D, ~flags, norm_dim, data->dim);

	data->str = xmalloc(D * sizeof(long));
	md_calc_strides(D, data->str, data->dim, CFL_SIZE);

#ifdef USE_CUDA
	data->tmp_norm = (gpu ? md_alloc_gpu : md_alloc)(D, norm_dim, CFL_SIZE);
#else
	assert(!gpu);
	data->tmp_norm = md_alloc(D, norm_dim, CFL_SIZE);
#endif

	return operator_p_create(D, data->dim, data->dim, data, softthresh_apply, thresh_del);

}


/**
 * Proximal operator for l1-norm with unitary transform: f(x) = lambda || T x ||_1
 *
 * @param D number of dimensions
 * @param dim dimensions of x
 * @param lambda threshold parameter
 * @param unitary_op unitary linear operator
 * @param flags bitmask for joint soft-thresholding
 * @param gpu true if using gpu, false if using cpu
 */
extern const struct operator_p_s* prox_unithresh_create(unsigned int D, const struct linop_s* unitary_op, const float lambda, const unsigned long flags, _Bool gpu)
{
	struct thresh_s* data = xmalloc(sizeof(struct thresh_s));

	data->lambda = lambda;
	data->D = D;
	data->flags = flags;
	data->unitary_op = unitary_op;

	data->dim = xmalloc(D * sizeof(long));
	md_copy_dims(D, data->dim, linop_domain(unitary_op)->dims);

	data->str = xmalloc(D * sizeof(long));
	md_calc_strides(D, data->str, data->dim, CFL_SIZE);

	// norm dimensions are the flagged transform dimensions
	long norm_dim[D];
	md_select_dims(D, ~flags, norm_dim, linop_codomain(unitary_op)->dims);

#ifdef USE_CUDA
	data->tmp_norm = (gpu ? md_alloc_gpu : md_alloc)(D, norm_dim, CFL_SIZE);
#else
	assert(!gpu);
	data->tmp_norm = md_alloc(D, norm_dim, CFL_SIZE);
#endif

	return operator_p_create(D, data->dim, data->dim, data, unisoftthresh_apply, thresh_del);
}


void thresh_free(const struct operator_p_s* o)
{
	operator_p_free(o);
}




/**
 * Convenience function for applying soft thresholding
 *
 * @param o operator_p_s prox function from prox_thresh_create
 * @param optr destination -- soft thresholded values
 * @param iptr source -- values to be soft thresholded
 */
void softthresh(const struct operator_p_s* o, complex float* optr, const complex float* iptr)
{
	debug_printf(DP_WARN, "Warning: This function should not be used with italgos. Use operator_p_apply instead\n");
	operator_p_apply_unchecked(o, 1., optr, iptr);
}




void md_zsoftthresh_core2(unsigned int D, const long dims[D], float lambda, unsigned int flags, complex float* tmp_norm, const long ostrs[D], complex float* optr, const long istrs[D], const complex float* iptr)
{
	long norm_dims[D];
	long norm_strs[D];

	md_select_dims(D, ~flags, norm_dims, dims);
	md_calc_strides(D, norm_strs, norm_dims, sizeof(complex float));

	md_rss(D, dims, flags, tmp_norm, iptr);
	md_zsoftthresh_half2(D, norm_dims, lambda, norm_strs, tmp_norm, norm_strs, tmp_norm);
	md_zmul2(D, dims, ostrs, optr, norm_strs, tmp_norm, istrs, iptr);
}



/**
 * Soft thresholding using norm along arbitrary dimension (with strides)
 *
 * y = ST(x, lambda)
 * 1) computes resid = MAX( (norm(x) - lambda)/norm(x), 0 )
 * 2) multiplies y = resid * x
 *
 * @param D number of dimensions
 * @param dims dimensions of input/output
 * @param lambda threshold parameter
 * @param flags jointly thresholded dimensions
 * @param optr destination -- soft thresholded values
 * @param iptr source -- values to be soft thresholded
 */
void md_zsoftthresh2(unsigned int D, const long dims[D], float lambda, unsigned int flags, const long ostrs[D], complex float* optr, const long istrs[D], const complex float* iptr)
{
	long norm_dims[D];
	md_select_dims(D, ~flags, norm_dims, dims);

	complex float* tmp_norm = md_alloc_sameplace(D, norm_dims, sizeof(complex float), iptr);

	md_zsoftthresh_core2(D, dims, lambda, flags, tmp_norm, ostrs, optr, istrs, iptr);

	md_free(tmp_norm);
}



/**
 * Soft thresholding using norm along arbitrary dimension (without strides)
 *
 * y = ST(x, lambda)
 * 1) computes resid = MAX( (norm(x) - lambda)/norm(x), 0 )
 * 2) multiplies y = resid * x
 *
 * @param D number of dimensions
 * @param dims dimensions of input/output
 * @param lambda threshold parameter
 * @param flags jointly thresholded dimensions
 * @param optr destination -- soft thresholded values
 * @param iptr source -- values to be soft thresholded
 */
void md_zsoftthresh(unsigned int D, const long dims[D], float lambda, unsigned int flags, complex float* optr, const complex float* iptr)
{
	long strs[D];
	md_calc_strides(D, strs, dims, sizeof(complex float));
	md_zsoftthresh2(D, dims, lambda, flags, strs, optr, strs, iptr);
}



/**
 * Change the threshold parameter of the soft threshold function
 *
 * @param o soft threshold prox operator
 * @param lambda new threshold parameter
 */
void set_thresh_lambda(const struct operator_p_s* o, const float lambda)
{
	struct thresh_s* data = operator_p_get_data(o);
	data->lambda = lambda;
}


