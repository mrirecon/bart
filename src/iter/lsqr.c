/* Copyright 2014. The Regents of the University of California.
 * Copyright 2016-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2012-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014      Frank Ong <frankong@berkeley.edu>
 * 2014      Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#include <complex.h>
#include <assert.h>
#include <stdlib.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops.h"
#include "num/iovec.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "misc/debug.h"
#include "misc/types.h"
#include "misc/misc.h"

#include "iter/iter.h"
#include "iter/iter2.h"
#include "iter/itop.h"

#include "lsqr.h"


const struct lsqr_conf lsqr_defaults = { .lambda = 0., .it_gpu = false };


struct lsqr_data {

	INTERFACE(operator_data_t);

	float l2_lambda;
	long size;

	const struct linop_s* model_op;
};

DEF_TYPEID(lsqr_data);

static void normaleq_l2_apply(const operator_data_t* _data, unsigned int N, void* args[static N])
{
	const struct lsqr_data* data = CAST_DOWN(lsqr_data, _data);

	assert(2 == N);

	linop_normal_unchecked(data->model_op, args[0], args[1]);

	md_axpy(1, MD_DIMS(data->size), args[0], data->l2_lambda, args[1]);
}

static void normaleq_del(const operator_data_t* _data)
{
	const struct lsqr_data* data = CAST_DOWN(lsqr_data, _data);

	linop_free(data->model_op);

	xfree(data);
}



/**
 * Operator for iterative, multi-regularized least-squares reconstruction
 */
const struct operator_s* lsqr2_create(const struct lsqr_conf* conf,
				      italgo_fun2_t italgo, iter_conf* iconf,
				      const float* init,
				      const struct linop_s* model_op,
				      const struct operator_s* precond_op,
			              unsigned int num_funs,
				      const struct operator_p_s* prox_funs[num_funs],
				      const struct linop_s* prox_linops[num_funs])
{
	PTR_ALLOC(struct lsqr_data, data);
	SET_TYPEID(lsqr_data, data);

	const struct iovec_s* iov = operator_domain(model_op->forward);

	data->l2_lambda = conf->lambda;
	data->model_op = linop_clone(model_op);
	data->size = 2 * md_calc_size(iov->N, iov->dims);	// FIXME: assume complex

	const struct operator_s* normaleq_op = operator_create(iov->N, iov->dims, iov->N, iov->dims, CAST_UP(PTR_PASS(data)), normaleq_l2_apply, normaleq_del);
	const struct operator_s* adjoint = operator_ref(model_op->adjoint);

	if (NULL != precond_op) {

		const struct operator_s* tmp;

		tmp = normaleq_op;
		normaleq_op = operator_chain(normaleq_op, precond_op);
		operator_free(tmp);

		tmp = adjoint;
		adjoint = operator_chain(adjoint, precond_op);
		operator_free(tmp);
	}

	const struct operator_s* itop_op = itop_create(italgo, iconf, init, normaleq_op, num_funs, prox_funs, prox_linops);

	if (conf->it_gpu)
		itop_op = operator_gpu_wrapper(itop_op);

	const struct operator_s* lsqr_op = operator_chain(adjoint, itop_op);

	operator_free(normaleq_op);
	operator_free(itop_op);
	operator_free(adjoint);

	return lsqr_op;
}



/**
 * Perform iterative, multi-regularized least-squares reconstruction
 */
void lsqr2(unsigned int N, const struct lsqr_conf* conf,
	   italgo_fun2_t italgo, iter_conf* iconf,
	   const struct linop_s* model_op,
	   unsigned int num_funs,
	   const struct operator_p_s* prox_funs[num_funs],
	   const struct linop_s* prox_linops[num_funs],
	   const long x_dims[static N], complex float* x,
	   const long y_dims[static N], const complex float* y,
	   const struct operator_s* precond_op,
	   struct iter_monitor_s* monitor)
{
#if 1
	// -----------------------------------------------------------
	// normal equation right hand side

	debug_printf(DP_DEBUG1, "lsqr: right hand side\n");

	complex float* x_adj = md_alloc_sameplace(N, x_dims, CFL_SIZE, y);
	linop_adjoint(model_op, N, x_dims, x_adj, N, y_dims, y);

	if (NULL != precond_op)
		operator_apply(precond_op, N, x_dims, x_adj, N, x_dims, x_adj);


	struct lsqr_data data = {

		.l2_lambda = conf->lambda,
		.model_op = model_op,
		.size = 2 * md_calc_size(N, x_dims),
	};

	SET_TYPEID(lsqr_data, &data);

	// -----------------------------------------------------------
	// run recon

	const struct operator_s* normaleq_op = operator_create(N, x_dims, N, x_dims, CAST_UP(&data), normaleq_l2_apply, NULL);

	if (NULL != precond_op) {

		const struct operator_s* tmp = normaleq_op;
		
		normaleq_op = operator_chain(normaleq_op, precond_op);
		operator_free(tmp);
	}
	

	debug_printf(DP_DEBUG1, "lsqr: solve normal equations\n");

	italgo(iconf, normaleq_op, num_funs, prox_funs, prox_linops, NULL, 
			NULL, data.size, (float*)x, (const float*)x_adj,
			monitor);


	// -----------------------------------------------------------
	// clean up
	
	md_free(x_adj);
	operator_free(normaleq_op);
#else
	// nicer, but is still missing some features
	const struct operator_s* op = lsqr2_create(conf, italgo, iconf, NULL, model_op, precond_op,
						num_funs, prox_funs, prox_linops);

	operator_apply(op, N, x_dims, x, N, y_dims, y);
	operator_free(op);
#endif
}




/**
 * Perform iterative, regularized least-squares reconstruction.
 */
void lsqr(unsigned int N,
	  const struct lsqr_conf* conf,
	  italgo_fun_t italgo,
	  iter_conf* iconf,
	  const struct linop_s* model_op,
	  const struct operator_p_s* thresh_op,
	  const long x_dims[static N],
	  complex float* x,
	  const long y_dims[static N],
	  const complex float* y,
	  const struct operator_s* precond_op)
{
	lsqr2(N, conf, iter2_call_iter, CAST_UP(&((struct iter_call_s){ { &TYPEID(iter_call_s) }, italgo, iconf })),
		model_op, (NULL != thresh_op) ? 1 : 0, &thresh_op, NULL,
		x_dims, x, y_dims, y, precond_op, NULL);
}


const struct operator_s* wlsqr2_create(	const struct lsqr_conf* conf,
					italgo_fun2_t italgo, iter_conf* iconf,
					const float* init,
					const struct linop_s* model_op,
					const struct linop_s* weights,
					const struct operator_s* precond_op,
					unsigned int num_funs,
					const struct operator_p_s* prox_funs[num_funs],
					const struct linop_s* prox_linops[num_funs])
{
	struct linop_s* op = linop_chain(model_op, weights);

	const struct operator_s* lsqr_op = lsqr2_create(conf, italgo, iconf, init,
						op, precond_op,
						num_funs, prox_funs, prox_linops);

	const struct operator_s* wlsqr_op = operator_chain(weights->forward, lsqr_op);

	operator_free(lsqr_op);
	linop_free(op);

	return wlsqr_op;
}


void wlsqr2(unsigned int N, const struct lsqr_conf* conf,
	    italgo_fun2_t italgo, iter_conf* iconf,
	    const struct linop_s* model_op,
	    unsigned int num_funs,
	    const struct operator_p_s* prox_funs[num_funs],
	    const struct linop_s* prox_linops[num_funs],
	    const long x_dims[static N], complex float* x,
	    const long y_dims[static N], const complex float* y,
	    const long w_dims[static N], const complex float* w,
	    const struct operator_s* precond_op)
{
	unsigned int flags = 0;
	for (unsigned int i = 0; i < N; i++)
		if (1 < w_dims[i])
			flags = MD_SET(flags, i);

	struct linop_s* weights = linop_cdiag_create(N, y_dims, flags, w);
#if 1
	struct linop_s* op = linop_chain(model_op, weights);

	complex float* wy = md_alloc_sameplace(N, y_dims, CFL_SIZE, y);

	linop_forward(weights, N, y_dims, wy, N, y_dims, y);

	lsqr2(N, conf, italgo, iconf, op, num_funs, prox_funs, prox_linops, x_dims, x, y_dims, wy, precond_op, NULL);

	md_free(wy);

	linop_free(op);
#else
	const struct operator_s* op = wlsqr2_create(conf, italgo, iconf, model_op, weights, precond_op,
						num_funs, prox_funs, prox_linops);

	operator_apply(op, N, x_dims, x, N, y_dims, y);
#endif
	linop_free(weights);
}

//  A^H W W A - A^H W W y
void wlsqr(unsigned int N, const struct lsqr_conf* conf,
	   italgo_fun_t italgo, iter_conf* iconf,
	   const struct linop_s* model_op,
	   const struct operator_p_s* thresh_op,
	   const long x_dims[static N], complex float* x,
	   const long y_dims[static N], const complex float* y,
	   const long w_dims[static N], const complex float* w,
	   const struct operator_s* precond_op)
{
	wlsqr2(N, conf, iter2_call_iter, CAST_UP(&((struct iter_call_s){ { &TYPEID(iter_call_s) }, italgo, iconf })),
	       model_op, (NULL != thresh_op) ? 1 : 0, &thresh_op, NULL,
	       x_dims, x, y_dims, y, w_dims, w, precond_op);
}
