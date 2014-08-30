/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2012-2014 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2014      Frank Ong <frankong@berkeley.edu>
 * 2014      Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <stdlib.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/linop.h"
#include "num/ops.h"
#include "num/someops.h"

#include "misc/debug.h"
#include "misc/misc.h"

#include "iter/iter.h"
#include "iter/iter2.h"

#include "lsqr.h"


const struct lsqr_conf lsqr_defaults = { 0. };


union italgo_fun_u {

	italgo_fun_t italgo;
	italgo_fun2_t italgo2;
};


struct lsqr_data {

	float l2_lambda;
	long size;

	const struct linop_s* model_op;
	const struct operator_p_s** prox_ops;
	const struct linop_s** G_ops;
};


static void normaleq_l2_apply(const void* _data, complex float* dst, const complex float* src)
{
	const struct lsqr_data* data = _data;

	linop_normal_unchecked(data->model_op, dst, src);

	md_axpy(1, MD_DIMS(data->size), (float*)dst, data->l2_lambda, (const float*)src);
}


/**
 * Helper function to be able to call lsqr and lsqr2 with different iter interfaces
 */
static void lsqr_priv( unsigned int N, const struct lsqr_conf* conf,
		const union italgo_fun_u* iter_u, void* iconf, bool use_iter,
		const struct linop_s* model_op,
		unsigned int num_funs,
		const struct operator_p_s** prox_funs,
		const struct linop_s** prox_linops,
		const long x_dims[N], _Complex float* x, 
		const long y_dims[N], const _Complex float* y,
		const complex float* x_truth,
		void* obj_eval_data,
		float (*obj_eval)(const void*, const float*))
{
	// -----------------------------------------------------------
	// normal equation right hand side

	complex float* x_adj = md_alloc_sameplace(N, x_dims, CFL_SIZE, y);
	linop_adjoint(model_op, N, x_dims, x_adj, y_dims, y);


	// -----------------------------------------------------------
	// initialize data: struct to hold all data and operators

	struct lsqr_data data;

	data.l2_lambda = conf->lambda;
	data.model_op = model_op;
	data.G_ops = prox_linops;
	data.prox_ops = prox_funs;
	data.size = 2 * md_calc_size(N, x_dims);


	// -----------------------------------------------------------
	// run recon

	const struct operator_s* normaleq_op = operator_create(N, x_dims, x_dims, (void*)&data, normaleq_l2_apply, NULL);

	if (use_iter) {
		assert(num_funs < 2);
		iter_u->italgo(iconf, normaleq_op, (NULL == prox_funs) ? NULL : prox_funs[0], data.size, (float*)x, (const float*)x_adj, (const float*)x_truth, obj_eval_data, obj_eval);
	}
	else
		iter_u->italgo2(iconf, normaleq_op, num_funs, prox_funs, prox_linops, NULL, data.size, (float*)x, (const float*)x_adj, (const float*)x_truth, obj_eval_data, obj_eval);


	// -----------------------------------------------------------
	// clean up
	
	md_free(x_adj);
	operator_free(normaleq_op);
}


/**
 * Perform iterative, multi-regularized least-squares reconstruction
 */
void lsqr2(	unsigned int N, const struct lsqr_conf* conf,
		italgo_fun2_t italgo, void* iconf,
		const struct linop_s* model_op,
		unsigned int num_funs,
		const struct operator_p_s** prox_funs,
		const struct linop_s** prox_linops,
		const long x_dims[N], _Complex float* x, 
		const long y_dims[N], const _Complex float* y,
		const complex float* x_truth,
		void* obj_eval_data,
		float (*obj_eval)(const void*, const float*))
{

	const union italgo_fun_u iter_u = { .italgo2 = italgo };

	lsqr_priv(N, conf, &iter_u, iconf, false, model_op, num_funs, prox_funs, prox_linops, x_dims, x, y_dims, y, x_truth, obj_eval_data, obj_eval);
}


/**
 * Perform iterative, regularized least-squares reconstruction.
 */
void lsqr(	unsigned int N, const struct lsqr_conf* conf,
		italgo_fun_t italgo, void* iconf,
		const struct linop_s* model_op,
		const struct operator_p_s* thresh_op,
		const long x_dims[N], complex float* x, 
		const long y_dims[N], const complex float* y)
{
	const union italgo_fun_u iter_u = { .italgo = italgo };
	
	lsqr_priv(N, conf, &iter_u, iconf, true, model_op, thresh_op == NULL ? 0 : 1, thresh_op == NULL ? NULL : &thresh_op, NULL, x_dims, x, y_dims, y, NULL, NULL, NULL);
}


//  A^H W W A - A^H W W y
void wlsqr(	unsigned int N, const struct lsqr_conf* conf,
		italgo_fun_t italgo, void* iconf,
		const struct linop_s* model_op,
		const struct operator_p_s* thresh_op,
		const long x_dims[N], complex float* x,
		const long y_dims[N], const complex float* y,
		const long w_dims[N], const complex float* w)
{
	unsigned int flags = 0;
	for (unsigned int i = 0; i < N; i++)
		if (1 < w_dims[i])
			flags |= (1 << i);

	struct linop_s* weights = linop_cdiag_create(N, y_dims, flags, w);
	struct linop_s* op = linop_chain(model_op, weights);

	complex float* wy = md_alloc_sameplace(N, y_dims, CFL_SIZE, y);

	linop_forward(weights, N, y_dims, wy, y_dims, y);

	lsqr(N, conf, italgo, iconf, op, thresh_op, x_dims, x, y_dims, wy);

	md_free(wy);

	linop_free(op);
	linop_free(weights);
}


/**
 * Wrapper for lsqr on GPU
 */
#ifdef USE_CUDA
extern void lsqr_gpu(	unsigned int N, const struct lsqr_conf* conf,
			italgo_fun_t italgo, void* iconf,
			const struct linop_s* model_op,
			const struct operator_p_s* thresh_op,
			const long x_dims[N], complex float* x, 
			const long y_dims[N], const complex float* y)
{

	complex float* gpu_y = md_gpu_move(N, y_dims, y, CFL_SIZE);
	complex float* gpu_x = md_gpu_move(N, x_dims, x, CFL_SIZE);

	lsqr(N, conf, italgo, iconf, model_op, thresh_op, x_dims, gpu_x, y_dims, gpu_y);

	md_copy(N, x_dims, x, gpu_x, CFL_SIZE);

	md_free(gpu_x);
	md_free(gpu_y);
}


#endif





/**
 * Wrapper for wlsqr on GPU
 */
#ifdef USE_CUDA
extern void wlsqr_gpu(	unsigned int N, const struct lsqr_conf* conf,
			italgo_fun_t italgo, void* iconf,
			const struct linop_s* model_op,
			const struct operator_p_s* thresh_op,
			const long x_dims[N], complex float* x, 
			const long y_dims[N], const complex float* y,
			const long w_dims[N], const complex float* w)
{

	complex float* gpu_y = md_gpu_move(N, y_dims, y, CFL_SIZE);
	complex float* gpu_x = md_gpu_move(N, x_dims, x, CFL_SIZE);
	complex float* gpu_w = md_gpu_move(N, w_dims, w, CFL_SIZE);

	wlsqr(N, conf, italgo, iconf, model_op, thresh_op, x_dims, gpu_x, y_dims, gpu_y, w_dims, gpu_w);

	md_copy(N, x_dims, x, gpu_x, CFL_SIZE);

	md_free(gpu_x);
	md_free(gpu_y);
	md_free(gpu_w);
}


#endif

/**
 * Wrapper for lsqr2 on GPU
 */
#ifdef USE_CUDA
extern void lsqr2_gpu(	unsigned int N, const struct lsqr_conf* conf,
			italgo_fun2_t italgo, void* iconf,
			const struct linop_s* model_op,
			unsigned int num_funs,
			const struct operator_p_s** prox_funs,
			const struct linop_s** prox_linops,
			const long x_dims[N], complex float* x,
			const long y_dims[N], const complex float* y,
			const complex float* x_truth,
			void* obj_eval_data,
			float (*obj_eval)(const void*, const float*))
{

	complex float* gpu_y = md_gpu_move(N, y_dims, y, CFL_SIZE);
	complex float* gpu_x = md_gpu_move(N, x_dims, x, CFL_SIZE);
	complex float* gpu_x_truth = md_gpu_move(N, x_dims, x_truth, CFL_SIZE);

	lsqr2(N, conf, italgo, iconf, model_op, num_funs, prox_funs, prox_linops, x_dims, gpu_x, y_dims, gpu_y, gpu_x_truth, obj_eval_data, obj_eval);

	md_copy(N, x_dims, x, gpu_x, CFL_SIZE);

	md_free(gpu_x_truth);
	md_free(gpu_x);
	md_free(gpu_y);
}


#endif
