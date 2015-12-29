/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * 2012-2014 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2014      Frank Ong <frankong@berkeley.edu>
 */

#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "linops/linop.h"

#include "misc/debug.h"
#include "misc/misc.h"

#include "iter/iter.h"
#include "iter/lsqr.h"

#include "lad.h"



const struct lad_conf lad_defaults = { 5, 0.1, ~0u, &lsqr_defaults };








/**
 * Perform iterative, regularized least-absolute derivation reconstruction.
 */
void lad2(	unsigned int N, const struct lad_conf* conf,
		italgo_fun2_t italgo, void* iconf,
		const struct linop_s* model_op,
		unsigned int num_funs,
		const struct operator_p_s* thresh_op[num_funs],
		const struct linop_s* thresh_funs[num_funs],
		const long x_dims[N], complex float* x, 
		const long y_dims[N], const complex float* y)
{
	long w_dims[N];
	md_select_dims(N, conf->wflags, w_dims, y_dims);
	
	complex float* weights = md_alloc_sameplace(N, w_dims, CFL_SIZE, y);
	complex float* tmp2 = md_alloc_sameplace(N, y_dims, CFL_SIZE, y);

	// use iterative reweigted least-squares
	// ADMM may be a better choice though...

	for (int i = 0; i < conf->rwiter; i++) {

		// recompute weights

		linop_forward(model_op, N, y_dims, tmp2, N, x_dims, x);
		md_zsub(N, y_dims, tmp2, tmp2, y);

		md_zrss(N, y_dims, ~(conf->wflags), weights, tmp2);

		for (long l = 0; l < md_calc_size(N, w_dims); l++)
			if (weights[l] != 0.)
				weights[l] = 1. / sqrtf(MAX(conf->gamma, cabsf(weights[l])));

		// solve weighted least-squares

		wlsqr2(N, conf->lsqr_conf, italgo, iconf, model_op,
				1, thresh_op, thresh_funs,
				x_dims, x, y_dims, y, w_dims, weights);
	}
		
	md_free(tmp2);
	md_free(weights);
}


/**
 * Perform iterative, regularized least-absolute derivation reconstruction.
 */
void lad(	unsigned int N, const struct lad_conf* conf,
		italgo_fun_t italgo, void* iconf,
		const struct linop_s* model_op,
		const struct operator_p_s* thresh_op,
		const long x_dims[N], complex float* x,
		const long y_dims[N], const complex float* y)
{
	lad2(N, conf, iter2_call_iter, &(struct iter_call_s){ italgo, iconf },
		model_op, (NULL != thresh_op) ? 1 : 0, &thresh_op, NULL,
		x_dims, x, y_dims, y);
}


/**
 * Wrapper for lsqr on GPU
 */
#ifdef USE_CUDA
extern void lad_gpu(	unsigned int N, const struct lad_conf* conf,
			italgo_fun_t italgo, void* iconf,
			const struct linop_s* model_op,
			const struct operator_p_s* thresh_op,
			const long x_dims[N], complex float* x,
			const long y_dims[N], const complex float* y)
{

	complex float* gpu_y = md_gpu_move(N, y_dims, y, CFL_SIZE);
	complex float* gpu_x = md_gpu_move(N, x_dims, x, CFL_SIZE);

	lad(N, conf, italgo, iconf, model_op, thresh_op, x_dims, gpu_x, y_dims, gpu_y);

	md_copy(N, x_dims, x, gpu_x, CFL_SIZE);

	md_free(gpu_x);
	md_free(gpu_y);
}
#endif

