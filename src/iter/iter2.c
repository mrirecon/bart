/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012, 2014 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2014	Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#include <complex.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define  NUM_INTERNAL

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/vecops.h"
#include "num/gpuops.h"
#include "num/iovec.h"
#include "num/ops.h"

#include "linops/linop.h"

#include "iter/italgos.h"
#include "iter/iter.h"
#include "iter/prox.h"
#include "iter/admm.h"
#include "iter/vec.h"

#include "iter2.h"







static bool checkeps(float eps)
{
	if (0. == eps) {

		debug_printf(DP_WARN, "Warning: data empty\n");
		return true;
	}

	if (!isnormal(eps)) {

		debug_printf(DP_WARN, "Warning: data corrupted\n");
		return true;
	}

	return false;
}


void iter2_conjgrad(void* _conf,
		const struct operator_s* normaleq_op,
		unsigned int D,
		const struct operator_p_s** prox_ops,
		const struct linop_s** ops,
		const struct operator_p_s* xupdate_op,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* obj_eval_data,
		float (*obj_eval)(const void*, const float*))
{

	assert(0 == D);
	assert(NULL == prox_ops);
	assert(NULL == ops);
	UNUSED(xupdate_op);

	struct iter_conjgrad_conf* conf = _conf;

	float eps = md_norm(1, MD_DIMS(size), image_adj);

	if (checkeps(eps))
		goto cleanup;

	conjgrad(conf->maxiter, conf->l2lambda, eps * conf->tol, size, (void*)normaleq_op, select_vecops(image_adj), operator_iter, image, image_adj, image_truth, obj_eval_data, obj_eval);

cleanup:
	;
}


void iter2_ist(void* _conf,
		const struct operator_s* normaleq_op,
		unsigned int D,
		const struct operator_p_s** prox_ops,
		const struct linop_s** ops,
		const struct operator_p_s* xupdate_op,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* obj_eval_data,
		float (*obj_eval)(const void*, const float*))
{

	assert(D == 1);
#if 0
	assert(NULL == ops);
#else
	UNUSED(ops);
#endif
	UNUSED(xupdate_op);

	struct iter_ist_conf* conf = _conf;

	float eps = md_norm(1, MD_DIMS(size), image_adj);

	if (checkeps(eps))
		goto cleanup;

	assert((conf->continuation >= 0.) && (conf->continuation <= 1.));

	ist(conf->maxiter, eps * conf->tol, conf->step, conf->continuation, conf->hogwild, size, (void*)normaleq_op, select_vecops(image_adj), operator_iter, operator_p_iter, (void*)prox_ops[0], image, image_adj, image_truth, obj_eval_data, obj_eval);


cleanup:
	;
}


void iter2_fista(void* _conf,
		const struct operator_s* normaleq_op,
		unsigned int D,
		const struct operator_p_s** prox_ops,
		const struct linop_s** ops,
		const struct operator_p_s* xupdate_op,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* obj_eval_data,
		float (*obj_eval)(const void*, const float*))
{

	assert(D == 1);
#if 0
	assert(NULL == ops);
#else
	UNUSED(ops);
#endif
	UNUSED(xupdate_op);

	struct iter_fista_conf* conf = _conf;

	float eps = md_norm(1, MD_DIMS(size), image_adj);

	if (checkeps(eps))
		goto cleanup;

	assert((conf->continuation >= 0.) && (conf->continuation <= 1.));

	fista(conf->maxiter, eps * conf->tol, conf->step, conf->continuation, conf->hogwild, size, (void*)normaleq_op, select_vecops(image_adj), operator_iter, operator_p_iter, (void*)prox_ops[0], image, image_adj, image_truth, obj_eval_data, obj_eval);

cleanup:
	;
}



void iter2_admm(void* _conf,
		const struct operator_s* normaleq_op,
		unsigned int D,
		const struct operator_p_s** prox_ops,
		const struct linop_s** ops,
		const struct operator_p_s* xupdate_op,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* obj_eval_data,
		float (*obj_eval)(const void*, const float*))
{
	struct iter_admm_conf* conf = _conf;

	struct admm_plan_s admm_plan = {

		.maxiter = conf->maxiter,
		.maxitercg = conf->maxitercg,
		.rho = conf->rho,
		.image_truth = image_truth,
		.num_funs = D,
		.do_warmstart = conf->do_warmstart,
		.dynamic_rho = conf->dynamic_rho,
		.hogwild = conf->hogwild,
		.ABSTOL = conf->ABSTOL,
		.RELTOL = conf->RELTOL,
		.alpha = conf->alpha,
		.tau = conf->tau,
		.mu = conf->mu,
		.fast = conf->fast,
	};


	struct admm_op a_ops[D];
	struct admm_prox_op a_prox_ops[D];

	for (unsigned int i = 0; i < D; i++) {

		a_ops[i].forward = linop_forward_iter;
		a_ops[i].normal = linop_normal_iter;
		a_ops[i].adjoint =  linop_adjoint_iter;
		a_ops[i].data = (void*)ops[i];

		a_prox_ops[i].prox_fun = operator_p_iter;
		a_prox_ops[i].data = (void*)prox_ops[i];
	}

	admm_plan.ops = a_ops;
	admm_plan.prox_ops = a_prox_ops;

	admm_plan.xupdate_fun = operator_p_iter;
	admm_plan.xupdate_data = (void*)xupdate_op;


	struct admm_history_s admm_history;

	long z_dims[D];

	for (unsigned int i = 0; i < D; i++)
		z_dims[i] = 2 * md_calc_size(linop_codomain(ops[i])->N, linop_codomain(ops[i])->dims);

	if (NULL != image_adj) {

		float eps = md_norm(1, MD_DIMS(size), image_adj);

		if (checkeps(eps))
			goto cleanup;
	}

	admm(&admm_history, &admm_plan, admm_plan.num_funs, z_dims, size, (float*)image, image_adj, select_vecops(image), operator_iter, (void*)normaleq_op, obj_eval_data, obj_eval);

cleanup:
	;

}


void iter2_pocs(void* _conf,
		const struct operator_s* normaleq_op,
		unsigned int D,
		const struct operator_p_s** prox_ops,
		const struct linop_s** ops,
		const struct operator_p_s* xupdate_op,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* obj_eval_data,
		float (*obj_eval)(const void*, const float*))
{

	const struct iter_pocs_conf* conf = _conf;

	assert(NULL == normaleq_op);
	assert(NULL == ops);
	assert(NULL == image_adj);

	UNUSED(xupdate_op);
	UNUSED(image_adj);
	
	struct pocs_proj_op proj_ops[D];

	for (unsigned int i = 0; i < D; i++) {

		proj_ops[i].proj_fun = operator_p_iter;
		proj_ops[i].data = (void*)prox_ops[i];
	}

	pocs(conf->maxiter, D, proj_ops, select_vecops(image), size, image, image_truth, obj_eval_data, obj_eval);
}


void iter2_call_iter(void* _conf,
		const struct operator_s* normaleq_op,
		unsigned int D,
		const struct operator_p_s** prox_ops,
		const struct linop_s** ops,
		const struct operator_p_s* xupdate_op,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* obj_eval_data,
		float (*obj_eval)(const void*, const float*))
{
	assert(D <= 1);
	assert(NULL == ops);

	UNUSED(xupdate_op);

	struct iter_call_s* it = _conf;
	it->fun(it->_conf, normaleq_op, (1 == D) ? prox_ops[0] : NULL,
		size, image, image_adj,
		image_truth, obj_eval_data, obj_eval);
}



