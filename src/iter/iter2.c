/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2016-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014	Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#include <complex.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/ops.h"

#include "linops/linop.h"

#include "iter/italgos.h"
#include "iter/iter.h"
#include "iter/prox.h"
#include "iter/admm.h"
#include "iter/vec.h"

#include "iter2.h"


void operator_iter(iter_op_data* _data, float* dst, const float* src)
{
	struct iter_op_op* data = CAST_DOWN(iter_op_op, _data);
	operator_apply_unchecked(data->op, (complex float*)dst, (const complex float*)src);
}

void operator_p_iter(iter_op_data* _data, float rho, float* dst, const float* src)
{
	struct iter_op_p_op* data = CAST_DOWN(iter_op_p_op, _data);
	operator_p_apply_unchecked(data->op, rho, (complex float*)dst, (const complex float*)src);
}




DEF_TYPEID(iter2_call_s);


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


void iter2_conjgrad(iter_conf* _conf,
		const struct operator_s* normaleq_op,
		unsigned int D,
		const struct operator_p_s* prox_ops[D],
		const struct linop_s* ops[D],
		const float* biases[D],
		const struct operator_p_s* xupdate_op,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor)
{

	assert(0 == D);
	assert(NULL == prox_ops);
	assert(NULL == ops);
	assert(NULL == biases);
	UNUSED(xupdate_op);

	struct iter_conjgrad_conf* conf = CAST_DOWN(iter_conjgrad_conf, _conf);

	float eps = md_norm(1, MD_DIMS(size), image_adj);

	if (checkeps(eps))
		goto cleanup;

	conjgrad(conf->maxiter, conf->l2lambda, eps * conf->tol, size, select_vecops(image_adj),
			OPERATOR2ITOP(normaleq_op), image, image_adj, monitor);

cleanup:
	;
}


void iter2_ist(iter_conf* _conf,
		const struct operator_s* normaleq_op,
		unsigned int D,
		const struct operator_p_s* prox_ops[D],
		const struct linop_s* ops[D],
		const float* biases[D],
		const struct operator_p_s* xupdate_op,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor)
{

	assert(D == 1);
	assert(NULL != prox_ops[0]);
	assert(NULL == biases);
#if 0
	assert(NULL == ops);
#else
	UNUSED(ops);
#endif
	UNUSED(xupdate_op);

	struct iter_ist_conf* conf = CAST_DOWN(iter_ist_conf, _conf);

	float eps = md_norm(1, MD_DIMS(size), image_adj);

	if (checkeps(eps))
		goto cleanup;

	assert((conf->continuation >= 0.) && (conf->continuation <= 1.));

	ist(conf->maxiter, eps * conf->tol, conf->step, conf->continuation, conf->hogwild, size, select_vecops(image_adj),
		OPERATOR2ITOP(normaleq_op), OPERATOR_P2ITOP(prox_ops[0]), image, image_adj, monitor);


cleanup:
	;
}


void iter2_fista(iter_conf* _conf,
		const struct operator_s* normaleq_op,
		unsigned int D,
		const struct operator_p_s* prox_ops[D],
		const struct linop_s* ops[D],
		const float* biases[D],
		const struct operator_p_s* xupdate_op,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor)
{

	assert(D == 1);
	assert(NULL == biases);
#if 0
	assert(NULL == ops);
#else
	UNUSED(ops);
#endif
	UNUSED(xupdate_op);

	struct iter_fista_conf* conf = CAST_DOWN(iter_fista_conf, _conf);

	float eps = md_norm(1, MD_DIMS(size), image_adj);

	if (checkeps(eps))
		goto cleanup;

	assert((conf->continuation >= 0.) && (conf->continuation <= 1.));

	fista(conf->maxiter, eps * conf->tol, conf->step, conf->continuation, conf->hogwild, size, select_vecops(image_adj),
		OPERATOR2ITOP(normaleq_op), OPERATOR_P2ITOP(prox_ops[0]), image, image_adj, monitor);

cleanup:
	;
}



void iter2_admm(iter_conf* _conf,
		const struct operator_s* normaleq_op,
		unsigned int D,
		const struct operator_p_s* prox_ops[D],
		const struct linop_s* ops[D],
		const float* biases[D],
		const struct operator_p_s* xupdate_op,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor)
{
	struct iter_admm_conf* conf = CAST_DOWN(iter_admm_conf, _conf);

	struct admm_plan_s admm_plan = {

		.maxiter = conf->maxiter,
		.maxitercg = conf->maxitercg,
		.cg_eps = conf->cg_eps,
		.rho = conf->rho,
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
		.biases = biases,
	};


	struct admm_op a_ops[D];
	struct iter_op_p_s a_prox_ops[D];

	for (unsigned int i = 0; i < D; i++) {

		a_ops[i].forward = OPERATOR2ITOP(ops[i]->forward),
		a_ops[i].normal = OPERATOR2ITOP(ops[i]->normal);
		a_ops[i].adjoint = OPERATOR2ITOP(ops[i]->adjoint);

		a_prox_ops[i] = OPERATOR_P2ITOP(prox_ops[i]);
	}

	admm_plan.ops = a_ops;
	admm_plan.prox_ops = a_prox_ops;

	admm_plan.xupdate = OPERATOR_P2ITOP(xupdate_op);


	long z_dims[D];

	for (unsigned int i = 0; i < D; i++)
		z_dims[i] = 2 * md_calc_size(linop_codomain(ops[i])->N, linop_codomain(ops[i])->dims);

	if (NULL != image_adj) {

		float eps = md_norm(1, MD_DIMS(size), image_adj);

		if (checkeps(eps))
			goto cleanup;
	}

	admm(&admm_plan, admm_plan.num_funs, z_dims, size, (float*)image, image_adj, select_vecops(image), OPERATOR2ITOP(normaleq_op), monitor);

cleanup:
	;

}


void iter2_pocs(iter_conf* _conf,
		const struct operator_s* normaleq_op,
		unsigned int D,
		const struct operator_p_s* prox_ops[D],
		const struct linop_s* ops[D],
		const float* biases[D],
		const struct operator_p_s* xupdate_op,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor)
{
	struct iter_pocs_conf* conf = CAST_DOWN(iter_pocs_conf, _conf);

	assert(NULL == normaleq_op);
	assert(NULL == ops);
	assert(NULL == biases);
	assert(NULL == image_adj);

	UNUSED(xupdate_op);
	UNUSED(image_adj);
	
	struct iter_op_p_s proj_ops[D];

	for (unsigned int i = 0; i < D; i++)
		proj_ops[i] = OPERATOR_P2ITOP(prox_ops[i]);

	pocs(conf->maxiter, D, proj_ops, select_vecops(image), size, image, monitor);
}


void iter2_call_iter(iter_conf* _conf,
		const struct operator_s* normaleq_op,
		unsigned int D,
		const struct operator_p_s* prox_ops[D],
		const struct linop_s* ops[D],
		const float* biases[D],
		const struct operator_p_s* xupdate_op,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor)
{
	assert(D <= 1);
	assert(NULL == ops);
	assert(NULL == biases);

	UNUSED(xupdate_op);

	struct iter_call_s* it = CAST_DOWN(iter_call_s, _conf);

	it->fun(it->_conf, normaleq_op, (1 == D) ? prox_ops[0] : NULL,
		size, image, image_adj, monitor);
}



