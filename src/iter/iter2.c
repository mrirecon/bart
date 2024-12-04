/* Copyright 2013-2018. The Regents of the University of California.
 * Copyright 2016-2019. Martin Uecker.
 * Copyright 2017. University of Oxford.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014-2018 Jon Tamir <jtamir@eecs.berkeley.edu>
 * 2017 Sofia Dimoudi <sofia.dimoudi@cardiov.ox.ac.uk>
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
#include "num/ops_p.h"
#include "num/ops.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "iter/italgos.h"
#include "iter/iter.h"
#include "iter/prox.h"
#include "iter/admm.h"
#include "iter/vec.h"
#include "iter/niht.h"
#include "iter/misc.h"

#include "iter2.h"


void operator_iter(iter_op_data* _data, float* dst, const float* src)
{
	auto data = CAST_DOWN(iter_op_op, _data);
	operator_apply_unchecked(data->op, (complex float*)dst, (const complex float*)src);
}

void operator_p_iter(iter_op_data* _data, float rho, float* dst, const float* src)
{
	auto data = CAST_DOWN(iter_op_p_op, _data);
	operator_p_apply_unchecked(data->op, rho, (complex float*)dst, (const complex float*)src);
}




DEF_TYPEID(iter_op_op);
DEF_TYPEID(iter_op_p_op);
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


static bool check_ops(long size,
	const struct operator_s* normaleq_op,
	int D,
	const struct operator_p_s* prox_ops[D],
	const struct linop_s* ops[D])
{
	bool ret = true;

	const struct iovec_s* iov = NULL;

	if (NULL != normaleq_op) {

		auto dom = operator_domain(normaleq_op);

		iov = operator_domain(normaleq_op);

		if (size != 2 * md_calc_size(dom->N, dom->dims))
			ret = false;

		auto cod = operator_codomain(normaleq_op);

		if (!iovec_compare(iov, cod))
			ret = false;
	}

	for (int i = 0; i < D; i++) {

		const struct iovec_s* cod = NULL;

		if ((NULL != ops) && (NULL != ops[i])) {

			auto dom = linop_domain(ops[i]);

			if (size != 2 * md_calc_size(dom->N, dom->dims))
				ret = false;

			if (NULL != iov) {

				if (!iovec_compare(iov, dom))
					ret = false;
			} else {

				iov = dom;
			}

			cod = linop_codomain(ops[i]);
		}

		if ((NULL != prox_ops) && (NULL != prox_ops[i])) {

			auto dom2 = operator_p_domain(prox_ops[i]);

			if (NULL != cod) {

				if (!iovec_compare(cod, dom2))
					ret = false;
			} else {

				if (size != 2 * md_calc_size(dom2->N, dom2->dims))
					ret = false;

				if (NULL != iov) {

					if (!iovec_compare(iov, dom2))
						ret = false;
				} else {

					iov = dom2;
				}
			}
		}
	}

	if (!ret) {

		iov = NULL;

		if (NULL != normaleq_op) {

			debug_printf(DP_INFO, "normaleq_op:\n");

			iov = operator_domain(normaleq_op);
			debug_print_dims(DP_INFO, iov->N, iov->dims);
			iov = operator_codomain(normaleq_op);
			debug_print_dims(DP_INFO, iov->N, iov->dims);
		}

		for (int i = 0; i < D; i++) {

			if ((NULL != ops) && (NULL != ops[i])) {

				debug_printf(DP_INFO, "linop[%d]:\n", i);
				iov = linop_domain(ops[i]);
				debug_print_dims(DP_INFO, iov->N, iov->dims);
				iov = linop_codomain(ops[i]);
				debug_print_dims(DP_INFO, iov->N, iov->dims);
			}

			if ((NULL != prox_ops) && (NULL != prox_ops[i])) {

				debug_printf(DP_INFO, "prox[%d]:\n", i);
				iov = operator_p_domain(prox_ops[i]);
				debug_print_dims(DP_INFO, iov->N, iov->dims);
				iov = operator_p_codomain(prox_ops[i]);
				debug_print_dims(DP_INFO, iov->N, iov->dims);
			}
		}

	}

	return ret;
}



void iter2_conjgrad(const iter_conf* _conf,
		const struct operator_s* normaleq_op,
		int D,
		const struct operator_p_s* prox_ops[D],
		const struct linop_s* ops[D],
		const float* biases[D],
		const struct operator_p_s* /*xupdate_op*/,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor)
{
	assert(0 == D);
	assert(NULL == prox_ops);
	assert(NULL == ops);
	assert(NULL == biases);

	assert(check_ops(size, normaleq_op, D, prox_ops, ops));

	auto conf = CAST_DOWN(iter_conjgrad_conf, _conf);

	long Bo = conf->Bo;
	long Bi = conf->Bi;

	if (1 == Bo * Bi) {

		float eps = md_norm(1, MD_DIMS(size), image_adj);

		if (checkeps(eps))
			goto cleanup;

		conjgrad(conf->maxiter, conf->INTERFACE.alpha * conf->l2lambda, eps * conf->tol, size, select_vecops(image_adj),
				OPERATOR2ITOP(normaleq_op), image, image_adj, monitor);
	} else {

		assert(0 == size % (Bo * Bi));

		conjgrad_batch(conf->maxiter, conf->INTERFACE.alpha * conf->l2lambda, conf->tol, size / (Bo * Bi * 2), Bi, Bo, select_vecops(image_adj),
			OPERATOR2ITOP(normaleq_op), image, image_adj, monitor);
	}

cleanup:
}


void iter2_ist(const iter_conf* _conf,
		const struct operator_s* normaleq_op,
		int D,
		const struct operator_p_s* prox_ops[D],
		const struct linop_s* ops[D],
		const float* biases[D],
		const struct operator_p_s* /*xupdate_op*/,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor)
{
	assert(D == 1);
	assert(NULL != prox_ops[0]);
	assert(NULL == biases);
#if 0
	assert(NULL == ops);
#endif

	assert(check_ops(size, normaleq_op, D, prox_ops, ops));

	auto conf = CAST_DOWN(iter_ist_conf, _conf);

	float eps = md_norm(1, MD_DIMS(size), image_adj);

	if (checkeps(eps))
		goto cleanup;

	// This was probably broken for IST until v0.4.04
	// better turn of it off with an error
	assert(1 == conf->continuation);

	// Let's see whether somebody uses it...
	assert(!conf->hogwild);

	ist(conf->maxiter, eps * conf->tol, conf->INTERFACE.alpha * conf->step, size, select_vecops(image_adj),
		NULL, OPERATOR2ITOP(normaleq_op), OPERATOR_P2ITOP(prox_ops[0]), image, image_adj, monitor);

cleanup:
	;
}


void iter2_eulermaruyama(const iter_conf* _conf,
		const struct operator_s* normaleq_op,
		int D,
		const struct operator_p_s* prox_ops[D],
		const struct linop_s* ops[D],
		const float* biases[D],
		const struct operator_p_s* /*xupdate_op*/,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor)
{
	assert(D == 1);
	assert(NULL != prox_ops[0]);
	assert(NULL == biases);
#if 0
	assert(NULL == ops);
#endif

	assert(check_ops(size, normaleq_op, D, prox_ops, ops));

	auto conf = CAST_DOWN(iter_eulermaruyama_conf, _conf);

	eulermaruyama(conf->maxiter, conf->INTERFACE.alpha, conf->step, size, select_vecops(image_adj),
		OPERATOR2ITOP(normaleq_op), &OPERATOR_P2ITOP(prox_ops[0]), image, image_adj, monitor);
}



void iter2_fista(const iter_conf* _conf,
		const struct operator_s* normaleq_op,
		int D,
		const struct operator_p_s* prox_ops[D],
		const struct linop_s* ops[D],
		const float* biases[D],
		const struct operator_p_s* /*xupdate_op*/,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor)
{
	assert(D == 1);
	assert(NULL == biases);
#if 0
	assert(NULL == ops);
#endif

	auto conf = CAST_DOWN(iter_fista_conf, _conf);

	float eps = md_norm(1, MD_DIMS(size), image_adj);

	assert(check_ops(size, normaleq_op, D, prox_ops, ops));

	if (checkeps(eps))
		return; // clang limitation
	//	goto cleanup;

	assert((conf->continuation >= 0.) && (conf->continuation <= 1.));

	__block int hogwild_k = 0;
	__block int hogwild_K = 10;

	NESTED(void, continuation, (struct ist_data* itrdata))
	{
		float a = logf(conf->continuation) / (float)itrdata->maxiter;
		itrdata->scale = expf(a * (float)itrdata->iter);

		if (conf->hogwild) {

			/* this is not exactly identical to what was implemented
			 * before as tau is now reduced at the beginning. But this
			 * seems more correct. */

			hogwild_k++;

			if (hogwild_k == hogwild_K) {

				hogwild_k = 0;
				hogwild_K *= 2;
				itrdata->tau /= 2;
			}
		}
	};

	double maxeigen = 1.;
	if (0 != conf->maxeigen_iter)
		maxeigen = estimate_maxeigenval_sameplace(normaleq_op, conf->maxeigen_iter, image_adj);

	fista(conf->maxiter, eps * conf->tol, conf->INTERFACE.alpha * conf->step / maxeigen, conf->last,
		(struct ravine_conf){ conf->p, conf->q, conf->r }, size, select_vecops(image_adj),
		continuation, OPERATOR2ITOP(normaleq_op), OPERATOR_P2ITOP(prox_ops[0]), image, image_adj, monitor);

// cleanup:
	;
}


/* Chambolle Pock Primal Dual algorithm. Solves G(x) + sum F_i(A_ix)
 * Assumes that G is in prox_ops[0] and ops[0] is NULL or identity, else G(x) = 0 is used.
 */
void iter2_chambolle_pock(const iter_conf* _conf,
		const struct operator_s* normaleq_op,
		int D,
		const struct operator_p_s* prox_ops[D],
		const struct linop_s* ops[D],
		const float* biases[D],
		const struct operator_p_s* /*xupdate_op*/,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor)
{
	assert(NULL == biases);
	assert((NULL == normaleq_op) == (NULL == image_adj));

	auto conf = CAST_DOWN(iter_chambolle_pock_conf, _conf);

	assert(check_ops(size, normaleq_op, D, prox_ops, ops));

	normaleq_op = operator_ref(normaleq_op);

	const struct operator_p_s* prox_G = NULL;

	if ((NULL == ops[0]) || linop_is_identity(ops[0])) {

		prox_G = operator_p_ref(prox_ops[0]);
		D--;
		prox_ops += 1;
		ops += 1;
	} else {
		const struct iovec_s* iov = NULL;

		if (NULL != normaleq_op)
			iov = operator_domain(normaleq_op);

		for (int i = 0; i < D; i++)
			if (NULL != ops[i])
				iov = linop_domain(ops[i]);

		assert(NULL != iov);

		prox_G = prox_zero_create(iov->N, iov->dims);
	}

	long M[D?:1];

	struct iter_op_s lop_frw[D?:1];
	struct iter_op_s lop_adj[D?:1];
	struct iter_op_p_s it_prox[D?:1];

	for (int i = 0; i < D; i++) {

		const struct iovec_s* ov = linop_codomain(ops[i]);
		M[i] = 2 * md_calc_size(ov->N, ov->dims);

		lop_frw[i] = OPERATOR2ITOP(ops[i]->forward);
		lop_adj[i] = OPERATOR2ITOP(ops[i]->adjoint);
		it_prox[i] = OPERATOR_P2ITOP(prox_ops[i]);
	}

	float eps = 1.;

	double maxeigen = 1.;
	if (0 != conf->maxeigen_iter) {

		auto iov = operator_p_domain(prox_G);
		const struct linop_s* lop_zero = linop_null_create(iov->N, iov->dims, iov->N, iov->dims);

		const struct operator_s* me_normal = operator_ref(normaleq_op) ?: operator_ref(lop_zero->normal);

		linop_free(lop_zero);

		for (int i = 0; i < D; i++) {

			auto tmp = me_normal;
			me_normal = operator_plus_create(me_normal, ops[i]->normal);
			operator_free(tmp);
		}

		debug_printf(DP_INFO, "Estimating max eigenvalue...\n");
		maxeigen = estimate_maxeigenval_sameplace(me_normal, conf->maxeigen_iter, image);
		debug_printf(DP_INFO, "Max eigenvalue: %e\n", maxeigen);

		operator_free(me_normal);
	}


	// FIXME: conf->INTERFACE.alpha * c
	chambolle_pock(conf->maxiter, eps * conf->tol, conf->tau / sqrtf(maxeigen), conf->sigma / sqrtf(maxeigen), conf->theta, conf->decay, D, size, M, select_vecops(image),
			OPERATOR2ITOP(normaleq_op), lop_frw, lop_adj, it_prox, OPERATOR_P2ITOP(prox_G),
			image, image_adj, monitor);

	operator_p_free(prox_G);
	operator_free(normaleq_op);

	//cleanup:
	//;
}



void iter2_admm(const iter_conf* _conf,
		const struct operator_s* normaleq_op,
		int D,
		const struct operator_p_s* prox_ops[D],
		const struct linop_s* ops[D],
		const float* biases[D],
		const struct operator_p_s* xupdate_op,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor)
{
	auto conf = CAST_DOWN(iter_admm_conf, _conf);

	assert(check_ops(size, normaleq_op, D, prox_ops, ops));

	struct admm_plan_s admm_plan = {

		.maxiter = conf->maxiter,
		.maxitercg = conf->maxitercg,
		.cg_eps = conf->cg_eps,
		.rho = conf->rho,
		.num_funs = D,
		.do_warmstart = conf->do_warmstart,
		.dynamic_rho = conf->dynamic_rho,
		.dynamic_tau = conf->dynamic_tau,
		.relative_norm = conf->relative_norm,
		.hogwild = conf->hogwild,
		.ABSTOL = conf->ABSTOL,
		.RELTOL = conf->RELTOL,
		.alpha = conf->alpha,
		.lambda = conf->INTERFACE.alpha,
		.tau = conf->tau,
		.tau_max = conf->tau_max,
		.mu = conf->mu,
		.fast = conf->fast,
		.biases = biases,
	};


	struct admm_op a_ops[D ?:1];
	struct iter_op_p_s a_prox_ops[D ?:1];

	for (int i = 0; i < D; i++) {

		a_ops[i].forward = OPERATOR2ITOP(ops[i]->forward),
		a_ops[i].normal = OPERATOR2ITOP(ops[i]->normal);
		a_ops[i].adjoint = OPERATOR2ITOP(ops[i]->adjoint);

		a_prox_ops[i] = OPERATOR_P2ITOP(prox_ops[i]);
	}

	admm_plan.ops = a_ops;
	admm_plan.prox_ops = a_prox_ops;

	admm_plan.xupdate = OPERATOR_P2ITOP(xupdate_op);


	long z_dims[D ?: 1];

	for (int i = 0; i < D; i++)
		z_dims[i] = 2 * md_calc_size(linop_codomain(ops[i])->N, linop_codomain(ops[i])->dims);

	if (NULL != image_adj) {

		float eps = md_norm(1, MD_DIMS(size), image_adj);

		if (checkeps(eps))
			goto cleanup;
	}

	admm(&admm_plan, admm_plan.num_funs, z_dims, size, image, image_adj, select_vecops(image), OPERATOR2ITOP(normaleq_op), monitor);

cleanup:
	;
}


void iter2_pocs(const iter_conf* _conf,
		const struct operator_s* normaleq_op,
		int D,
		const struct operator_p_s* prox_ops[D],
		const struct linop_s* ops[D],
		const float* biases[D],
		const struct operator_p_s* /*xupdate_op*/,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor)
{
	auto conf = CAST_DOWN(iter_pocs_conf, _conf);

	assert(NULL == normaleq_op);
	assert(NULL == ops);
	assert(NULL == biases);
	assert(NULL == image_adj);

	assert(check_ops(size, normaleq_op, D, prox_ops, ops));

	struct iter_op_p_s proj_ops[D];

	for (int i = 0; i < D; i++)
		proj_ops[i] = OPERATOR_P2ITOP(prox_ops[i]);

	pocs(conf->maxiter, D, proj_ops, select_vecops(image), size, image, monitor);
}


void iter2_niht(const iter_conf* _conf,
		const struct operator_s* normaleq_op,
		int D,
		const struct operator_p_s* prox_ops[D],
		const struct linop_s* ops[D],
		const float* /*biases*/[D],
		const struct operator_p_s* /*xupdate_op*/,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor)
{
	assert(D == 1);

	auto conf = CAST_DOWN(iter_niht_conf, _conf);

	struct niht_conf_s niht_conf = {

		.maxiter = conf->maxiter,
		.N = size,
		.trans = 0,
		.do_warmstart = conf->do_warmstart,
	};

	struct niht_transop trans;

	if (NULL != ops) {

		trans.forward = OPERATOR2ITOP(ops[0]->forward);
		trans.adjoint = OPERATOR2ITOP(ops[0]->adjoint);
		trans.N = 2 * md_calc_size(linop_codomain(ops[0])->N, linop_codomain(ops[0])->dims);
		niht_conf.trans = 1;
	}

	float eps = md_norm(1, MD_DIMS(size), image_adj);

	if (checkeps(eps))
		goto cleanup;

	niht_conf.epsilon = eps * conf->tol;

	niht(&niht_conf, &trans, select_vecops(image_adj), OPERATOR2ITOP(normaleq_op), OPERATOR_P2ITOP(prox_ops[0]), image, image_adj, monitor);

cleanup:
}

void iter2_call_iter(const iter_conf* _conf,
		const struct operator_s* normaleq_op,
		int D,
		const struct operator_p_s* prox_ops[D],
		const struct linop_s* ops[D],
		const float* biases[D],
		const struct operator_p_s* /*xupdate_op*/,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor)
{
	assert(D <= 1);
	assert(NULL == ops);
	assert(NULL == biases);

	auto it = CAST_DOWN(iter_call_s, _conf);

	it->fun(it->_conf, normaleq_op, (1 == D) ? prox_ops[0] : NULL,
		size, image, image_adj, monitor);
}



