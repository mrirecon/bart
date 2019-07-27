/* Copyright 2017-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <assert.h>
#include <math.h>

#include "num/ops.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "num/iovec.h"

#include "nlops/nlop.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"

#include "iter/italgos.h"
#include "iter/vec.h"
#include "iter/iter3.h"
#include "iter/iter2.h"

#include "iter4.h"


struct iter4_nlop_s {

	INTERFACE(iter_op_data);

	struct nlop_s nlop;
};

DEF_TYPEID(iter4_nlop_s);


static void nlop_for_iter(iter_op_data* _o, float* _dst, const float* _src)
{
	const auto nlop = CAST_DOWN(iter4_nlop_s, _o);

	assert(2 == operator_nr_args(nlop->nlop.op));
	operator_apply_unchecked(nlop->nlop.op, (complex float*)_dst, (const complex float*)_src);
}

static void nlop_der_iter(iter_op_data* _o, float* _dst, const float* _src)
{
	const auto nlop = CAST_DOWN(iter4_nlop_s, _o);

	assert(2 == operator_nr_args(nlop->nlop.op));
	linop_forward_unchecked(nlop->nlop.derivative[0], (complex float*)_dst, (const complex float*)_src);
}

static void nlop_adj_iter(iter_op_data* _o, float* _dst, const float* _src)
{
	const auto nlop = CAST_DOWN(iter4_nlop_s, _o);

	assert(2 == operator_nr_args(nlop->nlop.op));
	linop_adjoint_unchecked(nlop->nlop.derivative[0], (complex float*)_dst, (const complex float*)_src);
}


struct irgnm_s {

	INTERFACE(iter_op_data);

	struct iter_op_s der;
	struct iter_op_s adj;

	float* tmp;

	long size;

	int cgiter;
	float cgtol;
	bool nlinv_legacy;
};

DEF_TYPEID(irgnm_s);

static void normal(iter_op_data* _data, float* dst, const float* src)
{
	auto data = CAST_DOWN(irgnm_s, _data);

	iter_op_call(data->der, data->tmp, src);
	iter_op_call(data->adj, dst, data->tmp);
}

static void inverse(iter_op_data* _data, float alpha, float* dst, const float* src)
{
	auto data = CAST_DOWN(irgnm_s, _data);

	md_clear(1, MD_DIMS(data->size), dst, FL_SIZE);

        float eps = data->cgtol * md_norm(1, MD_DIMS(data->size), src);


	/* The original (Matlab) nlinv implementation uses
	 * "sqrt(rsnew) < 0.01 * rsnot" as termination condition.
	 */
	if (data->nlinv_legacy)
		eps = powf(eps, 2.);

        conjgrad(data->cgiter, alpha, eps, data->size, select_vecops(src),
			(struct iter_op_s){ normal, CAST_UP(data) }, dst, src, NULL);
}


void iter4_irgnm(const iter3_conf* _conf,
		struct nlop_s* nlop,
		long N, float* dst, const float* ref,
		long M, const float* src,
		const struct operator_p_s* pinv,
		struct iter_op_s cb)
{
	struct iter4_nlop_s data = { { &TYPEID(iter4_nlop_s) }, *nlop };

	auto cd = nlop_codomain(nlop);
	auto dm = nlop_domain(nlop);

	assert(NULL == pinv); // better we allow this only with irgnm2

	assert(M * sizeof(float) == md_calc_size(cd->N, cd->dims) * cd->size);
	assert(N * sizeof(float) == md_calc_size(dm->N, dm->dims) * dm->size);

	auto conf = CAST_DOWN(iter3_irgnm_conf, _conf);

	struct iter_op_s frw = { nlop_for_iter, CAST_UP(&data) };
	struct iter_op_s der = { nlop_der_iter, CAST_UP(&data) };
	struct iter_op_s adj = { nlop_adj_iter, CAST_UP(&data) };

	float* tmp = md_alloc_sameplace(1, MD_DIMS(M), FL_SIZE, src);

	struct irgnm_s data2 = { { &TYPEID(irgnm_s) }, der, adj, tmp, N, conf->cgiter, conf->cgtol, conf->nlinv_legacy };

	struct iter_op_p_s inv = { inverse, CAST_UP(&data2) };

	irgnm(conf->iter, conf->alpha, conf->alpha_min, conf->redu, N, M, select_vecops(src),
		frw, adj, inv,
		dst, ref, src, cb, NULL);

	md_free(tmp);
}



void iter4_landweber(const iter3_conf* _conf,
		struct nlop_s* nlop,
		long N, float* dst, const float* ref,
		long M, const float* src,
		const struct operator_p_s* inv,
		struct iter_op_s cb)
{
	assert(NULL == ref);
	assert(NULL == inv);

	struct iter4_nlop_s data = { { &TYPEID(iter4_nlop_s) }, *nlop };

	auto conf = CAST_DOWN(iter3_landweber_conf, _conf);

	float* tmp = md_alloc_sameplace(1, MD_DIMS(N), FL_SIZE, src);

	struct iter_op_s frw = { nlop_for_iter, CAST_UP(&data) };
	struct iter_op_s adj = { nlop_adj_iter, CAST_UP(&data) };

	landweber(conf->iter, conf->epsilon, conf->alpha, N, M,
		select_vecops(src), frw, adj, dst, src, cb, NULL);

	md_free(tmp);
}





static void inverse2(iter_op_data* _data, float alpha, float* dst, const float* src)
{
	auto data = CAST_DOWN(irgnm_s, _data);

	float* tmp = md_alloc_sameplace(1, MD_DIMS(data->size), FL_SIZE, src);

	iter_op_call(data->adj, tmp, src);

	inverse(_data, alpha, dst, tmp);

	md_free(tmp);
}



void iter4_irgnm2(const iter3_conf* _conf,
		struct nlop_s* nlop,
		long N, float* dst, const float* ref,
		long M, const float* src,
		const struct operator_p_s* lsqr,
		struct iter_op_s cb)
{
	struct iter4_nlop_s data = { { &TYPEID(iter4_nlop_s) }, *nlop };

	auto cd = nlop_codomain(nlop);
	auto dm = nlop_domain(nlop);

	assert(M * sizeof(float) == md_calc_size(cd->N, cd->dims) * cd->size);
	assert(N * sizeof(float) == md_calc_size(dm->N, dm->dims) * dm->size);

	auto conf = CAST_DOWN(iter3_irgnm_conf, _conf);

	float* tmp = md_alloc_sameplace(1, MD_DIMS(M), FL_SIZE, src);

	struct iter_op_s frw = { nlop_for_iter, CAST_UP(&data) };
	struct iter_op_s der = { nlop_der_iter, CAST_UP(&data) };
	struct iter_op_s adj = { nlop_adj_iter, CAST_UP(&data) };

	struct irgnm_s data2 = { { &TYPEID(irgnm_s) }, der, adj, tmp, N, conf->cgiter, conf->cgtol, conf->nlinv_legacy };

	// one limitation is that we currently cannot warm start the inner solver

	struct iter_op_p_s inv2 = { inverse2, CAST_UP(&data2) };

	irgnm2(conf->iter, conf->alpha, conf->alpha_min, 0., conf->redu, N, M, select_vecops(src),
		frw, der, (NULL == lsqr) ? inv2 : OPERATOR_P2ITOP(lsqr),
		dst, ref, src, cb, NULL);

	md_free(tmp);
}



