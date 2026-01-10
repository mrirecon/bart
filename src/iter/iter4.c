/* Copyright 2017-2021. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2023-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <assert.h>
#include <math.h>

#include "num/ops.h"
#include "num/ops_p.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/vptr.h"

#include "nlops/nlop.h"

#include "misc/types.h"

#include "iter/italgos.h"
#include "iter/vec.h"
#include "iter/iter3.h"
#include "iter/iter2.h"

#include "iter4.h"


struct iter4_nlop_s {

	iter_op_data super;

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

static void nlop_nrm_iter(iter_op_data* _o, float* _dst, const float* _src)
{
	const auto nlop = CAST_DOWN(iter4_nlop_s, _o);

	assert(2 == operator_nr_args(nlop->nlop.op));
	linop_normal_unchecked(nlop->nlop.derivative[0], (complex float*)_dst, (const complex float*)_src);
}


struct irgnm_s {

	iter_op_data super;

	struct iter_op_s der;
	struct iter_op_s adj;
	struct iter_op_s nrm;

	long size;

	int cgiter;
	float cgtol;
	bool nlinv_legacy;
};

DEF_TYPEID(irgnm_s);

static void normal(iter_op_data* _data, float* dst, const float* src)
{
	auto data = CAST_DOWN(irgnm_s, _data);

	iter_op_call(data->nrm, dst, src);
}

static void inverse(iter_op_data* _data, float alpha, float* dst, const float* src)
{
	auto data = CAST_DOWN(irgnm_s, _data);

	const struct vec_iter_s* vops = select_vecops(src);

	vops->clear(data->size, dst);

        float eps = data->cgtol * vops->norm(data->size, src);


	/* The original (Matlab) nlinv implementation uses
	 * "sqrt(rsnew) < 0.01 * rsnot" as termination condition.
	 */
	if (data->nlinv_legacy)
		eps = powf(eps, 2.);

        conjgrad(data->cgiter, alpha, eps, data->size, vops,
			(struct iter_op_s){ normal, CAST_UP(data) }, dst, src, NULL);
}


void iter4_irgnm(const iter3_conf* _conf,
		const struct nlop_s* _nlop,
		long N, float* dst, const float* ref,
		long M, const float* src,
		const struct operator_p_s* pinv,
		struct iter_op_s cb)
{
	const struct nlop_s* nlop = is_vptr(src) ? nlop_vptr_set_dims_wrapper(_nlop, 1, (const void*[1]) { src }, 1, (const void*[1]) { dst }, vptr_get_hint(dst)) : nlop_clone(_nlop);

	struct iter4_nlop_s data = { { &TYPEID(iter4_nlop_s) }, *nlop };

	auto cd = nlop_codomain(nlop);
	auto dm = nlop_domain(nlop);

	assert(NULL == pinv); // better we allow this only with irgnm2

	assert((long)sizeof(float[M]) == md_calc_size(cd->N, cd->dims) * (long)cd->size);
	assert((long)sizeof(float[N]) == md_calc_size(dm->N, dm->dims) * (long)dm->size);

	auto conf = CAST_DOWN(iter3_irgnm_conf, _conf);

	struct iter_op_s frw = { nlop_for_iter, CAST_UP(&data) };
	struct iter_op_s der = { nlop_der_iter, CAST_UP(&data) };
	struct iter_op_s adj = { nlop_adj_iter, CAST_UP(&data) };
	struct iter_op_s nrm = { nlop_nrm_iter, CAST_UP(&data) };

	struct irgnm_s data2 = { { &TYPEID(irgnm_s) }, der, adj, nrm, N, conf->cgiter, conf->cgtol, conf->nlinv_legacy };

	struct iter_op_p_s inv = { inverse, CAST_UP(&data2) };

	irgnm(conf->iter, conf->alpha, conf->alpha_min, conf->redu, N, M, select_vecops(src),
		frw, adj, inv,
		dst, ref, src, cb, NULL);

	nlop_free(nlop);
}



void iter4_landweber(const iter3_conf* _conf,
		const struct nlop_s* nlop,
		long N, float* dst, const float* ref,
		long M, const float* src,
		const struct operator_p_s* inv,
		struct iter_op_s cb)
{
	assert(NULL == ref);
	assert(NULL == inv);

	struct iter4_nlop_s data = { { &TYPEID(iter4_nlop_s) }, *nlop };

	auto conf = CAST_DOWN(iter3_landweber_conf, _conf);

	const struct vec_iter_s* vops = select_vecops(src);

	float* tmp = vops->allocate(N);

	struct iter_op_s frw = { nlop_for_iter, CAST_UP(&data) };
	struct iter_op_s adj = { nlop_adj_iter, CAST_UP(&data) };

	landweber(conf->iter, conf->epsilon, conf->alpha, N, M,
		vops, frw, adj, dst, src, cb, NULL);

	vops->del(tmp);
}





void iter4_irgnm2(const iter3_conf* _conf,
		const struct nlop_s* _nlop,
		long N, float* dst, const float* ref,
		long M, const float* src,
		const struct operator_p_s* lsqr,
		struct iter_op_s cb)
{
	const struct nlop_s* nlop = is_vptr(src) ? nlop_vptr_set_dims_wrapper(_nlop, 1, (const void*[1]) { src }, 1, (const void*[1]) { dst }, vptr_get_hint(dst)) : nlop_clone(_nlop);

	struct iter4_nlop_s data = { { &TYPEID(iter4_nlop_s) }, *nlop };

	auto cd = nlop_codomain(nlop);
	auto dm = nlop_domain(nlop);

	assert((long)sizeof(float[M]) == md_calc_size(cd->N, cd->dims) * (long)cd->size);
	assert((long)sizeof(float[N]) == md_calc_size(dm->N, dm->dims) * (long)dm->size);

	auto conf = CAST_DOWN(iter3_irgnm_conf, _conf);

	struct iter_op_s frw = { nlop_for_iter, CAST_UP(&data) };
	struct iter_op_s der = { nlop_der_iter, CAST_UP(&data) };
	struct iter_op_s adj = { nlop_adj_iter, CAST_UP(&data) };
	struct iter_op_s nrm = { nlop_nrm_iter, CAST_UP(&data) };

	struct irgnm_s data2 = { { &TYPEID(irgnm_s) }, der, adj, nrm, N, conf->cgiter, conf->cgtol, conf->nlinv_legacy };

	// one limitation is that we currently cannot warm start the inner solver

	struct iter_op_p_s inv2 = { inverse, CAST_UP(&data2) };

	const struct operator_p_s* vlsqr = (NULL == lsqr) ? NULL : is_vptr(dst) ? operator_p_vptr_set_dims_wrapper(lsqr, dst, dst, vptr_get_hint(dst)) : operator_p_ref(lsqr);

	irgnm2(conf->iter, conf->alpha, conf->alpha_min, conf->alpha_min0, conf->redu, N, M, select_vecops(src),
		frw, der, adj, (NULL == lsqr) ? inv2 : OPERATOR_P2ITOP(vlsqr),
		dst, ref, src, cb, NULL);

	nlop_free(nlop);
	operator_p_free(vlsqr);
}

void iter4_lbfgs(const iter3_conf* _conf,
		const struct nlop_s* nlop,
		long N, float* dst, const float* ref,
		long M, const float* src,
		const struct operator_p_s* lsqr,
		struct iter_op_s cb)
{
	struct iter4_nlop_s data = { { &TYPEID(iter4_nlop_s) }, *nlop };

	auto cd = nlop_codomain(nlop);
	auto dm = nlop_domain(nlop);

	assert((long)sizeof(float[M]) == md_calc_size(cd->N, cd->dims) * (long)cd->size);
	assert((long)sizeof(float[N]) == md_calc_size(dm->N, dm->dims) * (long)dm->size);
	assert(2 == M);
	assert(NULL == src);
	assert(NULL == ref);
	assert(NULL == lsqr);
	(void)cb;

	auto conf = CAST_DOWN(iter3_lbfgs_conf, _conf);

	struct iter_op_s frw = { nlop_for_iter, CAST_UP(&data) };
	struct iter_op_s adj = { nlop_adj_iter, CAST_UP(&data) };

	lbfgs(conf->iter, conf->M, conf->step, conf->ftol, conf->gtol, conf->c1, conf->c2, frw, adj, N, dst, select_vecops(dst));
}


void iter4_levenberg_marquardt(const iter3_conf* _conf,
		const struct nlop_s* nlop,
		long N, float* dst, const float* ref,
		long M, const float* src,
		const struct operator_p_s* lsqr,
		struct iter_op_s cb)
{
	struct iter4_nlop_s data = { { &TYPEID(iter4_nlop_s) }, *nlop };

	assert(NULL == ref);
	assert(NULL == lsqr);

	auto cd = nlop_codomain(nlop);
	auto dm = nlop_domain(nlop);

	assert((long)sizeof(float[M]) == md_calc_size(cd->N, cd->dims) * (long)cd->size);
	assert((long)sizeof(float[N]) == md_calc_size(dm->N, dm->dims) * (long)dm->size);

	auto conf = CAST_DOWN(iter3_levenberg_marquardt_conf, _conf);

	struct iter_op_s frw = { nlop_for_iter, CAST_UP(&data) };
	struct iter_op_s adj = { nlop_adj_iter, CAST_UP(&data) };
	struct iter_op_s nrm = { nlop_nrm_iter, CAST_UP(&data) };

	levenberg_marquardt(conf->iter, MIN(conf->cgiter, N / conf->Bi / conf->Bo), conf->l2lambda, conf->redu,
				N / 2 / conf->Bi / conf->Bo, M / 2 / conf->Bi / conf->Bo, conf->Bi, conf->Bo,
				select_vecops(dst), frw, adj, nrm, dst, src, cb, NULL);
}

