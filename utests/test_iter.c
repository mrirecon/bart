/* Copyright 2019-2020. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops_p.h"
#include "num/ops.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/types.h"

#include "iter/italgos.h"
#include "iter/iter3.h"
#include "iter/iter4.h"
#include "iter/lsqr.h"
#include "iter/thresh.h"

#include "linops/someops.h"

#include "nlops/zexp.h"
#include "nlops/nlop.h"
#include "nlops/cast.h"

#include "utest.h"





static bool test_iter_irgnm0(bool v2, bool ref)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	complex float* dst1 = md_alloc(N, dims, CFL_SIZE);
	complex float* src1 = md_alloc(N, dims, CFL_SIZE);
	complex float* src2 = md_alloc(N, dims, CFL_SIZE);

	md_zfill(N, dims, src1, 1.);

	struct nlop_s* zexp = nlop_zexp_create(N, dims);

	nlop_apply(zexp, N, dims, dst1, N, dims, src1);

	md_zfill(N, dims, src2, 0.);

	(v2 ? iter4_irgnm2 : iter4_irgnm)(CAST_UP(&iter3_irgnm_defaults), zexp,
		2 * md_calc_size(N, dims), (float*)src2, ref ? (const float*)src1 : NULL,
		2 * md_calc_size(N, dims), (const float*)dst1, NULL,
		(struct iter_op_s){ NULL, NULL });

	double err = md_znrmse(N, dims, src2, src1);

	nlop_free(zexp);

	md_free(src1);
	md_free(dst1);
	md_free(src2);

	UT_ASSERT(err < (ref ? 1.E-7 : 0.01));
}



static bool test_iter_irgnm_lsqr0(bool ref)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	complex float* dst1 = md_alloc(N, dims, CFL_SIZE);
	complex float* src1 = md_alloc(N, dims, CFL_SIZE);
	complex float* src2 = md_alloc(N, dims, CFL_SIZE);
	complex float* src3 = md_alloc(N, dims, CFL_SIZE);

	md_zfill(N, dims, src1, 1.);

	struct nlop_s* zexp = nlop_zexp_create(N, dims);

	nlop_apply(zexp, N, dims, dst1, N, dims, src1);

	md_zfill(N, dims, src2, 0.);
	md_zfill(N, dims, src3, 0.);

	const struct operator_p_s* lsqr = NULL;

	iter4_irgnm2(CAST_UP(&iter3_irgnm_defaults), zexp,
		2 * md_calc_size(N, dims), (float*)src2, ref ? (const float*)src1 : NULL,
		2 * md_calc_size(N, dims), (const float*)dst1, lsqr,
		(struct iter_op_s){ NULL, NULL });

	struct iter_conjgrad_conf conf = iter_conjgrad_defaults;
	conf.maxiter = 100;
	conf.l2lambda = 1.;
	conf.tol = 0.1;

	auto der = linop_clone(&zexp->derivative[0][0]);

	lsqr = lsqr2_create(&lsqr_defaults,
				iter2_conjgrad, CAST_UP(&conf),
				NULL, der, NULL,
				0, NULL, NULL, NULL);

	linop_free(der);

	iter4_irgnm2(CAST_UP(&iter3_irgnm_defaults), zexp,
		2 * md_calc_size(N, dims), (float*)src3, ref ? (const float*)src1 : NULL,
		2 * md_calc_size(N, dims), (const float*)dst1, lsqr,
		(struct iter_op_s){ NULL, NULL });

	double err = md_znrmse(N, dims, src2, src3);

	operator_p_free(lsqr);
	nlop_free(zexp);

	md_free(src1);
	md_free(dst1);
	md_free(src2);
	md_free(src3);

	UT_ASSERT(err < 1.E-10);
}


static bool test_iter_irgnm_lsqr1(bool ref, bool regu)
{
	enum { N = 3 };
	long dims[N]  = { 10, 7, 3 };
	long dims1[N] = { 10, 7, 1 };
	long dims2[N] = { 10, 7, 2 };

	complex float* dst1 = md_alloc(N, dims, CFL_SIZE);
	complex float* src1 = md_alloc(N, dims, CFL_SIZE);
	complex float* src2 = md_alloc(N, dims, CFL_SIZE);

	md_zfill(N, dims, src1, 1.);

	struct nlop_s* zexp = nlop_zexp_create(N, dims);

	nlop_apply(zexp, N, dims, dst1, N, dims, src1);

	md_zfill(N, dims, src2, 0.);

	const struct operator_p_s* lsqr = NULL;
	struct iter_admm_conf conf = iter_admm_defaults;
	conf.rho = 1.E-5;

	auto p1 = prox_thresh_create(3, dims1, 0.5, 0u);
	auto p2 = prox_thresh_create(3, dims2, 0.5, 0u);

	const struct operator_p_s* prox_ops[1] = { operator_p_stack(2, 2, p1, p2) };

	operator_p_free(p1);
	operator_p_free(p2);

	auto der = linop_clone(&zexp->derivative[0][0]);

	const struct linop_s* trafos[1] = { linop_identity_create(3, dims) };

	lsqr = lsqr2_create(&lsqr_defaults,
				iter2_admm, CAST_UP(&conf),
				NULL, der, NULL,
				regu ? 1 : 0,
				regu ? prox_ops : NULL,
				regu ? trafos : NULL,
				NULL);

	linop_free(der);
	linop_free(trafos[0]);
	operator_p_free(prox_ops[0]);

	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;
	irgnm_conf.iter = 4;

	iter4_irgnm2(CAST_UP(&irgnm_conf), zexp,
		2 * md_calc_size(N, dims), (float*)src2, ref ? (const float*)src1 : NULL,
		2 * md_calc_size(N, dims), (const float*)dst1, lsqr,
		(struct iter_op_s){ NULL, NULL });

	double err = md_znrmse(N, dims, src1, src2);

	operator_p_free(lsqr);
	nlop_free(zexp);

	md_free(src1);
	md_free(dst1);
	md_free(src2);

	UT_ASSERT(err < 1.E-3);
}


static bool test_iter_irgnm(void)
{
	return    test_iter_irgnm0(false, false)
	       && test_iter_irgnm0(false, true);
}

UT_REGISTER_TEST(test_iter_irgnm);

static bool test_iter_irgnm2(void)
{
	return    test_iter_irgnm0(true, false)
	       && test_iter_irgnm0(true, true);
}

UT_REGISTER_TEST(test_iter_irgnm2);

static bool test_iter_irgnm_lsqr(void)
{
	return    test_iter_irgnm_lsqr0(false)
	       && test_iter_irgnm_lsqr0(true);
}

UT_REGISTER_TEST(test_iter_irgnm_lsqr);

static bool test_iter_irgnm_lsqr_l1(void)
{
	return    test_iter_irgnm_lsqr1(false, true)
	       && test_iter_irgnm_lsqr1(false, false)
	       && test_iter_irgnm_lsqr1(true,  true)
	       && test_iter_irgnm_lsqr1(true,  false);
}

UT_REGISTER_TEST(test_iter_irgnm_lsqr_l1);


static bool test_iter_irgnm_l1(void)
{
	enum { N = 3 };
	long dims[N] = { 4, 2, 3 };
	long dims1[N] = { 4, 2, 1 };
	long dims2[N] = { 4, 2, 2 };

	complex float* dst1 = md_alloc(N, dims, CFL_SIZE);
	complex float* src1 = md_alloc(N, dims, CFL_SIZE);
	complex float* src2 = md_alloc(N, dims, CFL_SIZE);
	complex float* src3 = md_alloc(N, dims, CFL_SIZE);

	md_zfill(N, dims, src1, 1.);

	const struct linop_s* id = linop_identity_create(3, dims);

	struct nlop_s* nlid = nlop_from_linop(id);

	nlop_apply(nlid, N, dims, dst1, N, dims, src1);

	md_zfill(N, dims, src2, 0.975);
	md_zfill(N, dims, src3, 0.);

	const struct operator_p_s* lsqr = NULL;

	struct iter_fista_conf conf = iter_fista_defaults;
	conf.maxiter = 10;

	const struct linop_s* trafos[1] = { id };

	auto p1 = prox_thresh_create(3, dims1, 0.5, 0u);
	auto p2 = prox_thresh_create(3, dims2, 0.5, 0u);

	const struct operator_p_s* prox_ops[1] = { operator_p_stack(2, 2, p1, p2) };

	operator_p_free(p1);
	operator_p_free(p2);

	lsqr = lsqr2_create(&lsqr_defaults,
				iter2_fista, CAST_UP(&conf),
				NULL, &nlid->derivative[0][0], NULL,
				1, prox_ops, trafos, NULL);

	struct iter3_irgnm_conf conf2 = iter3_irgnm_defaults;
	conf2.redu = 1.;
	conf2.iter = 1.;

	iter4_irgnm2(CAST_UP(&conf2), nlid,
		2 * md_calc_size(N, dims), (float*)src3, NULL,
		2 * md_calc_size(N, dims), (const float*)dst1, lsqr,
		(struct iter_op_s){ NULL, NULL });

	double err = md_znrmse(N, dims, src2, src3);

	nlop_free(nlid);

	linop_free(id);
	operator_p_free(prox_ops[0]);
	operator_p_free(lsqr);

	md_free(src1);
	md_free(dst1);
	md_free(src2);
	md_free(src3);

	UT_ASSERT(err < 1.E-10);
}


UT_REGISTER_TEST(test_iter_irgnm_l1);


static bool test_iter_lsqr_warmstart(void)
{
	enum { N = 3 };
	long dims[N] = { 4, 2, 3 };

	complex float* src = md_alloc(N, dims, CFL_SIZE);
	complex float* dst = md_alloc(N, dims, CFL_SIZE);

	md_zfill(N, dims, src, 1.66);
	md_zfill(N, dims, dst, 0.66);

	const struct linop_s* id = linop_identity_create(3, dims);

	struct lsqr_conf conf = lsqr_defaults;
	conf.warmstart = true;

	struct iter_conjgrad_conf cg_conf = iter_conjgrad_defaults;
	cg_conf.maxiter = 0;

	const struct operator_p_s* lsqr = lsqr2_create(&conf,
						iter2_conjgrad, CAST_UP(&cg_conf),
						NULL, id, NULL,
						0, NULL, NULL, NULL);

	operator_p_apply(lsqr, 0.3, N, dims, dst, N, dims, src);

	md_zfill(N, dims, src, 0.66);

	double err = md_znrmse(N, dims, src, dst);

	linop_free(id);

	md_free(src);
	md_free(dst);

	operator_p_free(lsqr);

	UT_ASSERT(err < UT_TOL);
}

UT_REGISTER_TEST(test_iter_lsqr_warmstart);
