/* Copyright 2021. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2020-2021 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops_p.h"
#include "num/ops.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/types.h"

#include "nlops/nlop.h"
#include "nlops/tenmul.h"
#include "nlops/chain.h"

#include "iter/prox2.h"
#include "iter/thresh.h"

#include "utest.h"




static bool test_nlgrad(void)
{
	enum { N = 1 };
	long dims[N] = { 1 };

	auto nlop = nlop_tenmul_create(N, dims, dims, dims);
	auto sq = nlop_dup(nlop, 0, 1);

	nlop_free(nlop);

	complex float* src = md_alloc(N, dims, CFL_SIZE);
	complex float* dst = md_alloc(N, dims, CFL_SIZE);

	md_zfill(N, dims, src, 1.);

	auto p = prox_nlgrad_create(sq, 30, 0.1, 1.);

	nlop_free(sq);

	// argmin_x 0.5 (x - 1)^2 + x^2 = 1.5 x^2 -1x + 0.5
	operator_p_apply(p, 1., N, dims, dst, N, dims, src);

	operator_p_free(p);

	md_zfill(N, dims, src, 1. / 3.);

	float err = md_znrmse(N, dims, dst, src);

	md_free(src);
	md_free(dst);

	UT_ASSERT(err < 1.E-4);
}

UT_REGISTER_TEST(test_nlgrad);



static bool test_auto_norm(void)
{
	enum { N = 3 };
	long dims[N] = { 2, 4, 3 };

	complex float* src = md_alloc(N, dims, CFL_SIZE);
	complex float* dst = md_alloc(N, dims, CFL_SIZE);

	md_zfill(N, dims, src, 3.);

	auto p = prox_thresh_create(N, dims, 0.5, 0u);
	auto n = op_p_auto_normalize(p, MD_BIT(1), NORM_L2);

	operator_p_free(p);

	operator_p_apply(n, 0.5, N, dims, dst, N, dims, src);

	operator_p_free(n);

	md_zfill(N, dims, src, 3. * 0.5);

	float err = md_znrmse(N, dims, dst, src);

	md_free(src);
	md_free(dst);

#ifdef  __clang__
	UT_ASSERT(err < 1.E-6);
#else
#if __GNUC__ >= 10
	UT_ASSERT(err < 1.E-7);
#else
	UT_ASSERT(err < 1.E-10);
#endif
#endif
}

UT_REGISTER_TEST(test_auto_norm);



