/* Copyright 2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops_p.h"
#include "num/ops.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/types.h"

#include "iter/prox.h"
#include "iter/thresh.h"

#include "utest.h"



static bool test_thresh(void)
{
	enum { N = 3 };
	long dims[N] = { 4, 2, 3 };

	complex float* src = md_alloc(N, dims, CFL_SIZE);
	complex float* dst = md_alloc(N, dims, CFL_SIZE);

	md_zfill(N, dims, src, 1.);

	auto p = prox_thresh_create(N, dims, 0.5, 0u);

	operator_p_apply(p, 0.5, N, dims, dst, N, dims, src);

	operator_p_free(p);

	md_zfill(N, dims, src, 0.75);

	float err = md_znrmse(N, dims, dst, src);

	md_free(src);
	md_free(dst);

	UT_ASSERT(err < 1.E-10);
}

UT_REGISTER_TEST(test_thresh);




static bool test_auto_norm(void)
{
	enum { N = 3 };
	long dims[N] = { 2, 4, 3 };

	complex float* src = md_alloc(N, dims, CFL_SIZE);
	complex float* dst = md_alloc(N, dims, CFL_SIZE);

	md_zfill(N, dims, src, 3.);

	auto p = prox_thresh_create(N, dims, 0.5, 0u);
	auto n = op_p_auto_normalize(p, MD_BIT(1));

	operator_p_free(p);

	operator_p_apply(n, 0.5, N, dims, dst, N, dims, src);

	operator_p_free(n);

	md_zfill(N, dims, src, 3. * 0.5);

	float err = md_znrmse(N, dims, dst, src);

	md_free(src);
	md_free(dst);

	UT_ASSERT(err < 1.E-10);
}

UT_REGISTER_TEST(test_auto_norm);


