/* Copyright 2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/ops_p.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/types.h"

#include "utest.h"



static bool test_op_p_scale(void)
{
	enum { N = 3 };
	long dims[N] = { 8, 4, 1 };

	complex float* in = md_alloc(N, dims, CFL_SIZE);
	complex float* out = md_alloc(N, dims, CFL_SIZE);

	md_zfill(N, dims, in, 1.);
	md_zfill(N, dims, out, 100.);

	const struct operator_p_s* a = operator_p_scale(N, dims);
	operator_p_apply(a, 2., N, dims, out, N, dims, in);
	operator_p_free(a);

	md_zfill(N, dims, in, 2.);

	float err = md_znrmse(N, dims, out, in);

	md_free(in);
	md_free(out);

	return (err < UT_TOL);
}

UT_REGISTER_TEST(test_op_p_scale);



static bool test_op_p_stack(void)
{
	enum { N = 3 };
	long dims[N] = { 8, 4, 1 };
	long dims2[N] = { 8, 4, 2 };

	auto a = operator_p_scale(N, dims);
	auto a2 = operator_p_scale(N, dims);
	auto b = operator_p_stack(2, 2, a, a2);

	operator_p_free(a);
	operator_p_free(a2);

	complex float* in = md_alloc(N, dims2, CFL_SIZE);
	complex float* out = md_alloc(N, dims2, CFL_SIZE);

	md_zfill(N, dims2, in, 1.);
	md_zfill(N, dims2, out, 100.);

	operator_p_apply(b, 2., N, dims2, out, N, dims2, in);
	operator_p_free(b);

	md_zfill(N, dims2, in, 2.);

	float err = md_znrmse(N, dims2, out, in);

	md_free(in);
	md_free(out);

	return (err < UT_TOL);
}

UT_REGISTER_TEST(test_op_p_stack);



