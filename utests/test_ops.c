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
#include "num/ops.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "utest.h"




static bool test_op_stack(void)
{
	enum { N = 3 };
	long dims[N] = { 8, 4, 1 };
	long dims2[N] = { 8, 4, 2 };

	const auto a = operator_identity_create(N, dims);
	const auto b = operator_zero_create(N, dims);
	const auto c = operator_null_create(N, dims);
	const auto d = operator_combi_create(2, MAKE_ARRAY(OP_PASS(b), OP_PASS(c)));
	const auto e = operator_stack(2, 2, OP_PASS(a), OP_PASS(d));

	complex float* in = md_alloc(N, dims2, CFL_SIZE);
	complex float* out = md_alloc(N, dims2, CFL_SIZE);

	md_zfill(N, dims2, in, 1.);
	md_zfill(N, dims2, out, 100.);

	operator_apply(e, N, dims2, out, N, dims2, in);

	double err = fabsf(md_znorm(N, dims2, in) - sqrtf(2.) * md_znorm(N, dims2, out));

	operator_free(e);

	md_free(in);
	md_free(out);

	return (err < UT_TOL);
}

UT_REGISTER_TEST(test_op_stack);



static bool test_op_extract(void)
{
	enum { N = 3 };
	long dims[N] = { 8, 4, 1 };
	long dims2[N] = { 8, 4, 2 };

	const auto a = operator_identity_create(N, dims);
	const auto b = operator_zero_create(N, dims);
	const auto c = operator_null_create(N, dims);
	const auto d = operator_combi_create(2, MAKE_ARRAY(OP_PASS(b), OP_PASS(c)));
	const auto e = operator_extract_create(OP_PASS(a), 0, N, dims2, (long[]){ 0, 0, 0 });
	const auto f = operator_extract_create(OP_PASS(e), 1, N, dims2, (long[]){ 0, 0, 0 });
	const auto g = operator_extract_create(OP_PASS(d), 0, N, dims2, (long[]){ 0, 0, 1 });
	const auto h = operator_extract_create(OP_PASS(g), 1, N, dims2, (long[]){ 0, 0, 1 });
	const auto i = operator_combi_create(2, MAKE_ARRAY(OP_PASS(f), OP_PASS(h)));
	const auto j = operator_dup_create(OP_PASS(i), 0, 2);
	const auto k = operator_dup_create(OP_PASS(j), 1, 2);

	complex float* in = md_alloc(N, dims2, CFL_SIZE);
	complex float* out = md_alloc(N, dims2, CFL_SIZE);

	md_zfill(N, dims2, in, 1.);
	md_zfill(N, dims2, out, 100.);

	operator_apply(k, N, dims2, out, N, dims2, in);

	double err = fabsf(md_znorm(N, dims2, in) - sqrtf(2.) * md_znorm(N, dims2, out));

	operator_free(k);

	md_free(in);
	md_free(out);

	return (err < UT_TOL);
}

UT_REGISTER_TEST(test_op_extract);




