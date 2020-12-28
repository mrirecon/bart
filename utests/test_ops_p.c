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


static bool test_op_p_stack2(void)
{
	enum { N = 4 };
	long dims[N] = { 8, 4, 1, 4 };
	long dims2[N] = { 8, 4, 2, 4 };

	long dims_no3[N] = { 8, 4, 1, 1 };

	auto a = operator_p_scale(N, dims_no3);
	auto a2 = operator_p_scale(N, dims_no3);

	auto stack = operator_p_stack(2, 2, a, a2);

	auto b = operator_p_stack_FF(2, 2, a, a2); 

	long phases = dims[3];

	for (int k = 0; k < (phases - 1); k++) {

		auto tmp = operator_p_stack(3, 3, b, stack);
		operator_p_free(b);
                b = tmp;
	}

	operator_p_free(stack);

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

UT_REGISTER_TEST(test_op_p_stack2);

static bool test_op_p_stack3(void)
{
	enum { N = 4 };
	long dims[N] = { 8, 1, 4, 4 };
	long dims2[N] = { 8, 2, 4, 4 };
	long dims3[N] = { 8, 3, 4, 4 };
	long dims4[N] = { 8, 6, 4, 4 };

	auto a = operator_p_scale(N, dims);
	auto a2 = operator_p_scale(N, dims2);
	auto a3 = operator_p_scale(N, dims3);

	auto b = operator_p_stack_FF(1, 1, a, a2); 
	auto c = operator_p_stack_FF(1, 1, b, a3);

	complex float* in = md_alloc(N, dims4, CFL_SIZE);
	complex float* out = md_alloc(N, dims4, CFL_SIZE);

	md_zfill(N, dims4, in, 1.);
	md_zfill(N, dims4, out, 100.);

	operator_p_apply(c, 2., N, dims4, out, N, dims4, in);

	operator_p_free(c);

	md_zfill(N, dims4, in, 2.);

	float err = md_znrmse(N, dims4, out, in);

	md_free(in);
	md_free(out);

	return (err < UT_TOL);
}

UT_REGISTER_TEST(test_op_p_stack3);

static bool test_op_p_reshape(void)
{
	enum { N = 3 };
	long dims[N] = { 8, 4, 1 };
	long dims2[N] = { 1, 1, 32 };

	auto a = operator_p_scale(N, dims);
	auto b = operator_p_reshape_in_F(a, N, dims2);
	auto c = operator_p_reshape_out_F(b, N, dims2);

	complex float* in = md_alloc(N, dims2, CFL_SIZE);
	complex float* out = md_alloc(N, dims2, CFL_SIZE);

	md_zfill(N, dims2, in, 1.);
	md_zfill(N, dims2, out, 100.);

	operator_p_apply(c, 2., N, dims2, out, N, dims2, in);

	md_zfill(N, dims2, in, 2.);

	float err = md_znrmse(N, dims2, out, in);

	operator_p_free(c);

	md_free(in);
	md_free(out);

	return (err < UT_TOL);
}

UT_REGISTER_TEST(test_op_p_reshape);

static bool test_op_p_reshape_stack(void)
{
	enum { N = 3 };
	long dims[N] = { 8, 4, 4 };
	long dims2[N] = { 8, 1, 4 };
	long dims3[1] = { 8*5*4};

	const struct operator_p_s* a = operator_p_scale(N, dims);
	a = operator_p_flatten_F(a);
	
	const struct operator_p_s* b = operator_p_scale(N, dims2);
	b = operator_p_flatten_F(b);

	auto c = operator_p_stack_FF(0, 0, a, b);

	complex float* in = md_alloc(1, dims3, CFL_SIZE);
	complex float* out = md_alloc(1, dims3, CFL_SIZE);

	md_zfill(1, dims3, in, 1.);
	md_zfill(1, dims3, out, 100.);

	operator_p_apply(c, 2., 1, dims3, out, 1, dims3, in);

	md_zfill(1, dims3, in, 2.);

	float err = md_znrmse(1, dims3, out, in);

	operator_p_free(c);

	md_free(in);
	md_free(out);

	return (err < UT_TOL);
}

UT_REGISTER_TEST(test_op_p_reshape_stack);

