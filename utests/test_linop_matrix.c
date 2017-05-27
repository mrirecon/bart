/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"

#include "linops/someops.h"
#include "linops/linop.h"
#include "linops/lintest.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "utest.h"




static bool test_linop_matrix(void)
{
	enum { N = 3 };

	int A = 10;
	int B = 20;
	int C = 30;

	long odims[N] = { C, 1, A };
	long idims1[N] = { 1, B, A };
	long idims2[N] = { C, B, 1 };

	complex float* dst1 = md_alloc(N, odims, CFL_SIZE);
	complex float* dst2 = md_alloc(N, odims, CFL_SIZE);
	complex float* src1 = md_alloc(N, idims1, CFL_SIZE);
	complex float* src2 = md_alloc(N, idims2, CFL_SIZE);

	md_gaussian_rand(N, odims, dst1);
	md_gaussian_rand(N, odims, dst2);
	md_gaussian_rand(N, idims1, src1);
	md_gaussian_rand(N, idims2, src2);

	struct linop_s* mat = linop_matrix_create(N, odims, idims2, idims1, src1);

	md_zmatmul(N, odims, dst1, idims1, src1, idims2, src2);

	linop_forward(mat, N, odims, dst2, N, idims2, src2);

	double err = md_znrmse(N, odims, dst2, dst1);

	linop_free(mat);

	md_free(src1);
	md_free(src2);
	md_free(dst1);
	md_free(dst2);

	return (err < UT_TOL);
}



static bool test_linop_matrix_adjoint(void)
{
	enum { N = 3 };

	int A = 10;
	int B = 20;
	int C = 30;

	long odims[N] = { C, 1, A };
	long idims1[N] = { 1, B, A };
	long idims2[N] = { C, B, 1 };

	complex float* src1 = md_alloc(N, idims1, CFL_SIZE);

	md_gaussian_rand(N, idims1, src1);

	struct linop_s* mat = linop_matrix_create(N, odims, idims2, idims1, src1);

	float diff = linop_test_adjoint(mat);

	debug_printf(DP_DEBUG1, "adjoint diff: %f\n", diff);

	bool ret = (diff < 1.E-4f);

	linop_free(mat);

	return ret;
}


static bool test_linop_matrix_normal(void)
{
	enum { N = 3 };

	int A = 10;
	int B = 15;
	int C = 30;

	long odims[N] = { C, 1, A };
	long idims1[N] = { 1, B, A };
	long idims2[N] = { C, B, 1 };

	complex float* src1 = md_alloc(N, idims1, CFL_SIZE);

	md_gaussian_rand(N, idims1, src1);

	struct linop_s* mat = linop_matrix_create(N, odims, idims2, idims1, src1);

	float nrmse = linop_test_normal(mat);

	debug_printf(DP_DEBUG1, "normal nrmse: %f\n", nrmse);

	bool ret = (nrmse < 1.E-6f);

	linop_free(mat);

	return ret;
}



UT_REGISTER_TEST(test_linop_matrix);
UT_REGISTER_TEST(test_linop_matrix_adjoint);
UT_REGISTER_TEST(test_linop_matrix_normal);

