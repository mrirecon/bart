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

	int A = 2;
	int B = 3;
	int C = 4;

	long odims[N] = { C, 1, A };
	long idims1[N] = { 1, B, A };
	long idims2[N] = { C, B, 1 };

	complex float* dst1 = md_alloc(N, odims, CFL_SIZE);
	complex float* dst2 = md_alloc(N, odims, CFL_SIZE);
	complex float* src1 = md_alloc(N, idims1, CFL_SIZE);
	complex float* src2 = md_alloc(N, idims2, CFL_SIZE);

	md_gaussian_rand(N, odims, dst1); // test complete fill
	md_gaussian_rand(N, odims, dst2); // test complete fill
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

	int A = 2;
	int B = 3;
	int C = 4;

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

	int A = 2;
	int B = 3;
	int C = 4;

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


static bool test_linop_matrix_chain(void)
{
	int A = 9;
	int B = 7;
	int C = 3;
	int D = 2;
	int E = 5;

	enum { N = 8 };
	long odims[N] =  { D, C, 1, 1, 1, 1, C, D };
	long idims0[N] = { D, 1, 1, A, E, 1, A, D };
	long idims1[N] = { D, 1, B, A, 1, B, A, D };
	long tdims[N] =  { D, 1, B, 1, E, B, 1, D };
	long idims2[N] = { D, C, B, 1, E, B, C, D };

	complex float* dst1 = md_alloc(N, odims, CFL_SIZE);
	complex float* dst2 = md_alloc(N, odims, CFL_SIZE);
	complex float* src0 = md_alloc(N, idims0, CFL_SIZE);
	complex float* src1 = md_alloc(N, idims1, CFL_SIZE);
	complex float* src2 = md_alloc(N, idims2, CFL_SIZE);

	md_gaussian_rand(N, odims, dst1); // test complete fill
	md_gaussian_rand(N, odims, dst2); // test complete fill
	md_gaussian_rand(N, idims0, src0);
	md_gaussian_rand(N, idims1, src1);
	md_gaussian_rand(N, idims2, src2);

	struct linop_s* mat1 = linop_matrix_create(N, tdims, idims0, idims1, src1);
	struct linop_s* mat2 = linop_matrix_create(N, odims, tdims, idims2, src2);

	struct linop_s* matA = linop_chain(mat1, mat2);

	linop_forward(matA, N, odims, dst1, N, idims0, src0);

	linop_free(matA);

	struct linop_s* matB = linop_matrix_chain(mat1, mat2);

	linop_forward(matB, N, odims, dst2, N, idims0, src0);

	linop_free(matB);

	double err = md_znrmse(N, odims, dst2, dst1);

	linop_free(mat1);
	linop_free(mat2);

	md_free(src0);
	md_free(src1);
	md_free(src2);
	md_free(dst1);
	md_free(dst2);

	return (err < 1.E-5);
}


UT_REGISTER_TEST(test_linop_matrix);
UT_REGISTER_TEST(test_linop_matrix_adjoint);
UT_REGISTER_TEST(test_linop_matrix_normal);
UT_REGISTER_TEST(test_linop_matrix_chain);

