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
#include "num/blas.h"
#include "num/rand.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "utest.h"



static void matrix_mult(int A, int B, int C, complex float (*dst)[A][C], const complex float (*src1)[A][B], const complex float (*src2)[B][C])
{
	for (int i = 0; i < A; i++) {

		for (int k = 0; k < C; k++) {

			(*dst)[i][k] = 0.;

			for (int j = 0; j < B; j++)
				(*dst)[i][k] += (*src1)[i][j] * (*src2)[j][k];
		}
	}
}

static bool test_blas_matrix_mult(void)
{
	int A = 10;
	int B = 20;
	int C = 30;

	long odims[3] = { A, 1, C };
	long idims1[3] = { 1, B, C };
	long idims2[3] = { A, B, 1 };

	complex float* dst1 = md_alloc(3, odims, CFL_SIZE);
	complex float* dst2 = md_alloc(3, odims, CFL_SIZE);
	complex float* src1 = md_alloc(3, idims1, CFL_SIZE);
	complex float* src2 = md_alloc(3, idims2, CFL_SIZE);

	md_gaussian_rand(3, odims, dst1);
	md_gaussian_rand(3, odims, dst2);
	md_gaussian_rand(3, idims1, src1);
	md_gaussian_rand(3, idims2, src2);

	blas_matrix_multiply(A, C, B, MD_CAST_ARRAY2(complex float, 3, odims, dst1, 0, 2),
			MD_CAST_ARRAY2(const complex float, 3, idims2, src2, 0, 1),
			MD_CAST_ARRAY2(const complex float, 3, idims1, src1, 1, 2));

	// (A^T B^T)^T = B A
	
	matrix_mult(C, B, A, &MD_CAST_ARRAY2(complex float, 3, odims, dst2, 0, 2),
			&MD_CAST_ARRAY2(const complex float, 3, idims1, src1, 1, 2),
			&MD_CAST_ARRAY2(const complex float, 3, idims2, src2, 0, 1));

	double err = md_znrmse(3, odims, dst2, dst1);

	md_free(src1);
	md_free(src2);
	md_free(dst1);
	md_free(dst2);

	return (err < UT_TOL);
}


UT_REGISTER_TEST(test_blas_matrix_mult);

