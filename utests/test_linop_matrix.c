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

#include "misc/misc.h"
#include "misc/debug.h"

#include "utest.h"




static bool test_linop_matrix(void)
{
	int A = 10;
	int B = 20;
	int C = 30;

	long odims[3] = { C, 1, A };
	long idims1[3] = { 1, B, A };
	long idims2[3] = { C, B, 1 };

	complex float* dst1 = md_alloc(3, odims, CFL_SIZE);
	complex float* dst2 = md_alloc(3, odims, CFL_SIZE);
	complex float* src1 = md_alloc(3, idims1, CFL_SIZE);
	complex float* src2 = md_alloc(3, idims2, CFL_SIZE);

	md_gaussian_rand(3, odims, dst1);
	md_gaussian_rand(3, odims, dst2);
	md_gaussian_rand(3, idims1, src1);
	md_gaussian_rand(3, idims2, src2);

	struct linop_s* mat = linop_matrix_create(3, odims, idims2, idims1, src1);

	md_zmatmul(3, odims, dst1, idims1, src1, idims2, src2);

	linop_forward(mat, 3, odims, dst2, 3, idims2, src2);

	double err = md_znrmse(3, odims, dst2, dst1);

	linop_free(mat);

	md_free(src1);
	md_free(src2);
	md_free(dst1);
	md_free(dst2);

	return (err < UT_TOL);
}




UT_REGISTER_TEST(test_linop_matrix);

