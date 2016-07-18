/* Copyright 2016. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/rand.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "utest.h"


// include test data
#include "test_flpmath_data.h"




static bool test_md_zfmacc2_flags(unsigned int D, const long idims[D], unsigned int flags, const complex float* in1, const complex float* in2, const complex float* out_ref)
{
	long odims[D];
	md_select_dims(D, ~flags, odims, idims);

	complex float* out = md_calloc(D, odims, CFL_SIZE);

	long istr[D];
	long ostr[D];

	md_calc_strides(D, istr, idims, CFL_SIZE);
	md_calc_strides(D, ostr, odims, CFL_SIZE);

	md_zfmacc2(D, idims, ostr, out, istr, in1, istr, in2);

	float err = md_znrmse(D, odims, out_ref, out);

	md_free(out);

	UT_ASSERT(err < UT_TOL);

	return true;
}


/*
 * Tests based on previously generated data included in the header file
 */
static bool test_md_zfmacc2(void)
{
	long idims[4] = { 3, 3, 3, 3 };

	bool ret = true;

	for (unsigned int flags = 0u; flags < 16u; flags++) {

		debug_printf(DP_DEBUG1, "Testing md_zfmacc2_flags with flags=%d\n", flags);

		ret &= test_md_zfmacc2_flags(4, idims, flags, test_md_in0, test_md_in1, test_md_zfmacc2_out[flags]);
	}

	return ret;
}


static bool test_md_zavg_flags(unsigned int D, const long idims[D], unsigned int flags, const complex float* in, const complex float* out_ref, bool wavg)
{
	long odims[D];
	md_select_dims(D, ~flags, odims, idims);

	complex float* out = md_alloc(D, odims, CFL_SIZE);

	(wavg ? md_zwavg : md_zavg)(D, idims, flags, out, in);

	float err = md_znrmse(D, odims, out_ref, out);

	md_free(out);

	UT_ASSERT(err < UT_TOL);

	return true;
}


/*
 * Tests based on previously generated data included in the header file
 */
static bool test_md_zwavg(void)
{
	long idims[4] = { 3, 3, 3, 3 };

	bool wavg = true;
	bool ret = true;

	for (unsigned int flags = 0u; flags < 16u; flags++) {

		debug_printf(DP_DEBUG1, "Testing md_zwavg_flags with flags=%d\n", flags);

		ret &= test_md_zavg_flags(4, idims, flags, test_md_in0, test_md_zwavg_out[flags], wavg);
	}

	return ret;
}


static bool test_md_zavg(void)
{
	long idims[4] = { 3, 3, 3, 3 };

	bool wavg = false;
	bool ret = true;

	for (unsigned int flags = 0u; flags < 16u; flags++) {

		debug_printf(DP_DEBUG1, "Testing md_zavg_flags with flags=%d\n", flags);

		ret &= test_md_zavg_flags(4, idims, flags, test_md_in0, test_md_zavg_out[flags], wavg);
	}

	return ret;
}



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

static bool test_md_zmatmul(void)
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

	md_zmatmul(3, odims, dst1, idims1, src1, idims2, src2);

	matrix_mult(A, B, C, &MD_CAST_ARRAY2(complex float, 3, odims, dst2, 0, 2),
			&MD_CAST_ARRAY2(const complex float, 3, idims1, src1, 1, 2),
			&MD_CAST_ARRAY2(const complex float, 3, idims2, src2, 0, 1));

	double err = md_znrmse(3, odims, dst2, dst1);

	md_free(src1);
	md_free(src2);
	md_free(dst1);
	md_free(dst2);

	return (err < UT_TOL);
}





UT_REGISTER_TEST(test_md_zfmacc2);
UT_REGISTER_TEST(test_md_zwavg);
UT_REGISTER_TEST(test_md_zavg);
UT_REGISTER_TEST(test_md_zmatmul);

