/* Copyright 2016. The Regents of the University of California.
 * Copyright 2016-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2016-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
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



static bool test_md_zhardthresh(void)
{
	complex float test_vec[] = { 1., 2., 3., 4., 5., 6., 7., 8., 9. };

	unsigned int N = ARRAY_SIZE(test_vec);
	complex float test_out[N];

	unsigned int k = 5;

	md_zhardthresh(1, (long[1]){ N }, k, 0, test_out, test_vec);

	bool ok = true;

	for (unsigned int i = 0; i < N - k; i++)
		ok &= (0. == test_out[i]);

	for (unsigned int i = N - k; i < N; i++)
		ok &= (test_vec[i] == test_out[i]);

	return ok;
}


static bool test_md_zvar(void)
{
	const complex float test_vec[] = { 1. -6.j, 2. - 5.j, 3. - 4.j, 4. - 3.j, 5. - 2.j, 6. - 1.j };
	const complex float ref[] = { 8., 8. };

	long idims[2] = { 2, 3 };
	long odims[2] = { 2, 1 };

	complex float* out = md_alloc(2, odims, CFL_SIZE);

	md_zvar(2, idims, MD_BIT(1), out, test_vec);

	double err = md_znrmse(2, odims, ref, out);

	md_free(out);

	return (err < UT_TOL);
}


static bool test_md_zcovar(void)
{
	const complex float test_vec1[] = { 1. - 6.j, 2. - 5.j, 3. - 4.j, 4. - 3.j, 5. - 2.j, 6. - 1.j };
	const complex float test_vec2[] = { 1. - 6.j, 2.j + 5., 3. - 4.j, 4.j + 3., 5. - 2.j, 6.j + 1. };
	const complex float ref[] = { 8., -8.j };

	long idims[2] = { 2, 3 };
	long odims[2] = { 2, 1 };

	complex float* out = md_alloc(2, odims, CFL_SIZE);

	md_zcovar(2, idims, MD_BIT(1), out, test_vec1, test_vec2);

	double err = md_znrmse(2, odims, ref, out);

	md_free(out);

	return (err < UT_TOL);
}



static bool test_md_zstd(void)
{
	const complex float test_vec[] = { 1. - 6.j, 2. - 5.j, 3. - 4.j, 4. - 3.j, 5. - 2.j, 6. - 1.j };
	const complex float ref[] = { 1., 1., 1. };

	long idims[2] = { 2, 3 };
	long odims[2] = { 1, 3 };

	complex float* out = md_alloc(2, odims, CFL_SIZE);

	md_zstd(2, idims, MD_BIT(0), out, test_vec);

	double err = md_znrmse(2, odims, ref, out);

	md_free(out);

	return (err < UT_TOL);
}


static bool test_md_zconv(void)
{
	enum { N = 1 };
	long idims[N] = { 10 };
	long kdims[N] = { 3 };
	long odims[N] = { 8 };

	complex float* x = md_calloc(N, idims, sizeof(complex float));
	complex float* y = md_calloc(N, odims, sizeof(complex float));
	complex float* z = md_calloc(N, odims, sizeof(complex float));

	x[5] = 1.;

	z[3] = 0.5;
	z[4] = 1.;
	z[5] = -0.5;

	complex float k[3] = { 0.5, 1., -0.5 };

	md_zconv(N, 1u, odims, y, kdims, &k[0], idims, x);

	float err = md_znrmse(N, odims, y, z);

	md_free(x);
	md_free(y);
	md_free(z);

	return (err < 1.E-6);
}

static bool test_md_complex_real_conversion(void)
{
	enum { N = 1 };
	long dims[N] = { 10 };

	complex float* src_comp = md_alloc(N, dims, CFL_SIZE);
	md_gaussian_rand(N, dims, src_comp);

	float* real = md_alloc(N, dims, FL_SIZE);
	float* imag = md_alloc(N, dims, FL_SIZE);

	md_real(N, dims, real, src_comp);
	md_imag(N, dims, imag, src_comp);

	complex float* dst1 = md_alloc(N, dims, CFL_SIZE);
	complex float* dst2 = md_alloc(N, dims, CFL_SIZE);

	float err = 0;

	md_zcmpl_real(N, dims, dst1, real);
	md_zreal(N, dims, dst2, src_comp);
	err += md_znrmse(N, dims, dst2, dst1);

	md_zcmpl_imag(N, dims, dst1, imag);
	md_zimag(N, dims, dst2, src_comp);
	err += md_znrmse(N, dims, dst2, dst1);

	md_zcmpl(N, dims, dst1, real, imag);
	err += md_znrmse(N, dims, src_comp, dst1);
	
	md_free(src_comp);
	md_free(real);
	md_free(imag);
	md_free(dst1);
	md_free(dst2);

	return (err < 1.E-8);
}







UT_REGISTER_TEST(test_md_zfmacc2);
UT_REGISTER_TEST(test_md_zwavg);
UT_REGISTER_TEST(test_md_zavg);
UT_REGISTER_TEST(test_md_zmatmul);
UT_REGISTER_TEST(test_md_zhardthresh);
UT_REGISTER_TEST(test_md_zvar);
UT_REGISTER_TEST(test_md_zcovar);
UT_REGISTER_TEST(test_md_zstd);
UT_REGISTER_TEST(test_md_zconv);
UT_REGISTER_TEST( test_md_complex_real_conversion);

