/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <assert.h>

#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"
#include "nlops/snlop.h"
#include "nlops/smath.h"

#include "utest.h"


static bool test_snlop_abs(void)
{
	enum { N = 4, M = 4 };
	long dims[N] = { M, 1, 1, 1 };

	arg_t x = snlop_input(N, dims, "x");
	arg_t abs_x = snlop_abs(x);

	arg_t iargs[1] = { x };
	arg_t oargs[1] = { abs_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(abs_x), 1, oargs, 1, iargs);

	complex float in[M] = { -1., 3. + 4.i, 4., 4. + 3.i };
	complex float out[M];
	complex float ref_out[M] = { 1., 5., 4., 5. };

	nlop_apply(nlop, N, dims, out, N, dims, in);

	nlop_free(nlop);

	UT_RETURN_ASSERT(0. == md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_abs);



static bool test_snlop_unary2(float eps, complex float (*fun)(complex float), arg_t (*snlop)(arg_t),
			int N, const long dims[N], int M, const complex float in[M])
{
	assert(M == md_calc_size(N, dims));

	arg_t x = snlop_input(N, dims, "x");
	arg_t snlop_x = snlop(x);

	arg_t iargs[1] = { x };
	arg_t oargs[1] = { snlop(x) };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(snlop_x), 1, oargs, 1, iargs);

	complex float out[M];
	complex float ref_out[M];

	for (int i = 0; i < M; i++)
		ref_out[i] = fun(in[i]);

	nlop_apply(nlop, N, dims, out, N, dims, in);

	nlop_free(nlop);

	UT_RETURN_ASSERT(eps >= md_zrmse(N, dims, ref_out, out));
}

static bool test_snlop_unary(complex float (*fun)(complex float), arg_t (*snlop)(arg_t), const complex float in[4])
{
	enum { N = 4, M = 4 };
	long dims[N] = { M, 1, 1, 1 };

	return test_snlop_unary2(0., fun, snlop, N, dims, M, in);
}

static bool test_snlop_exp(void)
{
	return test_snlop_unary(cexpf, snlop_exp, (complex float[4]){ 0., 1., -1., 1. + 1.i });
}

UT_REGISTER_TEST(test_snlop_exp);


static bool test_snlop_log(void)
{
	return test_snlop_unary(clogf, snlop_log, (complex float[4]){ 1l, 2.718282, -1., 1. + 1.i });
}

UT_REGISTER_TEST(test_snlop_log);


static bool test_snlop_cos(void)
{
	return test_snlop_unary(ccosf, snlop_cos, (complex float[4]){ 0., M_PI / 2., M_PI, 1. + 1.i });
}

UT_REGISTER_TEST(test_snlop_cos);


static bool test_snlop_sin(void)
{
	return test_snlop_unary(csinf, snlop_sin, (complex float[4]){ 0., M_PI / 2., M_PI, 1. + 1.i });
}

UT_REGISTER_TEST(test_snlop_sin);


static complex float crealf2(complex float x) { return crealf(x); }

static bool test_snlop_real(void)
{
	return test_snlop_unary(crealf2, snlop_real, (complex float[4]){ 1., 1.i, 1.i + 1.i, -1. + 1.i });
}

UT_REGISTER_TEST(test_snlop_real);


static bool test_snlop_conj(void)
{
	return test_snlop_unary(conjf, snlop_conj, (complex float[4]){ -1., 1.i, 1.i + 1.i, -1. + 1.i });
}

UT_REGISTER_TEST(test_snlop_conj);


static bool test_snlop_sqrt(void)
{
	long dims[4] = { 4, 1, 1, 1 };
	return test_snlop_unary2(1.e-7, csqrtf, snlop_sqrt, 4, dims, 4, (complex float[4]){ 1., 4., -1., 2. + 2.i });
}

UT_REGISTER_TEST(test_snlop_sqrt);


static complex float cinv(complex float x) { return 1. / x; }

static bool test_snlop_inv(void)
{
	return test_snlop_unary(cinv, snlop_inv, (complex float[4]){ 1., 2., 1. + 1.i, 1. - 1.i });
}

UT_REGISTER_TEST(test_snlop_inv);



static bool test_snlop_pow(void)
{
	enum { N = 4, M = 4 };
	long dims[N] = { M, 1, 1, 1 };

	arg_t x = snlop_input(N, dims, "x");
	arg_t pow_x = snlop_spow(x, 2.);

	arg_t iargs[1] = { x };
	arg_t oargs[1] = { pow_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(pow_x), 1, oargs, 1, iargs);

	complex float in[M] = { 1., 1.i, 1. + 1.i, -1. + 1.i };
	complex float out[M];
	complex float ref_out[M] = { 1., -1., 2.i, -2.i };

	nlop_apply(nlop, N, dims, out, N, dims, in);

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-7 > md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_pow);


static bool test_snlop_scale(void)
{
	enum { N = 4, M = 4 };
	long dims[N] = { M, 1, 1, 1 };

	arg_t x = snlop_input(N, dims, "x");
	arg_t scale_x = snlop_scale(x, 2.);

	arg_t iargs[1] = { x };
	arg_t oargs[1] = { scale_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(scale_x), 1, oargs, 1, iargs);

	complex float in[M] = { 1., 1.i, 1. + 1.i, -1. + 1. * 1.i };
	complex float out[M];
	complex float ref_out[M] = { 2., 2.i, 2. + 2.i, -2. + 2.i };

	nlop_apply(nlop, N, dims, out, N, dims, in);

	nlop_free(nlop);

	UT_RETURN_ASSERT(0. == md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_scale);


static bool test_snlop_cdiag(void)
{
	enum { N = 4 };
	long dims[N] = { 4, 3, 1, 1 };

	const complex float diag[] = {
	    1., 2., 3., 4.,
	    5., 6., 7., 8.,
	    9., 10., 11., 12.
	};

	arg_t x = snlop_input(N, dims, "x");
	arg_t cdiag_x = snlop_cdiag(x, N, dims, diag);

	arg_t iargs[] = { x };
	arg_t oargs[] = { cdiag_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(cdiag_x), 1., oargs, 1, iargs);

	complex float in[12] = { 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. };
	complex float out[12];
	complex float ref_out[12] = { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12. }; // Element-wise multiplication with diagonal

	nlop_apply(nlop, N, dims, out, N, dims, in);

	nlop_free(nlop);

	UT_RETURN_ASSERT(0. == md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_cdiag);


static bool test_snlop_fmac(void)
{
	enum { N = 4 };
	long dims[N] = { 2, 2, 1, 1 };

	const complex float ten[] = { 1., 2., 3., 4. };

	arg_t x = snlop_input(N, dims, "x");
	arg_t fmac_x = snlop_fmac(x, N, dims, ten, 0);

	arg_t iargs[] = { x };
	arg_t oargs[] = { fmac_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(fmac_x), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	const complex float in[4] = { 3., 3., 4., 4. };
	complex float out[4];
	complex float ref_out[4] = { 3., 6., 12., 16. };

	nlop_apply(nlop, N, dims, out, N, dims, in);

	nlop_free(nlop);

	UT_RETURN_ASSERT(0. == md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_fmac);


static bool test_snlop_stack(void)
{
	enum { N = 4 };
	long dims[N] = { 4, 1, 1, 1 };

	arg_t x1 = snlop_input(N, dims, "x1");
	arg_t x2 = snlop_input(N, dims, "x2");
	arg_t stack_x = snlop_stack(x1, x2, 0);

	arg_t iargs[2] = { x1, x2 };
	arg_t oargs[1] = { stack_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(stack_x), 1, oargs, 2, iargs);

	complex float in1[4] = { 1., 2., 3., 4. };
	complex float in2[4] = { 5., 6., 7., 8. };
	complex float out[8];
	complex float ref_out[8] = { 1., 2., 3., 4., 5., 6., 7., 8. };

	nlop_generic_apply_unchecked(nlop, 3, (void* [3]){ out, in1, in2 });

	nlop_free(nlop);

	UT_RETURN_ASSERT(0. == md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_stack);


static bool test_snlop_binary(arg_t (*snlop)(arg_t x1, arg_t x2),
		const complex float in1[4], const complex float in2[4], const complex float ref[4])
{
	enum { N = 4 };
	long dims[N] = { 2, 2, 1, 1 };

	arg_t x1 = snlop_input(N, dims, "x1");
	arg_t x2 = snlop_input(N, dims, "x2");
	arg_t snlop_x = snlop(x1, x2);

	arg_t iargs[2] = { x1, x2 };
	arg_t oargs[1] = { snlop_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(snlop_x), 1, oargs, 2, iargs);

	complex float out[4];

	nlop_generic_apply_unchecked(nlop, 3, (void*[3]){ out, (void*)in1, (void*)in2 });

	nlop_free(nlop);

	UT_RETURN_ASSERT(0. == md_zrmse(N, dims, ref, out));
}


static bool test_snlop_mul_simple(void)
{
	const complex float in1[] = { 1, 2, 3, 4 };
	const complex float in2[] = { 5, 6, 7, 8 };
	const complex float ref[] = { 5, 12, 21, 32 };

	return test_snlop_binary(snlop_mul_simple, in1, in2, ref);
}

UT_REGISTER_TEST(test_snlop_mul_simple);


static bool test_snlop_mul(void)
{
	enum { N = 4 };
	long dims[N] = { 2, 3, 1, 1 };

	long odims[N] = { 3, 1, 1, 1 };

	arg_t x1 = snlop_input(N, dims, "x1");
	arg_t x2 = snlop_input(N, dims, "x2");
	arg_t mul_x = snlop_mul(x1, x2, 1UL);

	arg_t iargs[2] = { x1, x2 };
	arg_t oargs[1] = { mul_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(mul_x), 1, oargs, 2, iargs);

	complex float in1[6] = { 1., 2., 3., 4., 5., 6. };
	complex float in2[6] = { 7., 8., 9., 10., 11., 12. };
	complex float out[3];
	complex float ref_out[3] = { 23., 67., 127. }; // [1*7 + 2*8, 3*9 + 4*10, 5*11 + 6*12]

	nlop_generic_apply_unchecked(nlop, 3, (void* [3]){ out, in1, in2 });

	nlop_free(nlop);

	UT_RETURN_ASSERT(0. == md_zrmse(N, odims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_mul);


static bool test_snlop_div_simple(void)
{
	const complex float in1[] = { 6., 12., 20., 30. };
	const complex float in2[] = { 2., 3., 4., 5. };
	const complex float ref[] = { 3., 4., 5., 6. };

	return test_snlop_binary(snlop_div_simple, in1, in2, ref);
}


UT_REGISTER_TEST(test_snlop_div_simple);


static bool test_snlop_div(void)
{
	enum { N = 4 };
	long dims[N] = { 2, 3, 1, 1 };

	long odims[N] = { 3, 1, 1, 1 };

	arg_t x1 = snlop_input(N, dims, "x1");
	arg_t x2 = snlop_input(N, dims, "x2");
	arg_t div = snlop_div(x1, x2, 1UL);

	arg_t iargs[2] = { x1, x2 };
	arg_t oargs[1] = { div };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(div), 1, oargs, 2, iargs);

	complex float in1[6] = { 6., 12., 20., 30., 42., 48. };
	complex float in2[6] = { 2., 3., 4., 5., 7., 8. };
	complex float out[6];
	complex float ref_out[3] = { 7., 11., 12. }; // [6/2 + 12/3, 20/4 + 30/ 5, 42/7 + 48/8]

	nlop_generic_apply_unchecked(nlop, 3, (void* [3]){ out, in1, in2 });

	nlop_free(nlop);

	UT_RETURN_ASSERT(0. == md_zrmse(N, odims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_div);


static bool test_snlop_add(void)
{
	const complex float in1[] = { 1., 2., 7., 8. };
	const complex float in2[] = { 10., 20., 70., 80. };
	const complex float ref[] = { 11., 22., 77., 88. };

	return test_snlop_binary(snlop_add, in1, in2, ref);
}

UT_REGISTER_TEST(test_snlop_add);


static bool test_snlop_sub(void)
{
	const complex float in1[] = { 1., 2., 7., 8. };
	const complex float in2[] = { 10., 20., 70., 80. };
	const complex float ref[] = { -9., -18., -63., -72. };

	return test_snlop_binary(snlop_sub, in1, in2, ref);
}

UT_REGISTER_TEST(test_snlop_sub);



static bool test_snlop_axpbz(void)
{
	enum { N = 4 };
	long dims[N] = { 2, 3, 1, 1 };

	arg_t a = snlop_input(N, dims, "a");
	arg_t b = snlop_input(N, dims, "b");

	complex float sa = 2.;
	complex float sb = 3.;
	arg_t axpbz = snlop_axpbz(a, b, sa, sb);

	arg_t iargs[2] = { a, b };
	arg_t oargs[1] = { axpbz };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(axpbz), 1, oargs, 2, iargs);

	complex float in1[6] = { 1., 2., 3., 4., 5., 6. };
	complex float in2[6] = { 10., 20., 30., 40., 50., 60. };
	complex float out[6];

	// Expected: sa*a + sb*b = 2*in1 + 3*in2
	complex float ref_out[] = {
	    2. * 1. + 3. * 10.,
	    2. * 2. + 3. * 20.,
	    2. * 3. + 3. * 30.,
	    2. * 4. + 3. * 40.,
	    2. * 5. + 3. * 50.,
	    2. * 6. + 3. * 60.,
	};

	nlop_generic_apply_unchecked(nlop, 3, (void* [3]){ out, in1, in2 });

	nlop_free(nlop);

	UT_RETURN_ASSERT(0. == md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_axpbz);



static bool test_snlop_add_scalar(void)
{
	enum { N = 4 };
	long dims[N] = { 4, 1, 1, 1 };

	// 1 + x
	arg_t x = snlop_input(N, dims, "x");
	arg_t as = snlop_scale(x, 0.);
	arg_t a1 = snlop_scalar(1.);
	arg_t a = snlop_add(as, a1);
	arg_t res = snlop_add(x, a);

	arg_t iargs[1] = { x };
	arg_t oargs[2] = { res, a };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(res), 2, oargs, 1, iargs);

	complex float in[4] = { 1., 2., 3., 4. };
	complex float out1[4];
	complex float out2[4];
	complex float ref_out1[4] = { 1., 1., 1., 1. };
	complex float ref_out2[4] = { 2., 3., 4., 5. };

	nlop_generic_apply_unchecked(nlop, 3, (void* [3]){ out2, out1, in });

	nlop_free(nlop);

	UT_RETURN_ON_FAILURE(0. == md_zrmse(N, dims, ref_out1, out1));
	UT_RETURN_ON_FAILURE(0. == md_zrmse(N, dims, ref_out2, out2));

	return true;
}

UT_REGISTER_TEST(test_snlop_add_scalar);



static bool test_snlop_sub_scalar(void)
{
	enum { N = 4 };
	long dims[N] = { 4, 1, 1, 1 };

	// x - 1
	arg_t x = snlop_input(N, dims, "x");
	arg_t as = snlop_scale(x, 0);
	arg_t a1 = snlop_scalar(1);
	arg_t a = snlop_add(as, a1);
	arg_t res = snlop_sub(x, a);

	arg_t iargs[1] = { x };
	arg_t oargs[2] = { res, a };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(res), 2, oargs, 1, iargs);

	complex float in[4] = { 1., 2., 3., 4. };
	complex float out1[4];
	complex float out2[4];
	complex float ref_out1[4] = { 1., 1., 1., 1. };
	complex float ref_out2[4] = { 0., 1., 2., 3. };

	nlop_generic_apply_unchecked(nlop, 3, (void* [3]){ out2, out1, in });

	nlop_free(nlop);

	UT_RETURN_ON_FAILURE(0. == md_zrmse(N, dims, ref_out1, out1));
	UT_RETURN_ON_FAILURE(0. == md_zrmse(N, dims, ref_out2, out2));

	return true;
}

UT_REGISTER_TEST(test_snlop_sub_scalar);



static bool test_snlop_fix_input(void)
{
	enum { N = 4 };
	long dims[] = { 4, 1, 1, 1 };

	arg_t x = snlop_input(N, dims, "x");
	arg_t zeros = snlop_scale(x, 0.);
	arg_t fixed = snlop_scalar(5.);
	arg_t res = snlop_add(zeros, fixed);

	arg_t iargs[] = { x };
	arg_t oargs[] = { res };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(res), 1, oargs, 1, iargs);

	const complex float in[4] = { 1., 2., 3., 4. };
	const complex float ref_out[4] = { 5., 5., 5., 5. };
	complex float out[4];

	nlop_apply(nlop, N, dims, out, N, dims, in);

	nlop_free(nlop);

	UT_RETURN_ASSERT(0. == md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_fix_input);

