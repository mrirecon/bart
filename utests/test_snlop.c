/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#include "misc/list.h"
#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"


#include "linops/linop.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/const.h"
#include "nlops/nltest.h"
#include "nlops/someops.h"
#include "nlops/snlop.h"
#include "nlops/smath.h"

#include "utest.h"


static bool test_snlop_abs(void)
{
	long dims[] = { 4, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	arg_t x = snlop_input(N, dims, "x");
	arg_t abs_x = snlop_abs(x);

	arg_t iargs[] = { x };
	arg_t oargs[] = { abs_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(abs_x), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in[] = { -1, 1 + 1 * I, 4, 2 + 2 * I };
	complex float out[N];
	complex float ref_out[] = { 1, 1.414214, 4, 2.828427 };

	nlop_apply(nlop, N, dims, out, N, dims, in);

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_abs);

static bool test_snlop_exp(void)
{
	long dims[] = { 4, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	arg_t x = snlop_input(N, dims, "x");
	arg_t exp_x = snlop_exp(x);

	arg_t iargs[] = { x };
	arg_t oargs[] = { exp_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(exp_x), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in[] = { 0, 1, -1, 1 + 1 * I };
	complex float out[N];
	complex float ref_out[] = { 1, 2.718282, 0.367879, 1.468694 + 2.287355 * I };

	nlop_apply(nlop, N, dims, out, N, dims, in);

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_exp);

static bool test_snlop_log(void)
{
	long dims[] = { 4, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	arg_t x = snlop_input(N, dims, "x");
	arg_t log_x = snlop_log(x);

	arg_t iargs[] = { x };
	arg_t oargs[] = { log_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(log_x), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in[] = { 1, 2.718282, -1, 1 + 1 * I };
	complex float out[N];
	complex float ref_out[] = { 0,1, 0 + M_PI * I, 0.346574 + 0.785398 * I };

	nlop_apply(nlop, N, dims, out, N, dims, in);

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_log);

static bool test_snlop_cos(void)
{
	long dims[] = { 4, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	arg_t x = snlop_input(N, dims, "x");
	arg_t cos_x = snlop_cos(x);

	arg_t iargs[] = { x };
	arg_t oargs[] = { cos_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(cos_x), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in[] = { 0, M_PI / 2, M_PI, 1 + 1 * I };
	complex float out[N];
	complex float ref_out[] = { 1, 0, -1, 0.833730 - 0.988898 * I };

	nlop_apply(nlop, N, dims, out, N, dims, in);

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_cos);

static bool test_snlop_sin(void)
{
	long dims[] = { 4, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	arg_t x = snlop_input(N, dims, "x");
	arg_t sin_x = snlop_sin(x);

	arg_t iargs[] = { x };
	arg_t oargs[] = { sin_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(sin_x), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in[] = { 0, M_PI / 2, M_PI, 1 + 1 * I };
	complex float out[N];
	complex float ref_out[] = { 0, 1, 0,  1.298458 + 0.634964 * I };

	nlop_apply(nlop, N, dims, out, N, dims, in);

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_sin);

static bool test_snlop_real(void)
{
	long dims[] = { 4, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	arg_t x = snlop_input(N, dims, "x");
	arg_t real_x = snlop_real(x);

	arg_t iargs[] = { x };
	arg_t oargs[] = { real_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(real_x), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in[] = { 1, 1 * I, 1 + 1 * I, -1 + 1 * I };
	complex float out[N];
	complex float ref_out[] = { 1, 0, 1,  -1 };

	nlop_apply(nlop, N, dims, out, N, dims, in);

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_real);

static bool test_snlop_conj(void)
{
	long dims[] = { 5, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	arg_t x = snlop_input(N, dims, "x");
	arg_t conj_x = snlop_conj(x);

	arg_t iargs[] = { x };
	arg_t oargs[] = { conj_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(conj_x), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in[] = { -1, 1 * I, 1 + 1 * I, -1 + 1 * I , -1 - 1 * I };
	complex float out[N];
	complex float ref_out[] = { -1, -1 * I, 1 - 1 * I, -1 - 1 * I, -1 + 1 * I };

	nlop_apply(nlop, N, dims, out, N, dims, in);

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_conj);

static bool test_snlop_sqrt(void)
{
	long dims[] = { 4, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	arg_t x = snlop_input(N, dims, "x");
	arg_t sqrt_x = snlop_sqrt(x);

	arg_t iargs[] = { x };
	arg_t oargs[] = { sqrt_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(sqrt_x), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in[] = { 1, 4, -1, 2 + 2 * I };
	complex float out[N];
	complex float ref_out[] = { 1, 2, I, 1.553774 + 0.643594 * I };

	nlop_apply(nlop, N, dims, out, N, dims, in);

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_sqrt);

static bool test_snlop_inv(void)
{
	long dims[] = { 6, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	arg_t x = snlop_input(N, dims, "x");
	arg_t inv_x = snlop_inv(x);

	arg_t iargs[] = { x };
	arg_t oargs[] = { inv_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(inv_x), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in[] = { 1, 2, 4, I, 1 + 1 * I, 1 - 1 * I };
	complex float out[N];
	complex float ref_out[] = { 1, 0.5, 0.25, -I, 0.5 - 0.5 * I, 0.5 + 0.5 * I };

	nlop_apply(nlop, N, dims, out, N, dims, in);

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_inv);


static bool test_snlop_pow(void)
{
	long dims[] = { 4, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	arg_t x = snlop_input(N, dims, "x");
	arg_t pow_x = snlop_spow(x, 2.);

	arg_t iargs[] = { x };
	arg_t oargs[] = { pow_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(pow_x), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in[] = { 1, I, 1 + I, -1 + 1 * I };
	complex float out[N];
	complex float ref_out[] = { 1, -1, 2 * I,  -2 * I };

	nlop_apply(nlop, N, dims, out, N, dims, in);

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_pow);

static bool test_snlop_scale(void)
{
	long dims[] = { 4, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	arg_t x = snlop_input(N, dims, "x");
	arg_t scale_x = snlop_scale(x, 2.);

	arg_t iargs[] = { x };
	arg_t oargs[] = { scale_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(scale_x), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in[] = { 1, I, 1 + I, -1 + 1 * I };
	complex float out[N];
	complex float ref_out[] = { 2, 2 * I, 2 + 2 * I,  -2 + 2 * I };

	nlop_apply(nlop, N, dims, out, N, dims, in);

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_scale);

static bool test_snlop_cdiag(void)
{
	long dims[] = { 4, 3, 1, 1 };
	int N = ARRAY_SIZE(dims);

	complex float diag[] = {
	    1, 2, 3, 4,
	    5, 6, 7, 8,
	    9, 10, 11, 12
	};

	arg_t x = snlop_input(N, dims, "x");
	arg_t cdiag_x = snlop_cdiag(x, N, dims, diag);

	arg_t iargs[] = { x };
	arg_t oargs[] = { cdiag_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(cdiag_x), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	complex float out[md_calc_size(N, dims)];
	complex float ref_out[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }; // Element-wise multiplication with diagonal

	nlop_apply(nlop, N, dims, out, N, dims, in);

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_cdiag);

static bool test_snlop_fmac(void)
{
	long dims[] = { 2, 2, 1, 1 };
	int N = ARRAY_SIZE(dims);

	complex float ten[] = { 1, 2, 3, 4 };

	arg_t x = snlop_input(N, dims, "x");
	arg_t fmac_x = snlop_fmac(x, N, dims, ten, 0);

	arg_t iargs[] = { x };
	arg_t oargs[] = { fmac_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(fmac_x), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in[] = { 3, 3, 4, 4 };
	complex float out[md_calc_size(N, dims)];
	complex float ref_out[] = { 3, 6, 12, 16 };

	nlop_apply(nlop, N, dims, out, N, dims, in);

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_fmac);

static bool test_snlop_stack(void)
{
	long dims[] = { 4, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	arg_t x1 = snlop_input(N, dims, "x1");
	arg_t x2 = snlop_input(N, dims, "x2");
	arg_t stack_x = snlop_stack(x1, x2, 0);

	arg_t iargs[] = { x1, x2 };
	arg_t oargs[] = { stack_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(stack_x), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in1[] = { 1, 2, 3, 4 };
	complex float in2[] = { 5, 6, 7, 8 };
	complex float out[md_calc_size(N, dims) * 2];
	complex float ref_out[] = { 1, 2, 3, 4, 5, 6, 7, 8 };

	nlop_generic_apply_unchecked(nlop, 3, (void* [3]) { out, in1, in2 });

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_stack);

static bool test_snlop_mul_simple(void)
{
	long dims[] = { 2, 2, 1, 1 };
	int N = ARRAY_SIZE(dims);

	arg_t x1 = snlop_input(N, dims, "x1");
	arg_t x2 = snlop_input(N, dims, "x2");
	arg_t mul_x = snlop_mul_simple(x1, x2);

	arg_t iargs[] = { x1, x2 };
	arg_t oargs[] = { mul_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(mul_x), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in1[] = { 1, 2, 3, 4 };
	complex float in2[] = { 5, 6, 7, 8 };
	complex float out[md_calc_size(N, dims)];
	complex float ref_out[] = { 5, 12, 21, 32 };

	nlop_generic_apply_unchecked(nlop, 3, (void* [3]) { out, in1, in2 });

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_mul_simple);

static bool test_snlop_mul(void)
{
	long dims[] = { 2, 3, 1, 1 };
	int N = ARRAY_SIZE(dims);

	long odims[] = { 3, 1, 1, 1 };

	arg_t x1 = snlop_input(N, dims, "x1");
	arg_t x2 = snlop_input(N, dims, "x2");
	arg_t mul_x = snlop_mul(x1, x2, 1);

	arg_t iargs[] = { x1, x2 };
	arg_t oargs[] = { mul_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(mul_x), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in1[] = { 1, 2, 3, 4, 5, 6 };
	complex float in2[] = { 7, 8, 9, 10, 11, 12 };
	complex float out[md_calc_size(ARRAY_SIZE(odims), odims)];
	complex float ref_out[] = { 23, 67, 127 }; // [1*7 + 2*8, 3*9 + 4*10, 5*11 + 6*12]

	nlop_generic_apply_unchecked(nlop, 3, (void* [3]) { out, in1, in2 });

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, odims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_mul);

static bool test_snlop_div_simple(void)
{
	long dims[] = { 2, 2, 1, 1 };
	int N = ARRAY_SIZE(dims);

	arg_t x1 = snlop_input(N, dims, "x1");
	arg_t x2 = snlop_input(N, dims, "x2");
	arg_t div_x = snlop_div_simple(x1, x2);

	arg_t iargs[] = { x1, x2 };
	arg_t oargs[] = { div_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(div_x), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in1[] = { 6, 12, 20, 30 };
	complex float in2[] = { 2, 3, 4, 5 };
	complex float out[md_calc_size(N, dims)];
	complex float ref_out[] = { 3, 4, 5, 6 };

	nlop_generic_apply_unchecked(nlop, 3, (void* [3]) { out, in1, in2 });

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_div_simple);

static bool test_snlop_div(void)
{
	long dims[] = { 2, 3, 1, 1 };
	int N = ARRAY_SIZE(dims);

	long odims[] = { 3, 1, 1, 1 };

	arg_t x1 = snlop_input(N, dims, "x1");
	arg_t x2 = snlop_input(N, dims, "x2");
	arg_t div = snlop_div(x1, x2, 1);

	arg_t iargs[] = { x1, x2 };
	arg_t oargs[] = { div };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(div), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in1[] = { 6, 12, 20, 30, 42, 48 };
	complex float in2[] = { 2, 3, 4, 5 , 7, 8 };
	complex float out[md_calc_size(ARRAY_SIZE(odims), odims)];
	complex float ref_out[] = { 7, 11, 12 }; // [6/2 + 12/3, 20/4 + 30/ 5, 42/7 + 48/8]

	nlop_generic_apply_unchecked(nlop, 3, (void* [3]) { out, in1, in2 });

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, odims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_div);

static bool test_snlop_add(void)
{
	long dims[] = { 2, 4, 1, 1 };
	int N = ARRAY_SIZE(dims);

	arg_t x1 = snlop_input(N, dims, "x1");
	arg_t x2 = snlop_input(N, dims, "x2");
	arg_t add_x = snlop_add(x1, x2);

	arg_t iargs[] = { x1, x2 };
	arg_t oargs[] = { add_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(add_x), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in1[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
	complex float in2[] = { 10, 20, 30, 40, 50, 60, 70, 80 };
	complex float out[md_calc_size(N, dims)];
	complex float ref_out[] = { 11, 22, 33, 44, 55, 66, 77, 88 };

	nlop_generic_apply_unchecked(nlop, 3, (void* [3]) { out, in1, in2 });

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_add);

static bool test_snlop_sub(void)
{
	long dims[] = { 2, 4, 1, 1 };
	int N = ARRAY_SIZE(dims);

	arg_t x1 = snlop_input(N, dims, "x1");
	arg_t x2 = snlop_input(N, dims, "x2");
	arg_t sub_x = snlop_sub(x1, x2);

	arg_t iargs[] = { x1, x2 };
	arg_t oargs[] = { sub_x };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(sub_x), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in1[] = { 10, 20, 30, 40, 50, 60, 70, 80 };
	complex float in2[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
	complex float out[md_calc_size(N, dims)];
	complex float ref_out[] = { 9, 18, 27, 36, 45, 54, 63, 72 };

	nlop_generic_apply_unchecked(nlop, 3, (void* [3]) { out, in1, in2 });

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_sub);

static bool test_snlop_axpbz(void)
{
	long dims[] = { 2, 3, 1, 1 };
	int N = ARRAY_SIZE(dims);

	arg_t a = snlop_input(N, dims, "a");
	arg_t b = snlop_input(N, dims, "b");

	complex float sa = 2.0;
	complex float sb = 3.0;
	arg_t axpbz = snlop_axpbz(a, b, sa, sb);

	arg_t iargs[] = { a, b };
	arg_t oargs[] = { axpbz };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(axpbz), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in1[] = { 1, 2, 3, 4, 5, 6 };
	complex float in2[] = { 10, 20, 30, 40, 50, 60 };
	complex float out[6];

	// Expected: sa*a + sb*b = 2*in1 + 3*in2
	complex float ref_out[] = {
	    2 * 1 + 3 * 10,
	    2 * 2 + 3 * 20,
	    2 * 3 + 3 * 30,
	    2 * 4 + 3 * 40,
	    2 * 5 + 3 * 50,
	    2 * 6 + 3 * 60
	};

	nlop_generic_apply_unchecked(nlop, 3, (void* [3]) { out, in1, in2 });

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_axpbz);

static bool test_snlop_add_scalar(void)
{
	long dims[] = { 4, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	// 1 + x
	arg_t x = snlop_input(N, dims, "x");
	arg_t as = snlop_scale(x, 0);
	arg_t a1 = snlop_scalar(1);
	arg_t a = snlop_add(as, a1);
	arg_t res = snlop_add(x, a);

	arg_t iargs[] = { x };
	arg_t oargs[] = { res, a };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(res), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in[] = { 1, 2, 3, 4 };
	complex float out1[md_calc_size(N, dims)];
	complex float out2[md_calc_size(N, dims)];
	complex float ref_out1[] = { 1, 1, 1, 1 };
	complex float ref_out2[] = { 2, 3, 4, 5 };

	nlop_generic_apply_unchecked(nlop, 3, (void* [3]) { out2, out1, in });

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out1, out1));
	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out2, out1));
}

UT_REGISTER_TEST(test_snlop_add_scalar);

static bool test_snlop_sub_scalar(void)
{
	long dims[] = { 4, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	// x - 1
	arg_t x = snlop_input(N, dims, "x");
	arg_t as = snlop_scale(x, 0);
	arg_t a1 = snlop_scalar(1);
	arg_t a = snlop_add(as, a1);
	arg_t res = snlop_sub(x, a);

	arg_t iargs[] = { x };
	arg_t oargs[] = { res, a };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(res), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in[] = { 1, 2, 3, 4 };
	complex float out1[md_calc_size(N, dims)];
	complex float out2[md_calc_size(N, dims)];
	complex float ref_out1[] = { 1, 1, 1, 1 };
	complex float ref_out2[] = { 0, 1, 2, 3 };

	nlop_generic_apply_unchecked(nlop, 3, (void* [3]) { out2, out1, in });

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out1, out1));
	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out2, out1));
}

UT_REGISTER_TEST(test_snlop_sub_scalar);

static bool test_snlop_fix_input(void)
{
	long dims[] = { 4, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	arg_t x = snlop_input(N, dims, "x");
	arg_t zeros = snlop_scale(x, 0);
	arg_t fixed = snlop_scalar(5);
	arg_t res = snlop_add(zeros, fixed);

	arg_t iargs[] = { x };
	arg_t oargs[] = { res };

	const struct nlop_s* nlop = nlop_from_snlop_F(snlop_from_arg(res), ARRAY_SIZE(oargs), oargs, ARRAY_SIZE(iargs), iargs);

	complex float in[] = { 1, 2, 3, 4 };
	complex float out[md_calc_size(N, dims)];
	complex float ref_out[] = { 5, 5, 5, 5 };

	nlop_apply(nlop, N, dims, out, N, dims, in);

	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, dims, ref_out, out));
}

UT_REGISTER_TEST(test_snlop_fix_input);