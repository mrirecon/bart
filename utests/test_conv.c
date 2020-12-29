/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <math.h>
#include <complex.h>

#include "num/conv.h"

#include "utest.h"



static bool test_conv_generic(enum conv_mode mode, enum conv_type type, int N, const complex float G[N])
{
	const complex float K[3] = { 0.5, 1., -1. };
	const complex float T[8] = { 0.5, 0., 0., 1., 0., 0., 0., 1.i };
	complex float O[N];

	conv(1, 1u, type, mode,
		(long[]){ N }, O, (long[]){ 8 }, T, (long[]){ 3 }, K);

	bool ok = true;

	for (int i = 0; i < N; i++)
		ok &= (1.E-4 > cabsf(O[i] - G[i]));

	return ok;
}


static bool test_conv_sy_tr(void)
{
	return test_conv_generic(CONV_SYMMETRIC, CONV_TRUNCATED,
		8, (const complex float[8]){ 0.5, -0.5, 0.5, 1., -1., 0., 0.5i, 1.i });
}

static bool test_conv_sy_cy(void)
{
	return test_conv_generic(CONV_SYMMETRIC, CONV_CYCLIC,
		8, (const complex float[8]){ 0.5 - 1.i, -0.5, 0.5, 1., -1., 0., 0.5i, 0.25 + 1.i });
}

static bool test_conv_ca_ex(void)
{
	return test_conv_generic(CONV_CAUSAL, CONV_EXTENDED,
		10, (const complex float[10]){ 0.25, 0.5, -0.5, 0.5, 1., -1., 0., 0.5i, 1.i, -1.i });
}

static bool test_conv_ca_vd(void)
{
	return test_conv_generic(CONV_CAUSAL, CONV_VALID,
		6, (const complex float[6]){ -0.5, 0.5, 1., -1., 0., 0.5i });
}

static bool test_conv_ca_tr(void)
{
	return test_conv_generic(CONV_CAUSAL, CONV_TRUNCATED,
		8, (const complex float[8]){ 0.25, 0.5, -0.5, 0.5, 1., -1., 0., 0.5i });
}

static bool test_conv_ca_cy(void)
{
	return test_conv_generic(CONV_CAUSAL, CONV_CYCLIC,
		8, (const complex float[8]){ 0.25 + 1.i, 0.5 - 1.i, -0.5, 0.5, 1., -1., 0., 0.5i });
}


static bool test_conv_ac_tr(void)
{
	return test_conv_generic(CONV_ANTICAUSAL, CONV_TRUNCATED,
		8, (const complex float[8]){ -0.5, 0.5, 1., -1., 0., 0.5i, 1.i, -1.i });
}

static bool test_conv_ac_cy(void)
{
	return test_conv_generic(CONV_ANTICAUSAL, CONV_CYCLIC,
		8, (const complex float[8]){ -0.5, 0.5, 1., -1., 0., 0.5i, 0.25 + 1.i, 0.5 - 1.i });
}






UT_REGISTER_TEST(test_conv_ca_ex);
UT_REGISTER_TEST(test_conv_ca_vd);
UT_REGISTER_TEST(test_conv_ca_tr);
UT_REGISTER_TEST(test_conv_ca_cy);

UT_REGISTER_TEST(test_conv_sy_tr);
UT_REGISTER_TEST(test_conv_sy_cy);

UT_REGISTER_TEST(test_conv_ac_tr);
UT_REGISTER_TEST(test_conv_ac_cy);



static bool test_conv2_generic(enum conv_mode mode, enum conv_type type, int N, const complex float G[N])
{
	const complex float K[4] = { 0.5, 1., -1., 0. };
	const complex float T[8] = { 0.5, 0., 0., 1., 0., 0., 0., 1.i };
	complex float O[N];

	conv(1, 1u, type, mode,
		(long[]){ N }, O, (long[]){ 8 }, T, (long[]){ 4 }, K);

	bool ok = true;

	for (int i = 0; i < N; i++)
		ok &= (1.E-4 > cabsf(O[i] - G[i]));

	return ok;
}


static bool test_conv2_sy_tr(void)
{
	return test_conv2_generic(CONV_SYMMETRIC, CONV_TRUNCATED,
		8, (const complex float[8]){ 0.5, -0.5, 0.5, 1., -1., 0., 0.5i, 1.i });
}

static bool test_conv2_sy_cy(void)
{
	return test_conv2_generic(CONV_SYMMETRIC, CONV_CYCLIC,
		8, (const complex float[8]){ 0.5 - 1.i, -0.5, 0.5, 1., -1., 0., 0.5i, 0.25 + 1.i });
}

static bool test_conv2_ca_ex(void)
{
	return test_conv2_generic(CONV_CAUSAL, CONV_EXTENDED,
		11, (const complex float[11]){ 0.25, 0.5, -0.5, 0.5, 1., -1., 0., 0.5i, 1.i, -1.i, 0. });
}

static bool test_conv2_ca_vd(void)
{
	return test_conv2_generic(CONV_CAUSAL, CONV_VALID,
		5, (const complex float[5]){ 0.5, 1., -1., 0., 0.5i });
}

static bool test_conv2_ca_tr(void)
{
	return test_conv2_generic(CONV_CAUSAL, CONV_TRUNCATED,
		8, (const complex float[8]){ 0.25, 0.5, -0.5, 0.5, 1., -1., 0., 0.5i });
}

static bool test_conv2_ca_cy(void)
{
	return test_conv2_generic(CONV_CAUSAL, CONV_CYCLIC,
		8, (const complex float[8]){ 0.25 + 1.i, 0.5 - 1.i, -0.5, 0.5, 1., -1., 0., 0.5i });
}


static bool test_conv2_ac_tr(void)
{
	return test_conv2_generic(CONV_ANTICAUSAL, CONV_TRUNCATED,
		8, (const complex float[8]){ 0.5, 1., -1., 0., 0.5i, 1.i, -1.i, 0. });
}

static bool test_conv2_ac_cy(void)
{
	return test_conv2_generic(CONV_ANTICAUSAL, CONV_CYCLIC,
		8, (const complex float[8]){ 0.5, 1., -1., 0., 0.5i, 0.25 + 1.i, 0.5 - 1.i, -0.5 });
}






UT_REGISTER_TEST(test_conv2_ca_ex);
UT_REGISTER_TEST(test_conv2_ca_vd);
UT_REGISTER_TEST(test_conv2_ca_tr);
UT_REGISTER_TEST(test_conv2_ca_cy);

UT_REGISTER_TEST(test_conv2_sy_tr);
UT_REGISTER_TEST(test_conv2_sy_cy);

UT_REGISTER_TEST(test_conv2_ac_tr);
UT_REGISTER_TEST(test_conv2_ac_cy);


