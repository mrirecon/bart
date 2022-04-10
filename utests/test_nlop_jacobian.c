/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2022. Institute of Biomedical Imaging. Graz University of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/iovec.h"
#include "num/ops.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"

#include "linops/linop.h"
#include "linops/lintest.h"
#include "linops/someops.h"

#include "nlops/zexp.h"
#include "nlops/tenmul.h"
#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/nltest.h"
#include "nlops/stack.h"
#include "nlops/const.h"
#include "nlops/nlop_jacobian.h"

#include "utest.h"


static bool test_nlop_zprecomp_jacobian(void)
{
	unsigned long oflag  = MD_BIT(0) | MD_BIT(1) | MD_BIT(2);
	unsigned long iflag1 = MD_BIT(0) | MD_BIT(1) | MD_BIT(3);
	unsigned long iflag2 = MD_BIT(0) | MD_BIT(2) | MD_BIT(3);

	enum { N = 5 };
	long dims[N] = { 3, 4, 5, 6, 7 };

	long odims[N];
	long idims1[N];
	long idims2[N];

	md_select_dims(N, oflag, odims, dims);
	md_select_dims(N, iflag1, idims1, dims);
	md_select_dims(N, iflag2, idims2, dims);

	auto nlop1 = nlop_tenmul_create(N, odims, idims1, idims2);
	auto nlop2 = nlop_zprecomp_jacobian_F(nlop_tenmul_create(N, odims, idims1, idims2));

	bool ok = compare_nlops(nlop1, nlop2, true, true, true, UT_TOL);

	nlop_free(nlop1);
	nlop_free(nlop2);

	UT_ASSERT(ok);
}

UT_REGISTER_TEST(test_nlop_zprecomp_jacobian);


static bool test_nlop_zprecomp_jacobian2(void)
{
	unsigned long oflag  = MD_BIT(0) | MD_BIT(1) | MD_BIT(2);
	unsigned long iflag1 = MD_BIT(0) | MD_BIT(1) | MD_BIT(3);
	unsigned long iflag2 = MD_BIT(0) | MD_BIT(2) | MD_BIT(3);

	enum { N = 5 };
	long dims[N] = { 3, 4, 5, 6, 7 };

	long odims[N];
	long idims1[N];
	long idims2[N];

	md_select_dims(N, oflag, odims, dims);
	md_select_dims(N, iflag1, idims1, dims);
	md_select_dims(N, iflag2, idims2, dims);

	auto nlop1 = nlop_chain2_FF(nlop_tenmul_create(N, odims, idims1, idims2), 0, nlop_from_linop_F(linop_identity_create(N, odims)), 0);
	auto nlop2 = nlop_zprecomp_jacobian_F(nlop_chain2_FF(nlop_tenmul_create(N, odims, idims1, idims2), 0, nlop_from_linop_F(linop_identity_create(N, odims)), 0));

	bool ok = compare_nlops(nlop1, nlop2, true, true, true, UT_TOL);

	nlop_free(nlop1);
	nlop_free(nlop2);

	UT_ASSERT(ok);
}

UT_REGISTER_TEST(test_nlop_zprecomp_jacobian2);


static bool test_nlop_zprecomp_jacobian3(void)
{
	unsigned long oflag  = MD_BIT(0) | MD_BIT(1) | MD_BIT(2);
	unsigned long iflag1 = MD_BIT(0) | MD_BIT(1) | MD_BIT(3);
	unsigned long iflag2 = MD_BIT(0) | MD_BIT(2) | MD_BIT(3);

	enum { N = 5 };
	long dims[N] = { 3, 4, 5, 6, 7 };

	long odims[N];
	long idims1[N];
	long idims2[N];

	md_select_dims(N, oflag, odims, dims);
	md_select_dims(N, iflag1, idims1, dims);
	md_select_dims(N, iflag2, idims2, dims);

	auto nlop1 = nlop_chain2_FF(nlop_tenmul_create(N, odims, idims1, idims2), 0, nlop_from_linop_F(linop_zconj_create(N, odims)), 0);
	auto nlop2 = nlop_zprecomp_jacobian_F(nlop_chain2_FF(nlop_tenmul_create(N, odims, idims1, idims2), 0, nlop_from_linop_F(linop_zconj_create(N, odims)), 0));

	bool ok = compare_nlops(nlop1, nlop2, true, true, true, UT_TOL);

	nlop_free(nlop1);
	nlop_free(nlop2);

	//real and imaginary part must be considered independently
	UT_ASSERT(ok);
}

UT_UNUSED_TEST(test_nlop_zprecomp_jacobian3);
// U T_REGISTER_TEST(test_nlop_zprecomp_jacobian3);


static bool test_nlop_zrprecomp_jacobian(void)
{
	unsigned long oflag  = MD_BIT(0) | MD_BIT(1) | MD_BIT(2);
	unsigned long iflag1 = MD_BIT(0) | MD_BIT(1) | MD_BIT(3);
	unsigned long iflag2 = MD_BIT(0) | MD_BIT(2) | MD_BIT(3);

	enum { N = 5 };
	long dims[N] = { 3, 4, 5, 6, 7 };

	long odims[N];
	long idims1[N];
	long idims2[N];

	md_select_dims(N, oflag, odims, dims);
	md_select_dims(N, iflag1, idims1, dims);
	md_select_dims(N, iflag2, idims2, dims);

	auto nlop1 = nlop_tenmul_create(N, odims, idims1, idims2);
	auto nlop2 = nlop_zrprecomp_jacobian_F(nlop_tenmul_create(N, odims, idims1, idims2));

	bool ok = compare_nlops(nlop1, nlop2, true, true, true, UT_TOL);

	nlop_free(nlop1);
	nlop_free(nlop2);

	UT_ASSERT(ok);
}

UT_REGISTER_TEST(test_nlop_zrprecomp_jacobian);


static bool test_nlop_zrprecomp_jacobian2(void)
{
	unsigned long oflag  = MD_BIT(0) | MD_BIT(1) | MD_BIT(2);
	unsigned long iflag1 = MD_BIT(0) | MD_BIT(1) | MD_BIT(3);
	unsigned long iflag2 = MD_BIT(0) | MD_BIT(2) | MD_BIT(3);

	enum { N = 5 };
	long dims[N] = { 3, 4, 5, 6, 7 };

	long odims[N];
	long idims1[N];
	long idims2[N];

	md_select_dims(N, oflag, odims, dims);
	md_select_dims(N, iflag1, idims1, dims);
	md_select_dims(N, iflag2, idims2, dims);

	auto nlop1 = nlop_chain2_FF(nlop_tenmul_create(N, odims, idims1, idims2), 0, nlop_from_linop_F(linop_identity_create(N, odims)), 0);
	auto nlop2 = nlop_zrprecomp_jacobian_F(nlop_chain2_FF(nlop_tenmul_create(N, odims, idims1, idims2), 0, nlop_from_linop_F(linop_identity_create(N, odims)), 0));

	bool ok = compare_nlops(nlop1, nlop2, true, true, true, UT_TOL);

	nlop_free(nlop1);
	nlop_free(nlop2);

	UT_ASSERT(ok);
}

UT_REGISTER_TEST(test_nlop_zrprecomp_jacobian2);


static bool test_nlop_zrprecomp_jacobian3(void)
{
	unsigned long oflag  = MD_BIT(0) | MD_BIT(1) | MD_BIT(2);
	unsigned long iflag1 = MD_BIT(0) | MD_BIT(1) | MD_BIT(3);
	unsigned long iflag2 = MD_BIT(0) | MD_BIT(2) | MD_BIT(3);

	enum { N = 5 };
	long dims[N] = { 3, 4, 5, 6, 7 };

	long odims[N];
	long idims1[N];
	long idims2[N];

	md_select_dims(N, oflag, odims, dims);
	md_select_dims(N, iflag1, idims1, dims);
	md_select_dims(N, iflag2, idims2, dims);

	auto nlop1 = nlop_chain2_FF(nlop_tenmul_create(N, odims, idims1, idims2), 0, nlop_from_linop_F(linop_zconj_create(N, odims)), 0);
	auto nlop2 = nlop_zrprecomp_jacobian_F(nlop_chain2_FF(nlop_tenmul_create(N, odims, idims1, idims2), 0, nlop_from_linop_F(linop_zconj_create(N, odims)), 0));

	bool ok = compare_nlops(nlop1, nlop2, true, true, true, UT_TOL);

	nlop_free(nlop1);
	nlop_free(nlop2);

	UT_ASSERT(ok);
}

UT_REGISTER_TEST(test_nlop_zrprecomp_jacobian3);

