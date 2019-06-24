/* Copyright 2017-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"

#include "linops/linop.h"
#include "linops/lintest.h"

#include "noncart/nufft.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "utest.h"


enum { N = 8 };
static const long ksp_dims[N] = { 1, 5, 1, 1, 1, 1, 1, 1 };
static const long cim_dims[N] = { 8, 8, 1, 1, 1, 1, 1, 1 };
static const long trj_dims[N] = { 3, 5, 1, 1, 1, 1, 1, 1 };

static const complex float traj[5][3] = {
	{ 0., 0. , 0. },
	{ 1., 0. , 0. },
	{ 0., 1. , 0. },
	{ -1., 0., 0. },
	{ 0., -1., 0. },
};


static struct linop_s* create_nufft(bool toeplitz, bool use_weights)
{
	const complex float weights[5] = {
		0.5, 0.5, 0.5, 0.5, 0.5
	};

	struct nufft_conf_s conf = nufft_conf_defaults;
	conf.toeplitz = toeplitz;

	return nufft_create(N, ksp_dims, cim_dims, trj_dims, &traj[0][0], use_weights ? weights : NULL, conf);
}


static const long ci2_dims[N] = { 8, 8, 1, 1, 1, 1, 2, 1 };
static const long ks2_dims[N] = { 1, 5, 1, 1, 1, 1, 2, 1 };
static const long tr2_dims[N] = { 3, 5, 1, 1, 1, 1, 1, 1 };
static const long bas_dims[N] = { 1, 1, 1, 1, 1, 3, 2, 1 };
static const long wg2_dims[N] = { 1, 5, 1, 1, 1, 3, 1, 1 };



static struct linop_s* create_nufft2(bool toeplitz)
{
	const complex float weights[15] = {
		0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5,
	};

	const complex float basis[6] = {
		1., 0., 1., 0., 1., 0.,
	};

	struct nufft_conf_s conf = nufft_conf_defaults;
	conf.toeplitz = toeplitz;

	return nufft_create2(N, ks2_dims, ci2_dims, tr2_dims, &traj[0][0], wg2_dims, weights, bas_dims, basis, conf);
}



static bool test_nufft_forward(void)
{

	const complex float src[] = { 
		1., 1., 1., 1., 1., 1., 1., 1.,
		1., 1., 1., 1., 1., 1., 1., 1.,
		1., 1., 1., 1., 1., 1., 1., 1.,
		1., 1., 1., 1., 1., 1., 1., 1.,
		1., 1., 1., 1., 1., 1., 1., 1.,
		1., 1., 1., 1., 1., 1., 1., 1.,
		1., 1., 1., 1., 1., 1., 1., 1.,
		1., 1., 1., 1., 1., 1., 1., 1.,
	};

	struct linop_s* op = create_nufft(false, false);

	complex float dst[5];

	linop_forward(op, N, ksp_dims, dst, N, cim_dims, src);

	linop_free(op);

	return (cabsf(dst[0] - 8.f) < 0.02);	// !
}



static bool test_nufft_adjoint(void)
{
	struct linop_s* op = create_nufft(false, false);

	float diff = linop_test_adjoint(op);

	debug_printf(DP_DEBUG1, "adjoint diff: %f\n", diff);

	bool ret = (diff < 1.E-6f);

	linop_free(op);

	return ret;
}


static bool test_nufft_normal(void)
{
	struct linop_s* op = create_nufft(false, false);

	float nrmse = linop_test_normal(op);

	debug_printf(DP_DEBUG1, "normal nrmse: %f\n", nrmse);

	bool ret = (nrmse < 1.E-7f);

	linop_free(op);

	return ret;
}


static bool test_nufft_toeplitz(bool use_weights)
{
	complex float src[64];
	complex float dst1[64] = { 0 };
	complex float dst2[64] = { 0 };

	md_gaussian_rand(N, cim_dims, src);

	struct linop_s* op1 = create_nufft(false, use_weights);
	linop_normal(op1, N, cim_dims, dst1, src);
	linop_free(op1);

	struct linop_s* op2 = create_nufft(true, use_weights);
	linop_normal(op2, N, cim_dims, dst2, src);
	linop_free(op2);

	return md_znrmse(N, cim_dims, dst1, dst2) < 0.01;
}


static bool test_nufft_toeplitz_noweights(void)
{
	return test_nufft_toeplitz(false);
}

static bool test_nufft_toeplitz_weights(void)
{
	return test_nufft_toeplitz(true);
}


static bool test_nufft_basis_adjoint(void)
{
	struct linop_s* op = create_nufft2(false);

	float diff = linop_test_adjoint(op);

	debug_printf(DP_DEBUG1, "adjoint diff: %f\n", diff);

	bool ret = (diff < 1.E-6f);

	linop_free(op);

	return ret;
}


static bool test_nufft_basis_normal(void)
{
	struct linop_s* op = create_nufft2(false);

	float nrmse = linop_test_normal(op);

	debug_printf(DP_DEBUG1, "normal nrmse: %f\n", nrmse);

	bool ret = (nrmse < 1.E-7f);

	linop_free(op);

	return ret;
}


static bool test_nufft_basis_toeplitz(void)
{
	complex float src[128];
	complex float dst1[128];
	complex float dst2[128];

	assert(128 == md_calc_size(N, ci2_dims));

	md_gaussian_rand(N, ci2_dims, src);

	struct linop_s* op1 = create_nufft2(false);
	linop_normal(op1, N, ci2_dims, dst1, src);
	linop_free(op1);

	struct linop_s* op2 = create_nufft2(true);
	linop_normal(op2, N, ci2_dims, dst2, src);
	linop_free(op2);

	complex float sc = md_zscalar(N, ci2_dims, dst2, dst1);
	float n = md_znorm(N, ci2_dims, dst1);

	md_zsmul(N, ci2_dims, dst1, dst1, sc / (n * n));

	return md_znrmse(N, ci2_dims, dst1, dst2) < 1.E-4;
}





UT_REGISTER_TEST(test_nufft_forward);
UT_REGISTER_TEST(test_nufft_adjoint);
UT_REGISTER_TEST(test_nufft_normal);
UT_REGISTER_TEST(test_nufft_toeplitz_weights);
UT_REGISTER_TEST(test_nufft_toeplitz_noweights);
UT_REGISTER_TEST(test_nufft_basis_adjoint);
UT_REGISTER_TEST(test_nufft_basis_normal);
UT_REGISTER_TEST(test_nufft_basis_toeplitz);


