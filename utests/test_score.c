/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <complex.h>

#include "nn/nn.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/iovec.h"

#include "misc/mmio.h"

#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/const.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/nltest.h"
#include "nlops/someops.h"
#include "nlops/gmm.h"

#include "nn/nn.h"
#include "nn/chain.h"

#include "networks/score.h"

#include "utest.h"

static const struct nlop_s* get_gaussian_score(int N, const long dims[N], const complex float var, const float mean)
{
	complex float* meana = md_alloc(N, dims, CFL_SIZE);
	md_zfill(N, dims, meana, mean);

	complex float wgh = 1.f;

	const struct nlop_s* ret = nlop_gmm_score_create(N, dims, dims, meana, MD_SINGLETON_DIMS(N), &var, MD_SINGLETON_DIMS(N), &wgh);

	md_free(meana);

	return ret;
}

static bool test_nlop_score_to_expect(void)
{
	// if the gaussian variance is much smaller than the additional noise,
	// the expectation should be the mean
	enum { N = 2 };
	long dims[N] = { 2, 3 };
	complex float mean = 10;

	const struct nlop_s* score_nlop = get_gaussian_score(2, dims, 0.1f, mean);
	const struct nlop_s* expect_nlop = nlop_score_to_expectation(score_nlop);

	complex float* in = md_alloc(N, dims, CFL_SIZE);
	md_gaussian_rand(N, dims, in);

	complex float var = 100.f;
	complex float* exp = md_alloc(N, dims, CFL_SIZE);

	void* args[3] = { exp, in, &var };
	nlop_generic_apply_unchecked(expect_nlop, 3, args);
	nlop_free(expect_nlop);

	md_zfill(N, dims, in, mean);

	float err = md_znrmse(N, dims, exp, in);
	md_free(in);
	md_free(exp);

	UT_RETURN_ASSERT_TOL(err, 1.e-5);
}

UT_REGISTER_TEST(test_nlop_score_to_expect);


static bool test_nlop_score_to_expect_reverse(void)
{
	enum { N = 2 };
	long dims[N] = { 2, 3 };

	const struct nlop_s* score_nlop1 = get_gaussian_score(2, dims, 0.1f, 0.3);
	const struct nlop_s* score_nlop2 = nlop_expectation_to_score(nlop_score_to_expectation(get_gaussian_score(2, dims, 0.1f, 0.3)));

	score_nlop1 = nlop_prepend_FF(nlop_zss_create(N, MD_SINGLETON_DIMS(N), 0), score_nlop1, 1);
	score_nlop2 = nlop_prepend_FF(nlop_zss_create(N, MD_SINGLETON_DIMS(N), 0), score_nlop2, 1);

	bool ret = compare_nlops(score_nlop1, score_nlop2, true, false, false, UT_TOL);
	nlop_free(score_nlop1);
	nlop_free(score_nlop2);

	UT_RETURN_ASSERT(ret);
}

UT_REGISTER_TEST(test_nlop_score_to_expect_reverse);


static const struct nn_s* get_gaussian_score_nn(int N, const long dims[N], const complex float var, const float mean)
{
	return nn_from_nlop_F(get_gaussian_score(N, dims, var, mean));
}

static bool test_nn_score_to_expect(void)
{
	// if the gaussian variance is much smaller than the additional noise,
	// the expectation should be the mean
	enum { N = 2 };
	long dims[N] = { 2, 3 };
	complex float mean = 10;

	const struct nn_s* score_nlop = get_gaussian_score_nn(2, dims, 0.1f, mean);
	const struct nn_s* expect_nlop = nn_score_to_expectation(score_nlop);

	complex float* in = md_alloc(N, dims, CFL_SIZE);
	md_gaussian_rand(N, dims, in);

	complex float var = 100.f;
	complex float* exp = md_alloc(N, dims, CFL_SIZE);

	void* args[3] = { exp, in, &var };
	nlop_generic_apply_unchecked(expect_nlop->nlop, 3, args);
	nn_free(expect_nlop);

	md_zfill(N, dims, in, mean);

	float err = md_znrmse(N, dims, exp, in);
	md_free(in);
	md_free(exp);

	UT_RETURN_ASSERT_TOL(err, 1.e-5);
}

UT_REGISTER_TEST(test_nn_score_to_expect);


static bool test_nn_score_to_expect_reverse(void)
{
	enum { N = 2 };
	long dims[N] = { 2, 3 };

	const struct nn_s* score_nlop1 = get_gaussian_score_nn(2, dims, 0.1f, 0.3);
	const struct nn_s* score_nlop2 = nn_expectation_to_score(nn_score_to_expectation(get_gaussian_score_nn(2, dims, 0.1f, 0.3)));

	score_nlop1 = nn_chain2_FF(nn_from_nlop_F(nlop_zss_create(N, MD_SINGLETON_DIMS(N), 0)), 0, NULL, score_nlop1, 1, NULL);
	score_nlop2 = nn_chain2_FF(nn_from_nlop_F(nlop_zss_create(N, MD_SINGLETON_DIMS(N), 0)), 0, NULL, score_nlop2, 1, NULL);

	bool ret = compare_nlops(score_nlop1->nlop, score_nlop2->nlop, true, false, false, UT_TOL);
	nn_free(score_nlop1);
	nn_free(score_nlop2);

	UT_RETURN_ASSERT(ret);
}

UT_REGISTER_TEST(test_nn_score_to_expect_reverse);


