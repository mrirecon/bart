/* Copyright 2022. Uecker Lab, University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2022 Moritz Blumenthal
 */

#include <complex.h>
#include <assert.h>

#include "num/flpmath.h"
#include "num/rand.h"
#include "num/multind.h"
#include "num/init.h"
#include "num/shuffle.h"

#include "utest.h"

#ifndef CFL_SIZE
#define CFL_SIZE	sizeof(complex float)
#endif

struct ptr_cpugpu {

	complex float* cpu;
	complex float* gpu;
};

static struct ptr_cpugpu alloc_pair_rand(int N, const long dims[N])
{
	struct ptr_cpugpu ret = {

		.cpu = md_alloc(N, dims, CFL_SIZE),
		.gpu = md_alloc_gpu(N, dims, CFL_SIZE)
	};

	md_gaussian_rand(N, dims, ret.cpu);
	md_copy(N, dims, ret.gpu, ret.cpu, CFL_SIZE);

	return ret;
}

static struct ptr_cpugpu alloc_pair_zero(int N, const long dims[N])
{
	struct ptr_cpugpu ret = {

		.cpu = md_alloc(N, dims, CFL_SIZE),
		.gpu = md_alloc_gpu(N, dims, CFL_SIZE)
	};

	md_clear(N, dims, ret.cpu, CFL_SIZE);
	md_clear(N, dims, ret.gpu, CFL_SIZE);

	return ret;
}

static float cmp_pair_F(int N, const long dims[N], struct ptr_cpugpu x)
{
	complex float* tmp = md_alloc(N, dims, CFL_SIZE);
	md_copy(N, dims, tmp, x.gpu, CFL_SIZE);

	float result = md_znrmse(N, dims, x.cpu, tmp);

	md_free(tmp);
	md_free(x.cpu);
	md_free(x.gpu);

	return result;
}

static void free_pair(struct ptr_cpugpu x)
{
	md_free(x.cpu);
	md_free(x.gpu);
}



static bool test_cuda_decompose1(void)
{
	num_init_gpu();
	enum { N = 5 };

	const long dims[N] = { 4, 1, 9, 2, 2 };
	const long factors[N] = { 2, 1, 3, 1, 1 };
	const long odims[N + 1] = { 2, 1, 3, 2, 2, 6};

	struct ptr_cpugpu in = alloc_pair_rand(N, dims);
	struct ptr_cpugpu out = alloc_pair_zero(N + 1, odims);

	md_decompose(N, factors, odims, out.cpu, dims, in.cpu, CFL_SIZE);
	md_decompose(N, factors, odims, out.gpu, dims, in.gpu, CFL_SIZE);

	free_pair(in);

	UT_ASSERT(0. == cmp_pair_F(N + 1, odims, out));
}

static bool test_cuda_decompose2(void)
{
	num_init_gpu();
	enum { N = 5 };

	const long dims[N] = { 4, 1, 9, 2, 2 };
	const long factors[N] = { 2, 1, 3, 2, 1 };
	const long odims[N + 1] = { 2, 1, 3, 1, 2, 12};

	struct ptr_cpugpu in = alloc_pair_rand(N, dims);
	struct ptr_cpugpu out = alloc_pair_zero(N + 1, odims);

	md_decompose(N, factors, odims, out.cpu, dims, in.cpu, CFL_SIZE);
	md_decompose(N, factors, odims, out.gpu, dims, in.gpu, CFL_SIZE);

	free_pair(in);

	UT_ASSERT(0. == cmp_pair_F(N + 1, odims, out));
}

static bool test_cuda_recompose1(void)
{
	num_init_gpu();
	enum { N = 5 };

	const long dims[N] = { 4, 1, 9, 2, 2 };
	const long factors[N] = { 2, 1, 3, 1, 1 };
	const long odims[N + 1] = { 2, 1, 3, 2, 2, 6};

	struct ptr_cpugpu in = alloc_pair_rand(N + 1, odims);
	struct ptr_cpugpu out = alloc_pair_zero(N, dims);

	md_recompose(N, factors, dims, out.cpu, odims, in.cpu, CFL_SIZE);
	md_recompose(N, factors, dims, out.gpu, odims, in.gpu, CFL_SIZE);

	free_pair(in);

	UT_ASSERT(0. == cmp_pair_F(N + 1, odims, out));
}

static bool test_cuda_recompose2(void)
{
	num_init_gpu();
	enum { N = 5 };

	const long dims[N] = { 4, 1, 9, 2, 2 };
	const long factors[N] = { 2, 1, 3, 2, 1 };
	const long odims[N + 1] = { 2, 1, 3, 1, 2, 12};

	struct ptr_cpugpu in = alloc_pair_rand(N + 1, odims);
	struct ptr_cpugpu out = alloc_pair_zero(N, dims);

	md_recompose(N, factors, dims, out.cpu, odims, in.cpu, CFL_SIZE);
	md_recompose(N, factors, dims, out.gpu, odims, in.gpu, CFL_SIZE);

	free_pair(in);

	UT_ASSERT(0. == cmp_pair_F(N + 1, odims, out));
}

UT_GPU_REGISTER_TEST(test_cuda_decompose1);
UT_GPU_REGISTER_TEST(test_cuda_decompose2);
UT_GPU_REGISTER_TEST(test_cuda_recompose1);
UT_GPU_REGISTER_TEST(test_cuda_recompose2);
