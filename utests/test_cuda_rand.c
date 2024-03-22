/* Copyright 2024. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2024 Christian Holme <holme@tugraz.at>
 */

#include <complex.h>
#include <assert.h>
#include <limits.h>

#include "num/rand.h"

#include "num/gpuops.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/debug.h"
#include "misc/misc.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "utest.h"

// #define DO_SPEEDTEST

enum { N = 5 };
long dims[N] = { 10, 7, 3, 16,128 };


static bool test_cuda_uniform_rand(void)
{
#ifndef USE_CUDA
	return true;
#else
	// TODO: detect if GPU works


	complex float* cpu = md_alloc(N, dims, CFL_SIZE);

	complex float* gpu = md_alloc_gpu(N, dims, CFL_SIZE);
	complex float* gpu_result = md_alloc(N, dims, CFL_SIZE);


	num_rand_init(0xDEADBEEF);
	md_uniform_rand(N, dims, cpu);

	num_rand_init(0xDEADBEEF);
	md_uniform_rand(N, dims, gpu);

	md_copy(N, dims, gpu_result, gpu, CFL_SIZE);

	float err = md_znrmse(N, dims, cpu, gpu_result);
	if (0 != err)
		debug_printf(DP_ERROR, "%e\n", err);
	UT_RETURN_ON_FAILURE(0 == err);


	md_free(cpu);
	md_free(gpu);
	md_free(gpu_result);

	return true;
#endif
}
UT_GPU_REGISTER_TEST(test_cuda_uniform_rand);

static bool test_cuda_gaussian_rand(void)
{
#ifndef USE_CUDA
	return true;
#else
	// TODO: detect if GPU works


	complex float* cpu = md_alloc(N, dims, CFL_SIZE);

	complex float* gpu = md_alloc_gpu(N, dims, CFL_SIZE);
	complex float* gpu_result = md_alloc(N, dims, CFL_SIZE);



	num_rand_init(0xDEADBEEF);
	md_gaussian_rand(N, dims, cpu);

	num_rand_init(0xDEADBEEF);
	md_gaussian_rand(N, dims, gpu);

	md_copy(N, dims, gpu_result, gpu, CFL_SIZE);

	UT_RETURN_ON_FAILURE(0 == md_znrmse(N, dims, cpu, gpu_result));


	md_free(cpu);
	md_free(gpu);
	md_free(gpu_result);

	return true;
#endif
}
UT_GPU_REGISTER_TEST(test_cuda_gaussian_rand);


static bool test_cuda_rand_one(void)
{
#ifndef USE_CUDA
	return true;
#else
	// TODO: detect if GPU works

	complex float* cpu = md_alloc(N, dims, CFL_SIZE);

	complex float* gpu = md_alloc_gpu(N, dims, CFL_SIZE);
	complex float* gpu_result = md_alloc(N, dims, CFL_SIZE);

	double p = 0.123;


	num_rand_init(0xDEADBEEF);
	md_rand_one(N, dims, cpu, p);

	num_rand_init(0xDEADBEEF);
	md_rand_one(N, dims, gpu, p);

	md_copy(N, dims, gpu_result, gpu, CFL_SIZE);

	UT_RETURN_ON_FAILURE(0 == md_znrmse(N, dims, cpu, gpu_result));


	md_free(cpu);
	md_free(gpu);
	md_free(gpu_result);

	return true;
	#endif
}
UT_GPU_REGISTER_TEST(test_cuda_rand_one);




#ifndef DO_SPEEDTEST
enum { rounds = 1 };
enum { N2 = 5 };
long dims2[N2] = { 10, 7, 3, 16,128 };
#else
enum { rounds = 5 };
#if 1
// 2 GiB
enum { N2 = 5 };
long dims2[N2] = { 128,64,64,8,64};
#else
// 64 GiB
enum { N2 = 6 };
long dims2[N2] = { 1024,64,64,8,16,16};
#endif
#endif

typedef void (*md_rand_t)(int D, const long dims[D], complex float* dst);

static bool test_cuda_rand(md_rand_t function, const char* name, double tol)
{
	complex float* mt = md_calloc(N2, dims2, CFL_SIZE);
	complex float* mt_gpu = md_alloc_gpu(N2, dims2, CFL_SIZE);
	complex float* mt_gpu_cpu = md_calloc(N2, dims2, CFL_SIZE);

#ifdef _OPENMP
	int old_omp_dynamic = omp_get_dynamic();
	int old_omp_threads = omp_get_num_threads();

	omp_set_num_threads(1);
#endif


	int some_threads = 1;
#ifdef _OPENMP
	some_threads = 12;
	omp_set_num_threads(some_threads);
#endif
	num_rand_init(0xDEADBEEF);
	double start = timestamp();
	for (int i = 0; i < rounds; ++i)
		function(N2, dims2, mt);
	double mtt = timestamp() - start;

	num_rand_init(0xDEADBEEF);
	cuda_sync_stream();
	start = timestamp();
	for (int i = 0; i < rounds; ++i)
		function(N2, dims2, mt_gpu);

	cuda_sync_stream();
	double gput = timestamp() - start;
#ifdef _OPENMP
	omp_set_dynamic(old_omp_dynamic);
	omp_set_num_threads(old_omp_threads);
#else
	(void) some_threads;
#endif

#ifdef DO_SPEEDTEST
	double gibi = (double) md_calc_size(N2, dims2) * CHAR_BIT * CFL_SIZE / (1ULL << 30) / 8;
	debug_printf(DP_INFO, "times (%s, %ld elements, ~%.2f GiB, %d rounds): %d threads: %f, GPU: %f\n", name, md_calc_size(N2, dims2), gibi, rounds, some_threads, mtt/rounds, gput/rounds);
#else
	(void) mtt;
	(void) gput;
	(void) name;
#endif

	md_copy(N2, dims2, mt_gpu_cpu, mt_gpu, CFL_SIZE);
	double err = md_znrmse(N2, dims2, mt, mt_gpu_cpu);

	md_free(mt);
	md_free(mt_gpu);
	md_free(mt_gpu_cpu);


	debug_printf(DP_DEBUG1, "test_cuda: %s, error: %.8e, tol %.1e\n", name, err, tol);
	UT_RETURN_ASSERT(err <= tol);
}

static bool test_mt_cuda_uniform_rand(void) { return test_cuda_rand(md_uniform_rand, " uniform", 0.0);}
UT_GPU_REGISTER_TEST(test_mt_cuda_uniform_rand);

// since the floating point functions we use for generating Gaussian-distributed random numbers do not always give
// bit-identical results on the CPU compared to the CPU version, we allow a small error here
static bool test_mt_cuda_gaussian_rand(void) { return test_cuda_rand(md_gaussian_rand, "gaussian", 1e-11);}
UT_GPU_REGISTER_TEST(test_mt_cuda_gaussian_rand);
