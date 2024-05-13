/* Copyright 2024. Institute of Biomedical Imaging. Graz University of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2024 Christian Holme
 */

#include <complex.h>
#include <math.h>
#include <limits.h>

#include "misc/debug.h"
#include "misc/bench.h"
#include "misc/nested.h"


#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/rand.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "utest.h"

// #define DO_SPEEDTEST

#ifndef DO_SPEEDTEST
enum { rounds = 1 };
enum { N = 5 };
long dims[N] = { 10, 7, 3, 16,128 };
#else
enum { rounds = 5 };
#if 1
// 2 GiB
enum { N = 5 };
long dims[N] = { 128,64,64,8,64};
#else
// 64 GiB
enum { N = 6 };
long dims[N] = { 1024,64,64,8,16,16};
#endif
#endif

typedef void (*md_rand_t)(int D, const long dims[D], complex float* dst);

static bool test_threads_rand(md_rand_t function, const char* name)
{
	complex float* st = md_calloc(N, dims, CFL_SIZE);
	complex float* mt = md_calloc(N, dims, CFL_SIZE);
	complex float* mt2 = md_calloc(N, dims, CFL_SIZE);

	double gibi = (double) md_calc_size(N, dims) * CHAR_BIT * CFL_SIZE / (1ULL << 30) / 8;

	bool sync_gpu = false; // not need, as it is CPU code
	bool print_bench = false;
#ifdef DO_SPEEDTEST
	print_bench = true;
#endif

#ifdef _OPENMP
	int old_omp_dynamic = omp_get_dynamic();
	int old_omp_threads = omp_get_num_threads();

	omp_set_num_threads(1);
#endif
	NESTED(void, f_st, (void))
	{
		function(N, dims, st);
	};

	num_rand_init(0xDEADBEEF);
	if (print_bench)
		bart_printf("times (%s, %ld elements, ~%.2f GiB, %2d rounds):\tsingle thread: ", name, md_calc_size(N, dims), gibi, rounds);
	run_bench(rounds, print_bench, sync_gpu, f_st);



	int some_threads = 1;
#ifdef _OPENMP
	some_threads = 12;
	omp_set_num_threads(some_threads);
#endif
	NESTED(void, f_mt, (void))
	{
		function(N, dims, mt);
	};
	num_rand_init(0xDEADBEEF);
	if (print_bench)
		bart_printf("\t\t\t\t\t\t\t\t%5d threads: ", some_threads);
	run_bench(rounds, print_bench, sync_gpu, f_mt);

	int many_threads = 1;
#ifdef _OPENMP
	many_threads = 128;
	omp_set_num_threads(many_threads);
#endif

	NESTED(void, f_mt2, (void))
	{
		function(N, dims, mt2);
	};
	num_rand_init(0xDEADBEEF);
	if (print_bench)
		bart_printf("\t\t\t\t\t\t\t\t%5d threads: ", many_threads);
	run_bench(rounds, print_bench, sync_gpu, f_mt2);

#ifdef _OPENMP
	omp_set_dynamic(old_omp_dynamic);
	omp_set_num_threads(old_omp_threads);
#else
	(void) some_threads;
	(void) many_threads;
#endif

	UT_RETURN_ON_FAILURE(md_compare(N, dims, st, mt, CFL_SIZE));
	UT_RETURN_ON_FAILURE(md_compare(N, dims, st, mt2, CFL_SIZE));


	md_free(st);
	md_free(mt);
	md_free(mt2);


	return true;

}

static bool test_uniform_rand_threads(void) { return test_threads_rand(md_uniform_rand, " uniform");}
UT_REGISTER_TEST(test_uniform_rand_threads);

static bool test_gaussian_rand_threads(void) { return test_threads_rand(md_gaussian_rand, "gaussian");}
UT_REGISTER_TEST(test_gaussian_rand_threads);


static int dcomp(const void* _a, const void* _b)
{
	const double* a = _a;
	const double* b = _b;
	return copysign(1., (*a - *b));
}


enum distribution {uniform, gaussian};


static double gaussian_cdf(double x)
{
	return 0.5 * (1. + erf(x * sqrt(2)/2.));
}

static double uniform_cdf(double x)
{
	return x;
}

static bool kolmogorov_smirnov(long N, double* x, enum distribution dist, const char* testname)
{

	qsort(x, (size_t) N, DL_SIZE, dcomp);


	double ks_stat = -1;

	for (long i = 0; i < N; ++i) {


		double cdfi;

		if (gaussian == dist)
			cdfi = gaussian_cdf(x[i]);
		else if (uniform == dist)
			cdfi = uniform_cdf(x[i]);
		else
			error("Invalid distribution!\n");

		double ks_dist0 = fabs((double) i / N - cdfi);
		double ks_dist1 = fabs((double) (i + 1) / N - cdfi);

		ks_stat = MAX(ks_stat, MAX(ks_dist0, ks_dist1));
	}

	// For large N, the critical value of the Kolmogorov-Smirnov distribution for a p-value
	// of 0.05 is 1.36/sqrt(N).
	double ks_crit = 1.36 / sqrt(N);

	debug_printf(DP_DEBUG2, "%s:\tks_stat: %f, ks_crit: %f, percent: %3.2f\n", testname, ks_stat, ks_crit, ks_stat/ks_crit*100.);

	return ks_stat < ks_crit;


}


enum { ks_N = 50000};

static bool test_ks_uniform_integers()
{
	num_rand_init(0xDEADBEEF);

	long range = ks_N/10;

	double* x = md_alloc(1, MD_DIMS(ks_N), DL_SIZE);
	for (long i = 0; i < md_calc_size(1, MD_DIMS(ks_N)); i++) {

		x[i] = (double) rand_range(range) / (range - 1);
	}


	UT_RETURN_ON_FAILURE(kolmogorov_smirnov(ks_N, x, uniform, "ks_integers"));

	md_free(x);

	return true;
}


static bool test_ks_uniform()
{
	num_rand_init(0xDEADBEEF);

	complex float* _x = md_alloc(1, MD_DIMS(ks_N), CFL_SIZE);
	md_uniform_rand(1, MD_DIMS(ks_N), _x);
	double* x = md_alloc(1, MD_DIMS(ks_N), DL_SIZE);
	for (int i = 0; i < ks_N; ++i)
		x[i] = crealf(_x[i]);
	md_free(_x);


	UT_RETURN_ON_FAILURE(kolmogorov_smirnov(ks_N, x, uniform, "ks_uniform"));

	md_free(x);

	return true;
}


static bool test_ks_gaussian()
{
	num_rand_init(0xDEADBEEF);

	complex float* _x = md_alloc(1, MD_DIMS(ks_N), CFL_SIZE);
	md_gaussian_rand(1, MD_DIMS(ks_N), _x);
	double* x = md_alloc(1, MD_DIMS(ks_N), DL_SIZE);
	for (int i = 0; i < ks_N; ++i)
		x[i] = crealf(_x[i]);
	md_free(_x);


	UT_RETURN_ON_FAILURE(kolmogorov_smirnov(ks_N, x, gaussian, "ks_gaussian"));

	md_free(x);

	return true;
}


UT_REGISTER_TEST(test_ks_uniform_integers);
UT_REGISTER_TEST(test_ks_uniform);
UT_REGISTER_TEST(test_ks_gaussian);


// basic sanity test
static bool test_rand_range()
{

	unsigned int range = 4;

	for (int i = 0; i < 1e6; ++i) {

		unsigned int r = rand_range(range);
		if (r >= range)
			return false;
	}

	return true;

}


static bool test_var(void)
{
	enum { N = 1};
	const long dims[N] = { 1000 };

	complex float* data = md_alloc(N, dims, CFL_SIZE);
	complex float var;

	md_gaussian_rand(N, dims, data);	
	md_zvar(N, dims, ~0UL, &var, data);

	md_free(data);

	UT_RETURN_ASSERT(cabsf(var - 2) < 0.2);
}


UT_REGISTER_TEST(test_rand_range);
UT_REGISTER_TEST(test_var);
