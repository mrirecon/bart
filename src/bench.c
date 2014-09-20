/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors: 
 * 2014 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <string.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/init.h"
#include "num/ops.h"

#include "wavelet2/wavelet.h"
#ifdef BERKELEY_SVN
#include "wavelet3/wavthresh.h"
#endif

#include "misc/debug.h"
#include "misc/misc.h"

#define DIMS 8


const char* usage_str = "";
const char* help_str = "Run micro-benchmarks.\n";




static double bench_generic_copy(long dims[DIMS])
{
	long strs[DIMS];

	md_calc_strides(DIMS, strs, dims, CFL_SIZE);
	md_calc_strides(DIMS, strs, dims, CFL_SIZE);

	complex float* x = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dims, CFL_SIZE);

	md_gaussian_rand(DIMS, dims, x);

	double tic = timestamp();

	md_copy2(DIMS, dims, strs, y, strs, x, CFL_SIZE);

	double toc = timestamp();

	md_free(x);
	md_free(y);

	return toc - tic;
}

	
static double bench_generic_matrix_multiply(long dims[DIMS])
{
	long dimsX[DIMS];
	long dimsY[DIMS];
	long dimsZ[DIMS];

	md_select_dims(DIMS, 2 * 3 + 17, dimsX, dims);	// 1 110 1
	md_select_dims(DIMS, 2 * 6 + 17, dimsY, dims);	// 1 011 1
	md_select_dims(DIMS, 2 * 5 + 17, dimsZ, dims);	// 1 101 1

	long strsX[DIMS];
	long strsY[DIMS];
	long strsZ[DIMS];

	md_calc_strides(DIMS, strsX, dimsX, CFL_SIZE);
	md_calc_strides(DIMS, strsY, dimsY, CFL_SIZE);
	md_calc_strides(DIMS, strsZ, dimsZ, CFL_SIZE);

	complex float* x = md_alloc(DIMS, dimsX, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dimsY, CFL_SIZE);
	complex float* z = md_alloc(DIMS, dimsZ, CFL_SIZE);

	md_gaussian_rand(DIMS, dimsX, x);
	md_gaussian_rand(DIMS, dimsY, y);

	md_clear(DIMS, dimsZ, z, CFL_SIZE);


	double tic = timestamp();

	md_zfmac2(DIMS, dims, strsZ, z, strsX, x, strsY, y);

	double toc = timestamp();


	md_free(x);
	md_free(y);
	md_free(z);

	return toc - tic;
}


static double bench_generic_add(long dims[DIMS], unsigned int flags, bool forloop)
{
	long dimsX[DIMS];
	long dimsY[DIMS];

	long dimsC[DIMS];

	md_select_dims(DIMS, flags, dimsX, dims);
	md_select_dims(DIMS, ~flags, dimsC, dims);
	md_select_dims(DIMS, ~0u, dimsY, dims);

	long strsX[DIMS];
	long strsY[DIMS];

	md_calc_strides(DIMS, strsX, dimsX, CFL_SIZE);
	md_calc_strides(DIMS, strsY, dimsY, CFL_SIZE);

	complex float* x = md_alloc(DIMS, dimsX, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dimsY, CFL_SIZE);

	md_gaussian_rand(DIMS, dimsX, x);
	md_gaussian_rand(DIMS, dimsY, y);

	long L = md_calc_size(DIMS, dimsC);
	long T = md_calc_size(DIMS, dimsX);

	double tic = timestamp();

	if (forloop) {

		for (long i = 0; i < L; i++) {

			for (long j = 0; j < T; j++) {

				y[i + j*L] += x[j];
			}
		}

	} else {

		md_zaxpy2(DIMS, dims, strsY, y, 1., strsX, x);
	}

	double toc = timestamp();


	md_free(x);
	md_free(y);

	return toc - tic;
}


static double bench_generic_sum(long dims[DIMS], unsigned int flags, bool forloop)
{
	long dimsX[DIMS];
	long dimsY[DIMS];
	long dimsC[DIMS];

	md_select_dims(DIMS, ~0u, dimsX, dims);
	md_select_dims(DIMS, flags, dimsY, dims);
	md_select_dims(DIMS, ~flags, dimsC, dims);

	long strsX[DIMS];
	long strsY[DIMS];

	md_calc_strides(DIMS, strsX, dimsX, CFL_SIZE);
	md_calc_strides(DIMS, strsY, dimsY, CFL_SIZE);

	complex float* x = md_alloc(DIMS, dimsX, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dimsY, CFL_SIZE);

	md_gaussian_rand(DIMS, dimsX, x);
	md_clear(DIMS, dimsY, y, CFL_SIZE);

	long L = md_calc_size(DIMS, dimsC);
	long T = md_calc_size(DIMS, dimsY);

	double tic = timestamp();

	if (forloop) {
		for (long i = 0; i < L; i++) {

			for (long j = 0; j < T; j++) {

				y[j] = y[j] + x[i + j*L];
			}
		}

	}
	else
		md_zaxpy2(DIMS, dims, strsY, y, 1., strsX, x);

	double toc = timestamp();


	md_free(x);
	md_free(y);

	return toc - tic;
}

static double bench_copy1(void)
{
	long dims[DIMS] = { 1, 128, 128, 1, 1, 16, 1, 16 };
	return bench_generic_copy(dims);
}

static double bench_copy2(void)
{
	long dims[DIMS] = { 262144, 16, 1, 1, 1, 1, 1, 1 };
	return bench_generic_copy(dims);
}


static double bench_matrix_multiply(void)
{
	long dims[DIMS] = { 1, 256, 256, 256, 1, 1, 1, 1 };
	return bench_generic_matrix_multiply(dims);
}



static double bench_batch_matmul1(void)
{
	long dims[DIMS] = { 30000, 8, 8, 8, 1, 1, 1, 1 };
	return bench_generic_matrix_multiply(dims);
}



static double bench_batch_matmul2(void)
{
	long dims[DIMS] = { 1, 8, 8, 8, 30000, 1, 1, 1 };
	return bench_generic_matrix_multiply(dims);
}


static double bench_tall_matmul1(void)
{
	long dims[DIMS] = { 1, 8, 8, 100000, 1, 1, 1, 1 };
	return bench_generic_matrix_multiply(dims);
}


static double bench_tall_matmul2(void)
{
	long dims[DIMS] = { 1, 100000, 8, 8, 1, 1, 1, 1 };
	return bench_generic_matrix_multiply(dims);
}


static double bench_add(void)
{
	long dims[DIMS] = { 65536, 1, 50, 1, 1, 1, 1, 1 };
	return bench_generic_add(dims, (1 << 2), false);
}

static double bench_addf(void)
{
	long dims[DIMS] = { 65536, 1, 50, 1, 1, 1, 1, 1 };
	return bench_generic_add(dims, (1 << 2), true);
}

static double bench_add2(void)
{
	long dims[DIMS] = { 50, 1, 65536, 1, 1, 1, 1, 1 };
	return bench_generic_add(dims, (1 << 0), false);
}

static double bench_sum2(void)
{
	long dims[DIMS] = { 50, 1, 65536, 1, 1, 1, 1, 1 };
	return bench_generic_sum(dims, (1 << 0), false);
}

static double bench_sum(void)
{
	long dims[DIMS] = { 65536, 1, 50, 1, 1, 1, 1, 1 };
	return bench_generic_sum(dims, (1 << 2), false);
}

static double bench_sumf(void)
{
	long dims[DIMS] = { 65536, 1, 50, 1, 1, 1, 1, 1 };
	return bench_generic_sum(dims, (1 << 2), true);
}


static double bench_transpose(void)
{
	long dims[DIMS] = { 2000, 2000, 1, 1, 1, 1, 1, 1 };

	complex float* x = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dims, CFL_SIZE);
	
	md_gaussian_rand(DIMS, dims, x);
	md_clear(DIMS, dims, y, CFL_SIZE);

	double tic = timestamp();

	md_transpose(DIMS, 0, 1, dims, y, dims, x, CFL_SIZE);

	double toc = timestamp();

	md_free(x);
	md_free(y);
	
	return toc - tic;
}



static double bench_resize(void)
{
	long dimsX[DIMS] = { 2000, 1000, 1, 1, 1, 1, 1, 1 };
	long dimsY[DIMS] = { 1000, 2000, 1, 1, 1, 1, 1, 1 };

	complex float* x = md_alloc(DIMS, dimsX, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dimsY, CFL_SIZE);
	
	md_gaussian_rand(DIMS, dimsX, x);
	md_clear(DIMS, dimsY, y, CFL_SIZE);

	double tic = timestamp();

	md_resize(DIMS, dimsY, y, dimsX, x, CFL_SIZE);

	double toc = timestamp();

	md_free(x);
	md_free(y);
	
	return toc - tic;
}


static double bench_norm(int s)
{
	long dims[DIMS] = { 256, 256, 1, 16, 1, 1, 1, 1 };
#if 0
	complex float* x = md_alloc_gpu(DIMS, dims, CFL_SIZE);
	complex float* y = md_alloc_gpu(DIMS, dims, CFL_SIZE);
#else
	complex float* x = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dims, CFL_SIZE);
#endif
	
	md_gaussian_rand(DIMS, dims, x);
	md_gaussian_rand(DIMS, dims, y);

	double tic = timestamp();

	switch (s) {
	case 0:
		md_zscalar(DIMS, dims, x, y);
		break;
	case 1:
		md_zscalar_real(DIMS, dims, x, y);
		break;
	case 2:
		md_znorm(DIMS, dims, x);
		break;
	case 3:
		md_z1norm(DIMS, dims, x);
		break;
	}

	double toc = timestamp();

	md_free(x);
	md_free(y);
	
	return toc - tic;
}

static double bench_zscalar(void)
{
	return bench_norm(0);
}

static double bench_zscalar_real(void)
{
	return bench_norm(1);
}

static double bench_znorm(void)
{
	return bench_norm(2);
}

static double bench_zl1norm(void)
{
	return bench_norm(3);
}


static double bench_wavelet_thresh(int version)
{
	long dims[DIMS] = { 1, 256, 256, 1, 16, 1, 1, 1 };
	long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
	minsize[0] = MIN(dims[0], 16);
	minsize[1] = MIN(dims[1], 16);
	minsize[2] = MIN(dims[2], 16);

	const struct operator_p_s* p;

	switch (version) {
	case 2:
		p = prox_wavethresh_create(DIMS, dims, 7, minsize, 1.1, true, false);
		break;
	case 3:
#ifdef BERKELEY_SVN
		p = prox_wavelet3_thresh_create(DIMS, dims, 6, minsize, 1.1, true);
		break;
#endif
	default:
		assert(0);
	}

	complex float* x = md_alloc(DIMS, dims, CFL_SIZE);
	md_gaussian_rand(DIMS, dims, x);

	double tic = timestamp();

	operator_p_apply_unchecked(p, 0.98, x, x);

	double toc = timestamp();

	md_free(x);
	operator_p_free(p);

	return toc - tic;
}

static double bench_wavelet2_thresh(void)
{
	return bench_wavelet_thresh(2);
}

#ifdef BERKELEY_SVN
static double bench_wavelet3_thresh(void)
{
	return bench_wavelet_thresh(3);
}
#endif

static void do_test(double (*fun)(void), const char* str)
{
	printf("%30.30s |", str);
	
	int N = 5;
	double sum = 0.;
	double min = 1.E10;
	double max = 0.;

	for (int i = 0; i < N; i++) {

		double dt = fun();
		sum += dt;
		min = MIN(dt, min);
		max = MAX(dt, max);

		printf(" %3.4f", (float)dt);
		fflush(stdout);
	}

	printf(" | Avg: %3.4f Max: %3.4f Min: %3.4f\n", (float)(sum / N), max, min); 
}




int main(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 0, usage_str, help_str);
	num_init();

	do_test(bench_add,		"add (md_zaxpy)");
	do_test(bench_add2,		"add (md_zaxpy), contiguous");
	do_test(bench_addf,		"add (for loop)");
	do_test(bench_sum,   		"sum (md_zaxpy)");
	do_test(bench_sum2,   		"sum (md_zaxpy), contiguous");
	do_test(bench_sumf,   		"sum (for loop)");
	do_test(bench_transpose,	"complex transpose");
	do_test(bench_resize,   	"complex resize");
	do_test(bench_matrix_multiply,	"complex matrix multiply");
	do_test(bench_batch_matmul1,	"batch matrix multiply 1");
	do_test(bench_batch_matmul2,	"batch matrix multiply 2");
	do_test(bench_tall_matmul1,	"tall matrix multiply 1");
	do_test(bench_tall_matmul2,	"tall matrix multiply 2");
	do_test(bench_zscalar,		"complex dot product");
	do_test(bench_zscalar,		"complex dot product");
	do_test(bench_zscalar_real,	"real complex dot product");
	do_test(bench_znorm,		"l2 norm");
	do_test(bench_zl1norm,		"l1 norm");
	do_test(bench_copy1,		"copy 1");
	do_test(bench_copy2,		"copy 2");
	do_test(bench_wavelet2_thresh,	"wavelet soft thresh");
#ifdef BERKELEY_SVN
	do_test(bench_wavelet3_thresh,	"wavelet soft thresh");
#endif

	exit(0);
}


