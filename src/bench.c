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

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/init.h"
#include "num/ops.h"

#include "wavelet2/wavelet.h"

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


static double bench_wavelet_thresh(void)
{
	long dims[DIMS] = { 1, 256, 256, 1, 16, 1, 1, 1 };
	long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
	minsize[0] = MIN(dims[0], 16);
	minsize[1] = MIN(dims[1], 16);
	minsize[2] = MIN(dims[2], 16);

	const struct operator_p_s* p = prox_wavethresh_create(DIMS, dims, 7, minsize, 1.1, 1, 0);

	complex float* x = md_alloc(DIMS, dims, CFL_SIZE);
	md_gaussian_rand(DIMS, dims, x);

	double tic = timestamp();

	operator_p_apply_unchecked(p, 0.98, x, x);

	double toc = timestamp();

	md_free(x);
	operator_p_free(p);

	return toc - tic;
}


static void do_test(double (*fun)(void), const char* str)
{
	printf("%20.20s ", str);
	
	int N = 5;
	double sum = 0.;
	for (int i = 0; i < N; i++) {

		double dt = fun();
		sum += dt;

		printf(" %3.4f", (float)dt);
		fflush(stdout);
	}

	printf(" Avg: %3.4f\n", (float)(sum / N)); 
}




int main(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 0, usage_str, help_str);
	num_init();

	do_test(bench_transpose,	"complex transpose");
	do_test(bench_resize,   	"complex resize");
	do_test(bench_matrix_multiply,	"complex matrix multiply");
	do_test(bench_batch_matmul1,	"batch matrix multiply 1");
	do_test(bench_batch_matmul2,	"batch matrix multiply 2");
	do_test(bench_zscalar,		"complex dot product");
	do_test(bench_zscalar,		"complex dot product");
	do_test(bench_zscalar_real,	"real complex dot product");
	do_test(bench_znorm,		"l2 norm");
	do_test(bench_zl1norm,		"l1 norm");
	do_test(bench_copy1,	"copy 1");
	do_test(bench_copy2,	"copy 2");
	do_test(bench_wavelet_thresh,		"wavelet soft thresh");

	exit(0);
}


