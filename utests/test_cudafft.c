/* Copyright 2019. Uecker Lab, University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2019 Christian Holme <christian.holme@med.uni-goettingen.de>
 */

#include <complex.h>
#include <assert.h>

#include "num/fft.h"
#include "num/rand.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/debug.h"
#include "misc/misc.h"

#include "utest.h"




static bool run_cuda_fft_test(const unsigned int D, const long* dims, const unsigned long flags,
			       const complex float* in, complex float* cpu_inout,
			       complex float* gpu_inout, complex float* gpu_result)
{
	md_copy(D, dims, cpu_inout, in, CFL_SIZE);
	md_copy(D, dims, gpu_inout, in, CFL_SIZE);

	const struct operator_s* fftplan = fft_create(D, dims, flags, cpu_inout, cpu_inout, false);

	fft_exec(fftplan, cpu_inout, cpu_inout);
	fft_exec(fftplan, gpu_inout, gpu_inout);

	fft_free(fftplan);

	md_copy(D, dims, gpu_result, gpu_inout, CFL_SIZE);

	UT_ASSERT(md_znrmse(D, dims, cpu_inout, gpu_result) < UT_TOL);
}




static bool test_cuda_fft(void)
{
#ifndef USE_CUDA
	return true;
#else
	// TODO: detect if GPU works

	num_rand_init(5);
	num_init_gpu();

	enum { test_cuda_fft_dims = 7 };

	const long dims[test_cuda_fft_dims] = { 4, 4, 4, 4, 4, 4, 1 }; // in last dim != 1 works...

	const bool transform_dims[][test_cuda_fft_dims] = {
		{ 1, 1, 1, 0, 0, 0, 0 },
		{ 1, 1, 0, 0, 1, 0, 0 },
		{ 1, 0, 1, 0, 1, 0, 0 },
		{ 1, 1, 0, 1, 1, 1, 0 },
		{ 1, 1, 0, 1, 1, 0, 1 },
		{ 0, 0, 0, 0, 0, 0, 0 },
	};

	const unsigned int D = test_cuda_fft_dims;

	complex float* in = md_alloc(D, dims, CFL_SIZE);
	md_gaussian_rand(D, dims, in);

	complex float* cpu_inout = md_alloc(D, dims, CFL_SIZE);
	complex float* gpu_inout = md_alloc_gpu(D, dims, CFL_SIZE);
	complex float* gpu_result = md_alloc(D, dims, CFL_SIZE);


	for (unsigned int i = 0; i < ARRAY_SIZE(transform_dims); ++i) {

		unsigned long flags = 0;

		for (unsigned int j = 0; j < D; ++j)
			if (transform_dims[i][j])
				flags = MD_SET(flags, j);

		run_cuda_fft_test(D, dims, flags, in, cpu_inout, gpu_inout, gpu_result);
	}

	md_free(gpu_result);
	md_free(gpu_inout);
	md_free(cpu_inout);
	md_free(in);

	return true;
#endif
}
UT_GPU_REGISTER_TEST(test_cuda_fft);

static bool test_cuda_fftmod(void)
{
#ifndef USE_CUDA
	return true;
#else
	// TODO: detect if GPU works

	num_rand_init(5);
	num_init_gpu();
	
	enum { DIMS = 4 };
	const long dims[DIMS] = {3, 5, 9, 3};
	complex float* cpu1 = md_alloc(DIMS, dims, CFL_SIZE);
	md_gaussian_rand(DIMS, dims, cpu1);

	complex float* gpu = md_gpu_move(DIMS, dims, cpu1, CFL_SIZE);

	fftmod(DIMS, dims, 15, cpu1, cpu1);
	fftmod(DIMS, dims, 15, gpu, gpu);

	complex float* cpu2 = md_alloc(DIMS, dims, CFL_SIZE);
	md_copy(DIMS, dims, cpu2, gpu, CFL_SIZE);

	float err = md_znrmse(DIMS, dims, cpu2, cpu1);

	UT_ASSERT(err < UT_TOL);
#endif
}

UT_GPU_REGISTER_TEST(test_cuda_fftmod);


static bool test_cuda_fftmod2(void)
{
#ifndef USE_CUDA
	return true;
#else
	// TODO: detect if GPU works

	num_rand_init(5);
	num_init_gpu();
	
	enum { DIMS = 4 };
	const long dims[DIMS] = {16, 4, 16, 3};
	complex float* cpu1 = md_alloc(DIMS, dims, CFL_SIZE);
	md_gaussian_rand(DIMS, dims, cpu1);

	complex float* gpu = md_gpu_move(DIMS, dims, cpu1, CFL_SIZE);

	fftmod(DIMS, dims, 15, cpu1, cpu1);
	fftmod(DIMS, dims, 15, gpu, gpu);

	complex float* cpu2 = md_alloc(DIMS, dims, CFL_SIZE);
	md_copy(DIMS, dims, cpu2, gpu, CFL_SIZE);

	float err = md_znrmse(DIMS, dims, cpu2, cpu1);


	UT_ASSERT(err < UT_TOL);
#endif
}

UT_GPU_REGISTER_TEST(test_cuda_fftmod2);


