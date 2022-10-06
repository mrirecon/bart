/* Copyright 2020. Uecker Lab, University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2020 Christian Holme <christian.holme@med.uni-goettingen.de>
 */

#include <complex.h>
#include <assert.h>

#include "num/multind.h"
#include "num/init.h"
#include "num/mem.h"

#include "utest.h"

#ifndef CFL_SIZE
#define CFL_SIZE	sizeof(complex float)
#endif


static bool test_cuda_memcache_clear(void)
{
#ifndef USE_CUDA
	return true;
#else
	// TODO: detect if GPU works

	num_init_gpu();
	enum { test_dims = 7 };

	const long dims[test_dims] = { 4, 4, 4, 4, 4, 4, 1 };

	const unsigned int D = test_dims;

	complex float* ptr1 = md_alloc_gpu(D, dims, CFL_SIZE);
	complex float* ptr2 = md_alloc_gpu(D, dims, CFL_SIZE);

	md_clear(D, dims, ptr1, CFL_SIZE);
	md_clear(D, dims, ptr2, CFL_SIZE);

	md_free(ptr1);
	md_free(ptr2);

	num_deinit_gpu();

	return memcache_is_empty();
#endif
}

UT_GPU_REGISTER_TEST(test_cuda_memcache_clear);
