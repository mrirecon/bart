/* Copyright 2022. Uecker Lab, University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2022 Moritz Blumenthal
 */

#include <complex.h>
#include <assert.h>

#include "num/multind.h"
#include "num/rand.h"
#include "num/init.h"
#include "num/flpmath.h"

#include "utest.h"


static bool test_cuda_compress(void)
{
	num_init_gpu();
	enum { N = 5 };

	const long dims[N] = { 4, 1, 9, 2, 2 };

	complex float* _ptr1 = md_alloc_gpu(N, dims, CFL_SIZE);
	md_gaussian_rand(N, dims, _ptr1);

	float* ptr1 = (float*)_ptr1;
	md_sgreatequal(N, dims, ptr1, ptr1, 0.);

	void* compress = md_compress(N, dims, ptr1);

	float* ptr2 = md_alloc_gpu(N, dims, FL_SIZE);
	md_decompress(N, dims, ptr2, compress);

	float err = md_nrmse(N, dims, ptr2, ptr1);

	md_free(ptr1);
	md_free(ptr2);
	md_free(compress);

	
	UT_ASSERT(err < UT_TOL);
}

UT_GPU_REGISTER_TEST(test_cuda_compress);