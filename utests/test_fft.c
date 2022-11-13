/* Copyright 2022. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <complex.h>

#include "misc/debug.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/rand.h"

#include "utest.h"

static bool test_fftmod_optimize(int N, const long dims[N], unsigned long flags, bool inv)
{
	
	complex float* ptr1 = md_alloc(N, dims, CFL_SIZE);
	md_gaussian_rand(N, dims, ptr1);

	complex float* ptr2 = md_alloc(N, dims, CFL_SIZE);
	md_copy(N, dims, ptr2, ptr1, CFL_SIZE);

	(inv ? ifftmod : fftmod)(N, dims, flags, ptr1, ptr1);

	for (int i = 0; i < N; i++)
		(inv ? ifftmod : fftmod)(N, dims, flags & MD_BIT(i), ptr2, ptr2);
	
	float err = md_znrmse(N, dims, ptr1, ptr2);

	md_free(ptr1);
	md_free(ptr2);

	UT_ASSERT(err < UT_TOL);
}


static bool test_fftmod1(void) { return test_fftmod_optimize(4, MD_DIMS(16, 16, 1, 4), 7, false); }
static bool test_fftmod2(void) { return test_fftmod_optimize(4, MD_DIMS(16, 16, 4, 4), 7, false); }
static bool test_fftmod3(void) { return test_fftmod_optimize(4, MD_DIMS(16, 17, 4, 4), 7, false); }

static bool test_ifftmod1(void) { return test_fftmod_optimize(4, MD_DIMS(16, 16, 1, 4), 7, true); }
static bool test_ifftmod2(void) { return test_fftmod_optimize(4, MD_DIMS(16, 16, 4, 4), 7, true); }
static bool test_ifftmod3(void) { return test_fftmod_optimize(4, MD_DIMS(16, 17, 4, 4), 7, true); }




UT_REGISTER_TEST(test_fftmod1);
UT_REGISTER_TEST(test_fftmod2);
UT_REGISTER_TEST(test_fftmod3);

UT_REGISTER_TEST(test_ifftmod1);
UT_REGISTER_TEST(test_ifftmod2);
UT_REGISTER_TEST(test_ifftmod3);

