/* Copyright 2023. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Bernhard Rapp
 *
*/

#include <complex.h>
#include <assert.h>
#include <stdio.h>

#include "misc/debug.h"
#include "num/mpi_ops.h"
#include "num/rand.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"

#include "utest.h"

#define N 10

typedef void (*fun_t)(int D, const long dimensions[D], unsigned long flags, complex float* dst, const complex float* src);

static bool test_mpi_fft_variants(fun_t test_fun, unsigned long fft_flags, bool inplace)
{
	const unsigned long flags = 8UL;
	const long dims[N] = { 32, 32, 1, 3, 1, 1, 1, 1, 1, 1};
	long strs[N];
	md_calc_strides(N, strs, dims, 1);

	long sdims[N];
	md_select_dims(N, ~flags, sdims, dims);

	complex float* ptr1 = md_alloc(N, dims, CFL_SIZE);
	md_gaussian_rand(N, dims, ptr1);

	complex float* ptr1_dist = md_mpi_move(N, flags, dims, ptr1, CFL_SIZE);
	complex float* ptr2_dist = md_alloc_mpi(N, flags, dims, CFL_SIZE);

	complex float* ptr2 = md_alloc(N, dims, CFL_SIZE);
	md_copy(N, dims, ptr2, ptr1, CFL_SIZE);

	test_fun(N, dims, fft_flags, ptr2, ptr2);

	float err;

	if (inplace) {

		test_fun(N, dims, fft_flags, ptr1_dist, ptr1_dist);

		complex float* ptr1_cmp = md_alloc(N, dims, CFL_SIZE);
		md_copy(N, dims, ptr1_cmp, ptr1_dist, CFL_SIZE);
		
		err = md_znrmse(N, dims, ptr1_cmp, ptr2);

		md_free(ptr1_cmp);
	} else {

		test_fun(N, dims, fft_flags, ptr2_dist, ptr1_dist);

		complex float* ptr2_cmp = md_alloc(N, dims, CFL_SIZE);
		md_copy(N, dims, ptr2_cmp, ptr2_dist, CFL_SIZE);

		err = md_znrmse(N, dims, ptr2_cmp, ptr2);

		md_free(ptr2_cmp);
	}

	md_free(ptr1);
	md_free(ptr1_dist);
	md_free(ptr2);
	md_free(ptr2_dist);

	UT_RETURN_ASSERT(err < UT_TOL);
}

static bool test_fft(void)	{ return test_mpi_fft_variants(fft,	7UL, false); }
static bool test_ifft(void)	{ return test_mpi_fft_variants(ifft,	7UL, false); }
static bool test_fftc(void)	{ return test_mpi_fft_variants(fftc,	7UL, false); }
static bool test_ifftc(void)	{ return test_mpi_fft_variants(ifftc,	7UL, false); }
static bool test_fftu(void)	{ return test_mpi_fft_variants(fftu,	7UL, false); }
static bool test_ifftu(void)	{ return test_mpi_fft_variants(ifftu,	7UL, false); }


static bool test_fft_inplace(void)	{ return test_mpi_fft_variants(fft,	7UL, true); }
static bool test_ifft_inplace(void)	{ return test_mpi_fft_variants(ifft,	7UL, true); }
static bool test_fftc_inplace(void)	{ return test_mpi_fft_variants(fftc,	7UL, true); }
static bool test_ifftc_inplace(void)	{ return test_mpi_fft_variants(ifftc,	7UL, true); }
static bool test_fftu_inplace(void)	{ return test_mpi_fft_variants(fftu,	7UL, true); }
static bool test_ifftu_inplace(void)	{ return test_mpi_fft_variants(ifftu,	7UL, true); }

UT_REGISTER_TEST(test_fft);
UT_REGISTER_TEST(test_ifft);
UT_REGISTER_TEST(test_fftc);
UT_REGISTER_TEST(test_ifftc);
UT_REGISTER_TEST(test_fftu);
UT_REGISTER_TEST(test_ifftu);

UT_REGISTER_TEST(test_fft_inplace);
UT_REGISTER_TEST(test_ifft_inplace);
UT_REGISTER_TEST(test_fftc_inplace);
UT_REGISTER_TEST(test_ifftc_inplace);
UT_REGISTER_TEST(test_fftu_inplace);
UT_REGISTER_TEST(test_ifftu_inplace);


