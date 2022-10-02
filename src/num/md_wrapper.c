/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>

#include "misc/misc.h"

#include "num/blas.h"
#include "num/multind.h"

#ifdef USE_CUDA
#include "num/gpukrnls.h"
#include "num/gpuops.h"
#endif

#include "md_wrapper.h"


/****************************************************************************************************
 *
 * Wrappers for zfmac
 *
 ****************************************************************************************************/

/**
 * batched loop for zfmac (for block diagonal matrices)
 *
 * inner most dimension must be diagonal
 * strides of the following dims (up to three) must be given by selecting dims
 *
 * @param dims
 * @param ostr
 * @param optr
 * @param istr1
 * @param iptr1
 * @param istr1
 * @param iptr1
 **/
void zfmac_gpu_batched_loop(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
	long size = 8;

	assert(4 >= N);

	long todims[N];
	long tidims1[N];
	long tidims2[N];

	long tostrs[N];
	long tistrs1[N];
	long tistrs2[N];

	md_select_dims(N, MD_BIT(0) | md_nontriv_strides(N, ostr), todims, dims);
	md_select_dims(N, MD_BIT(0) | md_nontriv_strides(N, istr1), tidims1, dims);
	md_select_dims(N, MD_BIT(0) | md_nontriv_strides(N, istr2), tidims2, dims);

	md_calc_strides(N, tostrs, todims, size);
	md_calc_strides(N, tistrs1, tidims1, size);
	md_calc_strides(N, tistrs2, tidims2, size);

	for (int i = 0; i < N; i++) {

		assert(tostrs[i] == ostr[i]);
		assert(tistrs1[i] == istr1[i]);
		assert(tistrs2[i] == istr2[i]);
	}

	long tdims[3];
	md_singleton_dims(3, tdims);

	md_copy_dims(N - 1, tdims, dims + 1);

#ifdef USE_CUDA

	assert(cuda_ondevice(optr));
	assert(cuda_ondevice(iptr1));
	assert(cuda_ondevice(iptr2));

	cuda_zfmac_strided(dims[0], tdims,
				md_nontriv_strides(N - 1, ostr + 1),
				md_nontriv_strides(N - 1, istr1 + 1),
				md_nontriv_strides(N - 1, istr2 + 1),
				optr, iptr1, iptr2);

#else
	UNUSED(optr);
	UNUSED(iptr1);
	UNUSED(iptr2);
	assert(0);
#endif
}

/****************************************************************************************************
 *
 * Wrappers for zfmacc
 *
 ****************************************************************************************************/

/**
 * batched loop for zfmacc (for block diagonal matrices)
 *
 * inner most dimension must be diagonal
 * strides of the following dims (up to three) must be given by selecting dims
 *
 * @param dims
 * @param ostr
 * @param optr
 * @param istr1
 * @param iptr1
 * @param istr1
 * @param iptr1
 **/
void zfmacc_gpu_batched_loop(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
	long size = 8;

	assert(4 >= N);

	long todims[N];
	long tidims1[N];
	long tidims2[N];

	long tostrs[N];
	long tistrs1[N];
	long tistrs2[N];

	md_select_dims(N, MD_BIT(0) | md_nontriv_strides(N, ostr), todims, dims);
	md_select_dims(N, MD_BIT(0) | md_nontriv_strides(N, istr1), tidims1, dims);
	md_select_dims(N, MD_BIT(0) | md_nontriv_strides(N, istr2), tidims2, dims);

	md_calc_strides(N, tostrs, todims, size);
	md_calc_strides(N, tistrs1, tidims1, size);
	md_calc_strides(N, tistrs2, tidims2, size);

	for (int i = 0; i < N; i++) {

		assert(tostrs[i] == ostr[i]);
		assert(tistrs1[i] == istr1[i]);
		assert(tistrs2[i] == istr2[i]);
	}

	long tdims[3];
	md_singleton_dims(3, tdims);

	md_copy_dims(N - 1, tdims, dims + 1);

#ifdef USE_CUDA

	assert(cuda_ondevice(optr));
	assert(cuda_ondevice(iptr1));
	assert(cuda_ondevice(iptr2));

	cuda_zfmacc_strided(dims[0], tdims,
				md_nontriv_strides(N - 1, ostr + 1),
				md_nontriv_strides(N - 1, istr1 + 1),
				md_nontriv_strides(N - 1, istr2 + 1),
				optr, iptr1, iptr2);

#else
	UNUSED(optr);
	UNUSED(iptr1);
	UNUSED(iptr2);
	assert(0);
#endif
}
