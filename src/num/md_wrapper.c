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
#include "num/gpukrnls_unfold.h"
#include "num/gpuops.h"
#endif

#include "md_wrapper.h"

#define CFL_SIZE sizeof(complex float)
#define FL_SIZE sizeof(float)


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


void zfmac_gpu_unfold(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
	assert(3 >= N);
	assert((optr != iptr1) ||  (md_check_equal_dims(N, ostr, MD_STRIDES(N, dims, CFL_SIZE), ~0u) && md_check_equal_dims(N, istr1, MD_STRIDES(N, dims, CFL_SIZE), ~0u)));
	assert((optr != iptr2) ||  (md_check_equal_dims(N, ostr, MD_STRIDES(N, dims, CFL_SIZE), ~0u) && md_check_equal_dims(N, istr2, MD_STRIDES(N, dims, CFL_SIZE), ~0u)));

#ifdef USE_CUDA
	cuda_zfmac_unfold(N, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
#else
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

void zfmacc_gpu_unfold(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
	assert(3 >= N);
	assert((optr != iptr1) ||  (md_check_equal_dims(N, ostr, MD_STRIDES(N, dims, CFL_SIZE), ~0u) && md_check_equal_dims(N, istr1, MD_STRIDES(N, dims, CFL_SIZE), ~0u)));
	assert((optr != iptr2) ||  (md_check_equal_dims(N, ostr, MD_STRIDES(N, dims, CFL_SIZE), ~0u) && md_check_equal_dims(N, istr2, MD_STRIDES(N, dims, CFL_SIZE), ~0u)));

#ifdef USE_CUDA
	cuda_zfmacc_unfold(N, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
#else
	assert(0);
#endif
}

/****************************************************************************************************
 *
 * Wrappers for fmac
 *
 ****************************************************************************************************/


void fmac_gpu_unfold(int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
	assert(3 >= N);
	assert((optr != iptr1) ||  (md_check_equal_dims(N, ostr, MD_STRIDES(N, dims, FL_SIZE), ~0u) && md_check_equal_dims(N, istr1, MD_STRIDES(N, dims, FL_SIZE), ~0u)));
	assert((optr != iptr2) ||  (md_check_equal_dims(N, ostr, MD_STRIDES(N, dims, FL_SIZE), ~0u) && md_check_equal_dims(N, istr2, MD_STRIDES(N, dims, FL_SIZE), ~0u)));

#ifdef USE_CUDA
	cuda_fmac_unfold(N, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
#else
	assert(0);
#endif
}




/****************************************************************************************************
 *
 * Wrappers for add
 *
 ****************************************************************************************************/

/**
 * md_add with input strides possibly 0, no inplace
 *
 * @param dims
 * @param ostr
 * @param optr
 * @param istr1
 * @param iptr1
 * @param istr1
 * @param iptr1
 **/
void add_gpu_unfold(int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
	assert(3 >= N);
	assert((optr != iptr1) ||  (md_check_equal_dims(N, ostr, MD_STRIDES(N, dims, FL_SIZE), ~0u) && md_check_equal_dims(N, istr1, MD_STRIDES(N, dims, FL_SIZE), ~0u)));
	assert((optr != iptr2) ||  (md_check_equal_dims(N, ostr, MD_STRIDES(N, dims, FL_SIZE), ~0u) && md_check_equal_dims(N, istr2, MD_STRIDES(N, dims, FL_SIZE), ~0u)));

#ifdef USE_CUDA
	cuda_add_unfold(N, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
#else
	assert(0);
#endif
}





/****************************************************************************************************
 *
 * Wrappers for zadd
 *
 ****************************************************************************************************/

/**
 * md_add with input strides possibly 0, no inplace
 *
 * @param dims
 * @param ostr
 * @param optr
 * @param istr1
 * @param iptr1
 * @param istr1
 * @param iptr1
 **/
void zadd_gpu_unfold(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
	assert(3 >= N);
	assert((optr != iptr1) ||  (md_check_equal_dims(N, ostr, MD_STRIDES(N, dims, CFL_SIZE), ~0u) && md_check_equal_dims(N, istr1, MD_STRIDES(N, dims, CFL_SIZE), ~0u)));
	assert((optr != iptr2) ||  (md_check_equal_dims(N, ostr, MD_STRIDES(N, dims, CFL_SIZE), ~0u) && md_check_equal_dims(N, istr2, MD_STRIDES(N, dims, CFL_SIZE), ~0u)));

#ifdef USE_CUDA
	cuda_zadd_unfold(N, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
#else
	assert(0);
#endif
}





/****************************************************************************************************
 *
 * Wrappers for mul
 *
 ****************************************************************************************************/

/**
 * md_add with input strides possibly 0, no inplace
 *
 * @param dims
 * @param ostr
 * @param optr
 * @param istr1
 * @param iptr1
 * @param istr1
 * @param iptr1
 **/
void mul_gpu_unfold(int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
	assert(3 >= N);
	assert((optr != iptr1) ||  (md_check_equal_dims(N, ostr, MD_STRIDES(N, dims, FL_SIZE), ~0u) && md_check_equal_dims(N, istr1, MD_STRIDES(N, dims, FL_SIZE), ~0u)));
	assert((optr != iptr2) ||  (md_check_equal_dims(N, ostr, MD_STRIDES(N, dims, FL_SIZE), ~0u) && md_check_equal_dims(N, istr2, MD_STRIDES(N, dims, FL_SIZE), ~0u)));

#ifdef USE_CUDA
	cuda_mul_unfold(N, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
#else
	assert(0);
#endif
}





/****************************************************************************************************
 *
 * Wrappers for zmul
 *
 ****************************************************************************************************/

/**
 * md_add with input strides possibly 0, no inplace
 *
 * @param dims
 * @param ostr
 * @param optr
 * @param istr1
 * @param iptr1
 * @param istr1
 * @param iptr1
 **/
void zmul_gpu_unfold(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
	assert(3 >= N);
	assert((optr != iptr1) ||  (md_check_equal_dims(N, ostr, MD_STRIDES(N, dims, CFL_SIZE), ~0u) && md_check_equal_dims(N, istr1, MD_STRIDES(N, dims, CFL_SIZE), ~0u)));
	assert((optr != iptr2) ||  (md_check_equal_dims(N, ostr, MD_STRIDES(N, dims, CFL_SIZE), ~0u) && md_check_equal_dims(N, istr2, MD_STRIDES(N, dims, CFL_SIZE), ~0u)));

#ifdef USE_CUDA
	cuda_zmul_unfold(N, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
#else
	assert(0);
#endif
}



/****************************************************************************************************
 *
 * Wrappers for zmulc
 *
 ****************************************************************************************************/

/**
 * md_add with input strides possibly 0, no inplace
 *
 * @param dims
 * @param ostr
 * @param optr
 * @param istr1
 * @param iptr1
 * @param istr1
 * @param iptr1
 **/
void zmulc_gpu_unfold(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
	assert(3 >= N);
	assert((optr != iptr1) ||  (md_check_equal_dims(N, ostr, MD_STRIDES(N, dims, CFL_SIZE), ~0u) && md_check_equal_dims(N, istr1, MD_STRIDES(N, dims, CFL_SIZE), ~0u)));
	assert((optr != iptr2) ||  (md_check_equal_dims(N, ostr, MD_STRIDES(N, dims, CFL_SIZE), ~0u) && md_check_equal_dims(N, istr2, MD_STRIDES(N, dims, CFL_SIZE), ~0u)));

#ifdef USE_CUDA
	cuda_zmulc_unfold(N, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
#else
	assert(0);
#endif
}
