/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>

#include "misc/misc.h"

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/blas_md_wrapper.h"

#include "reduce_md_wrapper.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#include "num/gpu_reduce.h"
#endif

/**
 *
 * @param dims dimension
 * @param ostr must be of the form {0, 1} or {0}
 * @param optr
 * @param istr1 must be of the form {0, 1} or {0}
 * @param iptr1 must equal optr
 * @param istr1 must be of the form {1, dim[0]} or {1}
 * @param iptr1
 **/
void reduce_zadd_inner_gpu(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
	long size = 8;

	assert((2 == N) || (1 == N));
	assert((0 == ostr[0]));
	assert((0 == istr1[0]));
	assert((size == istr2[0]));

	if ((2 == N) && (1 != dims[1])){

		assert((0 == ostr[0]) && (size == ostr[1]));
		assert((0 == istr1[0]) && (size == istr1[1]));
		assert((size == istr2[0]) && (size * dims[0] == istr2[1]));
	}

	assert(optr == iptr1);

#ifdef USE_CUDA
	assert(cuda_ondevice(optr) && cuda_ondevice(iptr2));
	cuda_reduce_zadd_inner(dims[0], (2 == N) ? dims[1] : 1, optr, iptr2);
#else
	UNUSED(iptr2);
	error("Compiled without gpu support!");
#endif
}

/**
 *
 * @param dims dimension
 * @param ostr must be of the form {1, 0}
 * @param optr
 * @param istr1 must be of the form {1, 0}
 * @param iptr1 must equal optr
 * @param istr1 must be of the form {1, dim[0]}
 * @param iptr1
 **/
void reduce_zadd_outer_gpu(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
	long size = 8;

	assert(2 == N) ;
	assert(((1 == dims[0]) || (size == ostr[0])) && (0 == ostr[1]));
	assert(((1 == dims[0]) || (size == istr1[0])) && (0 == istr1[1]));
	assert(((1 == dims[0]) || (size == istr2[0])) && (size * dims[0] == istr2[1]));

	assert(optr == iptr1);

#ifdef USE_CUDA
	assert(cuda_ondevice(optr) && cuda_ondevice(iptr2));
	cuda_reduce_zadd_outer(dims[1], dims[0], optr, iptr2);
#else
	UNUSED(iptr2);
	error("Compiled without gpu support!");
#endif
}

/**
 *
 * @param dims dimension
 * @param ostr must be of the form {0, 1} or {0}
 * @param optr
 * @param istr1 must be of the form {0, 1} or {0}
 * @param iptr1 must equal optr
 * @param istr1 must be of the form {1, dim[0]} or {1}
 * @param iptr1
 **/
void reduce_zadd_gemv(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
	long size = 8;

	assert(optr == iptr1);

	for (int i = 0; i < N; i++)
		assert(ostr[i] == istr1[i]);

	long one_dims[N];
	long one_strs[N];
	md_select_dims(N, ~md_nontriv_strides(N, ostr) & md_nontriv_strides(N, istr2), one_dims, dims);
	md_calc_strides(N, one_strs, one_dims, size);

	complex float* ones = md_alloc_sameplace(N, one_dims, size, optr);
	md_zfill(N, one_dims, ones, 1.);
	md_zfmac2(N, dims, ostr, optr, istr2, iptr2, one_strs, ones);
	md_free(ones);
}

/**
 *
 * @param dims dimension
 * @param ostr must be of the form {0, 1} or {0}
 * @param optr
 * @param istr1 must be of the form {0, 1} or {0}
 * @param iptr1 must equal optr
 * @param istr1 must be of the form {1, dim[0]} or {1}
 * @param iptr1
 **/
void reduce_add_inner_gpu(int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
	long size = 4;

	assert((2 == N) || (1 == N));
	assert((0 == ostr[0]));
	assert((0 == istr1[0]));
	assert((size == istr2[0]));

	if ((2 == N) && (1 != dims[1])){

		assert((0 == ostr[0]) && (size == ostr[1]));
		assert((0 == istr1[0]) && (size == istr1[1]));
		assert((size == istr2[0]) && (size * dims[0] == istr2[1]));
	}

	assert(optr == iptr1);

#ifdef USE_CUDA
	assert(cuda_ondevice(optr) && cuda_ondevice(iptr2));
	cuda_reduce_add_inner(dims[0], (2 == N) ? dims[1] : 1, optr, iptr2);
#else
	UNUSED(iptr2);
	error("Compiled without gpu support!");
#endif
}

/**
 *
 * @param dims dimension
 * @param ostr must be of the form {1, 0}
 * @param optr
 * @param istr1 must be of the form {1, 0}
 * @param iptr1 must equal optr
 * @param istr1 must be of the form {1, dim[0]}
 * @param iptr1
 **/
void reduce_add_outer_gpu(int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
	long size = 4;

	assert(2 == N) ;
	assert(((1 == dims[0]) || (size == ostr[0])) && (0 == ostr[1]));
	assert(((1 == dims[0]) || (size == istr1[0])) && (0 == istr1[1]));
	assert(((1 == dims[0]) || (size == istr2[0])) && (size * dims[0] == istr2[1]));

	assert(optr == iptr1);

#ifdef USE_CUDA
	assert(cuda_ondevice(optr) && cuda_ondevice(iptr2));
	cuda_reduce_add_outer(dims[1], dims[0], optr, iptr2);
#else
	UNUSED(iptr2);
	error("Compiled without gpu support!");
#endif
}

/**
 *
 * @param dims dimension
 * @param ostr must be of the form {0, 1} or {0}
 * @param optr
 * @param istr1 must be of the form {0, 1} or {0}
 * @param iptr1 must equal optr
 * @param istr1 must be of the form {1, dim[0]} or {1}
 * @param iptr1
 **/
void reduce_add_gemv(int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
	long size = 4;

	assert(optr == iptr1);

	for (int i = 0; i < N; i++)
		assert(ostr[i] == istr1[i]);

	long one_dims[N];
	long one_strs[N];
	md_select_dims(N, ~md_nontriv_strides(N, ostr) & md_nontriv_strides(N, istr2), one_dims, dims);
	md_calc_strides(N, one_strs, one_dims, size);

	float* ones = md_alloc_sameplace(N, one_dims, size, optr);
	float one = 1.;
	md_fill(N, one_dims, ones, &one, size);
	md_fmac2(N, dims, ostr, optr, istr2, iptr2, one_strs, ones);
	md_free(ones);
}

/**
 *
 * @param dims dimension
 * @param ostr must be of the form {0, 1} or {0}
 * @param optr
 * @param istr1 must be of the form {0, 1} or {0}
 * @param iptr1 must equal optr
 * @param istr1 must be of the form {1, dim[0]} or {1}
 * @param iptr1
 **/
void reduce_zmax_inner_gpu(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
	long size = 8;

	assert((2 == N) || (1 == N));
	assert((0 == ostr[0]));
	assert((0 == istr1[0]));
	assert((size == istr2[0]));

	if ((2 == N) && (1 != dims[1])){

		assert((0 == ostr[0]) && (size == ostr[1]));
		assert((0 == istr1[0]) && (size == istr1[1]));
		assert((size == istr2[0]) && (size * dims[0] == istr2[1]));
	}

	assert(optr == iptr1);

#ifdef USE_CUDA
	assert(cuda_ondevice(optr) && cuda_ondevice(iptr2));
	cuda_reduce_zmax_inner(dims[0], (2 == N) ? dims[1] : 1, optr, iptr2);
#else
	UNUSED(iptr2);
	error("Compiled without gpu support!");
#endif
}

/**
 *
 * @param dims dimension
 * @param ostr must be of the form {1, 0}
 * @param optr
 * @param istr1 must be of the form {1, 0}
 * @param iptr1 must equal optr
 * @param istr1 must be of the form {1, dim[0]}
 * @param iptr1
 **/
void reduce_zmax_outer_gpu(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
	long size = 8;

	assert(2 == N) ;
	assert(((1 == dims[0]) || (size == ostr[0])) && (0 == ostr[1]));
	assert(((1 == dims[0]) || (size == istr1[0])) && (0 == istr1[1]));
	assert(((1 == dims[0]) || (size == istr2[0])) && (size * dims[0] == istr2[1]));

	assert(optr == iptr1);

#ifdef USE_CUDA
	assert(cuda_ondevice(optr) && cuda_ondevice(iptr2));
	cuda_reduce_zmax_outer(dims[1], dims[0], optr, iptr2);
#else
	UNUSED(iptr2);
	error("Compiled without gpu support!");
#endif
}
