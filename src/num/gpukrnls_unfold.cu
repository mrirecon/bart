/* Copyright 2023. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2026. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>
#include <limits.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "num/gpuops.h"
#include "num/gpukrnls.h"
#include "num/multind.h"


#define BLOCKSIZE 512

static int blocksize(long N)
{
	return BLOCKSIZE;
}

static long gridsize(long N)
{
	// to ensure that "start" does not overflow we need to restrict gridsize!
	return MIN((N + BLOCKSIZE - 1) / BLOCKSIZE, 65536 - 1);
}


struct cuda_strides_3D {

	long dims[3];
	long ostrs[3];
	long istrs1[3];
	long istrs2[3];
	unsigned long total;
};

static struct cuda_strides_3D strs_ini = {

	.dims = { 1, 1, 1},
	.ostrs = { 0, 0, 0},
	.istrs1 = { 0, 0, 0},
	.istrs2 = { 0, 0, 0},
	.total = 1UL,
};

typedef void(*fOp)(float*, float, float);
typedef void(kern_fOp_unfold)(cuda_strides_3D strs, float* dst, const float* src1, const float* src2);

typedef void(*zOp)(cuFloatComplex*, cuFloatComplex, cuFloatComplex);
typedef void(kern_zOp_unfold)(cuda_strides_3D strs, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2);

template <fOp fop, int N>
__global__ static void kern_fop_unfold_generic(cuda_strides_3D strs, float* dst, const float* src1, const float* src2)
{

	unsigned long start = (unsigned long)blockIdx.x * (unsigned long)blockDim.x + (unsigned long)threadIdx.x;
	unsigned long stride = (unsigned long)blockDim.x * (unsigned long)gridDim.x;

	for (unsigned long i = start; i < strs.total; i += stride) {

		long ooffset = 0;
		long ioffset1 = 0;
		long ioffset2 = 0;

		unsigned long tmp = i;

		for (int j = 0; j < N; j++) {

			unsigned long id = tmp % (unsigned long)strs.dims[j];
			tmp /= (unsigned long)strs.dims[j];

			ooffset += (long)id * strs.ostrs[j];
			ioffset1 += (long)id * strs.istrs1[j];
			ioffset2 += (long)id * strs.istrs2[j];
		}

		fop(dst + ooffset, src1[ioffset1], src2[ioffset2]);
	}
}

template <zOp zop, int N>
__global__ static void kern_zop_unfold_generic(cuda_strides_3D strs, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	unsigned long start = (unsigned long)blockIdx.x * (unsigned long)blockDim.x + (unsigned long)threadIdx.x;
	unsigned long stride = (unsigned long)blockDim.x * (unsigned long)gridDim.x;

	for (unsigned long i = start; i < (unsigned long)strs.total; i += stride) {

		long ooffset = 0;
		long ioffset1 = 0;
		long ioffset2 = 0;

		unsigned long tmp = i;

		for (int j = 0; j < N; j++) {

			unsigned long id = tmp % (unsigned long)strs.dims[j];
			tmp /= (unsigned long)strs.dims[j];

			ooffset += (long)id * strs.ostrs[j];
			ioffset1 += (long)id * strs.istrs1[j];
			ioffset2 += (long)id * strs.istrs2[j];
		}

		zop(dst + ooffset, src1[ioffset1], src2[ioffset2]);
	}
}


static cuda_strides_3D get_strides(int D, const long dims[], const long ostrs[], const long istrs1[], const long istrs2[], size_t size)
{
	assert(D <= 3);

	cuda_strides_3D strs = strs_ini;

	for (int i = 0; i < D; i++) {

		strs.dims[i] = dims[i];
		strs.ostrs[i] = ostrs[i] / size;
		strs.istrs1[i] = istrs1[i] / size;
		strs.istrs2[i] = istrs2[i] / size;
	}

	/* precompute total number of elements */
	unsigned long long tot = 1ULL;
	for (int i = 0; i < D; i++)
		tot *= (unsigned long long)strs.dims[i];

	strs.total = (unsigned long)tot;

	return strs;
}

template <fOp fop, int N>
static void cuda_fop_unfoldt(int D, const long dims[], const long ostrs[], float* dst, const long istrs1[], const float* src1, const long istrs2[], const float* src2)
{
	assert(D <= 3);

	cuda_strides_3D strs = get_strides(D, dims, ostrs, istrs1, istrs2, sizeof(float));

	CUDA_ERROR_PTR(dst, src1, src2);

	kern_fop_unfold_generic<fop, N><<<gridsize(strs.total), blocksize(strs.total), 0, cuda_get_stream()>>>(strs, dst, src1, src2);
	CUDA_KERNEL_ERROR;
}

template <fOp fop>
static void cuda_fop_unfold(int D, const long dims[], const long ostrs[], float* dst, const long istrs1[], const float* src1, const long istrs2[], const float* src2)
{
	switch (D) {

		case 1:
			cuda_fop_unfoldt<fop, 1>(D, dims, ostrs, dst, istrs1, src1, istrs2, src2);
			break;
		case 2:
			cuda_fop_unfoldt<fop, 2>(D, dims, ostrs, dst, istrs1, src1, istrs2, src2);
			break;
		case 3:
			cuda_fop_unfoldt<fop, 3>(D, dims, ostrs, dst, istrs1, src1, istrs2, src2);
			break;
		default:
			error("D > 3 not supported\n");
	}
}

template <zOp zop, int N>
static void cuda_zop_unfoldt(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2)
{
	assert(D <= 3);

	cuda_strides_3D strs = get_strides(D, dims, ostrs, istrs1, istrs2, sizeof(cuFloatComplex));

	CUDA_ERROR_PTR(dst, src1, src2);

	kern_zop_unfold_generic<zop, N><<<gridsize(strs.total), blocksize(strs.total), 0, cuda_get_stream()>>>(strs, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
	CUDA_KERNEL_ERROR;
}

template <zOp zop>
static void cuda_zop_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2)
{
	switch (D) {

		case 1:
			cuda_zop_unfoldt<zop, 1>(D, dims, ostrs, dst, istrs1, src1, istrs2, src2);
			break;
		case 2:
			cuda_zop_unfoldt<zop, 2>(D, dims, ostrs, dst, istrs1, src1, istrs2, src2);
			break;
		case 3:
			cuda_zop_unfoldt<zop, 3>(D, dims, ostrs, dst, istrs1, src1, istrs2, src2);
			break;
		default:
			error("D > 3 not supported\n");
	}
}

__device__ __forceinline__ static void cuda_device_add(float* dst, float x, float y)
{
	*dst = x + y;
}

extern "C" void cuda_add_unfold(int D, const long dims[], const long ostrs[], float* dst, const long istrs1[], const float* src1, const long istrs2[], const float* src2)
{
	cuda_fop_unfold<cuda_device_add>(D, dims, ostrs, dst, istrs1, src1, istrs2, src2);
}

__device__ __forceinline__ static void cuda_device_zadd(cuFloatComplex* dst, cuFloatComplex x, cuFloatComplex y)
{
	*dst = make_cuFloatComplex(x.x + y.x, x.y + y.y);
}

extern "C" void cuda_zadd_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2)
{
	cuda_zop_unfold<cuda_device_zadd>(D, dims, ostrs, dst, istrs1, src1, istrs2, src2);
}

__device__ __forceinline__ static void cuda_device_mul(float* dst, float x, float y)
{
	*dst = x * y;
}

extern "C" void cuda_mul_unfold(int D, const long dims[], const long ostrs[], float* dst, const long istrs1[], const float* src1, const long istrs2[], const float* src2)
{
	cuda_fop_unfold<cuda_device_mul>(D, dims, ostrs, dst, istrs1, src1, istrs2, src2);
}

__device__ __forceinline__ static void cuda_device_zmul(cuFloatComplex* dst, cuFloatComplex x, cuFloatComplex y)
{
	*dst = cuCmulf(x, y);
}

extern "C" void cuda_zmul_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2)
{
	cuda_zop_unfold<cuda_device_zmul>(D, dims, ostrs, dst, istrs1, src1, istrs2, src2);
}

__device__ __forceinline__ static void cuda_device_zmulc(cuFloatComplex* dst, cuFloatComplex x, cuFloatComplex y)
{
	*dst = cuCmulf(x, cuConjf(y));
}

extern "C" void cuda_zmulc_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2)
{
	cuda_zop_unfold<cuda_device_zmulc>(D, dims, ostrs, dst, istrs1, src1, istrs2, src2);
}


__device__ __forceinline__ static void cuda_device_fmac(float* dst, float x, float y)
{
	*dst += x * y;
}

extern "C" void cuda_fmac_unfold(int D, const long dims[], const long ostrs[], float* dst, const long istrs1[], const float* src1, const long istrs2[], const float* src2)
{
	cuda_fop_unfold<cuda_device_fmac>(D, dims, ostrs, dst, istrs1, src1, istrs2, src2);
}

__device__ __forceinline__ static void cuda_device_zfmac(cuFloatComplex* dst, cuFloatComplex x, cuFloatComplex y)
{
	*dst = cuCaddf(*dst, cuCmulf(x, y));
}

extern "C" void cuda_zfmac_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2)
{
	cuda_zop_unfold<cuda_device_zfmac>(D, dims, ostrs, dst, istrs1, src1, istrs2, src2);
}

__device__ __forceinline__ static void cuda_device_zfmacc(cuFloatComplex* dst, cuFloatComplex x, cuFloatComplex y)
{
	*dst = cuCaddf(*dst, cuCmulf(x, cuConjf(y)));
}

extern "C" void cuda_zfmacc_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2)
{
	cuda_zop_unfold<cuda_device_zfmacc>(D, dims, ostrs, dst, istrs1, src1, istrs2, src2);
}
