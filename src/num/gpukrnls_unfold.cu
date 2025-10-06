/* Copyright 2023. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "num/gpuops.h"
#include "num/gpukrnls.h"
#include "num/multind.h"



struct cuda_strides_3D {

	int N;

	long dims[3];
	long ostrs[3];
	long istrs1[3];
	long istrs2[3];

	int blockDim[3];
	long sistrs1[3];
	long sistrs2[3];

	int shared1_size;
	int shared2_size;
	unsigned int const1;
	unsigned int const2;
};

static struct cuda_strides_3D strs_ini = {

	.N = 0,

	.dims = { 1, 1, 1},
	.ostrs = { 0, 0, 0},
	.istrs1 = { 0, 0, 0},
	.istrs2 = { 0, 0, 0},

	.blockDim = {0, 0, 0},
	.sistrs1 = {0, 0, 0},
	.sistrs2 = {0, 0, 0},
	.shared1_size = 0,
	.shared2_size = 0,
	.const1 = 0,
	.const2 = 0,
};

typedef void(*fOp)(float*, float, float);
typedef void(kern_fOp_unfold)(cuda_strides_3D strs, float* dst, const float* src1, const float* src2);

typedef void(*zOp)(cuFloatComplex*, cuFloatComplex, cuFloatComplex);
typedef void(kern_zOp_unfold)(cuda_strides_3D strs, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2);

#define BLK_SIZE(x) (1U << (x))

__device__ static long offset_reravel_index(const long dims[3], const long strs[3], const long pos[3], const int blk_dim[3], unsigned int cflag, int idx)
{
	long offset = 0;
	bool valid = true;

	for (int i = 0; i < 3; i++) {

		if (MD_IS_SET(cflag, i))
			continue;

		unsigned int idx_i = idx & ((1U << blk_dim[i]) - 1);
		valid = valid && (pos[i] + idx_i < dims[i]);
		offset += idx_i * strs[i];
		idx = idx >> blk_dim[i];
	}

	valid = valid && (0 == idx);

	return valid ? offset : -1;
}

template <fOp fop>
__global__ static void kern_fop_unfold_generic(cuda_strides_3D strs, float* dst, const float* src1, const float* src2)
{
	long idx[3];
	long idx_init[3];
	unsigned int stride[3];
	int N = strs.N;

	idx_init[0] = (0 < N) ? BLK_SIZE(strs.blockDim[0]) * blockIdx.x : 0; //if idx would contain thread id, the loop might diverge ->deadlock with syncthreads
	idx_init[1] = (1 < N) ? BLK_SIZE(strs.blockDim[1]) * blockIdx.y : 0;
	idx_init[2] = (2 < N) ? BLK_SIZE(strs.blockDim[2]) * blockIdx.z : 0;

	stride[0] = (0 < N) ? BLK_SIZE(strs.blockDim[0]) * gridDim.x : 1;
	stride[1] = (1 < N) ? BLK_SIZE(strs.blockDim[1]) * gridDim.y : 1;
	stride[2] = (2 < N) ? BLK_SIZE(strs.blockDim[2]) * gridDim.z : 1;

	for (idx[2] = idx_init[2]; idx[2] < ((2 < N) ? strs.dims[2] : 1); idx[2] += stride[2])
		for (idx[1] = idx_init[1]; idx[1] < ((1 < N) ? strs.dims[1] : 1); idx[1] += stride[1])
			for (idx[0] = idx_init[0]; idx[0] < ((0 < N) ? strs.dims[0] : 1); idx[0] += stride[0]) {

				float* tmp_dst = dst;
				const float* tmp_src1 = src1;
				const float* tmp_src2 = src2;

				for (int i = 0; i < N; i++) {

					tmp_dst += idx[i] * strs.ostrs[i];
					tmp_src1 += idx[i] * strs.istrs1[i];
					tmp_src2 += idx[i] * strs.istrs2[i];
				}

				extern __shared__ float shared_float[];
				float* tmp_float = shared_float;

				if (strs.const1) {

					long roffset = offset_reravel_index(strs.dims, strs.istrs1, idx, strs.blockDim, strs.const1, threadIdx.x);
					long soffset = offset_reravel_index(strs.dims, strs.sistrs1, idx, strs.blockDim, strs.const1, threadIdx.x);
					bool valid = -1 != roffset;

					if (valid)
						tmp_float[soffset] = tmp_src1[roffset];

					tmp_src1 = tmp_float;
					tmp_float += strs.shared1_size;
				}

				if (strs.const2) {

					long roffset = offset_reravel_index(strs.dims, strs.istrs2, idx, strs.blockDim, strs.const2, threadIdx.x);
					long soffset = offset_reravel_index(strs.dims, strs.sistrs2, idx, strs.blockDim, strs.const2, threadIdx.x);
					bool valid = -1 != roffset;

					if (valid)
						tmp_float[soffset] = tmp_src2[roffset];

					tmp_src2 = tmp_float;
				}

				if (strs.const1 || strs.const2)
					__syncthreads();

				long o_off = offset_reravel_index(strs.dims, strs.ostrs, idx, strs.blockDim, 0, threadIdx.x);
				long i1_off = offset_reravel_index(strs.dims, strs.sistrs1, idx, strs.blockDim, 0, threadIdx.x);
				long i2_off = offset_reravel_index(strs.dims, strs.sistrs2, idx, strs.blockDim, 0, threadIdx.x);

				if (-1 != o_off)
					fop(tmp_dst + o_off, tmp_src1[i1_off], tmp_src2[i2_off]);

				if (strs.const1 || strs.const2)
					__syncthreads();
			}
}

template <zOp zop>
__global__ static void kern_zop_unfold_generic(cuda_strides_3D strs, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	long idx[3];
	long idx_init[3];
	unsigned int stride[3];
	int N = strs.N;

	idx_init[0] = (0 < N) ? BLK_SIZE(strs.blockDim[0]) * blockIdx.x : 0; //if idx would contain thread id, the loop might diverge ->deadlock with syncthreads
	idx_init[1] = (1 < N) ? BLK_SIZE(strs.blockDim[1]) * blockIdx.y : 0;
	idx_init[2] = (2 < N) ? BLK_SIZE(strs.blockDim[2]) * blockIdx.z : 0;

	stride[0] = (0 < N) ? BLK_SIZE(strs.blockDim[0]) * gridDim.x : 1;
	stride[1] = (1 < N) ? BLK_SIZE(strs.blockDim[1]) * gridDim.y : 1;
	stride[2] = (2 < N) ? BLK_SIZE(strs.blockDim[2]) * gridDim.z : 1;

	for (idx[2] = idx_init[2]; idx[2] < ((2 < N) ? strs.dims[2] : 1); idx[2] += stride[2])
		for (idx[1] = idx_init[1]; idx[1] < ((1 < N) ? strs.dims[1] : 1); idx[1] += stride[1])
			for (idx[0] = idx_init[0]; idx[0] < ((0 < N) ? strs.dims[0] : 1); idx[0] += stride[0]) {

				cuFloatComplex* tmp_dst = dst;
				const cuFloatComplex* tmp_src1 = src1;
				const cuFloatComplex* tmp_src2 = src2;

				for (int i = 0; i < N; i++) {

					tmp_dst += idx[i] * strs.ostrs[i];
					tmp_src1 += idx[i] * strs.istrs1[i];
					tmp_src2 += idx[i] * strs.istrs2[i];
				}

				extern __shared__ cuFloatComplex shared_complex[];
				cuFloatComplex* tmp_float = shared_complex;

				if (strs.const1) {

					long roffset = offset_reravel_index(strs.dims, strs.istrs1, idx, strs.blockDim, strs.const1, threadIdx.x);
					long soffset = offset_reravel_index(strs.dims, strs.sistrs1, idx, strs.blockDim, strs.const1, threadIdx.x);
					bool valid = -1 != roffset;

					if (valid)
						tmp_float[soffset] = tmp_src1[roffset];

					tmp_src1 = tmp_float;
					tmp_float += strs.shared1_size;
				}

				if (strs.const2) {

					long roffset = offset_reravel_index(strs.dims, strs.istrs2, idx, strs.blockDim, strs.const2, threadIdx.x);
					long soffset = offset_reravel_index(strs.dims, strs.sistrs2, idx, strs.blockDim, strs.const2, threadIdx.x);
					bool valid = -1 != roffset;

					if (valid)
						tmp_float[soffset] = tmp_src2[roffset];

					tmp_src2 = tmp_float;
				}

				if (strs.const1 || strs.const2)
					__syncthreads();

				long o_off = offset_reravel_index(strs.dims, strs.ostrs, idx, strs.blockDim, 0, threadIdx.x);
				long i1_off = offset_reravel_index(strs.dims, strs.sistrs1, idx, strs.blockDim, 0, threadIdx.x);
				long i2_off = offset_reravel_index(strs.dims, strs.sistrs2, idx, strs.blockDim, 0, threadIdx.x);

				if (-1 != o_off)
					zop(tmp_dst + o_off, tmp_src1[i1_off], tmp_src2[i2_off]);

				if (strs.const1 || strs.const2)
					__syncthreads();
			}
}


static long gridsize_int(long N, int blocksize)
{
	return MIN(65535, (N + blocksize - 1) / blocksize); // 65535 is maximum for y and z dim
}


static cuda_strides_3D get_strides(int D, int grd_size[3], const long dims[], const long ostrs[], const long istrs1[], const long istrs2[], size_t size, int threads)
{
	assert(D <= 3);

	cuda_strides_3D strs = strs_ini;

	strs.N = 3;

	for (int i = 0; i < D; i++) {

		strs.dims[i] = dims[i];
		strs.ostrs[i] = ostrs[i] / size;
		strs.istrs1[i] = istrs1[i] / size;
		strs.istrs2[i] = istrs2[i] / size;
		strs.sistrs1[i] = istrs1[i] / size;
		strs.sistrs2[i] = istrs2[i] / size;
	}

	while ((threads >= 2) && (BLK_SIZE(strs.blockDim[0]) < strs.dims[0]) && (BLK_SIZE(strs.blockDim[0]) < 32)) {

		strs.blockDim[0]++;
		threads /= 2;
	}

	for (int i = 0; i < 20; i++) {

		int j = 1 + i % 2;

		if ((threads >= 2) && (BLK_SIZE(strs.blockDim[j]) < strs.dims[j])) {

			strs.blockDim[j]++;
			threads /= 2;
		}
	}

	while ((threads >= 2) && (BLK_SIZE(strs.blockDim[0]) < strs.dims[0])) {

		strs.blockDim[0]++;
		threads /= 2;
	}

	unsigned long const1 = (md_nontriv_dims(D, dims) & (~md_nontriv_strides(D, istrs1)));
	unsigned long const2 = (md_nontriv_dims(D, dims) & (~md_nontriv_strides(D, istrs2)));

	long block_dims[3];

	for (int i = 0; i < 3; i++) {

		block_dims[i] = BLK_SIZE(strs.blockDim[i]);
		grd_size[i] = gridsize_int(strs.dims[i], block_dims[i]);
	}

	long block_dims_src1[3];
	long block_dims_src2[3];

	md_select_dims(3, ~const1, block_dims_src1, block_dims);
	md_select_dims(3, ~const2, block_dims_src2, block_dims);

	strs.shared1_size = md_calc_size(3, block_dims_src1);
	strs.shared2_size = md_calc_size(3, block_dims_src2);

	long block_strides_src1[3];
	long block_strides_src2[3];
	md_calc_strides(3, block_strides_src1, block_dims_src1, 1);
	md_calc_strides(3, block_strides_src2, block_dims_src2, 1);

	for (int i = 0; i < 3; i++) {

		if (0 != const1)
			strs.sistrs1[i] = block_strides_src1[i];

		if (0 != const2)
			strs.sistrs2[i] = block_strides_src2[i];
	}

	strs.const1 = const1;
	strs.const2 = const2;

	return strs;
}

template <fOp fop>
static void cuda_fop_unfold(int D, const long dims[], const long ostrs[], float* dst, const long istrs1[], const float* src1, const long istrs2[], const float* src2)
{
	assert(D <= 3);

	static int threads = -1;

	if (-1 == threads) {

		cudaFuncAttributes attr;
		cudaFuncGetAttributes(&attr, kern_fop_unfold_generic<fop>);
		threads = attr.maxThreadsPerBlock;
	}

	int grd_size[3];
	cuda_strides_3D strs = get_strides(D, grd_size, dims, ostrs, istrs1, istrs2, sizeof(float), threads);

	CUDA_ERROR_PTR(dst, src1, src2);

	dim3 blockDim = dim3(threads, 1, 1);
	dim3 gridDim = dim3(grd_size[0], grd_size[1], grd_size[2]);

	kern_fop_unfold_generic<fop><<<gridDim, blockDim, sizeof(float) * (strs.shared1_size + strs.shared2_size), cuda_get_stream()>>>(strs, dst, src1, src2);
	CUDA_KERNEL_ERROR;
}

template <zOp zop>
static void cuda_zop_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2)
{
	assert(D <= 3);

	static int threads = -1;

	if (-1 == threads) {

		cudaFuncAttributes attr;
		cudaFuncGetAttributes(&attr, kern_zop_unfold_generic<zop>);
		threads = attr.maxThreadsPerBlock;
	}

	int grd_size[3];
	cuda_strides_3D strs = get_strides(D, grd_size, dims, ostrs, istrs1, istrs2, sizeof(cuFloatComplex), 1024);

	CUDA_ERROR_PTR(dst, src1, src2);

	dim3 blockDim = dim3(1024, 1, 1);
	dim3 gridDim = dim3(grd_size[0], grd_size[1], grd_size[2]);

	kern_zop_unfold_generic<zop><<<gridDim, blockDim, sizeof(cuFloatComplex) * (strs.shared1_size + strs.shared2_size), cuda_get_stream()>>>(strs, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
	CUDA_KERNEL_ERROR;
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
