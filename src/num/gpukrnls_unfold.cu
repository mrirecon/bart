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
};

static struct cuda_strides_3D strs_ini = {

	.N = 0,

	.dims = { 1, 1, 1},
	.ostrs = { 0, 0, 0},
	.istrs1 = { 0, 0, 0},
	.istrs2 = { 0, 0, 0},
};

typedef void(*fOp)(float*, float, float);
typedef void(kern_fOp_unfold)(cuda_strides_3D strs, float* dst, const float* src1, const float* src2);

typedef void(*zOp)(cuFloatComplex*, cuFloatComplex, cuFloatComplex);
typedef void(kern_zOp_unfold)(cuda_strides_3D strs, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2);


template <fOp fop, int N, unsigned int const1, unsigned int const2>
__global__ static void kern_fop_unfold_generic(cuda_strides_3D strs, float* dst, const float* src1, const float* src2)
{
	long idx[3];
	long idx_init[3];
	unsigned int stride[3];
	unsigned int thread[3];
	unsigned int block[3];

	thread[0] = (0 < N) ? threadIdx.x : 0;
	thread[1] = (1 < N) ? threadIdx.y : 0;
	thread[2] = (2 < N) ? threadIdx.z : 0;

	block[0] = (0 < N) ? blockDim.x : 1;
	block[1] = (1 < N) ? blockDim.y : 1;
	block[2] = (2 < N) ? blockDim.z : 1;

	idx_init[0] = (0 < N) ? blockDim.x * blockIdx.x : 0; //if idx would contain thread id, the loop might diverge ->deadlock with syncthreads
	idx_init[1] = (1 < N) ? blockDim.y * blockIdx.y : 0;
	idx_init[2] = (2 < N) ? blockDim.z * blockIdx.z : 0;

	stride[0] = (0 < N) ? blockDim.x * gridDim.x : 1;
	stride[1] = (1 < N) ? blockDim.y * gridDim.y : 1;
	stride[2] = (2 < N) ? blockDim.z * gridDim.z : 1;

	for (idx[2] = idx_init[2]; idx[2] < ((2 < N) ? strs.dims[2] : 1); idx[2] += stride[2])
		for (idx[1] = idx_init[1]; idx[1] < ((1 < N) ? strs.dims[1] : 1); idx[1] += stride[1])
			for (idx[0] = idx_init[0]; idx[0] < ((0 < N) ? strs.dims[0] : 1); idx[0] += stride[0]) {

				long o_off =  0;
				long i1_off = 0;
				long i2_off = 0;

				bool valid = true;

				for (int i = 0; i < N ; i++) {

					valid = valid && ((thread[i] + idx[i]) < strs.dims[i]);

					o_off +=  (thread[i] + idx[i]) * strs.ostrs[i];
					i1_off += (thread[i] + idx[i]) * strs.istrs1[i];
					i2_off += (thread[i] + idx[i]) * strs.istrs2[i];
				}


				int tmp1_size = 0;
				int tmp1_idx = 0;
				bool read1 = false;

				if (const1) {

					tmp1_size = 1;
					read1 = true;

					for (int i = 0; i < N; i++) {

						if (const1 & (1u << i)) {

							read1 = read1 && (0 == thread[i]);

						} else {

							tmp1_idx += tmp1_size * thread[i];
							tmp1_size *= block[i];
						}
					}
				}

				int tmp2_size = 0;
				int tmp2_idx = tmp1_size;
				_Bool read2 = false;

				if (const2) {

					tmp2_size = 1;
					read2 = true;

					for (int i = 0; i < N; i++) {

						if (const2 & (1u << i)) {

							read2 = read2 && (0 == thread[i]);

						} else {

							tmp2_idx += tmp2_size * thread[i];
							tmp2_size *= block[i];
						}
					}
				}

				extern __shared__ float tmp_float[];

				if (valid && read1)
					tmp_float[tmp1_idx] = src1[i1_off];

				if (valid && read2)
					tmp_float[tmp2_idx] = src2[i2_off];

				if (const1 || const2)
					__syncthreads();

				if (valid)
					fop(&(dst[o_off]), (const1 ? tmp_float[tmp1_idx] : src1[i1_off]), (const2 ? tmp_float[tmp2_idx] : src2[i2_off]));
			}
}

#define run_fop_unfold(_N, _const1, _const2) if ((_N == N) && (_const1 == const1) && (_const2 == const2)) return kern_fop_unfold_generic<fop, _N, _const1, _const2>;

template <fOp fop>
static kern_fOp_unfold* get_kern_fop_unfold(int N, unsigned int const1, unsigned int const2)
{
	run_fop_unfold(1 , 0, 0);
	run_fop_unfold(1 , 1, 0);
	run_fop_unfold(1 , 0, 1);


	run_fop_unfold(2 , 0, 0);
	run_fop_unfold(2 , 0, 1);
	run_fop_unfold(2 , 0, 2);
	run_fop_unfold(2 , 0, 3);

	run_fop_unfold(2 , 1, 0);
	run_fop_unfold(2 , 1, 1);
	run_fop_unfold(2 , 1, 2);
	run_fop_unfold(2 , 1, 3);

	run_fop_unfold(2 , 2, 0);
	run_fop_unfold(2 , 2, 1);
	run_fop_unfold(2 , 2, 2);
	run_fop_unfold(2 , 2, 3);

	run_fop_unfold(2 , 3, 0);
	run_fop_unfold(2 , 3, 1);
	run_fop_unfold(2 , 3, 2);
	run_fop_unfold(2 , 3, 3);


	run_fop_unfold(3 , 0, 0);
	run_fop_unfold(3 , 0, 1);
	run_fop_unfold(3 , 0, 2);
	run_fop_unfold(3 , 0, 3);
	run_fop_unfold(3 , 0, 4);
	run_fop_unfold(3 , 0, 5);
	run_fop_unfold(3 , 0, 6);
	run_fop_unfold(3 , 0, 7);

	run_fop_unfold(3 , 1, 0);
	run_fop_unfold(3 , 1, 1);
	run_fop_unfold(3 , 1, 2);
	run_fop_unfold(3 , 1, 3);
	run_fop_unfold(3 , 1, 4);
	run_fop_unfold(3 , 1, 5);
	run_fop_unfold(3 , 1, 6);
	run_fop_unfold(3 , 1, 7);

	run_fop_unfold(3 , 2, 0);
	run_fop_unfold(3 , 2, 1);
	run_fop_unfold(3 , 2, 2);
	run_fop_unfold(3 , 2, 3);
	run_fop_unfold(3 , 2, 4);
	run_fop_unfold(3 , 2, 5);
	run_fop_unfold(3 , 2, 6);
	run_fop_unfold(3 , 2, 7);

	run_fop_unfold(3 , 3, 0);
	run_fop_unfold(3 , 3, 1);
	run_fop_unfold(3 , 3, 2);
	run_fop_unfold(3 , 3, 3);
	run_fop_unfold(3 , 3, 4);
	run_fop_unfold(3 , 3, 5);
	run_fop_unfold(3 , 3, 6);
	run_fop_unfold(3 , 3, 7);

	run_fop_unfold(3 , 4, 0);
	run_fop_unfold(3 , 4, 1);
	run_fop_unfold(3 , 4, 2);
	run_fop_unfold(3 , 4, 3);
	run_fop_unfold(3 , 4, 4);
	run_fop_unfold(3 , 4, 5);
	run_fop_unfold(3 , 4, 6);
	run_fop_unfold(3 , 4, 7);

	run_fop_unfold(3 , 5, 0);
	run_fop_unfold(3 , 5, 1);
	run_fop_unfold(3 , 5, 2);
	run_fop_unfold(3 , 5, 3);
	run_fop_unfold(3 , 5, 4);
	run_fop_unfold(3 , 5, 5);
	run_fop_unfold(3 , 5, 6);
	run_fop_unfold(3 , 5, 7);

	run_fop_unfold(3 , 6, 0);
	run_fop_unfold(3 , 6, 1);
	run_fop_unfold(3 , 6, 2);
	run_fop_unfold(3 , 6, 3);
	run_fop_unfold(3 , 6, 4);
	run_fop_unfold(3 , 6, 5);
	run_fop_unfold(3 , 6, 6);
	run_fop_unfold(3 , 6, 7);

	run_fop_unfold(3 , 7, 0);
	run_fop_unfold(3 , 7, 1);
	run_fop_unfold(3 , 7, 2);
	run_fop_unfold(3 , 7, 3);
	run_fop_unfold(3 , 7, 4);
	run_fop_unfold(3 , 7, 5);
	run_fop_unfold(3 , 7, 6);
	run_fop_unfold(3 , 7, 7);

	assert(0);
	return NULL;
}


template <zOp zop, int N, unsigned int const1, unsigned int const2>
__global__ static void kern_zop_unfold_generic(cuda_strides_3D strs, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	long idx[3];
	long idx_init[3];
	unsigned int stride[3];
	unsigned int thread[3];
	unsigned int block[3];

	thread[0] = (0 < N) ? threadIdx.x : 0;
	thread[1] = (1 < N) ? threadIdx.y : 0;
	thread[2] = (2 < N) ? threadIdx.z : 0;

	block[0] = (0 < N) ? blockDim.x : 1;
	block[1] = (1 < N) ? blockDim.y : 1;
	block[2] = (2 < N) ? blockDim.z : 1;

	idx_init[0] = (0 < N) ? blockDim.x * blockIdx.x : 0; //if idx would contain thread id, the loop might diverge ->deadlock with syncthreads
	idx_init[1] = (1 < N) ? blockDim.y * blockIdx.y : 0;
	idx_init[2] = (2 < N) ? blockDim.z * blockIdx.z : 0;

	stride[0] = (0 < N) ? blockDim.x * gridDim.x : 1;
	stride[1] = (1 < N) ? blockDim.y * gridDim.y : 1;
	stride[2] = (2 < N) ? blockDim.z * gridDim.z : 1;

	for (idx[2] = idx_init[2]; idx[2] < ((2 < N) ? strs.dims[2] : 1); idx[2] += stride[2])
		for (idx[1] = idx_init[1]; idx[1] < ((1 < N) ? strs.dims[1] : 1); idx[1] += stride[1])
			for (idx[0] = idx_init[0]; idx[0] < ((0 < N) ? strs.dims[0] : 1); idx[0] += stride[0]) {

				long o_off =  0;
				long i1_off = 0;
				long i2_off = 0;

				bool valid = true;

				for (int i = 0; i < N ; i++) {

					valid = valid && ((thread[i] + idx[i]) < strs.dims[i]);

					o_off +=  (thread[i] + idx[i]) * strs.ostrs[i];
					i1_off += (thread[i] + idx[i]) * strs.istrs1[i];
					i2_off += (thread[i] + idx[i]) * strs.istrs2[i];
				}

				int tmp1_size = 0;
				int tmp1_idx = 0;
				bool read1 = false;

				if (const1) {

					tmp1_size = 1;
					read1 = true;

					for (int i = 0; i < N; i++) {

						if (const1 & (1u << i)) {

							read1 = read1 && (0 == thread[i]);

						} else {

							tmp1_idx += tmp1_size * thread[i];
							tmp1_size *= block[i];
						}
					}
				}

				int tmp2_size = 0;
				int tmp2_idx = tmp1_size;
				bool read2 = false;

				if (const2) {

					tmp2_size = 1;
					read2 = true;

					for (int i = 0; i < N; i++) {

						if (const2 & (1u << i)) {

							read2 = read2 && (0 == thread[i]);

						} else {

							tmp2_idx += tmp2_size * thread[i];
							tmp2_size *= block[i];
						}
					}
				}

				extern __shared__ cuFloatComplex tmp_complex[];

				if (valid && read1)
					tmp_complex[tmp1_idx] = src1[i1_off];

				if (valid && read2)
					tmp_complex[tmp2_idx] = src2[i2_off];

				if (const1 || const2)
					__syncthreads();

				if (valid)
					zop(&(dst[o_off]), (const1 ? tmp_complex[tmp1_idx] : src1[i1_off]), (const2 ? tmp_complex[tmp2_idx] : src2[i2_off]));
			}
}

#define run_zop_unfold(_N, _const1, _const2) if ((_N == N) && (_const1 == const1) && (_const2 == const2)) return kern_zop_unfold_generic<zop, _N, _const1, _const2>;

template <zOp zop>
static kern_zOp_unfold* get_kern_zop_unfold(int N, unsigned int const1, unsigned int const2)
{
	run_zop_unfold(1 , 0, 0);
	run_zop_unfold(1 , 1, 0);
	run_zop_unfold(1 , 0, 1);


	run_zop_unfold(2 , 0, 0);
	run_zop_unfold(2 , 0, 1);
	run_zop_unfold(2 , 0, 2);
	run_zop_unfold(2 , 0, 3);

	run_zop_unfold(2 , 1, 0);
	run_zop_unfold(2 , 1, 1);
	run_zop_unfold(2 , 1, 2);
	run_zop_unfold(2 , 1, 3);

	run_zop_unfold(2 , 2, 0);
	run_zop_unfold(2 , 2, 1);
	run_zop_unfold(2 , 2, 2);
	run_zop_unfold(2 , 2, 3);

	run_zop_unfold(2 , 3, 0);
	run_zop_unfold(2 , 3, 1);
	run_zop_unfold(2 , 3, 2);
	run_zop_unfold(2 , 3, 3);


	run_zop_unfold(3 , 0, 0);
	run_zop_unfold(3 , 0, 1);
	run_zop_unfold(3 , 0, 2);
	run_zop_unfold(3 , 0, 3);
	run_zop_unfold(3 , 0, 4);
	run_zop_unfold(3 , 0, 5);
	run_zop_unfold(3 , 0, 6);
	run_zop_unfold(3 , 0, 7);

	run_zop_unfold(3 , 1, 0);
	run_zop_unfold(3 , 1, 1);
	run_zop_unfold(3 , 1, 2);
	run_zop_unfold(3 , 1, 3);
	run_zop_unfold(3 , 1, 4);
	run_zop_unfold(3 , 1, 5);
	run_zop_unfold(3 , 1, 6);
	run_zop_unfold(3 , 1, 7);

	run_zop_unfold(3 , 2, 0);
	run_zop_unfold(3 , 2, 1);
	run_zop_unfold(3 , 2, 2);
	run_zop_unfold(3 , 2, 3);
	run_zop_unfold(3 , 2, 4);
	run_zop_unfold(3 , 2, 5);
	run_zop_unfold(3 , 2, 6);
	run_zop_unfold(3 , 2, 7);

	run_zop_unfold(3 , 3, 0);
	run_zop_unfold(3 , 3, 1);
	run_zop_unfold(3 , 3, 2);
	run_zop_unfold(3 , 3, 3);
	run_zop_unfold(3 , 3, 4);
	run_zop_unfold(3 , 3, 5);
	run_zop_unfold(3 , 3, 6);
	run_zop_unfold(3 , 3, 7);

	run_zop_unfold(3 , 4, 0);
	run_zop_unfold(3 , 4, 1);
	run_zop_unfold(3 , 4, 2);
	run_zop_unfold(3 , 4, 3);
	run_zop_unfold(3 , 4, 4);
	run_zop_unfold(3 , 4, 5);
	run_zop_unfold(3 , 4, 6);
	run_zop_unfold(3 , 4, 7);

	run_zop_unfold(3 , 5, 0);
	run_zop_unfold(3 , 5, 1);
	run_zop_unfold(3 , 5, 2);
	run_zop_unfold(3 , 5, 3);
	run_zop_unfold(3 , 5, 4);
	run_zop_unfold(3 , 5, 5);
	run_zop_unfold(3 , 5, 6);
	run_zop_unfold(3 , 5, 7);

	run_zop_unfold(3 , 6, 0);
	run_zop_unfold(3 , 6, 1);
	run_zop_unfold(3 , 6, 2);
	run_zop_unfold(3 , 6, 3);
	run_zop_unfold(3 , 6, 4);
	run_zop_unfold(3 , 6, 5);
	run_zop_unfold(3 , 6, 6);
	run_zop_unfold(3 , 6, 7);

	run_zop_unfold(3 , 7, 0);
	run_zop_unfold(3 , 7, 1);
	run_zop_unfold(3 , 7, 2);
	run_zop_unfold(3 , 7, 3);
	run_zop_unfold(3 , 7, 4);
	run_zop_unfold(3 , 7, 5);
	run_zop_unfold(3 , 7, 6);
	run_zop_unfold(3 , 7, 7);

	assert(0);
	return NULL;
}


#define WARPSIZE 32

static void getBlockSize3_internal(int block[3], const long dims[3], int threads)
{
	block[0] = 1;
	block[1] = 1;
	block[2] = 1;

	while ((threads >= 2) && (block[0] < dims[0])) {

		block[0] *= 2;
		threads /= 2;
	}

	while ((threads >= 2) && (block[1] < dims[1])) {

		block[1] *= 2;
		threads /= 2;
	}

	while ((threads >= 2) && (block[2] < dims[2])) {

		block[2] *= 2;
		threads /= 2;
	}
}

static dim3 getBlockSize3(const long dims[3], int threads)
{
	int block[3];

	getBlockSize3_internal(block, dims, threads);

	return dim3(block[0], block[1], block[2]);
}

static long gridsize_int(long N, int blocksize)
{
	return MIN(65535, (N + blocksize - 1) / blocksize); // 65535 is maximum for y and z dim
}

static dim3 getGridSize3(const long dims[3], int threads)
{
	int block[3];

	getBlockSize3_internal(block, dims, threads);

	return dim3(gridsize_int(dims[0], block[0]), gridsize_int(dims[1], block[1]), gridsize_int(dims[2], block[2]));
}

template <fOp fop>
static void cuda_fop_unfold(int D, const long dims[], const long ostrs[], float* dst, const long istrs1[], const float* src1, const long istrs2[], const float* src2)
{
	assert(D <= 3);

	cuda_strides_3D strs = strs_ini;

	strs.N = D;

	for (int i = 0; i < D; i++) {

		strs.dims[i] = dims[i];
		strs.ostrs[i] = ostrs[i] / sizeof(float);
		strs.istrs1[i] = istrs1[i] / sizeof(float);
		strs.istrs2[i] = istrs2[i] / sizeof(float);
	}

	CUDA_ERROR_PTR(dst, src1, src2);

	unsigned long const1 = md_nontriv_dims(D, dims) & (~md_nontriv_strides(D, istrs1));
	unsigned long const2 = md_nontriv_dims(D, dims) & (~md_nontriv_strides(D, istrs2));

	kern_fOp_unfold* func = get_kern_fop_unfold<fop>(D, (unsigned int)const1, (unsigned int)const2);

	static int threads = -1;

	if (-1 == threads) {

		cudaFuncAttributes attr;
		cudaFuncGetAttributes(&attr, func);
		threads = attr.maxThreadsPerBlock;
	}

	dim3 blockDim = getBlockSize3(strs.dims, threads);
	dim3 gridDim = getGridSize3(strs.dims, threads);

	int size1 = 0;
	int size2 = 0;

	if (const1)
		size1 = (!MD_IS_SET(const1, 0) ? blockDim.x : 1) * (!MD_IS_SET(const1, 1) ? blockDim.y : 1) * (!MD_IS_SET(const1, 2) ? blockDim.z : 1);
	
	if (const2)
		size2 = (!MD_IS_SET(const2, 0) ? blockDim.x : 1) * (!MD_IS_SET(const2, 1) ? blockDim.y : 1) * (!MD_IS_SET(const2, 2) ? blockDim.z : 1);


	func<<<gridDim, blockDim, sizeof(float) * (size1 + size2), cuda_get_stream()>>>(strs, dst, src1, src2);
	CUDA_KERNEL_ERROR;
}

template <zOp zop>
static void cuda_zop_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2)
{
	assert(D <= 3);

	cuda_strides_3D strs = strs_ini;

	strs.N = D;

	for (int i = 0; i < D; i++) {

		strs.dims[i] = dims[i];
		strs.ostrs[i] = ostrs[i] / sizeof(_Complex float);
		strs.istrs1[i] = istrs1[i] / sizeof(_Complex float);
		strs.istrs2[i] = istrs2[i] / sizeof(_Complex float);
	}

	CUDA_ERROR_PTR(dst, src1, src2);

	unsigned long const1 = md_nontriv_dims(D, dims) & (~md_nontriv_strides(D, istrs1));
	unsigned long const2 = md_nontriv_dims(D, dims) & (~md_nontriv_strides(D, istrs2));

	kern_zOp_unfold* func = get_kern_zop_unfold<zop>(D, (unsigned int)const1, (unsigned int)const2);

	static int threads = -1;

	if (-1 == threads) {

		cudaFuncAttributes attr;
		cudaFuncGetAttributes(&attr, func);
		threads = attr.maxThreadsPerBlock;
	}

	dim3 blockDim = getBlockSize3(strs.dims, threads);
	dim3 gridDim = getGridSize3(strs.dims, threads);

	int size1 = 0;
	int size2 = 0;

	if (const1)
		size1 = (!MD_IS_SET(const1, 0) ? blockDim.x : 1) * (!MD_IS_SET(const1, 1) ? blockDim.y : 1) * (!MD_IS_SET(const1, 2) ? blockDim.z : 1);
	
	if (const2)
		size2 = (!MD_IS_SET(const2, 0) ? blockDim.x : 1) * (!MD_IS_SET(const2, 1) ? blockDim.y : 1) * (!MD_IS_SET(const2, 2) ? blockDim.z : 1);

	func<<<gridDim, blockDim, sizeof(cuFloatComplex) * (size1 + size2), cuda_get_stream()>>>(strs, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
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
