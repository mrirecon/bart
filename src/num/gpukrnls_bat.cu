/* Copyright 2023. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
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


static dim3 getBlockSize2(long Bi, long Bo, const void* func)
{
	int block[3] = { 1, 1, 1};

	cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, func);
	int threads = attr.maxThreadsPerBlock;

	block[0] = 1;
	block[1] = 1;

	while ((threads >= 2) && (block[0] < Bi)) {

		block[0] *= 2;
		threads /= 2;
	}

	while ((threads >= 2) && (block[1] < Bo)) {

		block[1] *= 2;
		threads /= 2;
	}

	return dim3(block[0], block[1], block[2]);
}

static long gridsize_int(long N, int blocksize)
{
	return MIN(65535, (N + blocksize - 1) / blocksize); // 65535 is maximum for y and z dim
}

static dim3 getGridSize2(long Bi, long Bo, const void* func)
{
	int block[3] = { 1, 1, 1};

	cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, func);
	int threads = attr.maxThreadsPerBlock;

	block[0] = 1;
	block[1] = 1;

	while ((threads >= 2) && (block[0] < Bi)) {

		block[0] *= 2;
		threads /= 2;
	}

	while ((threads >= 2) && (block[1] < Bo)) {

		block[1] *= 2;
		threads /= 2;
	}

	return dim3(gridsize_int(Bi, block[0]), gridsize_int(Bo, block[1]), 1);
}



__global__ static void kern_xpay_bat(long Bi, long N, long Bo, const float* _beta, cuFloatComplex* _a, const cuFloatComplex* _x)
{
	long bi_sta = threadIdx.x + blockDim.x * blockIdx.x;
	long bi_str = blockDim.x * gridDim.x;

	long bo_sta = threadIdx.y + blockDim.y * blockIdx.y;
	long bo_str = blockDim.y * gridDim.y;

	for (long bi = bi_sta; bi < Bi; bi += bi_str) {
		for (long bo = bo_sta; bo < Bo; bo += bo_str) {

			float beta = _beta[bi + Bi * bo];

			for (long i = 0; i < N; i++) {

				long idx = bi + Bi * i + Bi * N * bo;

				cuFloatComplex x = _x[idx];
				cuFloatComplex a = _a[idx];

				a.x = a.x * beta + x.x;
				a.y = a.y * beta + x.y;

				_a[idx] = a;
			}
		}
	}
}

extern "C" void cuda_xpay_bat(long Bi, long N, long Bo, const float* beta, float* a, const float* x)
{
	if (1 == Bi && (Bo * 1000 < N)) {

		float* beta_cpu = (float*) xmalloc(sizeof(float) * Bo);
		cuda_memcpy(sizeof(float) * Bo, beta_cpu, beta);

		for (long bo = 0; bo < Bo; bo++)
			cuda_xpay(2 * N, beta_cpu[bo], a + 2 * N * bo, x + 2 * N * bo);
		free(beta_cpu);
		return;
	}

	dim3 blockDim = getBlockSize2(Bi, Bo, (const void*)kern_xpay_bat);
	dim3 gridDim = getGridSize2(Bi, Bo, (const void*)kern_xpay_bat);

	kern_xpay_bat<<<gridDim, blockDim>>>(Bi, N, Bo, beta, (cuFloatComplex*) a, (const cuFloatComplex*)x);
	CUDA_KERNEL_ERROR;
}

__global__ static void kern_axpy_bat(long Bi, long N, long Bo, cuFloatComplex* _a, const float* _alpha, const cuFloatComplex* _x)
{
	long bi_sta = threadIdx.x + blockDim.x * blockIdx.x;
	long bi_str = blockDim.x * gridDim.x;

	long bo_sta = threadIdx.y + blockDim.y * blockIdx.y;
	long bo_str = blockDim.y * gridDim.y;

	for (long bi = bi_sta; bi < Bi; bi += bi_str) {
		for (long bo = bo_sta; bo < Bo; bo += bo_str) {

			float alpha = _alpha[bi + Bi * bo];

			for (long i = 0; i < N; i++) {

				long idx = bi + Bi * i + Bi * N * bo;

				cuFloatComplex x = _x[idx];
				cuFloatComplex a = _a[idx];

				a.x = a.x + x.x * alpha;
				a.y = a.y + x.y * alpha;

				_a[idx] = a;
			}
		}
	}
}


extern "C" void cuda_axpy_bat(long Bi, long N, long Bo, float* a, const float* alpha, const float* x)
{
	if (1 == Bi && (Bo * 1000 < N)) {

		float* alpha_cpu = (float*) xmalloc(sizeof(float) * Bo);
		cuda_memcpy(sizeof(float) * Bo, alpha_cpu, alpha);

		for (long bo = 0; bo < Bo; bo++)
			cuda_axpbz(2 * N, a + 2 * N * bo, 1, a + 2 * N * bo, alpha_cpu[bo], x + 2 * N * bo);

		free(alpha_cpu);
		return;
	}

	dim3 blockDim = getBlockSize2(Bi, Bo, (const void*)kern_axpy_bat);
	dim3 gridDim = getGridSize2(Bi, Bo, (const void*)kern_axpy_bat);

	kern_axpy_bat<<<gridDim, blockDim>>>(Bi, N, Bo, (cuFloatComplex*) a, alpha, (const cuFloatComplex*)x);
	CUDA_KERNEL_ERROR;
}


__global__ static void kern_dot_bat(long Bi, long N, long Bo, float* dst, const cuFloatComplex* _src1, const cuFloatComplex* _src2)
{
	long bi_sta = threadIdx.x + blockDim.x * blockIdx.x;
	long bi_str = blockDim.x * gridDim.x;

	long bo_sta = threadIdx.y + blockDim.y * blockIdx.y;
	long bo_str = blockDim.y * gridDim.y;

	for (long bi = bi_sta; bi < Bi; bi += bi_str) {
		for (long bo = bo_sta; bo < Bo; bo += bo_str) {

			double ret = 0;

			for (long i = 0; i < N; i++) {

				long idx = bi + Bi * i + Bi * N * bo;

				cuFloatComplex src1 = _src1[idx];
				cuFloatComplex src2 = _src2[idx];

				ret += src1.x * src2.x;
				ret += src1.y * src2.y;
			}

			dst[bi + Bi * bo] = ret;
		}
	}
}

extern "C" void cuda_dot_bat(long Bi, long N, long Bo, float* dst, const float* x, const float* y)
{
	if (1 == Bi && (Bo * 1000 < N)) {

		float* dst_cpu = (float*) xmalloc(sizeof(float) * Bo);

		for (long bo = 0; bo < Bo; bo++)
			dst_cpu[bo] = cuda_dot(2 * N, x + 2 * N * bo, y + 2 * N * bo);

		cuda_memcpy(sizeof(float) * Bo, dst, dst_cpu);
		free(dst_cpu);

		return;
	}

	dim3 blockDim = getBlockSize2(Bi, Bo, (const void*)kern_dot_bat);
	dim3 gridDim = getGridSize2(Bi, Bo, (const void*)kern_dot_bat);

	kern_dot_bat<<<gridDim, blockDim>>>(Bi, N, Bo, dst, (const cuFloatComplex*)x, (const cuFloatComplex*)y);
	CUDA_KERNEL_ERROR;
}
