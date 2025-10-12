/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012 Dara Bahri <dbahri123@gmail.com>
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 */


#include <stdbool.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>

#include "misc/mri.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/gpuops.h"

#include "calibcu.h"


extern "C" void error(const char* str, ...);

#if 0
// pass &matrix[0][0]
void eigen_hermcu(int N, float* eigenval, complex float* matrix)
{
	culaInitialize();
	assert(culaCheev('V', 'U', N, (culaFloatComplex*) matrix, N, (culaFloat*) eigenval) == culaNoError);
	culaShutdown();
}
#endif

static __device__ __host__ inline cuFloatComplex cuFloatComplexScale(cuFloatComplex a, float s)
{
	cuFloatComplex c;
	c.x = s * a.x;
	c.y = s * a.y;
	return c;
}

static __device__ cuFloatComplex vec_dot(int N, const cuFloatComplex* vec1, const cuFloatComplex* vec2, cuFloatComplex* tmp)
{
	__syncthreads();

	if (threadIdx.y == 0) {

		cuFloatComplex dot = make_cuFloatComplex(0., 0.);

		for (int k = 0; k < N; k++)
			dot = cuCaddf(dot, cuCmulf(vec1[k], cuConjf(vec2[k])));

		tmp[0] = dot;
	}

	__syncthreads();

	return tmp[0];
}


static __device__ void gram_schmidtcu(int M, int N, cuFloatComplex* evals, cuFloatComplex* vecs, cuFloatComplex* vecs_tmp)
{
	for (int i = M - 1; i >= 0; i--) {

		for (int j = i + 1; j <= M - 1; j++) {

			cuFloatComplex dot = vec_dot(N, vecs + i * N, vecs + j * N, vecs_tmp);

			for (int k = threadIdx.y; k < N; k += blockDim.y)
				vecs[k + i * N] = cuCaddf(vecs[k + i * N], cuCmulf(vecs[k + j * N], cuFloatComplexScale(dot, -1.)));

			__syncthreads();
		}

		float norm = cuCrealf(vec_dot(N, vecs + i * N, vecs + i * N, vecs_tmp));
		norm = sqrtf(norm);

		evals[i] = make_cuFloatComplex(norm, 0.);

		for (int k = threadIdx.y; k < N; k += blockDim.y)
			vecs[k + i * N] = cuFloatComplexScale(vecs[k + i * N], 1. / norm);
	}
}

static __device__ long upper_triag_idx(long j, long i)
{
	if (i > j)
		return - (j + ((i + 1) * i) / 2);

	return i + ((j + 1) * j) / 2;
}

static __device__ inline void mat_mulcu_upperdiag(int M, int N, cuFloatComplex* A, cuFloatComplex* B, cuFloatComplex* C, long offset, long stride)
{
	for (int i = 0; i < M; i++) {

		for (int k = threadIdx.y; k < N; k += blockDim.y) {

			A[k + i * N] = make_cuFloatComplex(0., 0.);

			for (int j = 0; j < N; j++) {

				cuFloatComplex val;
				long idx = upper_triag_idx(j, k);
				val = (0 > idx) ? cuConjf(C[offset - idx * stride]) : C[offset + idx * stride];

				A[k + i * N] = cuCaddf(A[k + i * N], cuCmulf(B[j + i * N], val));
			}
		}
	}
}

static __global__ void eigenmapscu_kern(cuFloatComplex* in, cuFloatComplex* out, cuFloatComplex* vals, int iter, long V, int N, int M)
{
	for (long boffset = blockDim.x * blockIdx.x; boffset < V; boffset += blockDim.x * gridDim.x) {

		long offset = boffset + threadIdx.x;

		extern __shared__ cuFloatComplex sdata[];
		cuFloatComplex *tmp1, *tmp2, *evals;
		tmp1 = sdata + threadIdx.x * (2 * M * N + M);
		tmp2 = tmp1 + M * N;
		evals = tmp2 + M * N;

		for (int i = 0; i < M; i++)
			for (int k = threadIdx.y; k < N; k += blockDim.y)
				tmp1[k + i * N] = (k == i) ? make_cuFloatComplex(1., 0.) : make_cuFloatComplex(0., 0.);

		__syncthreads();

		for (int i = 0; i < iter; i++) {

			cuFloatComplex* tmp = tmp1;
			tmp1 = tmp2;
			tmp2 = tmp;

			if (offset < V)
				mat_mulcu_upperdiag(M, N, tmp1, tmp2, in, offset, V);

			__syncthreads();

			gram_schmidtcu(M, N, evals, tmp1, tmp2);
			__syncthreads();
		}

		for (int i = 0; (i < M) && (offset < V); i++)
			for (int k = threadIdx.y; k < N; k += blockDim.y)
				out[offset + (i * N + k) * V] = tmp1[N * (M - 1 - i) + k];

		if ((offset < V) && (threadIdx.y) == 0 && vals)
			for (int i = 0; i < M; i++)
				vals[offset + i * V] = evals[M - 1 - i];
	}
}



void eigenmapscu(const long dims[5], _Complex float* optr, _Complex float* eptr, const _Complex float* imgcov2, int num_orthiter)
{
	const int N = (int) dims[3];
	const int M = (int) dims[4];

	assert(M <= N);

	long imgcov2_dims[5];
	md_select_dims(5, ~(COIL_FLAG|MAPS_FLAG), imgcov2_dims, dims);
	imgcov2_dims[3] = N * (N + 1) / 2;

	long eptr_dims[5];
	md_select_dims(5, ~COIL_FLAG, eptr_dims, dims);

	long imgcov2_df_dims[5];
	md_select_dims(5, ~(COIL_FLAG|MAPS_FLAG), imgcov2_df_dims, dims);
	imgcov2_df_dims[3] = N * N;

	static bool printed = false;
	if (!printed) {

		debug_printf(DP_INFO, "CUDA Pointwise Eigendecomposition...\n");
		printed = true;
	}

	cuFloatComplex* optr_device = (cuFloatComplex*)md_alloc_gpu(5, dims, sizeof(cuFloatComplex));
	cuFloatComplex* imgcov2_device = (cuFloatComplex*)md_alloc_gpu(5, imgcov2_dims, sizeof(cuFloatComplex));
	cuFloatComplex* eptr_device = (cuFloatComplex*)md_alloc_gpu(5, eptr_dims, sizeof(cuFloatComplex));

	md_copy(5, imgcov2_dims, imgcov2_device, imgcov2, sizeof(cuFloatComplex));


	struct cudaDeviceProp mycudaDeviceProperties;
	cudaGetDeviceProperties(&mycudaDeviceProperties, 0);
	const int maxSharedMemPerBlock = mycudaDeviceProperties.sharedMemPerBlock;
	const int maxThreadsPerBlock = mycudaDeviceProperties.maxThreadsPerBlock;
	const int memPerPoint = (2*M*N + M) * sizeof(cuFloatComplex);
	int pointsPerBlock = MIN(maxThreadsPerBlock/N, maxSharedMemPerBlock/memPerPoint);
	const int maxRegsPerBlock = mycudaDeviceProperties.regsPerBlock;
	const int maxCmemPerBlock = mycudaDeviceProperties.totalConstMem;
	// determined by --ptxas-options="-v". cmem is constant mem used for 1) kernel args, 2) user defined constants, 3) compiler-generated constants
	const int regsPerThread = 36;
	const int cmemPerThread = 108;
	pointsPerBlock = MIN(pointsPerBlock, maxRegsPerBlock / (N * regsPerThread));
	pointsPerBlock = MIN(pointsPerBlock, maxCmemPerBlock / (N * cmemPerThread));

	int tmp = pointsPerBlock;
	pointsPerBlock = 1;
	while (2 * pointsPerBlock <= tmp)
		pointsPerBlock *= 2;

	assert(pointsPerBlock > 0);

	long V = md_calc_size(3, dims);

	dim3 threads(pointsPerBlock, N, 1);
	int numBlocks = (V + (pointsPerBlock - 1)) / pointsPerBlock;
	dim3 blocks(numBlocks); // if numBlocks > ~65,000, need to distribute over x, y, z dims
	size_t sharedMem = memPerPoint * pointsPerBlock;

	eigenmapscu_kern<<<blocks, threads, sharedMem, cuda_get_stream()>>>(imgcov2_device, optr_device, eptr_device, num_orthiter, V, N, M);
	CUDA_KERNEL_ERROR;

	md_copy(5, dims, optr, optr_device, sizeof(_Complex float));
	md_copy(5, eptr_dims, eptr, eptr_device, sizeof(_Complex float));

	md_free(imgcov2_device);
	md_free(optr_device);
	md_free(eptr_device);
}















