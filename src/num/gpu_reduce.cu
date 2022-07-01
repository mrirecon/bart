/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
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

#include "num/gpu_reduce.h"
#include "num/multind.h"
#include "num/gpuops.h"

#define CFL_SIZE 8
#define FL_SIZE 4

#define BLOCKSIZE 1024

static long gridsizeX(long N, unsigned int blocksize)
{
	return (N + blocksize - 1) / blocksize;
}
static unsigned int gridsizeY(long N, unsigned int blocksize)
{
	return (N + blocksize - 1) / blocksize;
}

#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a > b) ? a : b)

__device__ static __inline__ cuFloatComplex dev_zadd(cuFloatComplex arg1, cuFloatComplex arg2)
{
	return cuCaddf(arg1, arg2);
}

__device__ static __inline__ void dev_atomic_zadd(cuFloatComplex* arg, cuFloatComplex val)
{
	atomicAdd(&(arg->x), val.x);
	atomicAdd(&(arg->y), val.y);
}

__global__ static void kern_reduce_zadd_outer(long dim_reduce, long dim_batch, cuFloatComplex* dst, const cuFloatComplex* src)
{
	extern __shared__ cuFloatComplex sdata_c[];

	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	int idxx = blockIdx.x * blockDim.x + threadIdx.x;
	int idxy = blockIdx.y * blockDim.y + threadIdx.y;

	for (long ix = idxx; ix < dim_batch; ix += gridDim.x * blockDim.x){

		sdata_c[tidy * blockDim.x + tidx] = src[ idxy * dim_batch + ix];

		for (long j = blockDim.y * gridDim.y + idxy; j < dim_reduce; j += blockDim.y * gridDim.y)
			sdata_c[tidy * blockDim.x + tidx] = dev_zadd(sdata_c[tidy * blockDim.x + tidx], src[j * dim_batch + ix]);

		__syncthreads();

		for (unsigned int s = blockDim.y / 2; s > 0; s >>= 1){

			if (tidy < s)
				sdata_c[tidy * blockDim.x + tidx] = dev_zadd(sdata_c[tidy * blockDim.x + tidx], sdata_c[(tidy + s) * blockDim.x + tidx]);
			__syncthreads();
		}

		if (0 == tidy) dev_atomic_zadd(dst + ix, sdata_c[tidx]);
	}
}

extern "C" void cuda_reduce_zadd_outer(long dim_reduce, long dim_batch, _Complex float* dst, const _Complex float* src)
{
	long maxBlockSizeX_dim = 1;
	while (maxBlockSizeX_dim < dim_batch)
		maxBlockSizeX_dim *= 2;

	long maxBlockSizeY_dim = 1;
	while (8 * maxBlockSizeY_dim < dim_reduce)
		maxBlockSizeY_dim *= 2;

	long maxBlockSizeX_gpu = 32;
	unsigned int blockSizeX = MIN(maxBlockSizeX_gpu, maxBlockSizeX_dim);
	unsigned int blockSizeY = MIN(maxBlockSizeY_dim, BLOCKSIZE / blockSizeX);


	dim3 blockDim(blockSizeX, blockSizeY);
    	dim3 gridDim(gridsizeX(dim_batch, blockSizeX), gridsizeY(maxBlockSizeY_dim, blockSizeY));

	kern_reduce_zadd_outer<<<gridDim, blockDim, blockSizeX * blockSizeY * CFL_SIZE, cuda_get_stream()>>>(dim_reduce, dim_batch, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}


__global__ static void kern_reduce_zadd_inner(long dim_reduce, long dim_batch, cuFloatComplex* dst, const cuFloatComplex* src)
{
	extern __shared__ cuFloatComplex sdata_c[];

	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	int idxx = blockIdx.x * blockDim.x + threadIdx.x;
	int idxy = blockIdx.y * blockDim.y + threadIdx.y;

	for (long iy = idxy; iy < dim_batch; iy += gridDim.y * blockDim.y){

		sdata_c[tidy * blockDim.x + tidx] = src[ idxx + dim_reduce * iy];

		//printf("%d %ld\n", idxx, iy);

		for (long j = blockDim.x * gridDim.x + idxx; j < dim_reduce; j += blockDim.x * gridDim.x)
			sdata_c[tidy * blockDim.x + tidx] = dev_zadd(sdata_c[tidy * blockDim.x + tidx], src[j + dim_reduce * iy]);

		__syncthreads();

		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1){

			if (tidx < s)
				sdata_c[tidy * blockDim.x + tidx] = dev_zadd(sdata_c[tidy * blockDim.x + tidx], sdata_c[tidy * blockDim.x + tidx + s]);
			__syncthreads();
		}

		if (0 == tidx) dev_atomic_zadd(dst + iy, sdata_c[tidy * blockDim.x]);
	}
}

extern "C" void cuda_reduce_zadd_inner(long dim_reduce, long dim_batch, _Complex float* dst, const _Complex float* src)
{
	long maxBlockSizeX_dim = 1;
	while (8 * maxBlockSizeX_dim < dim_reduce)
		maxBlockSizeX_dim *= 2;

	long maxBlockSizeY_dim = 1;
	while (maxBlockSizeY_dim < dim_batch)
		maxBlockSizeY_dim *= 2;

	long maxBlockSizeX_gpu = 32;
	unsigned int blockSizeX = MIN(maxBlockSizeX_gpu, maxBlockSizeX_dim);
	unsigned int blockSizeY = MIN(maxBlockSizeY_dim, BLOCKSIZE / blockSizeX);

	dim3 blockDim(blockSizeX, blockSizeY);
    	dim3 gridDim(gridsizeX(maxBlockSizeX_dim, blockSizeX), gridsizeY(dim_batch, blockSizeY));

	kern_reduce_zadd_inner<<<gridDim, blockDim, blockSizeX * blockSizeY * CFL_SIZE, cuda_get_stream()>>>(dim_reduce, dim_batch, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}

__device__ static __inline__ float dev_add(float arg1, float arg2)
{
	return arg1 + arg2;
}

__device__ static __inline__ void dev_atomic_add(float* arg, float val)
{
	atomicAdd(arg, val);
}

__global__ static void kern_reduce_add_outer(long dim_reduce, long dim_batch, float* dst, const float* src)
{
	extern __shared__ float sdata_s[];

	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	int idxx = blockIdx.x * blockDim.x + threadIdx.x;
	int idxy = blockIdx.y * blockDim.y + threadIdx.y;

	for (long ix = idxx; ix < dim_batch; ix += gridDim.x * blockDim.x){

		sdata_s[tidy * blockDim.x + tidx] = src[ idxy * dim_batch + ix];

		for (long j = blockDim.y * gridDim.y + idxy; j < dim_reduce; j += blockDim.y * gridDim.y)
			sdata_s[tidy * blockDim.x + tidx] = dev_add(sdata_s[tidy * blockDim.x + tidx], src[j * dim_batch + ix]);

		__syncthreads();

		for (unsigned int s = blockDim.y / 2; s > 0; s >>= 1){

			if (tidy < s)
				sdata_s[tidy * blockDim.x + tidx] = dev_add(sdata_s[tidy * blockDim.x + tidx], sdata_s[(tidy + s) * blockDim.x + tidx]);
			__syncthreads();
		}

		if (0 == tidy) dev_atomic_add(dst + ix, sdata_s[tidx]);
	}
}

extern "C" void cuda_reduce_add_outer(long dim_reduce, long dim_batch, float* dst, const float* src)
{
	long maxBlockSizeX_dim = 1;
	while (maxBlockSizeX_dim < dim_batch)
		maxBlockSizeX_dim *= 2;

	long maxBlockSizeY_dim = 1;
	while (8 * maxBlockSizeY_dim < dim_reduce)
		maxBlockSizeY_dim *= 2;

	long maxBlockSizeX_gpu = 32;
	unsigned int blockSizeX = MIN(maxBlockSizeX_gpu, maxBlockSizeX_dim);
	unsigned int blockSizeY = MIN(maxBlockSizeY_dim, BLOCKSIZE / blockSizeX);


	dim3 blockDim(blockSizeX, blockSizeY);
    	dim3 gridDim(gridsizeX(dim_batch, blockSizeX), gridsizeY(maxBlockSizeY_dim, blockSizeY));

	kern_reduce_add_outer<<<gridDim, blockDim, blockSizeX * blockSizeY * FL_SIZE, cuda_get_stream()>>>(dim_reduce, dim_batch, dst, src);
}


__global__ static void kern_reduce_add_inner(long dim_reduce, long dim_batch, float* dst, const float* src)
{
	extern __shared__ float sdata_s[];

	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	int idxx = blockIdx.x * blockDim.x + threadIdx.x;
	int idxy = blockIdx.y * blockDim.y + threadIdx.y;

	for (long iy = idxy; iy < dim_batch; iy += gridDim.y * blockDim.y){

		sdata_s[tidy * blockDim.x + tidx] = src[ idxx + dim_reduce * iy];

		//printf("%d %ld\n", idxx, iy);

		for (long j = blockDim.x * gridDim.x + idxx; j < dim_reduce; j += blockDim.x * gridDim.x)
			sdata_s[tidy * blockDim.x + tidx] = dev_add(sdata_s[tidy * blockDim.x + tidx], src[j + dim_reduce * iy]);

		__syncthreads();

		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1){

			if (tidx < s)
				sdata_s[tidy * blockDim.x + tidx] = dev_add(sdata_s[tidy * blockDim.x + tidx], sdata_s[tidy * blockDim.x + tidx + s]);
			__syncthreads();
		}

		if (0 == tidx) dev_atomic_add(dst + iy, sdata_s[tidy * blockDim.x]);
	}
}

extern "C" void cuda_reduce_add_inner(long dim_reduce, long dim_batch, float* dst, const float* src)
{
	long maxBlockSizeX_dim = 1;
	while (8 * maxBlockSizeX_dim < dim_reduce)
		maxBlockSizeX_dim *= 2;

	long maxBlockSizeY_dim = 1;
	while (maxBlockSizeY_dim < dim_batch)
		maxBlockSizeY_dim *= 2;

	long maxBlockSizeX_gpu = 32;
	unsigned int blockSizeX = MIN(maxBlockSizeX_gpu, maxBlockSizeX_dim);
	unsigned int blockSizeY = MIN(maxBlockSizeY_dim, BLOCKSIZE / blockSizeX);

	dim3 blockDim(blockSizeX, blockSizeY);
    	dim3 gridDim(gridsizeX(maxBlockSizeX_dim, blockSizeX), gridsizeY(dim_batch, blockSizeY));

	kern_reduce_add_inner<<<gridDim, blockDim, blockSizeX * blockSizeY * FL_SIZE, cuda_get_stream()>>>(dim_reduce, dim_batch, dst, src);
}



__device__ static __inline__ cuFloatComplex dev_zmax(cuFloatComplex arg1, cuFloatComplex arg2)
{
	return make_cuFloatComplex(MAX(cuCrealf(arg1), cuCrealf(arg2)), 0.);
}

__device__ static __inline__ void dev_atomic_zmax(cuFloatComplex* arg, cuFloatComplex val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)arg;

	unsigned long long int old_ull = *address_as_ull;
	unsigned long long int assumed;
	unsigned long long int new_ull;
	cuFloatComplex new_cf;

	do {
		assumed = old_ull;
		new_cf = dev_zmax(*((cuFloatComplex*)(&old_ull)), val);
		new_ull = *((unsigned long long int*)(&new_cf));
		old_ull = atomicCAS(address_as_ull, assumed, new_ull);

	} while (assumed != old_ull);
}

__global__ static void kern_reduce_zmax_outer(long dim_reduce, long dim_batch, cuFloatComplex* dst, const cuFloatComplex* src)
{
	extern __shared__ cuFloatComplex sdata_c[];

	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	int idxx = blockIdx.x * blockDim.x + threadIdx.x;
	int idxy = blockIdx.y * blockDim.y + threadIdx.y;

	for (long ix = idxx; ix < dim_batch; ix += gridDim.x * blockDim.x){

		sdata_c[tidy * blockDim.x + tidx] = src[ idxy * dim_batch + ix];

		for (long j = blockDim.y * gridDim.y + idxy; j < dim_reduce; j += blockDim.y * gridDim.y)
			sdata_c[tidy * blockDim.x + tidx] = dev_zmax(sdata_c[tidy * blockDim.x + tidx], src[j * dim_batch + ix]);

		__syncthreads();

		for (unsigned int s = blockDim.y / 2; s > 0; s >>= 1){

			if (tidy < s)
				sdata_c[tidy * blockDim.x + tidx] = dev_zmax(sdata_c[tidy * blockDim.x + tidx], sdata_c[(tidy + s) * blockDim.x + tidx]);
			__syncthreads();
		}

		if (0 == tidy) dev_atomic_zmax(dst + ix, sdata_c[tidx]);
	}
}

extern "C" void cuda_reduce_zmax_outer(long dim_reduce, long dim_batch, _Complex float* dst, const _Complex float* src)
{
	long maxBlockSizeX_dim = 1;
	while (maxBlockSizeX_dim < dim_batch)
		maxBlockSizeX_dim *= 2;

	long maxBlockSizeY_dim = 1;
	while (8 * maxBlockSizeY_dim < dim_reduce)
		maxBlockSizeY_dim *= 2;

	long maxBlockSizeX_gpu = 32;
	unsigned int blockSizeX = MIN(maxBlockSizeX_gpu, maxBlockSizeX_dim);
	unsigned int blockSizeY = MIN(maxBlockSizeY_dim, BLOCKSIZE / blockSizeX);


	dim3 blockDim(blockSizeX, blockSizeY);
    	dim3 gridDim(gridsizeX(dim_batch, blockSizeX), gridsizeY(maxBlockSizeY_dim, blockSizeY));

	kern_reduce_zmax_outer<<<gridDim, blockDim, blockSizeX * blockSizeY * CFL_SIZE, cuda_get_stream()>>>(dim_reduce, dim_batch, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}


__global__ static void kern_reduce_zmax_inner(long dim_reduce, long dim_batch, cuFloatComplex* dst, const cuFloatComplex* src)
{
	extern __shared__ cuFloatComplex sdata_c[];

	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	int idxx = blockIdx.x * blockDim.x + threadIdx.x;
	int idxy = blockIdx.y * blockDim.y + threadIdx.y;

	for (long iy = idxy; iy < dim_batch; iy += gridDim.y * blockDim.y){

		sdata_c[tidy * blockDim.x + tidx] = src[ idxx + dim_reduce * iy];

		//printf("%d %ld\n", idxx, iy);

		for (long j = blockDim.x * gridDim.x + idxx; j < dim_reduce; j += blockDim.x * gridDim.x)
			sdata_c[tidy * blockDim.x + tidx] = dev_zmax(sdata_c[tidy * blockDim.x + tidx], src[j + dim_reduce * iy]);

		__syncthreads();

		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1){

			if (tidx < s)
				sdata_c[tidy * blockDim.x + tidx] = dev_zmax(sdata_c[tidy * blockDim.x + tidx], sdata_c[tidy * blockDim.x + tidx + s]);
			__syncthreads();
		}

		if (0 == tidx) dev_atomic_zmax(dst + iy, sdata_c[tidy * blockDim.x]);
	}
}

extern "C" void cuda_reduce_zmax_inner(long dim_reduce, long dim_batch, _Complex float* dst, const _Complex float* src)
{
	long maxBlockSizeX_dim = 1;
	while (8 * maxBlockSizeX_dim < dim_reduce)
		maxBlockSizeX_dim *= 2;

	long maxBlockSizeY_dim = 1;
	while (maxBlockSizeY_dim < dim_batch)
		maxBlockSizeY_dim *= 2;

	long maxBlockSizeX_gpu = 32;
	unsigned int blockSizeX = MIN(maxBlockSizeX_gpu, maxBlockSizeX_dim);
	unsigned int blockSizeY = MIN(maxBlockSizeY_dim, BLOCKSIZE / blockSizeX);

	dim3 blockDim(blockSizeX, blockSizeY);
    	dim3 gridDim(gridsizeX(maxBlockSizeX_dim, blockSizeX), gridsizeY(dim_batch, blockSizeY));

	kern_reduce_zmax_inner<<<gridDim, blockDim, blockSizeX * blockSizeY * CFL_SIZE, cuda_get_stream()>>>(dim_reduce, dim_batch, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}