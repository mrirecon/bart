/* Copyright 2023. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <stdbool.h>
#include <assert.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/types.h"

#include "num/gpuops.h"
#include "num/multind.h"
#include "num/gpukrnls_misc.h"

#include "gpukrnls_copy.h"

#define MAX_COPY_DIMS 8

struct cuda_strides_ND {

	int N;

	long dims[MAX_COPY_DIMS];
	long ostrs[MAX_COPY_DIMS];
	long istrs[MAX_COPY_DIMS];

	long size;
};


typedef void(kern_copy_t)(cuda_strides_ND strs, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2);

static __device__ void md_unravel_index(int D, long* pos, const long* dims, long index)
{
	long ind = index;

	for (int d = 0; d < D; ++d) {

		pos[d] = ind % dims[d];
		ind /= dims[d];
	}
}

template <typename T, int N>
__global__ static void kern_copy_strides(cuda_strides_ND strs, T* dst, const T* src)
{
	int start[3];
	long stride[3];

	long pos[3 > N ? 3 : N];

	start[0] = (0 < N) ? threadIdx.x + blockDim.x * blockIdx.x : 0;
	start[1] = (1 < N) ? threadIdx.y + blockDim.y * blockIdx.y : 0;
	start[2] = (2 < N) ? threadIdx.z + blockDim.z * blockIdx.z : 0;

	stride[0] = (0 < N) ? blockDim.x * gridDim.x : 1;
	stride[1] = (1 < N) ? blockDim.y * gridDim.y : 1;
	stride[2] = (2 < N) ? blockDim.z * gridDim.z : 1;

	for (long i = 0; i < strs.size; i++) {

		md_unravel_index(N - 3, pos + 3, strs.dims + 3, i);

		for (pos[2] = start[2]; pos[2] < ((2 < N) ? strs.dims[2] : 1); pos[2] += stride[2]) {
			for (pos[1] = start[1]; pos[1] < ((1 < N) ? strs.dims[1] : 1); pos[1] += stride[1]) {
				for (pos[0] = start[0]; pos[0] < ((0 < N) ? strs.dims[0] : 1); pos[0] += stride[0]) {

					long o_off = 0;
					long i_off = 0;

					for (int i = 0; i < N ; i++) {

						o_off += pos[i] * strs.ostrs[i];
						i_off += pos[i] * strs.istrs[i];
					}

					dst[o_off] = src[i_off];
				}
			}
		}
	}
}

template <typename T>
static void* get_kern_fop_unfold(int N)
{
	if (1 == N)
		return (void*)kern_copy_strides<T, 1>;

	if (2 == N)
		return (void*)kern_copy_strides<T, 2>;

	if (3 == N)
		return (void*)kern_copy_strides<T, 3>;

	if (4 == N)
		return (void*)kern_copy_strides<T, 4>;

	if (5 == N)
		return (void*)kern_copy_strides<T, 5>;

	if (6 == N)
		return (void*)kern_copy_strides<T, 6>;

	if (7 == N)
		return (void*)kern_copy_strides<T, 7>;

	if (8 == N)
		return (void*)kern_copy_strides<T, 8>;

	return NULL;
}


template <typename T>
static long get_copy_maxthreads(int N)
{
	static long mt[7] = { 0 };

	if (0 == mt[N - 1])
		mt[N - 1] = cuda_get_max_threads(get_kern_fop_unfold<T>(N));

	return mt[N - 1];
}

template <typename T>
static void cuda_copy_template_unfold(int D, const long dims[], const long ostrs[], void* dst, const long istrs[], const void* src)
{
	assert(D <= MAX_COPY_DIMS);

	cuda_strides_ND strs;

	int shift = (1 == dims[0] ? 1 : 0);
	strs.N = D - shift;

	for (int i = 0; i < D - shift; i++) {

		strs.dims[i] =  dims[i + shift];
		strs.ostrs[i] = ostrs[i + shift] / sizeof(T);
		strs.istrs[i] = istrs[i + shift] / sizeof(T);
	}

	for (int i = D - shift; i < MAX_COPY_DIMS; i++) {

		strs.dims[i] = 1;
		strs.ostrs[i] = 0;
		strs.istrs[i] = 0;
	}

	strs.size = md_calc_size(MAX_COPY_DIMS - 3, strs.dims + 3);

	CUDA_ERROR_PTR(dst, src);

	dim3 ex1 = getGridSize3(strs.dims, get_copy_maxthreads<T>(strs.N));
	dim3 ex2 = getBlockSize3(strs.dims, get_copy_maxthreads<T>(strs.N));

	if (1 == strs.N)
		kern_copy_strides<T, 1><<<ex1, ex2, 0, cuda_get_stream()>>>(strs, (T*)dst, (const T*)src);

	if (2 == strs.N)
		kern_copy_strides<T, 2><<<ex1, ex2, 0, cuda_get_stream()>>>(strs, (T*)dst, (const T*)src);

	if (3 == strs.N)
		kern_copy_strides<T, 3><<<ex1, ex2, 0, cuda_get_stream()>>>(strs, (T*)dst, (const T*)src);

	if (4 == strs.N)
		kern_copy_strides<T, 4><<<ex1, ex2, 0, cuda_get_stream()>>>(strs, (T*)dst, (const T*)src);

	if (5 == strs.N)
		kern_copy_strides<T, 5><<<ex1, ex2, 0, cuda_get_stream()>>>(strs, (T*)dst, (const T*)src);

	if (6 == strs.N)
		kern_copy_strides<T, 6><<<ex1, ex2, 0, cuda_get_stream()>>>(strs, (T*)dst, (const T*)src);

	if (7 == strs.N)
		kern_copy_strides<T, 7><<<ex1, ex2, 0, cuda_get_stream()>>>(strs, (T*)dst, (const T*)src);

	if (8 == strs.N)
		kern_copy_strides<T, 8><<<ex1, ex2, 0, cuda_get_stream()>>>(strs, (T*)dst, (const T*)src);

	CUDA_KERNEL_ERROR;
}

void cuda_copy_ND(int D, const long dims[], const long ostrs[], void* dst, const long istrs[], const void* src, size_t size)
{
	assert(D <= MAX_COPY_DIMS - 1);

	size_t ele_size = 1;

	bool cont = true;

	while (cont) {

		if (0 != size % (2 * ele_size))
			cont = false;

		for (int i = 0; i < D; i++) {

			if (0 != ostrs[i] % (2 * ele_size))
				cont = false;

			if (0 != istrs[i] % (2 * ele_size))
				cont = false;
		}

		if (8 == ele_size)
			cont = false;

		if (cont)
			ele_size *= 2;
	}

	long ndims[MAX_COPY_DIMS];
	long nostrs[MAX_COPY_DIMS];
	long nistrs[MAX_COPY_DIMS];

	ndims[0] = size / ele_size;
	nostrs[0] = ele_size;
	nistrs[0] = ele_size;

	for (int i = 0; i < D; i++) {

		ndims[i + 1] = dims[i];
		nostrs[i + 1] = ostrs[i];
		nistrs[i + 1] = istrs[i];
	}

	switch (ele_size) {

	case 1:
		cuda_copy_template_unfold<uint8_t>(D + 1, ndims, nostrs, dst, nistrs, src);
		break;
	case 2:
		cuda_copy_template_unfold<uint16_t>(D + 1, ndims, nostrs, dst, nistrs, src);
		break;
	case 4:
		cuda_copy_template_unfold<uint32_t>(D + 1, ndims, nostrs, dst, nistrs, src);
		break;
	case 8:
		cuda_copy_template_unfold<uint64_t>(D + 1, ndims, nostrs, dst, nistrs, src);
		break;
	}
}


static __device__ void mmemcpy(void* dst, const void* src, size_t size)
{
	switch (size) {

		case 1:
			*((uint8_t*)dst) = *((uint8_t*)src);
			break;

		case 2:
			*((uint16_t*)dst) = *((uint16_t*)src);
			break;

		case 4:
			*((uint32_t*)dst) = *((uint32_t*)src);
			break;

		case 8:
			*((uint64_t*)dst) = *((uint64_t*)src);
			break;

		default:
			memcpy(dst, src, size);
	}
}


#define BLOCKSIZE 1024

static int blocksize(long N)
{
	return BLOCKSIZE;
}

static long gridsize(long N)
{
	// to ensure that "start" does not overflow we need to restrict gridsize!
	return MIN((N + BLOCKSIZE - 1) / BLOCKSIZE, 65536 - 1);
}


__global__ static void decompress_kern(long istride, long N, long dcstrs, void* dst, long istrs, const long* index, const void* src, size_t size)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		if (index[istrs * i] >= 0)
			mmemcpy((uint8_t*)dst + dcstrs * i, (uint8_t*)src + index[istrs * i] * istride, size);
}

extern "C" void cuda_decompress(long stride, long N, long dcstrs, void* dst, long istrs, const long* index, const void* src, size_t size)
{
	decompress_kern<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(stride, N, dcstrs, dst, istrs, index, src, size);
	CUDA_KERNEL_ERROR;
}

__global__ static void compress_kern(long istride, long N, void* dst, long istrs, const long* index, long dcstrs, const void* src, size_t size)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		if (index[istrs * i] >= 0)
			mmemcpy((uint8_t*)dst + index[istrs * i] * istride, (uint8_t*)src + dcstrs * i, size);
}

extern "C" void cuda_compress(long stride, long N, void* dst, long istrs, const long* index, long dcstrs, const void* src, size_t size)
{
	compress_kern<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(stride, N, dst, istrs, index, dcstrs, src, size);
	CUDA_KERNEL_ERROR;
}

