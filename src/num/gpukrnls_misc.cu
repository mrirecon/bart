/* Copyright 2023. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <cuda_runtime_api.h>

#include "gpukrnls_misc.h"


// limited by hardware to 1024 on most devices
// should be a multiple of 32 (warp size)
#define BLOCKSIZE 1024
#define WARPSIZE 32

#define MIN(x, y) ({ __typeof(x) __x = (x); __typeof(y) __y = (y); (__x < __y) ? __x : __y; })
#define MAX(x, y) ({ __typeof(x) __x = (x); __typeof(y) __y = (y); (__x > __y) ? __x : __y; })

static long gridsize_int(long N, int blocksize)
{
	return MIN(65535, (N + blocksize - 1) / blocksize); // 65535 is maximum for y and z dim
}

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

dim3 getBlockSize3(const long dims[3], int threads)
{
	int block[3];

	getBlockSize3_internal(block, dims, threads);

	return dim3(block[0], block[1], block[2]);
}


dim3 getGridSize3(const long dims[3], int threads)
{
	int block[3];

	getBlockSize3_internal(block, dims, threads);

	return dim3(gridsize_int(dims[0], block[0]), gridsize_int(dims[1], block[1]), gridsize_int(dims[2], block[2]));
}

dim3 getBlockSize3(const long dims[3], const void* func)
{
	return getBlockSize3(dims, cuda_get_max_threads(func));
}


dim3 getGridSize3(const long dims[3], const void* func)
{
	return getGridSize3(dims, cuda_get_max_threads(func));
}


dim3 getBlockSize(long N, int threads)
{
	long dims[3] = { N, 1, 1 };
	return getBlockSize3(dims, threads);
}


dim3 getGridSize(long N, int threads)
{
	long dims[3] = { N, 1, 1 };
	return getGridSize3(dims, threads);
}

dim3 getBlockSize(long N, const void* func)
{
	long dims[3] = { N, 1, 1 };
	return getBlockSize3(dims, cuda_get_max_threads(func));
}


dim3 getGridSize(long N, const void* func)
{
	long dims[3] = { N, 1, 1 };
	return getGridSize3(dims, cuda_get_max_threads(func));
}

int cuda_get_max_threads(const void* func)
{
	struct cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, func);
	return attr.maxThreadsPerBlock;
}
