#include <stdbool.h>
#include <assert.h>
#include <complex.h>
#include <limits.h>
#include <stdio.h>

#include <cuda_runtime_api.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/gpuops.h"

#include "gpu_misc.h"


// limited by hardware to 1024 on most devices
// should be a multiple of 32 (warp size)
#define BLOCKSIZE 1024
#define WARPSIZE 32

static long gridsize(long N, int blocksize)
{
	return (N + blocksize - 1) / blocksize;
}

static void getBlockSize3_internal(int block[3], const long dims[3], const void* func)
{
	cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, func);
	int threads = attr.maxThreadsPerBlock;

	block[0] = WARPSIZE * (MIN(threads, dims[0] + WARPSIZE) / WARPSIZE);
	block[0] = MIN(BLOCKSIZE, block[0]);
	block[0] = MAX(1, block[0]);
	
	threads /= block[0];

	block[1] = WARPSIZE * (MIN(threads, dims[1] + WARPSIZE) / WARPSIZE);
	block[1] = MIN(BLOCKSIZE, block[1]);
	block[1] = MAX(1, block[1]);
	
	threads /= block[1];
	
	block[2] = WARPSIZE * (MIN(threads, dims[2] + WARPSIZE) / WARPSIZE);
	block[2] = MIN(BLOCKSIZE, block[2]);
	block[2] = MAX(1, block[2]);
}



dim3 getBlockSize3(const long dims[3], const void* func)
{
	int block[3];

	getBlockSize3_internal(block, dims, func);

	return dim3(block[0], block[1], block[2]);
}

dim3 getGridSize3(const long dims[3], const void* func)
{
	int block[3];

	getBlockSize3_internal(block, dims, func);

	return dim3(gridsize(dims[0], block[0]), gridsize(dims[1], block[1]), gridsize(dims[2], block[2]));
}

dim3 getBlockSize(long N, const void* func)
{
	const long dims[3] = {N, 1, 1};

	return getBlockSize3(dims, func);
}

dim3 getGridSize(long N, const void* func)
{
	const long dims[3] = {N, 1, 1};

	return getGridSize3(dims, func);
}

extern void print_dim3(dim3 dims)
{
	printf("[ %3d %3d %3d ]\n", dims.x, dims.y, dims.z);
}


