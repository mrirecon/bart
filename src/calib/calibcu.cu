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

#include "num/multind.h"
#include "num/flpmath.h"

#include "calibcu.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

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


static __device__ void gram_schmidtcu(int M, int N, cuFloatComplex* evals, cuFloatComplex* vecs)
{    
    cuFloatComplex val1;
    cuFloatComplex val2;
    for (int i = M-1; i >= 0; i--) {
        val1 = vecs[threadIdx.y + i*N];
        __syncthreads();
        for (int j = i+1; j <= M-1; j++) {
            val2 = vecs[threadIdx.y + j*N];
            __syncthreads();
            vecs[threadIdx.y + i*N] = cuCmulf(val1, cuConjf(val2));
            __syncthreads();
            if (threadIdx.y == 0) {
                cuFloatComplex tmp = make_cuFloatComplex(0.,0.);
                for (int k = 0; k < N; k++)
                    tmp = cuCaddf(tmp, vecs[k + i*N]);
                vecs[i*N] = cuFloatComplexScale(tmp, -1.);
            }
            __syncthreads();
            val1 = cuCaddf(val1, cuCmulf(val2, vecs[i*N]));
            __syncthreads();
        }
            vecs[threadIdx.y + i*N] = cuCmulf(val1, cuConjf(val1));
            __syncthreads();
            if (threadIdx.y == 0) {
                cuFloatComplex tmp = make_cuFloatComplex(0.,0.);
                for (int k = 0; k < N; k++)
                    tmp = cuCaddf(tmp, vecs[k + i*N]);
                evals[i] = make_cuFloatComplex(sqrt(cuCrealf(tmp)),0.);
            }
            __syncthreads();
            vecs[threadIdx.y + i*N] = cuFloatComplexScale(val1, 1./cuCrealf(evals[i]));
        }
}
      
static __device__ inline void mat_mulcu(int M, int N, cuFloatComplex* A, cuFloatComplex* B, cuFloatComplex* C, int offset)
{
    cuFloatComplex tmp;
    for (int i = 0; i < M; i++) {
        tmp = make_cuFloatComplex(0.,0.);
        for (int j = 0; j < N; j++)
			tmp = cuCaddf(tmp, cuCmulf(B[j + i*N], C[offset*N*N + threadIdx.y + j*N]));
		A[threadIdx.y + i*N] = tmp;
	}
}

static __global__ void eigenmapscu_kern(cuFloatComplex* in_filled, cuFloatComplex* in, cuFloatComplex* out, cuFloatComplex* vals, int iter, int x, int y, int z, int N, int M)
{
    const int offset = blockIdx.x * blockDim.x + threadIdx.x; 
    if (offset > x*y*z-1)
        return;
    extern __shared__ cuFloatComplex sdata[];
    cuFloatComplex *tmp1, *tmp2, *evals;
    tmp1 = sdata + threadIdx.x * (2*M*N + M);
    tmp2 = tmp1 + M*N;
    evals = tmp2 + M*N;

	if (threadIdx.y == 0) {
		int l = 0;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= i; j++)
				in_filled[offset*N*N + i*N + j] = in[offset + (l++)*x*y*z];
			for (int j = 0; j < i; j++)
			in_filled[offset*N*N + j*N + i] = cuConjf(in_filled[offset*N*N + i*N + j]);
		}
	}
	__syncthreads();

	for (int i = 0; i < M; i++)
	    tmp1[threadIdx.y + i*N] = (threadIdx.y == i) ? make_cuFloatComplex(1.,0.) : make_cuFloatComplex(0.,0.);
    __syncthreads();    

    for (int i = 0; i < iter; i++) {
    	
    	for (int j = 0; j < M; j++)
	        tmp2[threadIdx.y + j*N] = tmp1[threadIdx.y + j*N];
        __syncthreads();
        
        mat_mulcu(M, N, tmp1, tmp2, in_filled, offset);
        __syncthreads();
		
		gram_schmidtcu(M, N, evals, tmp1);
		__syncthreads();   
    }
    
    for (int i = 0; i < M; i++)
        out[offset + (i*N + threadIdx.y)*x*y*z] = tmp1[N * (M-1-i) + threadIdx.y];
    
    if (threadIdx.y == 0)    
        if (vals)
            for (int i = 0; i < M; i++)
                vals[offset + i*x*y*z] = evals[M-1-i];
}


        
void eigenmapscu(const long dims[5], _Complex float* optr, _Complex float* eptr, const _Complex float* imgcov2)
{
	const int iter = 30;
	const int x = (int) dims[0];
	const int y = (int) dims[1];
	const int z = (int) dims[2];
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


	printf("CUDA Pointwise Eigendecomposition...\n");
    
	cuFloatComplex* optr_device = (cuFloatComplex*)md_alloc_gpu(5, dims, sizeof(cuFloatComplex));
	cuFloatComplex* imgcov2_device = (cuFloatComplex*)md_alloc_gpu(5, imgcov2_dims, sizeof(cuFloatComplex));
	cuFloatComplex* imgcov2_device_filled = (cuFloatComplex*)md_alloc_gpu(5, imgcov2_df_dims, sizeof(cuFloatComplex));
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
	assert(pointsPerBlock > 0);    

	dim3 threads(pointsPerBlock, N, 1);
	int numBlocks = (x*y*z + (pointsPerBlock-1)) / pointsPerBlock;
	dim3 blocks(numBlocks); // if numBlocks > ~65,000, need to distribute over x, y, z dims
	size_t sharedMem = memPerPoint * pointsPerBlock;

	eigenmapscu_kern<<<blocks, threads, sharedMem>>>(imgcov2_device_filled, imgcov2_device, optr_device, eptr_device, iter, x, y, z, N, M);
	cudaThreadSynchronize();

	cudaError_t error = cudaGetLastError();

	if (error != cudaSuccess) {

		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
		abort();
	}

	md_copy(5, dims, optr, optr_device, sizeof(_Complex float));
	md_copy(5, eptr_dims, eptr, eptr_device, sizeof(_Complex float));

	md_free(imgcov2_device);
	md_free(imgcov2_device_filled);
	md_free(optr_device);
    	md_free(eptr_device);
}














    
