/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2014 Joseph Y Cheng <jycheng@stanford.edu>
 *
 * 
 * CUDA support functions. The file exports gpu_ops of type struct vec_ops
 * for basic operations on single-precision floating pointer vectors defined
 * in gpukrnls.cu. See vecops.c for the CPU version.
 */

#ifdef USE_CUDA

#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas.h>

#include "num/vecops.h"
#include "num/gpuops.h"
#include "num/gpukrnls.h"

#include "misc/misc.h"

#include "gpuops.h"

#if 1
#define CUDA_MEMCACHE
#endif

#if 1
#define ACCOUNTING
#endif

static void cuda_error(int line, cudaError_t code)
{
	const char *err_str = cudaGetErrorString(code);
	fprintf(stderr, "cuda error: %d %s \n", line, err_str);
	abort();
}

#define CUDA_ERROR(x)	({ cudaError_t errval = (x); if (cudaSuccess != errval) cuda_error(__LINE__, errval); })

int cuda_devices(void)
{
	int count;
	CUDA_ERROR(cudaGetDeviceCount(&count));
	return count;
}

#ifdef CUDA_MEMCACHE
static __thread int last_init = -1;
#endif

void cuda_init(int device)
{
#ifdef CUDA_MEMCACHE
	last_init = device;
#endif
	CUDA_ERROR(cudaSetDevice(device));
}

int cuda_init_memopt(void) 
{
	int num_devices = cuda_devices();
	int device;
	int max_device = 0;

	if (num_devices > 1) {

		size_t mem_max = 0;
		size_t mem_free;
		size_t mem_total;

		for (device = 0; device < num_devices; device++) {

			cuda_init(device);
			CUDA_ERROR(cudaMemGetInfo(&mem_free,&mem_total));
			//printf(" device (%d): %d\n", device, mem_available);

			if (mem_max < mem_free) {

				mem_max = mem_free;
				max_device = device;
			}
		}
		//printf(" max device: %d\n", max_device);
		CUDA_ERROR(cudaSetDevice(max_device));
	}

	return max_device;
}



void cuda_clear(long size, void* dst)
{
//	printf("CLEAR %x %ld\n", dst, size);
	CUDA_ERROR(cudaMemset(dst, 0, size));
}

static void cuda_float_clear(long size, float* dst)
{
	cuda_clear(size * sizeof(float), (void*)dst);
}

void cuda_memcpy(long size, void* dst, const void* src)
{
//	printf("COPY %x %x %ld\n", dst, src, size);
	CUDA_ERROR(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
}


void cuda_memcpy_strided(const long dims[2], long ostr, void* dst, long istr, const void* src)
{
	CUDA_ERROR(cudaMemcpy2D(dst, ostr, src, istr, dims[0], dims[1], cudaMemcpyDefault));
}

static void cuda_float_copy(long size, float* dst, const float* src)
{
	cuda_memcpy(size * sizeof(float), (void*)dst, (const void*)src);
}


#ifdef ACCOUNTING
struct cuda_mem_s {

	const void* ptr;
	size_t len;
	bool device;
#ifdef CUDA_MEMCACHE
	bool free;
	int device_id;
#endif
	struct cuda_mem_s* next;
};

//struct cuda_mem_s init = { NULL, 0, false, NULL };
static struct cuda_mem_s* cuda_mem_list = NULL;//&init;

static struct cuda_mem_s* search(const void* ptr, bool remove)
{
	struct cuda_mem_s* rptr = NULL;

	#pragma omp critical
	{
	struct cuda_mem_s** nptr = &cuda_mem_list;

	while (true) {

		rptr = *nptr;

		if (NULL == rptr)
			break;

		if ((ptr >= rptr->ptr) && (ptr < rptr->ptr + rptr->len)) {

			if (remove)
				*nptr = rptr->next;

			break;
		}

		nptr = &(rptr->next);
	}
	}

	return rptr;
}

#ifdef CUDA_MEMCACHE
static struct cuda_mem_s* find_free(size_t size)
{
	struct cuda_mem_s* rptr = NULL;

	#pragma omp critical
	{
	struct cuda_mem_s** nptr = &cuda_mem_list;

	while (true) {

		rptr = *nptr;

		if (NULL == rptr)
			break;

		if (rptr->free && (rptr->device_id == last_init) && (rptr->len >= size)) {

			rptr->free = false;
			break;
		}

		nptr = &(rptr->next);
	}
	}

	return rptr;
}
#endif

static void insert(const void* ptr, size_t len, bool device)
{
	#pragma omp critical
	{
	struct cuda_mem_s* nptr = xmalloc(sizeof(struct cuda_mem_s));
	nptr->ptr = ptr;
	nptr->len = len;
	nptr->device = device;
	nptr->next = cuda_mem_list;
#ifdef CUDA_MEMCACHE
	nptr->device_id = last_init;
	nptr->free = false;
#endif
	cuda_mem_list = nptr;
	}
}
#endif

void cuda_exit(void)
{
#ifdef CUDA_MEMCACHE
	struct cuda_mem_s* nptr;

	while (NULL != (nptr = find_free(0))) {

		assert(nptr->device);
		assert(!nptr->free);
		cudaFree((void*)nptr->ptr);
		free(nptr);
	}
#endif
	CUDA_ERROR(cudaThreadExit());
}

bool cuda_ondevice(const void* ptr)
{
	if (NULL == ptr)
		return false;
#ifdef ACCOUNTING
	struct cuda_mem_s* p = search(ptr, false);	
#if 0
	if (p != NULL) {

		if (p->device)
			printf("device %x\n", ptr);
		else
			printf("host %x\n", ptr);
	} else
		printf("not found %x\n", ptr);
#endif
	return ((NULL != p) && p->device);
#else
	struct cudaPointerAttributes attr;
	//CUDA_ERROR(cudaPointerGetAttributes(&attr, ptr));
	if (cudaSuccess != (cudaPointerGetAttributes(&attr, ptr)))
		return false;

	return (cudaMemoryTypeDevice == attr.memoryType);
#endif
}

bool cuda_accessible(const void* ptr)
{
#ifdef ACCOUNTING
	struct cuda_mem_s* p = search(ptr, false);	

#if 0
	if (p != NULL) {

		if (p->device)
			printf("device %x\n", ptr);
		else
			printf("host %x\n", ptr);
	} else
		printf("not found %x\n", ptr);
#endif

	return (NULL != p);
#else
	struct cudaPointerAttributes attr;
	//CUDA_ERROR(cudaPointerGetAttributes(&attr, ptr));
	if (cudaSuccess != (cudaPointerGetAttributes(&attr, ptr)))
		return false;

	return true;
#endif
}



void cuda_free(void* ptr)
{
#ifdef CUDA_MEMCACHE
	struct cuda_mem_s* nptr = search(ptr, false);
	assert(NULL != nptr);
	assert(nptr->ptr == ptr);
	assert(nptr->device);
	assert(!nptr->free);
	nptr->free = true;
#else
#ifdef ACCOUNTING
	struct cuda_mem_s* nptr = search(ptr, true);
	assert(NULL != nptr);
	assert(nptr->ptr == ptr);
	assert(nptr->device);
	free(nptr);
#endif
	cudaFree(ptr);
#endif
}

void* cuda_malloc(long size)
{
#ifdef CUDA_MEMCACHE
	struct cuda_mem_s* nptr = find_free(size);

	if (NULL != nptr) {

		assert(nptr->device);
		assert(!nptr->free);
		return (void*)(nptr->ptr);
	}
#endif
	void* ptr;
        CUDA_ERROR(cudaMalloc(&ptr, size));

//	printf("DEVICE %x %ld\n", ptr, size);
#ifdef ACCOUNTING
	insert(ptr, size, true);
#endif
	return ptr;
}




void* cuda_hostalloc(long N)
{
	void* ptr;
	if (cudaSuccess != cudaHostAlloc(&ptr, N, cudaHostAllocDefault))
		abort();

//	printf("HOST %x %ld\n", ptr, N);
#ifdef ACCOUNTING
	insert(ptr, N, false);
#endif
	return ptr;
}

void cuda_hostfree(void* ptr)
{
#ifdef ACCOUNTING
	struct cuda_mem_s* nptr = search(ptr, true);
	assert(nptr->ptr == ptr);
	assert(!nptr->device);
	free(nptr);
#endif
	cudaFreeHost(ptr);
}

static float* cuda_float_malloc(long size)
{
	return (float*)cuda_malloc(size * sizeof(float));
}

static void cuda_float_free(float* x)
{
	cuda_free((void*)x);
}

static double cuda_sdot(long size, const float* src1, const float* src2)
{
	assert(cuda_ondevice(src1));
	assert(cuda_ondevice(src2));
//	printf("SDOT %x %x %ld\n", src1, src2, size);
	return cublasSdot(size, src1, 1, src2, 1);
}


static double cuda_norm(long size, const float* src1)
{
#if 1
	// cublasSnrm2 produces NaN in some situations
	// e.g. nlinv -g -i8 utests/data/und2x2 o 
	// git rev: ab28a9a953a80d243511640b23501f964a585349
//	printf("cublas: %f\n", cublasSnrm2(size, src1, 1));
//	printf("GPU norm (sdot: %f)\n", sqrt(cuda_sdot(size, src1, src1)));
	return sqrt(cuda_sdot(size, src1, src1));
#else
	return cublasSnrm2(size, src1, 1);
#endif
}


static double cuda_asum(long size, const float* src)
{
	return cublasSasum(size, src, 1);
}


static void cuda_saxpy(long size, float* y, float alpha, const float* src)
{       
//	printf("SAXPY %x %x %ld\n", y, src, size);
        cublasSaxpy(size, alpha, src, 1, y, 1);
}

static void cuda_swap(long size, float* a, float* b)
{       
        cublasSswap(size, a, 1, b, 1);
}

const struct vec_ops gpu_ops = {

	.allocate = cuda_float_malloc,
	.del = cuda_float_free,
	.clear = cuda_float_clear,
	.copy = cuda_float_copy,
	.float2double = cuda_float2double,
	.double2float = cuda_double2float,
	.dot = cuda_sdot,
	.norm = cuda_norm,
	.asum = cuda_asum,
	.zl1norm = NULL,
	.axpy = cuda_saxpy,
	.xpay = cuda_xpay,
	.smul = cuda_smul,

	.add = cuda_add,
	.sub = cuda_sub,
	.mul = cuda_mul,
	.div = cuda_div,
	.fmac = cuda_fmac,
	.fmac2 = cuda_fmac2,

	.pow = cuda_pow,
	.sqrt = cuda_sqrt,

	.le = cuda_le,

	.zmul = cuda_zmul,
	.zdiv = cuda_zdiv,
	.zfmac = cuda_zfmac,
	.zfmac2 = cuda_zfmac2,
	.zmulc = cuda_zmulc,
	.zfmacc = cuda_zfmacc,
	.zfmacc2 = cuda_zfmacc2,

	.zpow = cuda_zpow,
	.zphsr = cuda_zphsr,
	.zconj = cuda_zconj,

	.zcmp = cuda_zcmp,
	.zdiv_reg = cuda_zdiv_reg,

	.zsoftthresh = cuda_zsoftthresh,
	.zsoftthresh_half = cuda_zsoftthresh_half,
	.softthresh = cuda_softthresh,
	.softthresh_half = cuda_softthresh_half,

	.swap = cuda_swap,
};


// defined in iter/vec.h
struct vec_iter_s {

	float* (*allocate)(long N);
	void (*del)(float* x);
	void (*clear)(long N, float* x);
	void (*copy)(long N, float* a, const float* x);
	void (*swap)(long N, float* a, float* x);

	double (*norm)(long N, const float* x);
	double (*dot)(long N, const float* x, const float* y);

	void (*sub)(long N, float* a, const float* x, const float* y);
	void (*add)(long N, float* a, const float* x, const float* y);

	void (*smul)(long N, float alpha, float* a, const float* x);
	void (*xpay)(long N, float alpha, float* a, const float* x);
	void (*axpy)(long N, float* a, float alpha, const float* x);
};

extern const struct vec_iter_s gpu_iter_ops;
const struct vec_iter_s gpu_iter_ops = {

	.allocate = cuda_float_malloc,
	.del = cuda_float_free,
	.clear = cuda_float_clear,
	.copy = cuda_float_copy,
	.dot = cuda_sdot,
	.norm = cuda_norm,
	.axpy = cuda_saxpy,
	.xpay = cuda_xpay,
	.smul = cuda_smul,
	.add = cuda_add,
	.sub = cuda_sub,
	.swap = cuda_swap,
};



#endif

