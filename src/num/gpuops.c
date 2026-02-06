/* Copyright 2013-2018. The Regents of the University of California.
 * Copyright 2014. Joseph Y Cheng.
 * Copyright 2016-2022. Uecker Lab. University Center Göttingen.
 * Copyright 2023. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2023	Martin Uecker
 * 2014 	Joseph Y Cheng
 * 2015-2018	Jon Tamir
 *
 *
 * CUDA support functions. The file exports gpu_ops of type struct vec_ops
 * for basic operations on single-precision floating pointer vectors defined
 * in gpukrnls.cu. See vecops.c for the CPU version.
 */

#ifdef USE_CUDA

#include <stdbool.h>
#include <assert.h>
#include <complex.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "num/vecops.h"
#include "num/gpukrnls.h"
#include "num/gpukrnls_bat.h"
#include "num/mem.h"
#include "num/multind.h"
#include "num/blas.h"
#include "num/rand.h"
#include "num/mpi_ops.h"
#include "num/vptr.h"

#ifdef USE_CUDNN
#include "num/cudnn_wrapper.h"
#endif

#include "misc/misc.h"
#include "misc/debug.h"

#include "gpuops.h"

#define MiBYTE (1024 * 1024)

static int cuda_stream_level = -1;
cudaStream_t cuda_streams[CUDA_MAX_STREAMS + 1];

static int cuda_device_id = -1;
static _Thread_local int cuda_device_id_thread = -1;

bool cuda_memcache = true;
bool cuda_global_memory = false;
int cuda_num_streams = 1;


void cuda_error(const char* file, int line, cudaError_t code)
{
	const char *err_str = cudaGetErrorString(code);
	error("CUDA Error: %s in %s:%d\n", err_str, file, line);
}

void cuda_gpu_check(const char* file, int line, const char* note)
{
#ifdef GPU_ASSERTS
	cudaError_t code = cudaStreamSynchronize(cuda_get_stream());

	if (cudaSuccess != code) {

		const char *err_str = cudaGetErrorString(code);

		if (0 == strlen(note))
			error("CUDA Error: %s in %s:%d\n", err_str, file, line);
		else
			error("CUDA Error: %s in %s:%d (%s)\n", err_str, file, line, note);
	}
#else
	(void)file;
	(void)line;
	(void)note;
#endif
}

void cuda_check_ptr(const char* file, int line, int N, const void* ptr[N])
{
#ifdef GPU_ASSERTS
	bool same_device = true;

	for (int i = 0; i < N; i++)
		if (!cuda_ondevice(ptr[i]))
			error("CUDA Error: Pointer not on device in %s:%d", file, line);
#else
	(void)file;
	(void)line;
	(void)N;
	(void)ptr;
#endif
}

// Print free and used memory on GPU.
void print_cuda_meminfo(void)
{
	size_t byte_tot;
	size_t byte_free;

	CUDA_ERROR(cudaMemGetInfo(&byte_free, &byte_tot));

	double dbyte_tot = (double)byte_tot;
	double dbyte_free = (double)byte_free;
	double dbyte_used = dbyte_tot - dbyte_free;

	debug_printf(DP_INFO , "GPU memory usage: used = %.4f MiB, free = %.4f MiB, total = %.4f MiB\n", dbyte_used/MiBYTE, dbyte_free/MiBYTE, dbyte_tot/MiBYTE);
}



//*************************************** CUDA Device Initialization *********************************************


static bool cuda_try_init(int device)
{
	// prevent reinitialization
	if (-1 != cuda_device_id)
		return true;

	cudaError_t errval = cudaSetDevice(device);

	if (cudaSuccess == errval)
		errval = cudaDeviceSynchronize();

	if (cudaSuccess == errval) {

		cuda_device_id = device;

		cuda_streams[CUDA_MAX_STREAMS] = cudaStreamLegacy;

		for (int i = 0; i < CUDA_MAX_STREAMS; i++)
			CUDA_ERROR(cudaStreamCreate(&(cuda_streams[i])));

		memcache_init();
		cublas_init();
#ifdef USE_CUDNN
		cudnn_init();
#endif
		return true;

	}

	if (cudaErrorDevicesUnavailable != errval) {

		const char *err_str = cudaGetErrorString(errval);
		debug_printf(DP_WARN, "Device %d could not be initialized: \"%s\"\n", device, err_str);
	}

	// clear last error
	cudaGetLastError();

	return false;
}

void cuda_init(void)
{
	int count;
	CUDA_ERROR(cudaGetDeviceCount(&count));

	int off = mpi_get_rank();

#pragma omp critical
	{
	for (int device = off; device < (count + off); ++device)
		if (cuda_try_init(device % count))
			break;
	}

	if (-1 == cuda_device_id)
		error("Could not allocate any GPU device!\n");

	cuda_device_id_thread = cuda_device_id;
}


int cuda_get_device_id(void)
{
	return cuda_device_id;
}


void cuda_exit(void)
{
	cuda_memcache_clear();
	memcache_destroy();
	cublas_deinit();
#ifdef USE_CUDNN
	cudnn_deinit();
#endif
	for (int i = 0; i < CUDA_MAX_STREAMS; i++)
		CUDA_ERROR(cudaStreamDestroy(cuda_streams[i]));

	cuda_device_id = -1;
	cuda_device_id_thread = -1;

	CUDA_ERROR(cudaDeviceReset());
}


int cuda_get_stream_id(void)
{
	if (-1 == cuda_device_id)
		error("CUDA not initialized!\n");

	if (cuda_device_id != cuda_device_id_thread) {

		CUDA_ERROR(cudaSetDevice(cuda_device_id));
		cuda_device_id_thread = cuda_device_id;
	}

#ifdef _OPENMP
	if (omp_get_level() < cuda_stream_level)
		cuda_stream_level = -1;
#endif

	if (-1 == cuda_stream_level)
		return CUDA_MAX_STREAMS;

#ifdef _OPENMP
	return (0 < CUDA_MAX_STREAMS) ? omp_get_ancestor_thread_num(cuda_stream_level) % CUDA_MAX_STREAMS : 0;
#else
	return 0;
#endif
}

bool cuda_is_stream_default(void)
{
	if (-1 == cuda_device_id)
		return true;

	return CUDA_MAX_STREAMS == cuda_get_stream_id();
}



int cuda_set_stream_level(void)
{
#ifdef _OPENMP
	if (0 < omp_get_active_level())
		return 1;

	if (1 == cuda_num_streams)
		return 1;

	if (-1 == cuda_stream_level)
		cuda_stream_level = omp_get_level() + 1;

	return MIN(cuda_num_streams, CUDA_MAX_STREAMS);
#else
	return 1;
#endif
}

cudaStream_t cuda_get_stream_by_id(int id)
{
	return cuda_streams[id];
}

cudaStream_t cuda_get_stream(void)
{
	return cuda_streams[cuda_get_stream_id()];
}


void cuda_sync_device(void)
{
	// do not initialize gpu just for syncing
	if (-1 == cuda_device_id)
		return;

	CUDA_ERROR(cudaDeviceSynchronize());
}

void cuda_sync_stream(void)
{
	// do not initialize gpu just for syncing
	if (-1 == cuda_device_id)
		return;

	CUDA_ERROR(cudaStreamSynchronize(cuda_get_stream()));
}


//*************************************** Memory Management *********************************************


static void* cuda_malloc_wrapper(size_t size)
{
	if (-1 == cuda_device_id)
		error("CUDA_ERROR: No gpu initialized, run \"num_init_gpu\"!\n");

	if (cuda_device_id == cuda_device_id_thread) {

		CUDA_ERROR(cudaSetDevice(cuda_device_id));
		cuda_device_id_thread = cuda_device_id;
	}

	void* ptr;

	if (cuda_global_memory) {

		CUDA_ERROR(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));

		int access;
		CUDA_ERROR(cudaDeviceGetAttribute(&access, cudaDevAttrConcurrentManagedAccess, cuda_device_id));

		if (0 != access) {
#if 13000 <= CUDART_VERSION
			struct cudaMemLocation mloc;
			mloc.type = cudaMemLocationTypeDevice;
			mloc.id = cuda_device_id;
			CUDA_ERROR(cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, mloc));
			CUDA_ERROR(cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, mloc));
			CUDA_ERROR(cudaMemPrefetchAsync(ptr, size, mloc, 0U, cuda_get_stream()));
#else
			CUDA_ERROR(cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, cuda_device_id));
			CUDA_ERROR(cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, cuda_device_id));
			CUDA_ERROR(cudaMemPrefetchAsync(ptr, size, cuda_device_id, cuda_get_stream()));
#endif
		}

	} else {

		cudaError_t err = cudaMalloc(&ptr, size);

		if (cudaSuccess != err) {

			if (cudaErrorMemoryAllocation == err) {

				cuda_memcache_clear();
				cudaGetLastError();
				err = cudaMalloc(&ptr, size);
			}

			if (cudaSuccess != err) {

				debug_print_memcache(DP_INFO);
				debug_printf(DP_WARN, "Trying to allocate %zu GB failed.\n", size / 1024 /1024 /1024);
				debug_printf(DP_WARN, "Try BART_GPU_GLOBAL_MEMORY=1 environment variable for memory oversubscription.\n");
				cuda_error(__FILE__, __LINE__, err);
			}
		}
	}

	return ptr;
}

static void* cuda_malloc_host_wrapper(size_t size)
{
	void* ptr;
	CUDA_ERROR(cudaMallocHost(&ptr, size));
	return ptr;
}

static void cuda_free_wrapper(const void* ptr, bool host)
{
	if (host)
		CUDA_ERROR(cudaFreeHost((void*)ptr));
	else
		CUDA_ERROR(cudaFree((void*)ptr));
}

void cuda_free(void* ptr)
{
	mem_device_free(ptr, cuda_free_wrapper);
}

void* cuda_malloc(long size)
{
	return mem_device_malloc((size_t)size, cuda_malloc_wrapper, false);
}

void* cuda_malloc_host(long size)
{
	return mem_device_malloc((size_t)size, cuda_malloc_host_wrapper, true);
}

void cuda_use_global_memory(void)
{
	cuda_global_memory = true;
}

void cuda_memcache_off(void)
{
	cuda_memcache = false;
}

void cuda_memcache_clear(void)
{
	memcache_clear(cuda_free_wrapper);
}


//#if CUDART_VERSION >= 10000
//#define CUDA_GET_CUDA_DEVICE_NUM
//#endif

static bool cuda_ondevice_int(const void* ptr)
{
#ifdef CUDA_GET_CUDA_DEVICE_NUM
// (We still don use this because it is slow. Why? Nivida, why?)
// Starting with CUDA 10 it has similar speed to the memcache but is
// faster if multiple threads access the memcache
// with our trees it's faster again...
	if (NULL == ptr)
		return false;

	if (-1 == cuda_device_id)
		return false;

	struct cudaPointerAttributes attr;
	if (cudaSuccess != (cudaPointerGetAttributes(&attr, ptr)))
	{
	/* The secret trick to make this work for arbitrary pointers
	   is to clear the error using cudaGetLastError. See end of:
	   http://www.alexstjohn.com/WP/2014/04/28/cuda-6-0-first-look/
	 */
		cudaGetLastError();
		return false;
	}

	if ((cudaMemoryTypeUnregistered == attr.type) || (cudaMemoryTypeHost == attr.type))
		return false;

	return 0 <= attr.device;
#else
	return mem_ondevice(ptr);
#endif
}

bool cuda_ondevice(const void* ptr)
{
	return cuda_ondevice_int(ptr) || is_vptr_gpu(ptr);
}


void cuda_clear(long size, void* dst)
{
	CUDA_ERROR_PTR(dst);
	CUDA_ERROR(cudaMemsetAsync(dst, 0, (size_t)size, cuda_get_stream()));
}

static void cuda_float_clear(long size, float* dst)
{
	cuda_clear(size * (long)sizeof(float), (void*)dst);
}

void cuda_memcpy(long size, void* dst, const void* src)
{
	CUDA_ERROR(cudaMemcpyAsync(dst, src, (size_t)size, cudaMemcpyDefault, cuda_get_stream()));
}

void cuda_memcpy_strided(const long dims[2], long ostr, void* dst, long istr, const void* src)
{
	CUDA_ERROR(cudaMemcpy2DAsync(dst, (size_t)ostr, src, (size_t)istr, (size_t)dims[0], (size_t)dims[1], cudaMemcpyDefault, cuda_get_stream()));
}

static void cuda_float_copy(long size, float* dst, const float* src)
{
	cuda_memcpy(size * (long)sizeof(float), (void*)dst, (const void*)src);
}


static float* cuda_float_malloc(long size)
{
	return (float*)cuda_malloc(size * (long)sizeof(float));
}

static void cuda_float_free(float* x)
{
	cuda_free((void*)x);
}



const struct vec_ops gpu_ops = {

	.float2double = cuda_float2double,
	.double2float = cuda_double2float,
	.dot = cuda_dot,
	.asum = cuda_asum,
	.zsum = cuda_zsum,
	.zl1norm = NULL,

	.zdot = cuda_cdot,

	.add = cuda_add,
	.sub = cuda_sub,
	.mul = cuda_mul,
	.div = cuda_div,
	.fmac = cuda_fmac,
	.fmacD = cuda_fmacD,

	.smul = cuda_smul,
	.sadd = cuda_sadd,

	.axpy = cuda_saxpy,

	.pow = cuda_pow,
	.sqrt = cuda_sqrt,
	.round = cuda_round,

	.le = cuda_le,

	.zsmul = cuda_zsmul,
	.zsadd = cuda_zsadd,
	.zsmax = cuda_zsmax,
	.zsmin = cuda_zsmin,

	.zmul = cuda_zmul,
	.zdiv = cuda_zdiv,
	.zfmac = cuda_zfmac,
	.zfmacD = cuda_zfmacD,
	.zmulc = cuda_zmulc,
	.zfmacc = cuda_zfmacc,
	.zfmaccD = cuda_zfmaccD,
	.zfsq2 = cuda_zfsq2,

	.zpow = cuda_zpow,
	.zphsr = cuda_zphsr,
	.zconj = cuda_zconj,
	.zexpj = cuda_zexpj,
	.zexp = cuda_zexp,
	.zsin = cuda_zsin,
	.zcos = cuda_zcos,
	.zsinh = cuda_zsinh,
	.zcosh = cuda_zcosh,
	.zlog = cuda_zlog,
	.zarg = cuda_zarg,
	.zabs = cuda_zabs,
	.zatanr = cuda_zatanr,
	.zatan2r = cuda_zatan2r,
	.zacosr = cuda_zacosr,

	.exp = cuda_exp,
	.log = cuda_log,

	.zcmp = cuda_zcmp,
	.zdiv_reg = cuda_zdiv_reg,
	.zfftmod = cuda_zfftmod,

	.zmax = cuda_zmax,
	.zle = cuda_zle,

	.zsetnanzero=NULL,
	.smax = cuda_smax,
	.max = cuda_max,
	.min = cuda_min,

	.zsoftthresh = cuda_zsoftthresh,
	.zsoftthresh_half = cuda_zsoftthresh_half,
	.softthresh = cuda_softthresh,
	.softthresh_half = cuda_softthresh_half,
	.zhardthresh = NULL,

	.pdf_gauss = cuda_pdf_gauss,

	.real = cuda_real,
	.imag = cuda_imag,
	.zcmpl_real = cuda_zcmpl_real,
	.zcmpl_imag = cuda_zcmpl_imag,
	.zcmpl = cuda_zcmpl,

	.zfill = cuda_zfill,
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
	void (*axpbz)(long N, float* out, const float a, const float* x, const float b, const float* z);
	void (*fmac)(long N, float* a, const float* x, const float* y);

	void (*mul)(long N, float* a, const float* x, const float* y);
	void (*div)(long N, float* a, const float* x, const float* y);
	void (*sqrt)(long N, float* a, const float* x);

	void (*smax)(long N, float alpha, float* a, const float* x);
	void (*smin)(long N, float alpha, float* a, const float* x);
	void (*sadd)(long N, float* x, float y);
	void (*sdiv)(long N, float* a, float x, const float* y);
	void (*le)(long N, float* a, const float* x, const float* y);

	void (*zmul)(long N, complex float* dst, const complex float* src1, const complex float* src2);
	void (*zsmax)(long N, float val, complex float* dst, const complex float* src1);

	void (*rand)(long N, float* dst);
	void (*uniform)(long N, float* dst);

	void (*xpay_bat)(long Bi, long N, long Bo, const float* beta, float* a, const float* x);
	void (*dot_bat)(long Bi, long N, long Bo, float* dst, const float* src1, const float* src2);
	void (*axpy_bat)(long Bi, long N, long Bo, float* a, const float* alpha, const float* x);

};


static void cuda_sadd_inpl(long N, float* dst, float val)
{
	cuda_sadd(N, val, dst, dst);
}


extern const struct vec_iter_s gpu_iter_ops;
const struct vec_iter_s gpu_iter_ops = {

	.allocate = cuda_float_malloc,
	.del = cuda_float_free,
	.clear = cuda_float_clear,
	.copy = cuda_float_copy,
	.dot = cuda_dot,
	.norm = cuda_norm,
	.axpy = cuda_saxpy,
	.xpay = cuda_xpay,
	.axpbz = cuda_axpbz,
	.smul = cuda_smul,
	.add = cuda_add,
	.sub = cuda_sub,
	.swap = cuda_swap,
	.zmul = cuda_zmul,
	.rand = gaussian_rand_vec,
	.uniform = uniform_rand_vec,
	.mul = cuda_mul,
	.fmac = cuda_fmac,
	.div = cuda_div,
	.sqrt = cuda_sqrt,
	.smax = cuda_smax,
	.smin = NULL,
	.sadd = cuda_sadd_inpl,
	.sdiv = NULL,
	.le = cuda_le,
	.zsmax = cuda_zsmax,
	.xpay_bat = cuda_xpay_bat,
	.dot_bat = cuda_dot_bat,
	.axpy_bat = cuda_axpy_bat,

};
#endif

