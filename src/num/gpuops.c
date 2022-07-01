/* Copyright 2013-2018. The Regents of the University of California.
 * Copyright 2014. Joseph Y Cheng.
 * Copyright 2016-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2019	Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014 	Joseph Y Cheng <jycheng@stanford.edu>
 * 2015-2018	Jon Tamir <jtamir@eecs.berkeley.edu>
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

#include "num/vecops.h"
#include "num/gpuops.h"
#include "num/gpukrnls.h"
#include "num/mem.h"
#include "num/multind.h"
#include "num/blas.h"

#ifdef USE_CUDNN
#include "num/cudnn_wrapper.h"
#endif

#include "misc/misc.h"
#include "misc/debug.h"

#include "gpuops.h"

#define MiBYTE (1024*1024)

static unsigned int reserved_gpus = 0U;
static int n_reserved_gpus = 0;

static int gpu_map[MAX_CUDA_DEVICES] = { [0 ... MAX_CUDA_DEVICES - 1] = -1 };

static bool gpu_peer_accees[MAX_CUDA_DEVICES][MAX_CUDA_DEVICES] = { [0 ... MAX_CUDA_DEVICES - 1] = { [0 ... MAX_CUDA_DEVICES - 1] = false} };


bool cuda_memcache = true;
bool cuda_global_memory = false;

static int num_cuda_devices_internal(void);
static int cuda_get_device_internal(void);
static void cuda_set_device_internal(int device);

static void cuda_stream_sync_deinit(void);
static void cuda_stream_sync_init(void);

void cuda_error(const char* file, int line, cudaError_t code)
{
	const char *err_str = cudaGetErrorString(code);
	error("CUDA Error on Device %d: %s in %s:%d\n", cuda_get_device_internal(), err_str, file, line);
}

void cuda_gpu_check(const char* file, int line, const char* note)
{
#ifdef GPU_ASSERTS
	cudaError_t code = cudaStreamSynchronize(cuda_default_stream);

	if (cudaSuccess != code) {
		
		const char *err_str = cudaGetErrorString(code);
		if (0 == strlen(note))
			error("CUDA Error on Device %d: %s in %s:%d\n", cuda_get_device_internal(), err_str, file, line);
		else
			error("CUDA Error on Device %d: %s in %s:%d (%s)\n", cuda_get_device_internal(), err_str, file, line, note);
	}
#else
	UNUSED(file);
	UNUSED(line);
	UNUSED(note);
#endif
}

void cuda_check_ptr(const char* file, int line, int N, const void* ptr[N])
{
#ifdef GPU_ASSERTS
	bool same_device = true;
	
	for (int i = 0; i < N; i++)
		if (cuda_get_device() != cuda_get_device_num(ptr[i]))
			same_device = false;
	
	if (!same_device) {
		for (int i = 0; i < N; i++)
			debug_printf(DP_WARN, "%d: %x on device %d\n", i, ptr[i], cuda_get_device_num(ptr[i]));
		error("CUDA Error on Device %d: Pointer not on current device in %s:%d", cuda_get_device(), file, line);
	}
#else
	UNUSED(file);
	UNUSED(line);
	UNUSED(N);
	UNUSED(ptr);
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


//*************************************** CUDA Device Selection ********************************************* 

static int cuda_get_internal_device(int device)
{
	if (-1 == device)
		return -1;

	return gpu_map[device];
}

static int cuda_get_external_device(int device)
{
	if (-1 == device)
		return device;

	assert(MD_IS_SET(reserved_gpus, device));

	for (int i = 0; i < cuda_num_devices(); i++)
		if (gpu_map[i] == device)
			return i;
	
	error("CUDA: Device (%d) not in GPU Map!\n", device);
}

static int num_cuda_devices_internal(void)
{
	int count;
	CUDA_ERROR(cudaGetDeviceCount(&count));
	return count;
}

int cuda_num_devices(void)
{
	assert((int)bitcount(reserved_gpus) == n_reserved_gpus);
	return n_reserved_gpus;
}

static int cuda_get_device_internal(void)
{
	int device;
	if (0 == n_reserved_gpus)
		device = -1;
	else
		CUDA_ERROR(cudaGetDevice(&device));


	return device;
}

int cuda_get_device_internal_unchecked(void)
{
	
	return cuda_get_device_internal();
}

int cuda_get_device(void)
{
	int device = cuda_get_device_internal();
	if (-1 == device)
		return -1;

	device = cuda_get_external_device(device);
	
	return device;
}

static void cuda_set_device_internal(int device)
{
	if (-1 == device)
		return;

	if (!MD_IS_SET(reserved_gpus, device))
		error("Trying to use non-reserved GPU (%d)! Reserve first by using cuda_try_init(device)\n", device);

	CUDA_ERROR(cudaSetDevice(device));
}

void cuda_set_device(int device)
{
	assert((0 <= device) || (0 == cuda_num_devices()));

	if (device >= cuda_num_devices())
		error("Tried to select device %d, but only %d device(s) initialized!", device, cuda_num_devices());

	if (0 <= device)
		cuda_set_device_internal(gpu_map[device]);
}

static void cuda_activate_p2p(void)
{

	for (int i = 0; i < cuda_num_devices(); i++) {
		for (int j = 0; j < cuda_num_devices(); j++) {

			int r;
			CUDA_ERROR(cudaDeviceCanAccessPeer(&r, gpu_map[i], gpu_map[j]));

			gpu_peer_accees[i][j] = (1 == r);

			if (gpu_peer_accees[i][j]) {

				cuda_set_device(i);
				CUDA_ERROR(cudaDeviceEnablePeerAccess(gpu_map[j], 0));
			}			
		}
	}
}

static void cuda_deactivate_p2p(void)
{
	for (int i = 0; i < cuda_num_devices(); i++) {
		for (int j = 0; j < cuda_num_devices(); j++) {

			if (gpu_peer_accees[i][j]) {

				cuda_set_device(i);
				CUDA_ERROR(cudaDeviceDisablePeerAccess(gpu_map[j]));
			}

			gpu_peer_accees[i][j] = false;
		}
	}
}

static void cuda_libraries_init(void)
{
	cuda_activate_p2p();
	cublas_init();
#ifdef USE_CUDNN
	cudnn_init();
#endif
}

static void cuda_libraries_deinit(void)
{
	cuda_deactivate_p2p();
	cublas_deinit();
#ifdef USE_CUDNN
	cudnn_deinit();
#endif
}

bool cuda_try_init(int device)
{
	int odev = cuda_get_device_internal();

	cudaError_t errval = cudaSetDevice(device);
	if (cudaSuccess == errval) {

		errval = cudaDeviceSynchronize();

		if (cudaSuccess == errval) {

			cuda_set_device_internal(odev);

			// only add to gpu_map if not already present.
			// This allows multiple calls to initialize cuda
			// to succeed without problems.
			if (!MD_IS_SET(reserved_gpus, device)) {

				cuda_libraries_deinit();

				gpu_map[n_reserved_gpus++] = device;
				reserved_gpus = MD_SET(reserved_gpus, device);

				cuda_set_device(0);
				cuda_libraries_init();
			}

			cuda_set_device(0);
			return true;

		} else {

			// clear last error
			cudaGetLastError();
			cuda_set_device_internal(odev);
		}
	}

	return false;
}


static void remove_from_gpu_map(int device)
{
	int device_index = -1;
	for (int i = 0; i < n_reserved_gpus; ++i) {

		if (device == gpu_map[i]) {

			device_index = i;
			break;
		}
	}

	for (int i = device_index; i < MIN(n_reserved_gpus, MAX_CUDA_DEVICES); ++i)
		gpu_map[i] = gpu_map[i + 1];

	gpu_map[n_reserved_gpus - 1] = -1;

}

static void cuda_deinit(int device)
{
	cuda_libraries_deinit();


	cuda_set_device_internal(device);
	CUDA_ERROR(cudaDeviceReset());
	remove_from_gpu_map(device);
	n_reserved_gpus--;
	reserved_gpus = MD_CLEAR(reserved_gpus, device);

	if (0 < n_reserved_gpus) {

		cuda_set_device(0);
		cuda_libraries_init();
		cuda_set_device(0);
	}
}


void cuda_init(void)
{
	int num_devices = num_cuda_devices_internal();
	for (int device = 0; device < num_devices; ++device)
		if (cuda_try_init(device))
			return;

	error("Could not allocate any GPU device\n");
}

void cuda_init_multigpu_select(unsigned long requested_gpus)
{

	int num_devices = num_cuda_devices_internal();
	for (int device = 0; device < num_devices; ++device) {

		if (MD_IS_SET(requested_gpus, device))
			cuda_try_init(device);
	}

	if (0UL == reserved_gpus )
		error("No GPUs could be allocated!\n");
	else if (reserved_gpus != (requested_gpus & (MD_BIT(num_devices) - 1)))
		debug_printf(DP_WARN, "Not all requested gpus could be allocated, continuing with fewer (%d)\n", cuda_num_devices());
}

void cuda_init_multigpu_number(int requested_gpus)
{

	int num_devices = num_cuda_devices_internal();
	for (int device = 0; device < num_devices; ++device) {

		if (cuda_num_devices() < requested_gpus)
			cuda_try_init(device);
	}

	if (0UL == reserved_gpus )
		error("No GPUs could be allocated!\n");
	else if (cuda_num_devices() != requested_gpus)
		debug_printf(DP_WARN, "Not all requested gpus could be allocated, continuing with fewer (%d)\n", cuda_num_devices());
}

int cuda_init_memopt(void)
{
	int num_devices = num_cuda_devices_internal();
	int device;
	int max_device = -1;

	if (num_devices > 1) {

		size_t mem_max = 0;
		size_t mem_free;
		size_t mem_total;

		for (device = 0; device < num_devices; device++) {

			if (!cuda_try_init(device))
				continue;

			CUDA_ERROR(cudaMemGetInfo(&mem_free, &mem_total));

			if (mem_max < mem_free) {

				mem_max = mem_free;
				max_device = device;
			}
		}

		if (-1 == max_device)
			error("Could not allocate any GPU device\n");

		for (device = 0; device < num_devices; device++) {

			if (MD_IS_SET(reserved_gpus, device) && (device != max_device))
				cuda_deinit(device);
		}

		cuda_set_device_internal(max_device);

	} else {

		cuda_init();
	}

	return max_device;
}

void cuda_exit(void)
{
	cuda_memcache_clear();
	for (int d = 0; d < n_reserved_gpus; d++)
		cuda_deinit(gpu_map[d]);

}



//*************************************** Stream Synchronisation ********************************************* 

cudaStream_t cuda_get_stream(void)
{
	return cudaStreamLegacy;
}



//*************************************** Host Synchonization ********************************************* 


void cuda_sync_device(void)
{
	// do not initialize gpu just for syncing
	if (-1 == cuda_get_device())
		return;

	CUDA_ERROR(cudaDeviceSynchronize());
}

void cuda_sync_devices(void)
{
	int olddevice = cuda_get_device();
	for (int i = 0; i < cuda_num_devices(); i++) {

		cuda_set_device(i);
		cuda_sync_device();
	}
	cuda_set_device(olddevice);
}



//*************************************** Memory Management ********************************************* 




static void* cuda_malloc_wrapper(size_t size)
{
	void* ptr;

	if (cuda_global_memory) {

		CUDA_ERROR(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));

	} else {

		CUDA_ERROR(cudaMalloc(&ptr, size));
	}

	return ptr;
}

static void cuda_free_wrapper(const void* ptr)
{
	CUDA_ERROR(cudaFree((void*)ptr));
}

void cuda_free(void* ptr)
{
	mem_device_free(ptr, cuda_free_wrapper);
}

void* cuda_malloc(long size)
{
	return mem_device_malloc(cuda_get_device_internal(), size, cuda_malloc_wrapper);
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
	if (!cuda_memcache)
		return;

	memcache_clear(cuda_get_device_internal(), cuda_free_wrapper);
}

#if 0
// We still don use this because it is slow. Why? Nivida, why?

static bool cuda_cuda_ondevice(const void* ptr)
{
	if (NULL == ptr)
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

	return (cudaMemoryTypeDevice == attr.memoryType);
}
#endif

bool cuda_accessible(const void* ptr)
{
	return mem_device_accessible(ptr);
}

bool cuda_ondevice(const void* ptr)
{
	return mem_ondevice(ptr);
}



void cuda_clear(long size, void* dst)
{
	CUDA_ERROR(cudaMemsetAsync(dst, 0, size, cuda_get_stream()));
}

static void cuda_float_clear(long size, float* dst)
{
	cuda_clear(size * sizeof(float), (void*)dst);
}

void cuda_memcpy(long size, void* dst, const void* src)
{
	CUDA_ERROR(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, cuda_get_stream()));
}

void cuda_memcpy_strided(const long dims[2], long ostr, void* dst, long istr, const void* src)
{
	CUDA_ERROR(cudaMemcpy2DAsync(dst, ostr, src, istr, dims[0], dims[1], cudaMemcpyDefault, cuda_get_stream()));
}

static void cuda_float_copy(long size, float* dst, const float* src)
{
	cuda_memcpy(size * sizeof(float), (void*)dst, (const void*)src);
}


static float* cuda_float_malloc(long size)
{
	return (float*)cuda_malloc(size * sizeof(float));
}

static void cuda_float_free(float* x)
{
	cuda_free((void*)x);
}



const struct vec_ops gpu_ops = {

	.float2double = cuda_float2double,
	.double2float = cuda_double2float,
	.dot = cuda_sdot,
	.asum = cuda_asum,
	.zsum = cuda_zsum,
	.zl1norm = NULL,

	.add = cuda_add,
	.sub = cuda_sub,
	.mul = cuda_mul,
	.div = cuda_div,
	.fmac = cuda_fmac,
	.fmac2 = cuda_fmac2,

	.smul = cuda_smul,
	.sadd = cuda_sadd,

	.axpy = cuda_saxpy,

	.pow = cuda_pow,
	.sqrt = cuda_sqrt,

	.le = cuda_le,

	.zsmul = cuda_zsmul,
	.zsadd = cuda_zsadd,
	.zsmax = cuda_zsmax,

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
	.zacos = cuda_zacos,

	.exp = cuda_exp,
	.log = cuda_log,

	.zcmp = cuda_zcmp,
	.zdiv_reg = cuda_zdiv_reg,
	.zfftmod = cuda_zfftmod,

	.zmax = cuda_zmax,
	.zle = cuda_zle,

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

	void (*zmul)(long N, complex float* dst, const complex float* src1, const complex float* src2);
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
	.axpbz = cuda_axpbz,
	.smul = cuda_smul,
	.add = cuda_add,
	.sub = cuda_sub,
	.swap = cuda_swap,
	.zmul = cuda_zmul,
};

#endif
