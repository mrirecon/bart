/* Copyright 2013-2018. The Regents of the University of California.
 * Copyright 2014. Joseph Y Cheng.
 * Copyright 2016-2022. Uecker Lab. University Center Göttingen.
 * Copyright 2023. Insitute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2023	Martin Uecker <uecker@tugraz.at>
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
int cuda_streams_per_device = 1;

static int gpu_map[MAX_CUDA_DEVICES] = { [0 ... MAX_CUDA_DEVICES - 1] = -1 };

static cudaStream_t gpu_streams[MAX_CUDA_DEVICES][MAX_CUDA_STREAMS + 1];
static cudaEvent_t gpu_streams_sync[MAX_CUDA_DEVICES][MAX_CUDA_DEVICES][MAX_CUDA_STREAMS + 1][MAX_CUDA_STREAMS + 1];

static bool gpu_peer_accees[MAX_CUDA_DEVICES][MAX_CUDA_DEVICES] = { [0 ... MAX_CUDA_DEVICES - 1] = { [0 ... MAX_CUDA_DEVICES - 1] = false} };

struct cuda_stream_id {

	int device;
	int stream;
};

// we check ourself which stream/device is associated to the current thread 
static _Thread_local struct cuda_stream_id thread_active_stream = { -1, 0 };
// we need to sync streams, if we change the stream/device and call a new kernel
// thus, we keep track of the last stream we placed a cuda call in 
static _Thread_local struct cuda_stream_id thread_last_issued_stream = { -1, 0 };


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
	cudaError_t code = cudaStreamSynchronize(cuda_get_stream());

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

	if (device != thread_active_stream.device)
		error("CUDA incosistent active device!\n");

	return device;
}

int cuda_get_device_internal_unchecked(void)
{
	return thread_active_stream.device;
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

	thread_active_stream.device = device;

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

void cuda_device_is_set(const char* file, int line)
{
	if ((0 < n_reserved_gpus) && (-1 == thread_active_stream.device))
		error(
		"CUDA Error on Device ?: Call without selected device! in %s:%d\n"
		"Probably CUDA is called within an OMP region without setting the device after entering!\n",
		file, line);
}



//*************************************** CUDA Device Initialization *********************************************


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
	cuda_stream_sync_init();
	memcache_init();
	cuda_activate_p2p();
	cublas_init();
#ifdef USE_CUDNN
	cudnn_init();
#endif
}

static void cuda_libraries_deinit(void)
{
	cuda_stream_sync_deinit();
	memcache_destroy();
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
			thread_last_issued_stream.device = thread_active_stream.device;
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


static void cuda_stream_sync_init(void)
{
	int num_device = cuda_num_devices();
	int device = cuda_get_device();

	for (int dev1 = 0; dev1 < num_device; dev1++) {

		cuda_set_device(dev1);

		for (int str1 = 0; str1 < MAX_CUDA_STREAMS; str1++) {

			CUDA_ERROR(cudaStreamCreate(&(gpu_streams[dev1][str1])));

			for (int dev2 = 0; dev2 < num_device; dev2++)
				for (int str2 = 0; str2 < MAX_CUDA_STREAMS; str2++) {

					CUDA_ERROR(cudaEventCreate(&(gpu_streams_sync[dev1][dev2][str1][str2])));
					CUDA_ERROR(cudaEventRecord(gpu_streams_sync[dev1][dev2][str1][str2], gpu_streams[dev1][str1]));
				}
		}

		thread_last_issued_stream.stream = 0;
		thread_active_stream.stream = 0;
	}

	cuda_set_device(device);
}

static void cuda_stream_sync_deinit(void)
{
	int num_device = cuda_num_devices();
	int device = cuda_get_device();

	for (int dev1 = 0; dev1 < num_device; dev1++) {

		cuda_set_device(dev1);

		for (int str1 = 0; str1 < MAX_CUDA_STREAMS; str1++) {

			for (int dev2 = 0; dev2 < num_device; dev2++)
				for (int str2 = 0; str2 < MAX_CUDA_STREAMS; str2++) {

					CUDA_ERROR(cudaEventDestroy((gpu_streams_sync[dev1][dev2][str1][str2])));

				}
			
			CUDA_ERROR(cudaStreamDestroy((gpu_streams[dev1][str1])));
		}
	}

	cuda_set_device(device);
}

static void cuda_stream_sync(void)
{
	struct cuda_stream_id new = thread_active_stream;
	struct cuda_stream_id old = thread_last_issued_stream;

	if ((old.device == new.device) && (old.stream == new.stream))
		return;

	int device = cuda_get_device_internal();
	assert(new.device == device);

	new.device = cuda_get_external_device(new.device);
	old.device = cuda_get_external_device(old.device);


	cuda_set_device(old.device);
	CUDA_ERROR(cudaEventRecord(gpu_streams_sync[old.device][new.device][old.stream][new.stream], gpu_streams[old.device][old.stream]));
	
	cuda_set_device(new.device);
	CUDA_ERROR(cudaStreamWaitEvent(gpu_streams[new.device][new.stream], gpu_streams_sync[old.device][new.device][old.stream][new.stream], 0));

	new.device = cuda_get_internal_device(new.device);
	thread_last_issued_stream = new;
}

void cuda_set_stream(int stream)
{
	thread_active_stream.stream = stream;
}

static cudaStream_t cuda_get_stream_internal(void)
{
	if (MAX_CUDA_STREAMS >= thread_active_stream.stream)
		return cudaStreamLegacy;
	
	if (   (0 > thread_active_stream.device)
	    || (0 > thread_active_stream.stream)
	    || (MAX_CUDA_DEVICES <= thread_active_stream.device)
	    || (MAX_CUDA_STREAMS <= thread_active_stream.stream))
		error("CUDA: active device/stream not initialized correctly!\n");

	return gpu_streams[thread_active_stream.device][thread_active_stream.stream];
}

cudaStream_t cuda_get_stream(void)
{
	cuda_stream_sync();
	return cuda_get_stream_internal();
}

int cuda_get_stream_id(void)
{
	return thread_active_stream.stream;
}

int cuda_num_streams()
{
	assert(cuda_streams_per_device <= MAX_CUDA_STREAMS);
	return cuda_streams_per_device;
}

struct cuda_threads_s {
	
	struct cuda_stream_id active;
	struct cuda_stream_id last;
};

struct cuda_threads_s* cuda_threads_create(void)
{
	if (0 == cuda_num_devices())
		return NULL;

	PTR_ALLOC(struct cuda_threads_s, x);
	
	x->active = thread_active_stream;
	x->last = thread_last_issued_stream;

	return PTR_PASS(x);
}

void cuda_threads_enter(struct cuda_threads_s* x)
{
	if(NULL == x)
		return;

	thread_last_issued_stream = x->last;
	
	cuda_set_device_internal(x->active.device);
	cuda_set_stream(x->active.stream);
}

void cuda_threads_leave(struct cuda_threads_s* x)
{
	if(NULL == x)
		return;

#if 0
	cuda_sync_stream();
#else
	cuda_set_device_internal(x->active.device);
	cuda_set_stream(x->active.stream);
	cuda_stream_sync();
#endif
	return;
}

void cuda_threads_free(struct cuda_threads_s* x)
{
	if(NULL == x)
		return;

	thread_last_issued_stream = x->active;

	cuda_set_device_internal(x->active.device);
	cuda_set_stream(x->active.stream);

	xfree(x);
}

//*************************************** Host Synchonization ********************************************* 

void cuda_sync_stream(void)
{
	// do not initialize gpu just for syncing
	if (-1 == cuda_get_device())
		return;

	CUDA_ERROR(cudaStreamSynchronize(cuda_get_stream()));
}

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
	if (0 == cuda_num_devices())
		error("CUDA_ERROR: No gpu initialized, run \"num_init_gpu\"!\n");

	void* ptr;

	if (cuda_global_memory) {

		CUDA_ERROR(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));

		int device = cuda_get_internal_device(cuda_get_device());
		
		for (int i = 0; i < cuda_num_devices(); i++) {

			int access;
			CUDA_ERROR(cudaDeviceGetAttribute(&access, cudaDevAttrConcurrentManagedAccess, cuda_get_internal_device(i)));

			if(0 != access)
				CUDA_ERROR(cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, cuda_get_internal_device(i)));
		}

		int access;
		CUDA_ERROR(cudaDeviceGetAttribute(&access, cudaDevAttrConcurrentManagedAccess, device));

		if (0 != access)
			CUDA_ERROR(cudaMemPrefetchAsync(ptr, size, device, cuda_get_stream()));
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
	return mem_device_malloc(cuda_get_device_internal(), cuda_get_stream_id(), size, cuda_malloc_wrapper);
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

	for (int d = 0; d < n_reserved_gpus; d++) {

		cuda_set_device(d);

		for (int i = 0; i < MAX_CUDA_STREAMS; i++)
			memcache_clear(gpu_map[d], i, cuda_free_wrapper);
	}
}


#if CUDART_VERSION >= 10000
#define CUDA_GET_CUDA_DEVICE_NUM
#endif

#ifdef CUDA_GET_CUDA_DEVICE_NUM
// (We still don use this because it is slow. Why? Nivida, why?)
// Starting with CUDA 10 it has similar speed to the memcache but is 
// faster if multiple threads access the memcache

static int cuda_cuda_get_device_num_internal(const void* ptr)
{
	if (NULL == ptr)
		return -1;

	if (0 == n_reserved_gpus)
		return -1;
	
	struct cudaPointerAttributes attr;
	if (cudaSuccess != (cudaPointerGetAttributes(&attr, ptr)))
	{
	/* The secret trick to make this work for arbitrary pointers
	   is to clear the error using cudaGetLastError. See end of:
	   http://www.alexstjohn.com/WP/2014/04/28/cuda-6-0-first-look/
	 */
		cudaGetLastError();
		return -1;
	}

	if ((cudaMemoryTypeUnregistered == attr.type) || (cudaMemoryTypeHost == attr.type))
		return -1;

	return attr.device;
}
#endif

bool cuda_ondevice(const void* ptr)
{
	return (-1 != cuda_get_device_num(ptr));
}

bool cuda_accessible(const void* ptr)
{
	return cuda_ondevice(ptr);
}

static int cuda_get_device_num_internal(const void* ptr)
{
	if (NULL == ptr)
		return -1;

#ifdef CUDA_GET_CUDA_DEVICE_NUM
	return cuda_cuda_get_device_num_internal(ptr);
#else
	return mem_device_num(ptr);
#endif

}

int cuda_get_device_num(const void* ptr)
{
	return cuda_get_external_device(cuda_get_device_num_internal(ptr));
}


void cuda_clear(long size, void* dst)
{
	CUDA_ERROR_PTR(dst);
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
