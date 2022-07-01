/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/cppwrap.h"

#ifdef USE_CUDA

#include <cuda_runtime_api.h>
void cuda_error(const char* file, int line, cudaError_t code);
void cuda_gpu_check(const char* file, int line, const char* note);
void cuda_check_ptr(const char* file, int line, int N, const void* ptr[__VLA(N)]);

#define CUDA_ASYNC_ERROR_NOTE(x)	({ cuda_gpu_check(__FILE__, __LINE__, (x)); })
#define CUDA_ASYNC_ERROR		CUDA_ASYNC_ERROR_NOTE("")
#define CUDA_ERROR(x)			({ cudaError_t errval = (x); if (cudaSuccess != errval) cuda_error(__FILE__, __LINE__, errval); })
#define CUDA_KERNEL_ERROR 		({ cudaError_t errval = cudaGetLastError(); if (cudaSuccess != errval) cuda_error(__FILE__, __LINE__, errval); CUDA_ASYNC_ERROR; })
#define CUDA_ERROR_PTR(...)		({ CUDA_ASYNC_ERROR; const void* _ptr[] = { __VA_ARGS__}; cuda_check_ptr(__FILE__, __LINE__, (sizeof(_ptr) / sizeof(_ptr[0])), _ptr); })

#endif

#define MAX_CUDA_DEVICES 16
#define MAX_CUDA_STREAMS 2
extern int cuda_streams_per_device;

extern const struct vec_ops gpu_ops;

//gpu initialisation
extern void cuda_init(void);
extern _Bool cuda_try_init(int device);
extern int cuda_init_memopt(void);
extern void cuda_init_multigpu_select(unsigned long requested_gpus);
extern void cuda_init_multigpu_number(int requested_gpus);
extern void cuda_exit(void);

//device selection
extern int cuda_num_devices(void);
extern void cuda_set_device(int device);
extern int cuda_get_device(void);
extern int cuda_get_device_internal_unchecked(void);

#ifdef USE_CUDA
extern cudaStream_t cuda_get_stream(void);
#endif
//synchronisation functions
extern void cuda_sync_device(void);
extern void cuda_sync_devices(void);

//cuda memory allocations
extern void* cuda_malloc(long N);
extern void cuda_free(void*);

extern _Bool cuda_ondevice(const void* ptr);
extern _Bool cuda_accessible(const void* ptr);
extern void cuda_clear(long size, void* ptr);
extern void cuda_memcpy(long size, void* dst, const void* src);
extern void cuda_memcpy_strided(const long dims[2], long ostr, void* dst, long istr, const void* src);

extern void cuda_memcache_off(void);
extern void cuda_memcache_clear(void);

extern void cuda_use_global_memory(void);
extern void print_cuda_meminfo(void);

#include "misc/cppwrap.h"
