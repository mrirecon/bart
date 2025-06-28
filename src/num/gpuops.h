
#include "misc/cppwrap.h"

#ifdef USE_CUDA

#include <cuda_runtime_api.h>
extern void cuda_error(const char* file, int line, cudaError_t code);
extern void cuda_gpu_check(const char* file, int line, const char* note);
extern void cuda_check_ptr(const char* file, int line, int N, const void* ptr[__VLA(N)]);

#define CUDA_ASYNC_ERROR_NOTE(x)	({ cuda_gpu_check(__FILE__, __LINE__, (x)); })
#define CUDA_ASYNC_ERROR		CUDA_ASYNC_ERROR_NOTE("")
#define CUDA_ERROR(x)			({ cudaError_t errval = (x); if (cudaSuccess != errval) cuda_error(__FILE__, __LINE__, errval); })
#define CUDA_KERNEL_ERROR 		({ cudaError_t errval = cudaGetLastError(); if (cudaSuccess != errval) cuda_error(__FILE__, __LINE__, errval); CUDA_ASYNC_ERROR; })
#define CUDA_ERROR_PTR(...)		({ CUDA_ASYNC_ERROR; const void* _ptr[] = { __VA_ARGS__}; cuda_check_ptr(__FILE__, __LINE__, (sizeof(_ptr) / sizeof(_ptr[0])), _ptr); })

#endif

#define CUDA_MAX_STREAMS 8
extern int cuda_num_streams;

extern const struct vec_ops gpu_ops;

extern void cuda_init(void);
extern void cuda_exit(void);
extern int cuda_get_device_id(void);

extern int cuda_get_stream_id(void);
#ifdef USE_CUDA
extern cudaStream_t cuda_get_stream_by_id(int id);
extern cudaStream_t cuda_get_stream(void);
#endif

extern int cuda_set_stream_level(void);
extern _Bool cuda_is_stream_default(void);

extern void cuda_sync_device(void);
extern void cuda_sync_stream(void);

extern void* cuda_malloc(long N);
extern void* cuda_malloc_host(long N);
extern void cuda_free(void*);
extern _Bool cuda_ondevice(const void* ptr);
extern void cuda_clear(long size, void* ptr);
extern void cuda_memcpy(long size, void* dst, const void* src);
extern void cuda_memcpy_strided(const long dims[2], long ostr, void* dst, long istr, const void* src);

extern void cuda_memcache_off(void);
extern void cuda_memcache_clear(void);

extern void cuda_use_global_memory(void);
extern void print_cuda_meminfo(void);

#include "misc/cppwrap.h"
