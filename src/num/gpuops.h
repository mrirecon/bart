/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern const struct vec_ops gpu_ops;
extern _Bool cuda_ondevice(const void* ptr);
extern _Bool cuda_accessible(const void* ptr);
extern void cuda_clear(long size, void* ptr);
extern void cuda_memcpy(long size, void* dst, const void* src);
extern void cuda_hostfree(void*);
extern void* cuda_hostalloc(long N);
extern void* cuda_malloc(long N);
extern void cuda_memcpy_strided(const long dims[2], long ostr, void* dst, long istr, const void* src);
extern void cuda_free(void*);
extern void cuda_init(int device);
extern int cuda_init_memopt(void);
extern void cuda_p2p_table(int n, _Bool table[n][n]);
extern void cuda_p2p(int a, int b);
extern void cuda_exit(void);
extern int cuda_devices(void);
extern void cuda_memcache_off(void);
extern void cuda_memcache_clear(void);

#ifdef __cplusplus
}
#endif
