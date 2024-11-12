
#include <stddef.h>

#include "misc/cppwrap.h"

extern long md_compress_mask_to_index(long N, const long dims[__VLA(N)], long* index, const _Complex float* mask);

extern void md_compress_dims(long N, long cdims[__VLA(N)], const long dcdims[__VLA(N)], const long mdims[__VLA(N)], long max);
extern void md_decompress_dims(long N, long dcdims[__VLA(N)], const long cdims[__VLA(N)], const long mdims[__VLA(N)]);

extern void md_decompress2(int N, const long odims[__VLA(N)], const long ostrs[__VLA(N)], void* dst, const long idims[__VLA(N)], const long istrs[__VLA(N)], const void* src, const long mdims[__VLA(N)], const long mstrs[__VLA(N)], const long* index, const void* fill, size_t size);
extern void md_decompress(int N, const long odims[__VLA(N)], void* dst, const long idims[__VLA(N)], const void* src, const long mdims[__VLA(N)], const long* index, const void* fill, size_t size);
extern void md_compress2(int N, const long odims[__VLA(N)], const long ostrs[__VLA(N)], void* dst, const long idims[__VLA(N)], const long istrs[__VLA(N)], const void* src, const long mdims[__VLA(N)], const long mstrs[__VLA(N)], const long* index, size_t size);
extern void md_compress(int N, const long odims[__VLA(N)], void* dst, const long idims[__VLA(N)], const void* src, const long mdims[__VLA(N)], const long* index, size_t size);

#include "misc/cppwrap.h"
