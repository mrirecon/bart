/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2016-2021. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2022. Insitute of Biomedical Imaging. Graz University of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef __MULTIND_H
#define __MULTIND_H	1

#include <string.h>
#include <assert.h>
#ifdef _WIN32
#include <malloc.h>
#else
#include <alloca.h>
#endif

#include <stdbool.h>

#include "misc/cppwrap.h"
#include "misc/nested.h"
#include "misc/types.h"

typedef void CLOSURE_TYPE(md_nary_fun_t)(void* ptr[]);
typedef void CLOSURE_TYPE(md_trafo_fun_t)(long N, long str, void* ptr);
typedef void CLOSURE_TYPE(md_loop_fun_t)(const long* pos);


extern void md_nary(int C, int D, const long dim[__VLA(D)], const long* str[__VLA(C)], void* ptr[__VLA(C)], md_nary_fun_t fun);

extern void md_parallel_nary(int C, int D, const long dim[__VLA(D)], unsigned long flags, const long* str[__VLA(C)], void* ptr[__VLA(C)], md_nary_fun_t fun);
extern void md_parallel_loop(int D, const long dim[__VLA(D)], unsigned long flags, md_loop_fun_t fun);

extern void md_loop(int D, const long dim[__VLA(D)], md_loop_fun_t fun);

extern void md_septrafo2(int D, const long dimensions[__VLA(D)], unsigned long flags, const long strides[__VLA(D)], void* ptr, md_trafo_fun_t fun);
extern void md_septrafo(int D, const long dimensions[__VLA(D)], unsigned long flags, void* ptr, size_t size, md_trafo_fun_t fun);


extern void md_clear2(int D, const long dim[__VLA(D)], const long str[__VLA(D)], void* ptr, size_t size);
extern void md_clear(int D, const long dim[__VLA(D)], void* ptr, size_t size);
extern void md_swap2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], void* optr, const long istr[__VLA(D)], void* iptr, size_t size);
extern void md_swap(int D, const long dim[__VLA(D)],  void* optr, void* iptr, size_t size);
extern void md_circular_swap2(int M, int D, const long dims[__VLA(D)], const long* strs[__VLA(M)], void* ptr[__VLA(M)], size_t size);
extern void md_circular_swap(int M, int D, const long dims[__VLA(D)], void* ptr[__VLA(M)], size_t size);

extern void md_copy2(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], void* optr, const long istr[__VLA(D)], const void* iptr, size_t size);
extern void md_copy(int D, const long dim[__VLA(D)],  void* optr, const void* iptr, size_t size);
extern void md_copy_block2(int D, const long pos[__VLA(D)], const long odim[__VLA(D)], const long ostr[__VLA(D)], void* optr, const long idim[__VLA(D)], const long istr[__VLA(D)], const void* iptr, size_t size);
extern void md_copy_block(int D, const long pos[__VLA(D)], const long odim[__VLA(D)], void* optr, const long idim[__VLA(D)], const void* iptr, size_t size);
extern void md_move_block2(int D, const long dim[__VLA(D)], const long opos[__VLA(D)], const long odim[__VLA(D)], const long ostr[__VLA(D)], void* optr, const long ipos[__VLA(D)], const long idim[__VLA(D)], const long istr[__VLA(D)], const void* iptr, size_t size);
extern void md_move_block(int D, const long dim[__VLA(D)], const long opos[__VLA(D)], const long odim[__VLA(D)], void* optr, const long ipos[__VLA(D)], const long idim[__VLA(D)], const void* iptr, size_t size);

extern void md_pad(int D, const void* val, const long odim[__VLA(D)], void* optr, const long idim[__VLA(D)], const void* iptr, size_t size);
extern void md_pad_center(int D, const void* val, const long odim[__VLA(D)], void* optr, const long idim[__VLA(D)], const void* iptr, size_t size);
extern void md_resize(int D, const long odim[__VLA(D)], void* optr, const long idim[__VLA(D)], const void* iptr, size_t size);
extern void md_resize_center(int D, const long odim[__VLA(D)], void* optr, const long idim[__VLA(D)], const void* iptr, size_t size);
extern void md_fill2(int D, const long dim[__VLA(D)], const long str[__VLA(D)], void* ptr, const void* iptr, size_t size);
extern void md_fill(int D, const long dim[__VLA(D)], void* ptr, const void* iptr, size_t size);
extern void md_slice2(int D, unsigned long flags, const long pos[__VLA(D)], const long dim[__VLA(D)], const long ostr[__VLA(D)], void* optr, const long istr[__VLA(D)], const void* iptr, size_t size);
extern void md_slice(int D, unsigned long flags, const long pos[__VLA(D)], const long dim[__VLA(D)], void* optr, const void* iptr, size_t size);
extern void md_transpose2(int D, int dim1, int dim2, const long odims[__VLA(D)], const long ostr[__VLA(D)], void* optr, const long idims[__VLA(D)], const long istr[__VLA(D)], const void* iptr, size_t size);
extern void md_transpose(int D, int dim1, int dim2, const long odims[__VLA(D)], void* optr, const long idims[__VLA(D)], const void* iptr, size_t size);
extern void md_permute2(int D, const int order[__VLA(D)], const long odims[__VLA(D)], const long ostr[__VLA(D)], void* optr, const long idims[__VLA(D)], const long istr[__VLA(D)], const void* iptr, size_t size);
extern void md_permute(int D, const int order[__VLA(D)], const long odims[__VLA(D)], void* optr, const long idims[__VLA(D)], const void* iptr, size_t size);
extern void md_flip2(int D, const long dims[__VLA(D)], unsigned long flags, const long ostr[__VLA(D)], void* optr, const long istr[__VLA(D)], const void* iptr, size_t size);
extern void md_flip(int D, const long dims[__VLA(D)], unsigned long flags, void* optr, const void* iptr, size_t size);

extern void md_swap_flip2(int D, const long dims[__VLA(D)], unsigned long flags, const long ostr[__VLA(D)], void* optr, const long istr[__VLA(D)], void* iptr, size_t size);
extern void md_swap_flip(int D, const long dims[__VLA(D)], unsigned long flags, void* optr, void* iptr, size_t size);

extern void md_reshape(int D, unsigned long flags, const long odims[__VLA(D)], void* optr, const long idims[__VLA(D)], const void* iptr, size_t size);
extern void md_reshape2(int D, unsigned long flags, const long odims[__VLA(D)], const long ostrs[__VLA(D)], void* optr, const long idims[__VLA(D)], const long istrs[__VLA(D)], const void* iptr, size_t size);

extern void md_copy_diag2(int D, const long dims[__VLA(D)], unsigned long flags, const long str1[__VLA(D)], void* dst, const long str2[__VLA(D)], const void* src, size_t size);
extern void md_copy_diag(int D, const long dims[__VLA(D)], unsigned long flags, void* dst, const void* src, size_t size);
extern void md_fill_diag(int D, const long dims[__VLA(D)], unsigned long flags, void* dst, const void* src, size_t size);

extern void md_circ_shift2(int D, const long dim[__VLA(D)], const long center[__VLA(D)], const long str1[__VLA(D)], void* dst, const long str2[__VLA(D)], const void* src, size_t size);
extern void md_circ_shift(int D, const long dim[__VLA(D)], const long center[__VLA(D)], void* dst, const void* src, size_t size);

extern void md_circ_ext2(int D, const long dims1[__VLA(D)], const long strs1[__VLA(D)], void* dst, const long dims2[__VLA(D)], const long strs2[__VLA(D)], const void* src, size_t size);
extern void md_circ_ext(int D, const long dims1[__VLA(D)], void* dst, const long dims2[__VLA(D)], const void* src, size_t size);


extern void md_periodic2(int D, const long dims1[__VLA(D)], const long strs1[__VLA(D)], void* dst, const long dims2[__VLA(D)], const long strs2[__VLA(D)], const void* src, size_t size);
extern void md_periodic(int D, const long dims1[__VLA(D)], void* dst, const long dims2[__VLA(D)], const void* src, size_t size);

extern bool md_compare2(int D, const long dims[__VLA(D)], const long str1[__VLA(D)], const void* src1,
			const long str2[__VLA(D)], const void* src2, size_t size);
extern bool md_compare(int D, const long dims[__VLA(D)], const void* src1, const void* src2, size_t size);


typedef void* (*md_alloc_fun_t)(int D, const long dimensions[__VLA(D)], size_t size);

extern void* md_alloc(int D, const long dimensions[__VLA(D)], size_t size);
extern void* md_calloc(int D, const long dimensions[__VLA(D)], size_t size);
#ifdef USE_CUDA
extern void* md_alloc_gpu(int D, const long dimensions[__VLA(D)], size_t size);
extern void* md_gpu_move(int D, const long dims[__VLA(D)], const void* ptr, size_t size);
#endif
extern void* md_alloc_sameplace(int D, const long dimensions[__VLA(D)], size_t size, const void* ptr);
extern bool md_is_sameplace(const void* ptr1, const void* ptr2);
extern void md_free(const void* p);

struct cuda_threads_s;
extern struct cuda_threads_s* gpu_threads_create(const void* ref);
extern void gpu_threads_enter(struct cuda_threads_s* x);
extern void gpu_threads_leave(struct cuda_threads_s* x);
extern void gpu_threads_free(struct cuda_threads_s* x);


extern long md_calc_size(int D, const long dimensions[__VLA(D)]);
extern long* md_calc_strides(int D, long str[__VLA(D)], const long dim[__VLA(D)], size_t size);
extern long md_calc_offset(int D, const long strides[__VLA(D)], const long position[__VLA(D)]);
extern int md_calc_blockdim(int D, const long dim[__VLA(D)], const long str[__VLA(D)], size_t size);
extern void md_select_dims(int D, unsigned long flags, long odims[__VLA(D)], const long idims[__VLA(D)]);
extern void md_copy_dims(int D, long odims[__VLA(D)], const long idims[__VLA(D)]);
extern void md_copy_strides(int D, long odims[__VLA(D)], const long idims[__VLA(D)]);
extern void md_merge_dims(int D, long odims[__VLA(D)], const long dims1[__VLA(D)], const long dims2[__VLA(D)]);
extern bool md_check_compat(int D, unsigned long flags, const long dim1[__VLA(D)], const long dim2[__VLA(D)]);
extern bool md_check_bounds(int D, unsigned long flags, const long dim1[__VLA(D)], const long dim2[__VLA(D)]);
extern void md_singleton_dims(int D, long dims[__VLA(D)]);
extern void md_singleton_strides(int D, long strs[__VLA(D)]);
extern void md_set_dims(int D, long dims[__VLA(D)], long val);
extern void md_min_dims(int D, unsigned long flags, long odims[__VLA(D)], const long idims1[__VLA(D)], const long idims2[__VLA(D)]);
extern void md_max_dims(int D, unsigned long flags, long odims[__VLA(D)], const long idims1[__VLA(D)], const long idims2[__VLA(D)]);
extern bool md_is_index(int D, const long pos[__VLA(D)], const long dims[__VLA(D)]);
extern bool md_check_dimensions(int N, const long dims[__VLA(N)], unsigned long flags);
extern bool md_check_equal_dims(int N, const long dims1[__VLA(N)], const long dims2[__VLA(N)], unsigned long flags);
extern void md_permute_dims(int D, const int order[__VLA(D)], long odims[__VLA(D)], const long idims[__VLA(D)]);
extern void md_transpose_dims(int D, int dim1, int dim2, long odims[__VLA(D)], const long idims[__VLA(D)]);
extern bool md_next(int D, const long dims[__VLA(D)], unsigned long flags, long pos[__VLA(D)]);

extern void* md_compress(int D, const long dims[__VLA(D)], const float* src);
extern void md_decompress(int D, const long dims[__VLA(D)], float* dst, const void* src);

extern unsigned long md_nontriv_dims(int D, const long dims[__VLA(D)]);
extern unsigned long md_nontriv_strides(int D, const long dims[__VLA(D)]);


#define MD_INIT_ARRAY(x, y) { [ 0 ... ((x) - 1) ] = (y) }
#define MD_MAKE_ARRAY(T, ...) ((T[]){ __VA_ARGS__ })
#define MD_DIMS(...) MD_MAKE_ARRAY(long, __VA_ARGS__)

#define MD_BIT(x) (1ul << (x))
#define MD_IS_SET(x, y)	((x) & MD_BIT(y))
#define MD_CLEAR(x, y) ((x) & ~MD_BIT(y))
#define MD_SET(x, y)	((x) | MD_BIT(y))

extern int md_max_idx(unsigned long flags);
extern int md_min_idx(unsigned long flags);

#define MD_CAST_ARRAY2_PTR(T, N, dims, x, a, b) \
({						\
	int _a = (a), _b = (b);			\
	const long* _dims = dims;		\
	assert(_a < _b);			\
	assert(!md_check_dimensions((N), _dims, (1 << _a) | (1 << _b))); \
	(T (*)[_dims[_b]][_dims[_a]])(x);	\
})
#define MD_CAST_ARRAY3_PTR(T, N, dims, x, a, b, c) \
({						\
	int _a = (a), _b = (b), _c = (c);	\
	const long* _dims = dims;		\
	assert((_a < _b) && (_b < _c));		\
	assert(!md_check_dimensions((N), _dims, (1 << _a) | (1 << _b | (1 << _c)))); \
	(T (*)[_dims[_c]][_dims[_b]][_dims[_a]])(x); \
})

#define MD_CAST_ARRAY2(T, N, dims, x, a, b) (*MD_CAST_ARRAY2_PTR(T, N, dims, x, a, b))
#define MD_CAST_ARRAY3(T, N, dims, x, a, b, c) (*MD_CAST_ARRAY3_PTR(T, N, dims, x, a, b, c))


#define MD_ACCESS(N, strs, pos, x)	(*({ auto _x = (x); &((_x)[md_calc_offset((N), (strs), (pos)) / (long)sizeof((_x)[0])]); }))

#define MD_STRIDES(N, dims, elsize)	(md_calc_strides(N, alloca(N * sizeof(long)), dims, elsize))

#define MD_SINGLETON_DIMS(N)				\
({							\
	int _N = (N);					\
	long* _dims = alloca(_N * sizeof(long));	\
	md_singleton_dims(_N, _dims);			\
	_dims;						\
})

#define MD_SINGLETON_STRS(N)				\
({							\
	int _N = (N);					\
	long* _dims = alloca(_N * sizeof(long)); 	\
	md_singleton_strides(_N, _dims); 		\
	_dims; 						\
})

#include "misc/cppwrap.h"

#endif // __MULTIND_H

