/* Copyright 2014. The Regents of the University of California.
 * Copyright 2016-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifdef __cplusplus
#error This file does not support C++
#endif

#include <stdlib.h>

extern void merge_dims(unsigned int D, unsigned int N, long dims[N], long (*ostrs[D])[N]);
extern unsigned int remove_empty_dims(unsigned int D, unsigned int N, long dims[N], long (*ostrs[D])[N]);

extern unsigned int simplify_dims(unsigned int D, unsigned int N, long dims[N], long (*strs[D])[N]);
extern unsigned int optimize_dims(unsigned int D, unsigned int N, long dims[N], long (*strs[D])[N]);
extern unsigned int min_blockdim(unsigned int D, unsigned int N, const long dims[N], long (*strs[D])[N], size_t size[D]);
extern unsigned long dims_parallel(unsigned int D, unsigned int io, unsigned int N, const long dims[N], long (*strs[D])[N], size_t size[D]);


struct vec_ops;

struct nary_opt_data_s {

	long size;
	const struct vec_ops* ops;
	void* data_ptr;
};



typedef void (*md_nary_opt_fun_t)(struct nary_opt_data_s* data, void* ptr[]);

extern void optimized_nop(unsigned int N, unsigned int io, unsigned int D, const long dim[D], const long (*nstr[N])[D], void* const nptr[N], size_t sizes[N], md_nary_opt_fun_t too, void* data_ptr);

