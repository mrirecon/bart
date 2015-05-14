/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifdef __cplusplus
#error This file does not support C++
#endif

#include <stdlib.h>

extern void merge_dims(unsigned int D, unsigned int N, long dims[N], long (*ostrs[D])[N]);
extern unsigned int remove_empty_dims(unsigned int D, unsigned int N, long dims[N], long (*ostrs[D])[N]);
extern unsigned int optimize_dims(unsigned int D, unsigned int N, long dims[N], long (*strs[D])[N]);
extern unsigned int min_blockdim(unsigned int D, unsigned int N, const long dims[N], long (*strs[D])[N], size_t size[D]);
extern unsigned int dims_parallel(unsigned int D, unsigned int io, unsigned int N, const long dims[N], long (*strs[D])[N], size_t size[D]);


