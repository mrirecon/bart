/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */ 

#include <complex.h>

#include "misc/cppwrap.h"

typedef complex float (*sample_fun_t)(void* _data, const long pos[]);



extern void md_zsample(unsigned int N, const long dims[__VLA(N)], complex float* z, void* data, sample_fun_t fun);
extern void md_parallel_zsample(unsigned int N, const long dims[__VLA(N)], complex float* z, void* data, sample_fun_t fun);

extern void md_zgradient(unsigned int N, const long dims[__VLA(N)], complex float* out, const complex float grad[__VLA(N)]);

typedef complex float (*map_fun_data_t)(void* _data, complex float arg);
typedef complex float (*map_fun_t)(complex float arg);


extern void md_zmap_data(unsigned int N, const long dims[__VLA(N)], complex float* out, const complex float* in, void* data, map_fun_data_t fun);
extern void md_zmap(unsigned int N, const long dims[__VLA(N)], complex float* out, const complex float* in, map_fun_t fun);


#include "misc/cppwrap.h"

