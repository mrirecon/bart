/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "num/multind.h"

#include "misc/cppwrap.h"

extern void md_wavtrafo2(int D, const long dims[__VLA(D)], unsigned int flags, const long strs[__VLA(D)], void* ptr, md_trafo_fun_t fun, _Bool inv, _Bool nosort);
extern void md_wavtrafo(int D, const long dims[__VLA(D)], unsigned int flags, void* ptr, size_t size, md_trafo_fun_t fun, _Bool inv, _Bool nosort);
extern void md_wavtrafoz2(int D, const long dims[__VLA(D)], unsigned int flags, const long strs[__VLA(D)], _Complex float* x, md_trafo_fun_t fun, _Bool inv, _Bool nosort);
extern void md_wavtrafoz(int D, const long dims[__VLA(D)], unsigned int flags, _Complex float* ptr, md_trafo_fun_t fun, _Bool inv, _Bool nosort);
extern void md_cdf97z(int D, const long dims[__VLA(D)], unsigned int flags, _Complex float* data);
extern void md_icdf97z(int D, const long dims[__VLA(D)], unsigned int flags, _Complex float* data);
extern void md_cdf97z2(int D, const long dims[__VLA(D)], unsigned int flags, const long strs[__VLA(D)], _Complex float* data);
extern void md_icdf97z2(int D, const long dims[__VLA(D)], unsigned int flags, const long strs[__VLA(D)], _Complex float* data);
extern void md_resortz(int D, const long dims[__VLA(D)], unsigned int flags, _Complex float* data);
extern void md_iresortz(int D, const long dims[__VLA(D)], unsigned int flags, _Complex float* data);


#include "misc/cppwrap.h"


