/* Copyright 2013-2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#ifndef _FINITE_DIFF
#define _FINITE_DIFF

#include "misc/cppwrap.h"


extern void md_zfinitediff(int D, const long dim[__VLA(D)], unsigned long flags, _Bool snip, _Complex float* optr, const _Complex float* iptr);
extern void md_zfinitediff2(int D, const long dim[__VLA(D)], unsigned long flags, _Bool snip, const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

extern void md_zcumsum(int D, const long dim[__VLA(D)], unsigned long flags, _Complex float* optr, const _Complex float* iptr);
extern void md_zcumsum2(int D, const long dim[__VLA(D)], unsigned long flags, const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);

struct linop_s;
extern const struct linop_s* linop_finitediff_create(int D, const long dim[__VLA(D)], const unsigned long flags, _Bool snip);

extern void fd_proj_noninc(const struct linop_s* o, _Complex float* optr, const _Complex float* iptr);


/**
 * Circular finite difference operator
 *      (without "snipping" or first elements)
 *
 * @param D number of dimensions
 * @param dim dimensions
 * @param diffdim specifies the direction to perform the operation
 * @param circular indicates whether it a circular operation
 *
 * Joseph Y Cheng (jycheng@stanford.edu)
 */
const struct linop_s* linop_zfinitediff_create(int D,
				    const long dims[__VLA(D)],
				    const long diffdim,
				    _Bool circular);



  
#include "misc/cppwrap.h"

#endif



