/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef __BLOCKPROC
#define __BLOCKPROC

#include "misc/cppwrap.h"

extern float lineproc2(unsigned int D, const long dims[__VLA(D)], const long blkdims[__VLA(D)], const long line_dims[__VLA(D)], const void * data,
			 float (*op)(const void* data, const long blkdims[__VLA(D)], complex float* dst, const complex float* src),
			 const long ostrs[__VLA(D)], complex float* dst, const long istrs[__VLA(D)], const complex float* src);


extern float lineproc(unsigned int D, const long dims[__VLA(D)], const long blkdims[__VLA(D)], const long line_dims[__VLA(D)], const void * data,
			float (*op)(const void* data, const long blkdims[__VLA(D)], complex float* dst, const complex float* src),
			complex float* dst, const complex float* src);


extern float blockproc2(unsigned int D, const long dims[__VLA(D)], const long blkdims[__VLA(D)], const void * data,
			 float (*op)(const void* data, const long blkdims[__VLA(D)], complex float* dst, const complex float* src),
			 const long ostrs[__VLA(D)], complex float* dst, const long istrs[__VLA(D)], const complex float* src);


extern float blockproc(unsigned int D, const long dims[__VLA(D)], const long blkdims[__VLA(D)], const void * data,
			float (*op)(const void* data, const long blkdims[__VLA(D)], complex float* dst, const complex float* src),
			complex float* dst, const complex float* src);


extern float blockproc_shift2(unsigned int D, const long dims[__VLA(D)], const long blkdims[__VLA(D)], const long shifts[__VLA(D)], const void* data,
			       float (*op)(const void* data, const long blkdims[__VLA(D)], complex float* dst, const complex float* src), 
			       const long ostrs[__VLA(D)], complex float* dst, const long istrs[__VLA(D)], const complex float* src);


extern float blockproc_shift(unsigned int D, const long dims[__VLA(D)], const long blkdims[__VLA(D)], const long shifts[__VLA(D)], const void* data,
			      float (*op)(const void* data, const long blkdims[__VLA(D)], complex float* dst, const complex float* src), 
			      complex float* dst, const complex float* src);




extern float blockproc_circshift(unsigned int D, const long dims[__VLA(D)], const long blkdims[__VLA(D)], const long shifts[__VLA(D)], const void* data,
			      float (*op)(const void* data, const long blkdims[__VLA(D)], complex float* dst, const complex float* src), 
			      complex float* dst, const complex float* src);


extern float blockproc_shift_mult2(unsigned int D, const long dims[__VLA(D)], const long blkdims[__VLA(D)], const long shifts[__VLA(D)], const long mult[__VLA(D)], const void* data,
			       float (*op)(const void* data, const long blkdims[__VLA(D)], complex float* dst, const complex float* src),
			       const long ostrs[__VLA(D)], complex float* dst, const long istrs[__VLA(D)], const complex float* src);


extern float blockproc_shift_mult(unsigned int D, const long dims[__VLA(D)], const long blkdims[__VLA(D)], const long shifts[__VLA(D)], const long mult[__VLA(D)], const void* data,
			      float (*op)(const void* data, const long blkdims[__VLA(D)], complex float* dst, const complex float* src),
			      complex float* dst, const complex float* src);

extern float stackproc2(unsigned int D, const long dims[__VLA(D)], const long blkdims[__VLA(D)], unsigned int stkdim, const void* data,
			float (*op)(const void* data, const long stkdims[__VLA(D)], complex float* dst, const complex float* src),
			const long ostrs[__VLA(D)], complex float* dst, const long istrs[__VLA(D)], const complex float* src);

extern float stackproc(unsigned int D, const long dims[__VLA(D)], const long blkdims[__VLA(D)], unsigned int stkdim, const void* data,
			float (*op)(const void* data, const long stkdims[__VLA(D)], complex float* dst, const complex float* src),
			complex float* dst, const complex float* src);


#include "misc/cppwrap.h"

#endif


