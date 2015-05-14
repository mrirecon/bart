/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2012-10-28 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <complex.h>

#include "misc/cppwrap.h"

#ifndef __CONV_ENUMS
#define __CONV_ENUMS
enum conv_mode { CONV_SYMMETRIC, CONV_CAUSAL, CONV_ANTICAUSAL };
enum conv_type { CONV_CYCLIC, CONV_TRUNCATED, CONV_VALID, CONV_EXTENDED };
#endif


struct conv_plan;

extern struct conv_plan* conv_plan(int N, unsigned int flags, enum conv_type ctype, enum conv_mode cmode, const long odims[__VLA(N)],  
		const long idims1[__VLA(N)], const long idims2[__VLA(N)], const complex float* src2);
extern void conv_exec(struct conv_plan* plan, complex float* dst, const complex float* src1);
extern void conv_adjoint(struct conv_plan* plan, complex float* dst, const complex float* src1);
extern void conv_free(struct conv_plan* plan);
extern void conv(int N, unsigned int flags, enum conv_type ctype, enum conv_mode cmode, const long odims[__VLA(N)], complex float* dst,
		const long idims1[__VLA(N)], const complex float* src1, const long idims2[__VLA(N)], const complex float* src2);
extern void convH(int N, unsigned int flags, enum conv_type ctype, enum conv_mode cmode, const long odims[__VLA(N)], complex float* dst, 
		const long idims1[__VLA(N)], const complex float* src1, const long idims2[__VLA(N)], const complex float* src2);

#include "misc/cppwrap.h"
