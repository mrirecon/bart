
#include <complex.h>

#include "misc/cppwrap.h"

#ifndef _CONV_ENUMS
#define _CONV_ENUMS
enum conv_mode { CONV_SYMMETRIC, CONV_CAUSAL, CONV_ANTICAUSAL };
enum conv_type { CONV_CYCLIC, CONV_TRUNCATED, CONV_VALID, CONV_EXTENDED };
#endif


struct conv_plan;

extern struct conv_plan* conv_plan(int N, unsigned long flags, enum conv_type ctype, enum conv_mode cmode, const long odims[__VLA(N)],  
		const long idims1[__VLA(N)], const long idims2[__VLA(N)], const complex float* src2);
extern void conv_exec(struct conv_plan* plan, complex float* dst, const complex float* src1);
extern void conv_adjoint(struct conv_plan* plan, complex float* dst, const complex float* src1);
extern void conv_free(struct conv_plan* plan);
extern void conv(int N, unsigned long flags, enum conv_type ctype, enum conv_mode cmode, const long odims[__VLA(N)], complex float* dst,
		const long idims1[__VLA(N)], const complex float* src1, const long idims2[__VLA(N)], const complex float* src2);
extern void convH(int N, unsigned long flags, enum conv_type ctype, enum conv_mode cmode, const long odims[__VLA(N)], complex float* dst, 
		const long idims1[__VLA(N)], const complex float* src1, const long idims2[__VLA(N)], const complex float* src2);

#include "misc/cppwrap.h"
