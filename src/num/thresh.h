/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef __THRESH
#define __THRESH

#ifdef __cplusplus

extern "C" {
#ifndef __VLA
#define __VLA(x)
#endif
#else
#ifndef __VLA
#define __VLA(x) static x
#endif
#endif


extern void md_zsoftthresh_core2(unsigned int D, const long dims[__VLA(D)], float lambda, unsigned int flags, _Complex float* tmp_norm, const long ostrs[__VLA(D)], _Complex float* optr, const long istrs[__VLA(D)], const _Complex float* iptr);

extern void md_zsoftthresh2(unsigned int D, const long dim[__VLA(D)], float lambda, unsigned int flags, const long ostr[__VLA(D)], _Complex float* optr, const long istr[__VLA(D)], const _Complex float* iptr);
extern void md_zsoftthresh(unsigned int D, const long dim[__VLA(D)], float lambda, unsigned int flags, _Complex float* optr, const _Complex float* iptr);

struct operator_p_s;
extern const struct operator_p_s* prox_thresh_create(unsigned int D, const long dim[__VLA(D)], const float lambda, const unsigned long flags, _Bool gpu);

extern void thresh_free(const struct operator_p_s* data);
	
extern void set_thresh_lambda(const struct operator_p_s* o, const float lambda);

extern void softthresh(const struct operator_p_s* o, _Complex float* optr, const _Complex float* iptr);


struct linop_s;

extern const struct operator_p_s* prox_unithresh_create(unsigned int D, const struct linop_s* unitary_op, const float lambda, const unsigned long flags, _Bool gpu);






#ifdef __cplusplus
}
#endif

#endif


