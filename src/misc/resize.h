/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

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


extern void sinc_resize(unsigned int D, const long out_dims[__VLA(D)], _Complex float* out, const long in_dims[__VLA(D)], const _Complex float* in);
extern void sinc_zeropad(unsigned int D, const long out_dims[__VLA(D)], _Complex float* out, const long in_dims[__VLA(D)], const _Complex float* in);
extern void fft_zeropad(unsigned int D, unsigned int flags, const long out_dims[__VLA(D)], _Complex float* out, const long in_dims[__VLA(D)], const _Complex float* in);
extern void fft_zeropadH(unsigned int D, unsigned int flags, const long out_dims[__VLA(D)], _Complex float* out, const long in_dims[__VLA(D)], const _Complex float* in);



#ifdef __cplusplus
extern "C" {
#undef __VLA
#endif

