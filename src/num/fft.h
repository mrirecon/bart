/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2011, 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#ifndef __FFT_H
#define __FFT_H

#include <stdbool.h>

#include "misc/cppwrap.h"

// similar to fftshift but modulates in the transform domain
extern void fftmod(int N, const long dims[__VLA(N)], unsigned long flags, _Complex float* dst, const _Complex float* src);
extern void fftmod2(int D, const long dimensions[__VLA(D)], unsigned long flags, const long ostrides[__VLA(D)], _Complex float* dst, const long istrides[__VLA(D)], const _Complex float* src);

// fftmod for ifft
extern void ifftmod(int N, const long dims[__VLA(N)], unsigned long flags, _Complex float* dst, const _Complex float* src);
extern void ifftmod2(int D, const long dimensions[__VLA(D)], unsigned long flags, const long ostrides[__VLA(D)], _Complex float* dst, const long istrides[__VLA(D)], const _Complex float* src);

// apply scaling necessary for unitarity
extern void fftscale(int N, const long dims[__VLA(N)], unsigned long flags, _Complex float* dst, const _Complex float* src);
extern void fftscale2(int D, const long dimensions[__VLA(D)], unsigned long flags, const long ostrides[__VLA(D)], _Complex float* dst, const long istrides[__VLA(D)], const _Complex float* src);

// fftshift
extern void fftshift(int N, const long dims[__VLA(N)], unsigned long flags, _Complex float* dst, const _Complex float* src);
extern void fftshift2(int D, const long dimensions[__VLA(D)], unsigned long flags, const long ostrides[__VLA(D)], _Complex float* dst, const long istrides[__VLA(D)], const _Complex float* src);

// ifftshift
extern void ifftshift(int N, const long dims[__VLA(N)], unsigned long flags, _Complex float* dst, const _Complex float* src);
extern void ifftshift2(int D, const long dimensions[__VLA(D)], unsigned long flags, const long ostrides[__VLA(D)], _Complex float* dst, const long istrides[__VLA(D)], const _Complex float* src);



// FFT
extern void fft(int D, const long dimensions[__VLA(D)], unsigned long flags, _Complex float* dst, const _Complex float* src);
extern void ifft(int D, const long dimensions[__VLA(D)], unsigned long flags, _Complex float* dst, const _Complex float* src);
extern void fft2(int D, const long dimensions[__VLA(D)], unsigned long flags, const long ostrides[__VLA(D)], _Complex float* dst, const long istrides[__VLA(D)], const _Complex float* src);
extern void ifft2(int D, const long dimensions[__VLA(D)], unsigned long flags, const long ostrides[__VLA(D)], _Complex float* dst, const long istrides[__VLA(D)], const _Complex float* src);

// centered
extern void fftc(int D, const long dimensions[__VLA(D)], unsigned long flags, _Complex float* dst, const _Complex float* src);
extern void ifftc(int D, const long dimensions[__VLA(D)], unsigned long flags, _Complex float* dst, const _Complex float* src);
extern void fftc2(int D, const long dimensions[__VLA(D)], unsigned long flags, const long ostrides[__VLA(D)], _Complex float* dst, const long istrides[__VLA(D)], const _Complex float* src);
extern void ifftc2(int D, const long dimensions[__VLA(D)], unsigned long flags, const long ostrides[__VLA(D)], _Complex float* dst, const long istrides[__VLA(D)], const _Complex float* src);

// unitary
extern void fftu(int D, const long dimensions[__VLA(D)], unsigned long flags, _Complex float* dst, const _Complex float* src);
extern void ifftu(int D, const long dimensions[__VLA(D)], unsigned long flags, _Complex float* dst, const _Complex float* src);
extern void fftu2(int D, const long dimensions[__VLA(D)], unsigned long flags, const long ostrides[__VLA(D)], _Complex float* dst, const long istrides[__VLA(D)], const _Complex float* src);
extern void ifftu2(int D, const long dimensions[__VLA(D)], unsigned long flags, const long ostrides[__VLA(D)], _Complex float* dst, const long istrides[__VLA(D)], const _Complex float* src);

// unitary and centered
extern void fftuc(int D, const long dimensions[__VLA(D)], unsigned long flags, _Complex float* dst, const _Complex float* src);
extern void ifftuc(int D, const long dimensions[__VLA(D)], unsigned long flags, _Complex float* dst, const _Complex float* src);
extern void fftuc2(int D, const long dimensions[__VLA(D)], unsigned long flags, const long ostrides[__VLA(D)], _Complex float* dst, const long istrides[__VLA(D)], const _Complex float* src);
extern void ifftuc2(int D, const long dimensions[__VLA(D)], unsigned long flags, const long ostrides[__VLA(D)], _Complex float* dst, const long istrides[__VLA(D)], const _Complex float* src);




struct operator_s;
extern const struct operator_s* fft_create(int D, const long dimensions[__VLA(D)], unsigned long flags, _Complex float* dst, const _Complex float* src, _Bool backwards);
extern const struct operator_s* fft_create2(int D, const long dimensions[__VLA(D)], unsigned long flags, const long ostrides[__VLA(D)], _Complex float* dst, const long istrides[__VLA(D)], const _Complex float* src, _Bool backwards);

extern const struct operator_s* fft_measure_create(int D, const long dimensions[__VLA(D)], unsigned long flags, _Bool inplace, _Bool backwards);



// interface using a plan
extern void fft_exec(const struct operator_s* plan, _Complex float* dst, const _Complex float* src);
extern void fft_free(const struct operator_s* plan);

extern _Bool use_fftw_wisdom;
extern void fft_set_num_threads(int n);


#include "misc/cppwrap.h"

#endif

