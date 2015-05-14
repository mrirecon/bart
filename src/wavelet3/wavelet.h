/* Copyright 2014-2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */


#include <complex.h>
#include <stdbool.h>

extern const float wavelet3_haar[2][2][2];
extern const float wavelet3_dau2[2][2][4];
extern const float wavelet3_cdf44[2][2][10];

// layer 1

extern void fwt1(unsigned int N, unsigned int d, const long dims[N], const long ostr[N], complex float* low, complex float* hgh, const long istr[N], const complex float* in, const long flen, const float filter[2][2][flen]);
extern void iwt1(unsigned int N, unsigned int d, const long dims[N], const long ostr[N], complex float* out, const long istr[N], const complex float* low, const complex float* hgh, const long flen, const float filter[2][2][flen]);

// layer 2

extern void fwtN(unsigned int N, unsigned int flags, const long shifts[N], const long dims[N], const long ostr[2 * N], complex float* out, const long istr[N], const complex float* in, const long flen, const float filter[2][2][flen]);
extern void iwtN(unsigned int N, unsigned int flags, const long shifts[N], const long dims[N], const long ostr[N], complex float* out, const long istr[2 * N], const complex float* in, const long flen, const float filter[2][2][flen]);

extern void wavelet_dims(unsigned int N, unsigned int flags, long odims[2 * N], const long dims[N], const long flen);

// layer 3

extern void fwt(unsigned int N, unsigned int flags, const long shifts[N], const long dims[N], complex float* out, const long istr[N], const complex float* in, const long minsize[N], const long flen, const float filter[2][2][flen]);
extern void iwt(unsigned int N, unsigned int flags, const long shifts[N], const long dims[N], const long ostr[N], complex float* out, const complex float* in, const long minsize[N], const long flen, const float filter[2][2][flen]);

extern long wavelet_num_levels(unsigned int N, unsigned int flags, const long dims[N], const long min[N], const long flen);
extern long wavelet_coeffs(unsigned int N, unsigned int flags, const long dims[N], const long min[N], const long flen);


extern void wavelet3_thresh(unsigned int N, float lambda, unsigned int flags, const long shifts[N], const long dims[N], complex float* out, const complex float* in, const long minsize[N], long flen, const float filter[2][2][flen]);


