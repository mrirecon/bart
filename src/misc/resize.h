/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/cppwrap.h"

extern void sinc_resize(int D, const long out_dims[__VLA(D)], _Complex float* out, const long in_dims[__VLA(D)], const _Complex float* in);
extern void sinc_zeropad(int D, const long out_dims[__VLA(D)], _Complex float* out, const long in_dims[__VLA(D)], const _Complex float* in);
extern void fft_zeropad(int D, long flags, const long out_dims[__VLA(D)], _Complex float* out, const long in_dims[__VLA(D)], const _Complex float* in);
extern void fft_zeropadH(int D, long flags, const long out_dims[__VLA(D)], _Complex float* out, const long in_dims[__VLA(D)], const _Complex float* in);

#include "misc/cppwrap.h"



