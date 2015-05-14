/* Copyright 2013-2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/cppwrap.h"

extern void casorati_dims(unsigned int N, long odim[2], const long dimk[__VLA(N)], const long dims[__VLA(N)]);
extern void casorati_matrix(unsigned int N, const long dimk[__VLA(N)], const long odim[2], _Complex float* optr, const long dim[__VLA(N)], const long str[__VLA(N)], const _Complex float* iptr);
extern void casorati_matrixH(unsigned int N, const long dimk[__VLA(N)], const long dim[__VLA(N)], const long str[__VLA(N)], _Complex float* optr, const long odim[2], const _Complex float* iptr);


extern void basorati_dims(unsigned int N, long odim[2], const long dimk[__VLA(N)], const long dims[__VLA(N)]);
extern void basorati_matrix(unsigned int N, const long dimk[__VLA(N)], const long odim[2], _Complex float* optr, const long dim[__VLA(N)], const long str[__VLA(N)], const _Complex float* iptr);
extern void basorati_matrixH(unsigned int N, const long dimk[__VLA(N)], const long dim[__VLA(N)], const long str[__VLA(N)], _Complex float* optr, const long odim[2], const _Complex float* iptr);

#include "misc/cppwrap.h"
