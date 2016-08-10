/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/cppwrap.h"

extern void grad(unsigned int D, const long dims[__VLA(D)], unsigned int flags, complex float* out, const complex float* in);
extern void grad_op(unsigned int D, const long dims[__VLA(D)], unsigned int flags, complex float* out, const complex float* in);
extern void grad_adjoint(unsigned int D, const long dims[__VLA(D)], unsigned int flags, complex float* out, const complex float* in);

extern struct linop_s* linop_grad_create(long N, const long dims[__VLA(N)], unsigned int flags);

#include "misc/cppwrap.h"

