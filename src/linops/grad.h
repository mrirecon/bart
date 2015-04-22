/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

extern void grad(unsigned int D, const long dims[D], unsigned int flags, complex float* out, const complex float* in);
extern void grad_op(unsigned int D, const long dims[D], unsigned int flags, complex float* out, const complex float* in);
extern void grad_adjoint(unsigned int D, const long dims[D], unsigned int flags, complex float* out, const complex float* in);

extern struct linop_s* grad_init(long N, const long dims[N], unsigned int flags);

