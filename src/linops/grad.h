/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/cppwrap.h"

extern struct linop_s* linop_grad_forward_create(long N, const long dims[__VLA(N)], int d, unsigned long flags);
extern struct linop_s* linop_grad_backward_create(long N, const long dims[__VLA(N)], int d, unsigned long flags);
extern struct linop_s* linop_grad_zentral_create(long N, const long dims[__VLA(N)], int d, unsigned long flags);

extern struct linop_s* linop_grad_create(long N, const long dims[__VLA(N)], int d, unsigned long flags);

#include "misc/cppwrap.h"

