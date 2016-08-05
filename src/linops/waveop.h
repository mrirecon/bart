/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/cppwrap.h"

extern struct linop_s* linop_wavelet3_create(unsigned int N, unsigned int flags, const long dims[__VLA(N)], const long istr[__VLA(N)], const long minsize[__VLA(N)]);

#include "misc/cppwrap.h"

