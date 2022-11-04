/* Copyright 2015. The Regents of the University of California.
 * Copyright 2022. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/cppwrap.h"

#ifndef WTYPE
#define WTYPE
enum wtype { WAVELET_HAAR, WAVELET_DAU2, WAVELET_CDF44 };
#endif

extern struct linop_s* linop_wavelet_create(int N, unsigned long flags, const long dims[__VLA(N)], const long istr[__VLA(N)],
						enum wtype wtype, const long minsize[__VLA(N)], _Bool randshift);

#include "misc/cppwrap.h"

