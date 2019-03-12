/* Copyright 2019. Massachusetts Institute of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2019 Siddharth Iyer <ssi@mit.edu>
 */

#include "misc/cppwrap.h"

extern struct linop_s* linop_decompose_complex_create(unsigned int N, unsigned int D, const long dims[__VLA(N)]);

#include "misc/cppwrap.h"
