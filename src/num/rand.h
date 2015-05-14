/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/cppwrap.h"

extern double uniform_rand(void);
extern _Complex double gaussian_rand(void);
extern void md_gaussian_rand(unsigned int D, const long dims[__VLA(D)], _Complex float* dst);

extern void num_rand_init(unsigned int seed);

#include "misc/cppwrap.h"

