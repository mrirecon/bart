/* Copyright 2014-2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/cppwrap.h"

struct operator_s;
struct linop_s;

extern const struct operator_s* nufft_precond_create(const struct linop_s* nufft_op);

#include "misc/cppwrap.h"

