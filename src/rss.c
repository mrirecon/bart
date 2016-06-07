/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2012 Martin Uecker <uecker@eecs.berkeley.edu.
 */

#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>


#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"

#ifndef DIMS
#define DIMS 16
#endif



static const char usage_str[] = "bitmask <input> <output>";
static const char help_str[] = "Calculates root of sum of squares along selected dimensions.\n";


int main_rss(int argc, char* argv[argc])
{
	mini_cmdline(argc, argv, 3, usage_str, help_str);

	num_init();

	long dims[DIMS];
	complex float* data = load_cfl(argv[2], DIMS, dims);

	int flags = atoi(argv[1]);

	assert(0 <= flags);

	long odims[DIMS];
	md_select_dims(DIMS, ~flags, odims, dims);

	complex float* out = create_cfl(argv[3], DIMS, odims);

	md_zrss(DIMS, dims, flags, out, data);

	unmap_cfl(DIMS, dims, data);
	unmap_cfl(DIMS, odims, out);

	exit(0);
}


