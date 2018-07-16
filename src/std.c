/* Copyright 2017. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2017	Jon Tamir <jtamir@eecs.berkeley.edu>
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/init.h"
#include "num/flpmath.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif



static const char usage_str[] = "bitmask <input> <output>";
static const char help_str[] = "Compute standard deviation along selected dimensions specified by the {bitmask}";



int main_std(int argc, char* argv[])
{
	mini_cmdline(&argc, argv, 3, usage_str, help_str);

	num_init();

	long flags = atoi(argv[1]);

	long idims[DIMS];
	long odims[DIMS];

	complex float* in = load_cfl(argv[2], DIMS, idims);

	md_select_dims(DIMS, ~flags, odims, idims);

	complex float* out = create_cfl(argv[3], DIMS, odims);

	md_zstd(DIMS, idims, flags, out, in);

	unmap_cfl(DIMS, idims, in);
	unmap_cfl(DIMS, odims, out);

	return 0;
}




