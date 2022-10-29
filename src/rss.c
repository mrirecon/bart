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
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif



static const char help_str[] = "Calculates root of sum of squares along selected dimensions.";


int main_rss(int argc, char* argv[argc])
{
	unsigned long flags = 0;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_ULONG(true, &flags, "bitmask"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = {};
	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long dims[DIMS];
	complex float* data = load_cfl(in_file, DIMS, dims);

	long odims[DIMS];
	md_select_dims(DIMS, ~flags, odims, dims);

	complex float* out = create_cfl(out_file, DIMS, odims);

	md_zrss(DIMS, dims, flags, out, data);

	unmap_cfl(DIMS, dims, data);
	unmap_cfl(DIMS, odims, out);

	return 0;
}


