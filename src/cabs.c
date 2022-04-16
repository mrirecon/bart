/* Copyright 2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2016 Siddharth Iyer <sid8795@gmail.com>
 */

#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdio.h>

#include "num/init.h"
#include "num/flpmath.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif

static const char help_str[] = "Absolute value of array (|<input>|).";


int main_cabs(int argc, char* argv[argc])
{
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = { };

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long dims[DIMS];

	complex float* idata = load_cfl(in_file, DIMS, dims);
	complex float* odata = create_cfl(out_file, DIMS, dims);
		
        md_zabs(DIMS, dims, odata, idata);

	unmap_cfl(DIMS, dims, idata);
	unmap_cfl(DIMS, dims, odata);

	return 0;
}


