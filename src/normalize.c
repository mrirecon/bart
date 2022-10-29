/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012, 2014 Martin Uecker <uecker@eecs.berkeley.edu>
 */
 
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/opts.h"
#include "misc/utils.h"


#ifndef DIMS
#define DIMS 16
#endif





static const char help_str[] = "Normalize along selected dimensions.";



int main_normalize(int argc, char* argv[argc])
{
	unsigned long flags = 0;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_ULONG(true, &flags, "flags"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	bool l1 = false;

	const struct opt_s opts[] = {

		OPT_SET('b', &l1, "l1"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	int N = DIMS;
	long dims[N];
	complex float* data = load_cfl(in_file, N, dims);

	complex float* out = create_cfl(out_file, N, dims);

	md_copy(N, dims, out, data, CFL_SIZE);

	(l1 ? normalizel1 : normalize)(N, flags, dims, out);

	unmap_cfl(N, dims, out);

	return 0;
}


