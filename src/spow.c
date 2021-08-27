/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2014 Martin Uecker <uecker@eecs.berkeley.edu>
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

static const char help_str[] = "Raise array to the power of {exponent}. The exponent can be a complex number.";


int main_spow(int argc, char* argv[argc])
{
	complex float expo = 0;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_CFL(true, &expo, "exponent"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};
	const struct opt_s opts[] = {};
	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	const int N = DIMS;
	long dims[N];
	complex float* idata = load_cfl(in_file, N, dims);
	complex float* odata = create_cfl(out_file, N, dims);
		
	md_zspow(N, dims, odata, idata, expo);

	unmap_cfl(N, dims, idata);
	unmap_cfl(N, dims, odata);
	return 0;
}


