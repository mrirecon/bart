/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <string.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/io.h"
#include "misc/misc.h"
#include "misc/opts.h"


static const char help_str[] = "Create a vector of values.";



int main_vec(int argc, char* argv[argc])
{
	long count = 0;
	complex float* vals = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_TUPLE(true, &count, 1, { OPT_CFL, sizeof(complex float), &vals, "val" }),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = {};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long dims[1] = { count };

	complex float* x = create_cfl(out_file, 1, dims);

	for (int i = 0; i < count; i++)
		x[i] = vals[i];

	unmap_cfl(1, dims, x);

	xfree(vals);

	return 0;
}


