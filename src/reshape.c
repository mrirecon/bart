/* Copyright 2015. The Regents of the University of California.
 * Copyright 2015-2020. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2015-2020 Martin Uecker
 */

#include <complex.h>
#include <assert.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#define DIMS 16

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char help_str[] = "Reshape selected dimensions.";


int main_reshape(int argc, char* argv[argc])
{
	unsigned long flags = 0;
	long count = 0;
	long* dims = NULL;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_ULONG(true, &flags, "flags"),
		ARG_TUPLE(true, &count, 1, TUPLE_LONG(&dims, "dim")),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = { };

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	int n = bitcount(flags);

	assert(n == count);

	long in_dims[DIMS];
	long out_dims[DIMS];

	complex float* in_data = load_cfl(in_file, DIMS, in_dims);

	md_copy_dims(DIMS, out_dims, in_dims);
	
	int j = 0;

	for (int i = 0; i < DIMS; i++)
		if (MD_IS_SET(flags, i))
			out_dims[i] = dims[j++];

	assert(j == n);

	complex float* out_data = create_cfl(out_file, DIMS, out_dims);

	md_reshape(DIMS, flags, out_dims, out_data, in_dims, in_data, CFL_SIZE);

	unmap_cfl(DIMS, in_dims, in_data);
	unmap_cfl(DIMS, out_dims, out_data);
	xfree(dims);

	return 0;
}


