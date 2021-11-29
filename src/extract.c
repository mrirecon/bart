/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#include <complex.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"


#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char help_str[] = "Extracts a sub-array along dims from index start to (not including) end.";


int main_extract(int argc, char* argv[argc])
{
	long count = 0;
	long* dims = NULL;
	long* starts = NULL;
	long* ends = NULL;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_TUPLE(true, &count, 3, TUPLE_LONG(&dims, "dim"),
					   TUPLE_LONG(&starts, "start"),
					   TUPLE_LONG(&ends, "end")),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};


	const struct opt_s opts[] = { };
	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long in_dims[DIMS];
	long out_dims[DIMS];
	
	complex float* in_data = load_cfl(in_file, DIMS, in_dims);
	md_copy_dims(DIMS, out_dims, in_dims);


	long pos2[DIMS] = { [0 ... DIMS - 1] = 0 };

	for (long i = 0; i < count; i++) {

		long dim = dims[i];
		long start = starts[i];
		long end = ends[i];

		assert((0 <= dim) && (dim < DIMS));
		assert(start >= 0);
		assert(start < end);
		assert(end <= in_dims[dim]);

		out_dims[dim] = end - start;
		pos2[dim] = start;
	}


	complex float* out_data = create_cfl(out_file, DIMS, out_dims);

	md_copy_block(DIMS, pos2, out_dims, out_data, in_dims, in_data, CFL_SIZE);

	unmap_cfl(DIMS, in_dims, in_data);
	unmap_cfl(DIMS, out_dims, out_data);
	xfree(dims);
	xfree(starts);
	xfree(ends);
	return 0;
}


