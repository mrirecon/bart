/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012 Martin Uecker <uecker@eecs.berkeley.edu>
 */


#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif


static const char help_str[] = "Extracts a sub-array corresponding to the central part of {size} along {dimension}";


int main_crop(int argc, char* argv[argc])
{
	int dim = 0;
	int count = 0;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INT(true, &dim, "dimension"),
		ARG_INT(true, &count, "size"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = {};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	int N = DIMS;
	long in_dims[N];
	long out_dims[N];
	
	complex float* in_data = load_cfl(in_file, N, in_dims);

	assert(dim < N);
	assert(count >= 1);
	
	for (int i = 0; i < N; i++)
		out_dims[i] = in_dims[i];

	out_dims[dim] = count;

	complex float* out_data = create_cfl(out_file, N, out_dims);

	md_resize_center(N, out_dims, out_data, in_dims, in_data, sizeof(complex float));

	unmap_cfl(N, in_dims, in_data);
	unmap_cfl(N, out_dims, out_data);

	return 0;
}


