/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors: 
 * 2012-06-04 Martin Uecker <uecker@eecs.berkeley.edu>
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


static const char help_str[] = "Create a zero-filled array with {dims} dimensions of size {dim1} to {dimn}.";



int main_zeros(int argc, char* argv[argc])
{
	long count = 0;
	long N = -1;
	long* dims = NULL;

	const char* out_file = NULL;


	struct arg_s args[] = {

		ARG_LONG(true, &N, "dims"),
		ARG_TUPLE(true, &count, 1, TUPLE_LONG(&dims, "dim")),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = { };

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	assert(N >= 0);
	assert(count == N);

	for (int i = 0; i < N; i++)
		assert(dims[i] >= 1);

	complex float* x = create_cfl(out_file, N, dims);

	md_clear(N, dims, x, sizeof(complex float));

	unmap_cfl(N, dims, x);

	xfree(dims);

	return 0;
}


