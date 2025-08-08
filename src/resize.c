/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2012-2016 Martin Uecker
 * 2014 Jonathan Tamir
 */

#include <complex.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/resize.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"


#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char help_str[] = "Resizes an array along dimensions to sizes by truncating or zero-padding. Please see doc/resize.txt for examples.";


int main_resize(int argc, char* argv[argc])
{
	int count = 0;
	int* dims = NULL;
	int* sizes = NULL;

	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_TUPLE(true, &count, 2, { OPT_PINT, sizeof(*dims), &dims, "dim" },
					   { OPT_PINT, sizeof(*sizes), &sizes, "size" }),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	enum mode { FRONT, CENTER, END } mode = END;;

	const struct opt_s opts[] = {

		OPT_SELECT('c', enum mode, &mode, CENTER, "center"),
		OPT_SELECT('f', enum mode, &mode, FRONT, "front"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();


	int N = DIMS;

	long in_dims[N];
	long out_dims[N];

	complex float* in_data = load_cfl(in_file, N, in_dims);
	md_copy_dims(N, out_dims, in_dims);
	
	for (int i = 0; i < count; i++) {

		long dim = dims[i];
		long size = sizes[i];

		assert(dim < N);
		assert(size >= 1);

		out_dims[dim] = size;
	}

	complex float* out_data = create_cfl(out_file, N, out_dims);

	switch (mode) {
	case FRONT:

		md_resize_front(N, out_dims, out_data, in_dims, in_data, CFL_SIZE);
		break;

	case CENTER:

		md_resize_center(N, out_dims, out_data, in_dims, in_data, CFL_SIZE);
		break;

	case END:
		md_resize(N, out_dims, out_data, in_dims, in_data, CFL_SIZE);
		break;
	}

	unmap_cfl(N, in_dims, in_data);
	unmap_cfl(N, out_dims, out_data);

	xfree(dims);
	xfree(sizes);

	return 0;
}

