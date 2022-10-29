/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014-2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <complex.h>

#include "num/multind.h"
#include "num/conv.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif


static const char help_str[] = "Performs a convolution along selected dimensions.";


int main_conv(int argc, char* argv[argc])
{
	unsigned long flags = 0;
	const char* in_file = NULL;
	const char* kern_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_ULONG(true, &flags, "bitmask"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_INFILE(true, &kern_file, "kernel"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = {};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	int N = DIMS;
	long dims[N];
	const complex float* in = load_cfl(in_file, N, dims);

	long krn_dims[N];
	const complex float* krn = load_cfl(kern_file, N, krn_dims);
	complex float* out = create_cfl(out_file, N, dims);

	struct conv_plan* plan = conv_plan(N, flags, CONV_CYCLIC, CONV_SYMMETRIC, dims, dims, krn_dims, krn);
	conv_exec(plan, out, in);
	conv_free(plan);

	unmap_cfl(N, dims, out);
	unmap_cfl(N, krn_dims, krn);
	unmap_cfl(N, dims, in);

	return 0;
}




