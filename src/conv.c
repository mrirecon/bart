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


static const char usage_str[] = "bitmask <input> <kernel> <output>";
static const char help_str[] = "Performs a convolution along selected dimensions.";


int main_conv(int argc, char* argv[])
{
	cmdline(&argc, argv, 4, 4, usage_str, help_str, 0, NULL);

	num_init();

	unsigned int flags = atoi(argv[1]);

	unsigned int N = DIMS;
	long dims[N];
	const complex float* in = load_cfl(argv[2], N, dims);

	long krn_dims[N];
	const complex float* krn = load_cfl(argv[3], N, krn_dims);
	complex float* out = create_cfl(argv[4], N, dims);

	struct conv_plan* plan = conv_plan(N, flags, CONV_CYCLIC, CONV_SYMMETRIC, dims, dims, krn_dims, krn);
	conv_exec(plan, out, in);
	conv_free(plan);

	unmap_cfl(N, dims, out);
	unmap_cfl(N, krn_dims, krn);
	unmap_cfl(N, dims, in);
	exit(0);
}




