/* Copyright 2013. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors: 
 * 2012, 2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

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


static const char usage_str[] = "<input1> <input2>";
static const char help_str[] = "Compute dot product along selected dimensions.";

int main_sdot(int argc, char* argv[])
{
	cmdline(&argc, argv, 2, 2, usage_str, help_str, 0, NULL);

	num_init();

	int N = DIMS;
	long in1_dims[N];
	long in2_dims[N];

	complex float* in1_data = load_cfl(argv[1], N, in1_dims);
	complex float* in2_data = load_cfl(argv[2], N, in2_dims);


	for (int i = 0; i < N; i++)
		assert(in1_dims[i] == in2_dims[i]);

	// compute scalar product
	complex float value = md_zscalar(N, in1_dims, in1_data, in2_data);
	printf("%+e%+ei\n", crealf(value), cimagf(value));

	unmap_cfl(N, in1_dims, in1_data);
	unmap_cfl(N, in2_dims, in2_data);
	exit(0);
}


