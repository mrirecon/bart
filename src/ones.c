/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors: 
 * 2012, 2014 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <string.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/io.h"
#include "misc/misc.h"


static const char usage_str[] = "dims dim1 ... dimn name";
static const char help_str[] = "Create an array filled with ones with {dims} dimensions of size {dim1} to {dimn}.\n";



int main_ones(int argc, char* argv[])
{
	mini_cmdline(argc, argv, -3, usage_str, help_str);

	num_init();

	int N = atoi(argv[1]);

	assert(N >= 0);
	assert(argc == 3 + N);

	long dims[N];

	for (int i = 0; i < N; i++) {

		dims[i] = atoi(argv[2 + i]);
		assert(dims[i] >= 1);
	}

	complex float* x = create_cfl(argv[2 + N], N, dims);
	md_zfill(N, dims, x, 1.);
	unmap_cfl(N, dims, x);
	exit(0);
}


