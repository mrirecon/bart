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

#ifndef DIMS
#define DIMS 16
#endif


static const char usage_str[] = "dimension size <input> <output>";
static const char help_str[] = "Extracts a sub-array corresponding to the central part of {size} along {dimension}\n";


int main_crop(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 4, usage_str, help_str);

	num_init();

	int N = DIMS;
	long in_dims[N];
	long out_dims[N];
	
	complex float* in_data = load_cfl(argv[3], N, in_dims);

	int dim = atoi(argv[1]);
	int count = atoi(argv[2]);

	assert(dim < N);
	assert(count >= 1);
	
	for (int i = 0; i < N; i++)
		out_dims[i] = in_dims[i];

	out_dims[dim] = count;

	complex float* out_data = create_cfl(argv[4], N, out_dims);

	md_resize_center(N, out_dims, out_data, in_dims, in_data, sizeof(complex float));

	unmap_cfl(N, in_dims, in_data);
	unmap_cfl(N, out_dims, out_data);
	exit(0);
}


