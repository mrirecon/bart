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


#define DIMS 16

static const char usage_str[] = "dimension start end <input> <output>";
static const char help_str[] = "Extracts a sub-array along {dim} from index {start} to {end}.\n";


int main_extract(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 5, usage_str, help_str);

	num_init();

	long in_dims[DIMS];
	long out_dims[DIMS];
	
	complex float* in_data = load_cfl(argv[4], DIMS, in_dims);

	int dim = atoi(argv[1]);
	int start = atoi(argv[2]);
	int end = atoi(argv[3]);

	assert((0 <= dim) && (dim < DIMS));
	assert(start >= 0);
	assert(start <= end);
	assert(end < in_dims[dim]);

	for (int i = 0; i < DIMS; i++)
		out_dims[i] = in_dims[i];

	out_dims[dim] = end - start + 1;

	complex float* out_data = create_cfl(argv[5], DIMS, out_dims);

	long pos2[DIMS] = { [0 ... DIMS - 1] = 0 };
	pos2[dim] = start;
	
	md_copy_block(DIMS, pos2, out_dims, out_data, in_dims, in_data, sizeof(complex float));

	unmap_cfl(DIMS, in_dims, in_data);
	unmap_cfl(DIMS, out_dims, out_data);
	exit(0);
}


