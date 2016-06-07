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



#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


static const char usage_str[] = "dimension position <input> <output>";
static const char help_str[] = "Extracts a slice from {position} along {dimension}.\n";


int main_slice(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 4, usage_str, help_str);

	num_init();

	long in_dims[DIMS];
	long out_dims[DIMS];
	
	complex float* in_data = load_cfl(argv[3], DIMS, in_dims);

	int dim = atoi(argv[1]);
	int pos = atoi(argv[2]);

	assert(dim < DIMS);
	assert(pos >= 0);
	assert(pos < in_dims[dim]);

	for (int i = 0; i < DIMS; i++)
		out_dims[i] = in_dims[i];

	out_dims[dim] = 1;

	complex float* out_data = create_cfl(argv[4], DIMS, out_dims);

	long pos2[DIMS] = { [0 ... DIMS - 1] = 0 };
	pos2[dim] = pos;
	md_slice(DIMS, MD_BIT(dim), pos2, in_dims, out_data, in_data, CFL_SIZE);

	unmap_cfl(DIMS, out_dims, out_data);
	unmap_cfl(DIMS, in_dims, in_data);
	exit(0);
}


