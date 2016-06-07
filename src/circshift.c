/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2012 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <string.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"


#ifndef DIMS
#define DIMS 16
#endif

static const char usage_str[] = "dim shift <input> <output>";
static const char help_str[] = "Perform circular shift along {dim} by {shift} elements.\n";



int main_circshift(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 4, usage_str, help_str);

	num_init();

	const int N = DIMS;
	long dims[N];

	int dim = atoi(argv[1]);
	int shift = atoi(argv[2]);

	assert((0 <= dim) && (dim < N));

	long center[N];
	memset(center, 0, N * sizeof(long));
	center[dim] = shift;

	complex float* idata = load_cfl(argv[3], N, dims);
	complex float* odata = create_cfl(argv[4], N, dims);

	md_circ_shift(N, dims, center, odata, idata, sizeof(complex float));

	unmap_cfl(N, dims, idata);
	unmap_cfl(N, dims, odata);
	exit(0);
}


