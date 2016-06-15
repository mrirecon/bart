/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 */
 
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"

#ifndef DIMS
#define DIMS 16
#endif

static const char usage_str[] = "scale <input1> <input2> <output>";
static const char help_str[] = "Multiply input1 with scale factor and add input2.\n";


int main_saxpy(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 4, usage_str, help_str);

	num_init();

	complex float scale;

	if (0 != parse_cfl(&scale, argv[1])) {
		
		fprintf(stderr, "ERROR: %s is not a complex number.\n", argv[1]);
		exit(1);
	}

	const int N = DIMS;
	long dims1[N];
	long dims2[N];

	complex float* data1 = load_cfl(argv[2], N, dims1);
	complex float* data2 = load_cfl(argv[3], N, dims2);

	for (int i = 0; i < N; i++)
		assert(dims1[i] == dims2[i]);

	complex float* out = create_cfl(argv[4], N, dims2);

	#pragma omp parallel for
	for (long i = 0; i < md_calc_size(N, dims1); i++)
		out[i] = scale * data1[i] + data2[i];

	unmap_cfl(N, dims1, data1);
	unmap_cfl(N, dims2, data2);
	unmap_cfl(N, dims2, out);
	exit(0);
}


