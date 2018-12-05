/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
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


static const char usage_str[] = "dim size name";
static const char help_str[] = "Create an array counting from 0 to {size-1} in dimensions {dim}.\n";



int main_index(int argc, char* argv[])
{
	mini_cmdline(&argc, argv, 3, usage_str, help_str);

	num_init();

	int N = atoi(argv[1]);
	int s = atoi(argv[2]);

	assert(N >= 0);
	assert(s >= 0);

	long dims[N + 1];

	for (int i = 0; i < N; i++)
		dims[i] = 1;

	dims[N] = s;

	complex float* x = create_cfl(argv[3], N + 1, dims);

	for (int i = 0; i < s; i++)
		x[i] = i;

	unmap_cfl(N + 1, dims, x);
	exit(0);
}


