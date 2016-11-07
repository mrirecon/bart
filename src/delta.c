/* Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
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


static const char usage_str[] = "dims flags size out";
static const char help_str[] = "Kronecker delta.\n";



int main_delta(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 4, usage_str, help_str);

	num_init();

	int N = atoi(argv[1]);
	long len = atoi(argv[3]);
	unsigned int flags = atoi(argv[2]);

	assert(N >= 0);

	long dims[N];

	for (int i = 0; i < N; i++)
		dims[i] = MD_IS_SET(flags, i) ? len : 1;

	complex float* x = create_cfl(argv[4], N, dims);

	md_clear(N, dims, x, CFL_SIZE);
	md_fill_diag(N, dims, flags, x, &(complex float){ 1. }, CFL_SIZE); 

	unmap_cfl(N, dims, x);
	exit(0);
}


