/* Copyright 2018. Massachusetts Institute of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2018 Siddharth Iyer <ssi@mit.edu>
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

#ifndef DIMS
#define DIMS 16
#endif

static const char usage_str[] = "L N a_0 a_1 ... a_N output";
static const char help_str[]  = "Evaluate polynomial p(x) = a_0 + a_1 x + a_2 x^2 ... a_N x^N at x = {0, 1, ... , L - 1} where a_i are floats.";

int main_poly(int argc, char* argv[])
{
	mini_cmdline(&argc, argv, -3, usage_str, help_str);

	num_init();

	int L = atoi(argv[1]);
	int N = atoi(argv[2]);

	assert(L >= 1);
	assert(L <= 256 * 256);
	assert(N >= 0);
	assert(N + 1 <= L);
	assert(argc == 5 + N);

	long p_dims[] = { [0 ... DIMS - 1] = 1 };
	p_dims[0] = L;
	complex float* p = create_cfl(argv[argc - 1], DIMS, p_dims);
	md_clear(DIMS, p_dims, p, CFL_SIZE);

	for (int x = 0; x < L; x++) {
		p[x] = (complex float) atof(argv[3]);
		for (int n = 1; n < N + 1; n++) {
			p[x] += ((complex float) atof(argv[3 + n])) * cpowf(x, n);
		}
	}

	unmap_cfl(N, p_dims, p);

	return 0;
}
