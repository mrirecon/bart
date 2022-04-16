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
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif

static const char help_str[]  = "Evaluate polynomial p(x) = a_1 + a_2 x + a_3 x^2 ... a_(N+1) x^N at x = {0, 1, ... , L - 1} where a_i are floats.";

int main_poly(int argc, char* argv[argc])
{
	int L = -1;
	int N = -1;
	long count = 0;
	float* as = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INT(true, &L, "L"),
		ARG_INT(true, &N, "N"),
		ARG_TUPLE(true, &count, 1, { OPT_FLOAT, sizeof(*as), &as, "a_" }),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = { };

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	assert(L >= 1);
	assert(L <= 256 * 256);
	assert(N >= 0);
	assert(N + 1 <= L);
	assert(N + 1 == count);

	long p_dims[] = { [0 ... DIMS - 1] = 1 };
	p_dims[0] = L;

	complex float* p = create_cfl(out_file, DIMS, p_dims);

	md_clear(DIMS, p_dims, p, CFL_SIZE);

	for (int x = 0; x < L; x++) {

		p[x] = (complex float) as[0];

		for (int n = 1; n < N + 1; n++) {

			p[x] += ((complex float) as[n]) * cpowf(x, n);
		}
	}

	unmap_cfl(N, p_dims, p);
	xfree(as);

	return 0;
}
