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
#include "misc/opts.h"


static const char help_str[] = "Create an array counting from 0 to {size-1} in dimensions {dim}.";



int main_index(int argc, char* argv[argc])
{
	int N = -1;
	int s = -1;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INT(true, &N, "dim"),
		ARG_INT(true, &s, "size"),
		ARG_OUTFILE(true, &out_file, "name"),
	};

	const struct opt_s opts[] = {};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	assert(N >= 0);
	assert(s >= 0);

	long dims[N + 1];

	for (int i = 0; i < N; i++)
		dims[i] = 1;

	dims[N] = s;

	complex float* x = create_cfl(out_file, N + 1, dims);

	for (int i = 0; i < s; i++)
		x[i] = i;

	unmap_cfl(N + 1, dims, x);

	return 0;
}


