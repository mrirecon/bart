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
#include "misc/opts.h"


static const char help_str[] = "Kronecker delta.";



int main_delta(int argc, char* argv[argc])
{
	int N = 0;
	unsigned long flags = 0;
	long len = 0;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INT(true, &N, "dims"),
		ARG_ULONG(true, &flags, "flags"),
		ARG_LONG(true, &len, "size"),
		ARG_OUTFILE(true, &out_file, "out"),
	};

	const struct opt_s opts[] = { };

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	assert(N >= 0);

	long dims[N];

	for (int i = 0; i < N; i++)
		dims[i] = MD_IS_SET(flags, i) ? len : 1;

	complex float* x = create_cfl(out_file, N, dims);

	md_clear(N, dims, x, CFL_SIZE);
	md_fill_diag(N, dims, flags, x, &(complex float){ 1. }, CFL_SIZE); 

	unmap_cfl(N, dims, x);

	return 0;
}


