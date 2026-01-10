/* Copyright 2013-2016. The Regents of the University of California.
 * Copyright 2017-2021. Uecker Lab. Unversity Medical Center Göttingen.
 * Copyright 2021-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/io.h"
#include "misc/misc.h"
#include "misc/opts.h"


static const char help_str[] = "Create a zero-filled array with {dims} dimensions of size {dim1} to {dimn}.";



int main_zeros(int argc, char* argv[argc])
{
	int count = 0;
	int N = -1;
	long* dims = NULL;

	const char* out_file = NULL;


	struct arg_s args[] = {

		ARG_INT(true, &N, "dims"),
		ARG_TUPLE(true, &count, 1, TUPLE_LONG(&dims, "dim")),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = { };

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	if (N < 0)
		error("Number of dimensions not specified\n");

	if (count != N)
		error("Incorrect number of dimensions\n");

	for (int i = 0; i < N; i++)
		if (dims[i] <= 0)
			error("Dimensions must be larger than zero\n");

	complex float* x = create_cfl(out_file, N, dims);

	md_clear(N, dims, x, sizeof(complex float));

	unmap_cfl(N, dims, x);

	xfree(dims);

	return 0;
}

