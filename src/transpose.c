/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <stdbool.h>
#include <complex.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/opts.h"


static const char help_str[] = "Transpose dimensions {dim1} and {dim2}.";

int main_transpose(int argc, char* argv[argc])
{
	int dim1 = -1;
	int dim2 = -1;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INT(true, &dim1, "dim1"),
		ARG_INT(true, &dim2, "dim2"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = { };

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	int N = DIMS;
	long idims[N];

	if ((0 > dim1) || (dim1 >= N))
		error("dim1 must be between 0 and %d", N - 1);

	if ((0 > dim2) || (dim2 >= N))
		error("dim2 must be between 0 and %d", N - 1);

	complex float* idata = load_cfl(in_file, N, idims);

	long odims[N];
	md_transpose_dims(N, dim1, dim2, odims, idims);

	complex float* odata = create_cfl(out_file, N, odims);

	md_transpose(N, dim1, dim2, odims, odata, idims, idata, sizeof(complex float));

	unmap_cfl(N, idims, idata);
	unmap_cfl(N, odims, odata);

	return 0;
}

