/* Copyright 2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors: 
 * 2019 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <string.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/io.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/opts.h"


static const char help_str[] = "Create a sampling pattern.";



int main_upat(int argc, char* argv[argc])
{
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_OUTFILE(true, &out_file, "output"),
	};

	long dims[DIMS] = { [0 ... DIMS - 1] = 1 };
        dims[PHS1_DIM] = 128;
	dims[PHS2_DIM] = 128;

	unsigned int undy = 1;
	unsigned int undz = 2;
	unsigned int center = 0;

	const struct opt_s opts[] = {

		OPT_LONG('Y', &dims[PHS1_DIM], "Y", "size Y"),
		OPT_LONG('Z', &dims[PHS2_DIM], "Z", "size Z"),
		OPT_UINT('y', &undy, "uy", "undersampling y"),
		OPT_UINT('z', &undz, "uz", "undersampling z"),
		OPT_UINT('c', &center, "cen", "size of k-space center"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	complex float* pat = create_cfl(out_file, DIMS, dims);

	long Y = dims[PHS1_DIM];
	long Z = dims[PHS2_DIM];

	for (long y = 0; y < Y; y++)
		for (long z = 0; z < Z; z++)
			pat[z * Y + y] = (   ((y % undy == 0) && (z % undz == 0))
					  || (   (labs(2 * y - Y) < 2 * center)
					      && (labs(2 * z - Z) < 2 * center)))  ? 1.  : 0.;

	unmap_cfl(DIMS, dims, pat);
	return 0;
}


