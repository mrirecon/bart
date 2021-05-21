/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <complex.h>
#include <stdlib.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/misc.h" 
#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/opts.h"

#include "ismrm/read.h"


static const char help_str[] = "Import ISMRM raw data files.";


int main_ismrmrd(int argc, char* argv[])
{
	const char* ismrm_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_STRING(true, &ismrm_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};
	const struct opt_s opts[] = {};
	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	long dims[DIMS];

	printf("Reading headers... "); fflush(stdout);

	if (-1 == ismrm_read(ismrm_file, dims, NULL)) {

		fprintf(stderr, "Reading headers failed.\n");
		return 1;
	}

	printf("done.\n");

	printf("Dimensions:");
	unsigned int i;
	for (i = 0; i < DIMS; i++)
		printf(" %ld", dims[i]);

	printf("\n");

	complex float* out = create_cfl(out_file, DIMS, dims);
	md_clear(DIMS, dims, out, CFL_SIZE);

	printf("Reading data... "); fflush(stdout);

	if (-1 == ismrm_read(ismrm_file, dims, out)) {

		fprintf(stderr, "Reading data failed.\n");
		return 1;
	}

	printf("done.\n");

	unmap_cfl(DIMS, dims, out);
	return 0;
}




