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

#include "ismrm/read.h"


static const char usage_str[] = "<ismrm-file> <output>";
static const char help_str[] = "Import ISMRM raw data files.\n";


int main_ismrmrd(int argc, char* argv[])
{
        mini_cmdline(argc, argv, 2, usage_str, help_str);

	long dims[DIMS];

	printf("Reading headers... "); fflush(stdout);

	if (-1 == ismrm_read(argv[1], dims, NULL)) {

		fprintf(stderr, "Reading headers failed.\n");
		exit(1);
	}

	printf("done.\n");

	printf("Dimensions:");
	unsigned int i;
	for (i = 0; i < DIMS; i++)
		printf(" %ld", dims[i]);

	printf("\n");

	complex float* out = create_cfl(argv[2], DIMS, dims);
	md_clear(DIMS, dims, out, CFL_SIZE);

	printf("Reading data... "); fflush(stdout);

	if (-1 == ismrm_read(argv[1], dims, out)) {

		fprintf(stderr, "Reading data failed.\n");
		exit(1);
	}

	printf("done.\n");

	unmap_cfl(DIMS, dims, out);
	exit(0);
}




