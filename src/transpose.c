/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/misc.h"


static const char usage_str[] = "dim1 dim2 <input> <output>";
static const char help_str[] = "Transpose dimensions {dim1} and {dim2}.\n";

int main_transpose(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 4, usage_str, help_str);

	num_init();

	int N = DIMS;
	long idims[N];

	int dim1 = atoi(argv[1]);
	int dim2 = atoi(argv[2]);

	assert((0 <= dim1) && (dim1 < N));
	assert((0 <= dim2) && (dim2 < N));

	complex float* idata = load_cfl(argv[3], N, idims);

	long odims[N];
	md_transpose_dims(N, dim1, dim2, odims, idims);

	complex float* odata = create_cfl(argv[4], N, odims);

	md_transpose(N, dim1, dim2, odims, odata, idims, idata, sizeof(complex float));

	unmap_cfl(N, idims, idata);
	unmap_cfl(N, odims, odata);
	exit(0);
}


