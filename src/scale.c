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
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"

#ifndef DIMS
#define DIMS 16
#endif

static const char usage_str[] = "factor <input> <output>";
static const char help_str[] = "Scale array by {factor}. The scale factor can be a complex number.\n";


int main_scale(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 3, usage_str, help_str);

	num_init();

	complex float scale;
	// = atof(argv[1]);
	if (0 != parse_cfl(&scale, argv[1])) {

		fprintf(stderr, "ERROR: scale factor %s is not a number.\n", argv[1]);
		exit(1);
	}

	const int N = DIMS;
	long dims[N];
	complex float* idata = load_cfl(argv[2], N, dims);
	complex float* odata = create_cfl(argv[3], N, dims);
		
	md_zsmul(N, dims, odata, idata, scale);

	unmap_cfl(N, dims, idata);
	unmap_cfl(N, dims, odata);
	exit(0);
}


