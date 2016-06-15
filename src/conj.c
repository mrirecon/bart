/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * 2012 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>

#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"

#ifndef DIMS
#define DIMS 16
#endif

static const char usage_str[] = "<input> <output>";
static const char help_str[] = "Compute complex conjugate.\n";


int main_conj(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 2, usage_str, help_str);

	num_init();

	const int N = 16;
	long dims[N];
	complex float* idata = load_cfl(argv[1], N, dims);
	complex float* odata = create_cfl(argv[2], N, dims);

	md_zconj(N, dims, odata, idata);

	unmap_cfl(N, dims, idata);
	unmap_cfl(N, dims, odata);
	exit(0);
}


