/* Copyright 2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2016 Siddharth Iyer <sid8795@gmail.com>
 */

#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdio.h>

#include "num/flpmath.h"

#include "misc/mmio.h"
#include "misc/misc.h"

#ifndef DIMS
#define DIMS 16
#endif

static const char usage_str[] = "<input> <output>";
static const char help_str[] = "Absolute value of array (|<input>|).\n";


int main_cabs(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 2, usage_str, help_str);

	long dims[DIMS];

	complex float* idata = load_cfl(argv[1], DIMS, dims);
	complex float* odata = create_cfl(argv[2], DIMS, dims);
		
        md_zabs(DIMS, dims, odata, idata);

	unmap_cfl(DIMS, dims, idata);
	unmap_cfl(DIMS, dims, odata);

	exit(0);
}


