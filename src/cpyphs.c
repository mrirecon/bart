/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2014 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"



#ifndef DIMS
#define DIMS 16
#endif


static const char usage_str[] = "<input> <output";
static const char help_str[] = "Copy phase from <input> to <output>.\n";


int main_cpyphs(int argc, char* argv[argc])
{
	mini_cmdline(argc, argv, 2, usage_str, help_str);

	num_init();

	long dims[DIMS];
	
	complex float* in_data = load_cfl(argv[1], DIMS, dims);
	complex float* out_data = create_cfl(argv[2], DIMS, dims);

	md_zphsr(DIMS, dims, out_data, in_data);

	unmap_cfl(DIMS, dims, in_data);
	unmap_cfl(DIMS, dims, out_data);
	exit(0);
}


