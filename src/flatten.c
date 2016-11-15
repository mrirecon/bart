/* Copyright 2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2016 Jon Tamir <jtamir@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
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


static const char usage_str[] = "<input> <output>";
static const char help_str[] = "Flatten array to one dimension.\n";


int main_flatten(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 2, usage_str, help_str);

	num_init();

	long idims[DIMS];

	complex float* idata = load_cfl(argv[1], DIMS, idims);

	long odims[DIMS] = MD_INIT_ARRAY(DIMS, 1);
	odims[0] = md_calc_size(DIMS, idims);

	complex float* odata = create_cfl(argv[2], DIMS, odims);

	md_copy(DIMS, idims, odata, idata, CFL_SIZE);

	unmap_cfl(DIMS, idims, idata);
	unmap_cfl(DIMS, odims, odata);

	exit(0);
}


