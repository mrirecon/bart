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
#define DIMS 32
#endif


static const char usage_str[] = "<input> <output>";
static const char help_str[] = "Remove singleton dimensions of array.\n";


int main_squeeze(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 2, usage_str, help_str);

	num_init();

	long idims[DIMS];
	long odims[DIMS] = MD_INIT_ARRAY(DIMS, 1);

	complex float* idata = load_cfl(argv[1], DIMS, idims);
		
	unsigned int j = 0;

	for (unsigned int i = 0; i < DIMS; i++)
		if (1 < idims[i])
			odims[j++] = idims[i];

	if (0 == j)
		j = 1;

	complex float* odata = create_cfl(argv[2], j, odims);

	md_copy(DIMS, idims, odata, idata, CFL_SIZE);

	unmap_cfl(DIMS, idims, idata);
	unmap_cfl(j, odims, odata);

	exit(0);
}


