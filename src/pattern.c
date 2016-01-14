/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2013 Martin Uecker <uecker@eecs.berkeley.edu>
 */


#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdio.h>

#include "num/multind.h"

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/misc.h"



static const char usage_str[] = "<kspace> <pattern>";
static const char help_str[] = "Compute sampling pattern from kspace\n";


int main_pattern(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 2, usage_str, help_str);

	unsigned int N = DIMS;
	long in_dims[N];
	long out_dims[N];
	
	unsigned int dim = COIL_DIM;
	assert(dim < N);

	complex float* kspace = load_cfl(argv[1], N, in_dims);

	md_select_dims(N, ~MD_BIT(dim), out_dims, in_dims);
	
	complex float* pattern = create_cfl(argv[2], N, out_dims);

	estimate_pattern(N, in_dims, dim, pattern, kspace);

	unmap_cfl(N, in_dims, kspace);
	unmap_cfl(N, out_dims, pattern);
	exit(0);
}




