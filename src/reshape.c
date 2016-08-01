/* Copyright 2015. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <assert.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#define DIMS 16

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char usage_str[] = "flags dim1 ... dimN <input> <output>";
static const char help_str[] = "Reshape selected dimensions.\n";


int main_reshape(int argc, char* argv[])
{
	cmdline(&argc, argv, 3, 100, usage_str, help_str, 0, NULL);

	num_init();

	unsigned int flags = atoi(argv[1]);
	unsigned int n = bitcount(flags);

	assert((int)n + 3 == argc - 1);

	long in_dims[DIMS];
	long in_strs[DIMS];

	long out_dims[DIMS];
	long out_strs[DIMS];

	complex float* in_data = load_cfl(argv[n + 2], DIMS, in_dims);

	md_calc_strides(DIMS, in_strs, in_dims, CFL_SIZE);

	md_copy_dims(DIMS, out_dims, in_dims);
	
	unsigned int j = 0;

	for (unsigned int i = 0; i < DIMS; i++)
		if (MD_IS_SET(flags, i))
			out_dims[i] = atoi(argv[j++ + 2]);

	assert(j == n);
	assert(md_calc_size(DIMS, in_dims) == md_calc_size(DIMS, out_dims));

	md_calc_strides(DIMS, out_strs, out_dims, CFL_SIZE);
	
	for (unsigned int i = 0; i < DIMS; i++)
		if (!(MD_IS_SET(flags, i) || (in_strs[i] == out_strs[i]))) 
			error("Dimensions are not consistent at index %d.\n", i);


	complex float* out_data = create_cfl(argv[n + 3], DIMS, out_dims);

	md_copy(DIMS, in_dims, out_data, in_data, CFL_SIZE);

	unmap_cfl(DIMS, in_dims, in_data);
	unmap_cfl(DIMS, out_dims, out_data);
	exit(0);
}


