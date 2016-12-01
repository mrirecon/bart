/* Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/resize.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"


#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


static const char usage_str[] = "dim1 pos1 ... dimn posn <input> <output>";
static const char help_str[] = "Copy an array to a given position in the output file (which must exist).";


int main_copy(int argc, char* argv[])
{
	const struct opt_s opts[] = { };
	cmdline(&argc, argv, 4, 1000, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();


	unsigned int N = DIMS;

	int count = argc - 3;
	assert((count > 0) && (count % 2 == 0));

	long in_dims[N];
	long out_dims[N];

	void* in_data = load_cfl(argv[argc - 2], N, in_dims);
	void* out_data = load_cfl(argv[argc - 1], N, out_dims);

	// reload
	unmap_cfl(N, out_dims, out_data);
	out_data = create_cfl(argv[argc - 1], N, out_dims);

	long position[N];

	for (unsigned int i = 0; i < N; i++)
		position[i] = 0;

	for (int i = 0; i < count; i += 2) {

		unsigned int dim = atoi(argv[i + 1]);
		long pos = atol(argv[i + 2]);

		assert(dim < N);
		assert((0 <= pos) && (pos < out_dims[dim]));

		position[dim] = pos;
	}

	md_copy_block(N, position, out_dims, out_data, in_dims, in_data, CFL_SIZE); 

	unmap_cfl(N, in_dims, in_data);
	unmap_cfl(N, out_dims, out_data);

	exit(0);
}


