/* Copyright 2016-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2016-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
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


static const char help_str[] = "Copy an array (to a given position in the output file - which then must exist).";


int main_copy(int argc, char* argv[argc])
{
	long count = 0;
	long* dims = NULL;
	long* poss = NULL;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_TUPLE(false, &count, 2, TUPLE_LONG(&dims, "dim"),
					    TUPLE_LONG(&poss, "pos")),
		ARG_INFILE(true, &in_file, "input"),
		ARG_INOUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = { };

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();


	int N = DIMS;

	assert(count >= 0);

	long in_dims[N];
	long out_dims[N];

	void* in_data = load_cfl(in_file, N, in_dims);

	if (count > 0) {

		// get dimensions
		void* out_data = load_cfl(out_file, N, out_dims);

		unmap_cfl(N, out_dims, out_data);

	} else {

		md_copy_dims(N, out_dims, in_dims);
	}

	void* out_data = create_cfl(out_file, N, out_dims);

	long position[N];

	for (int i = 0; i < N; i++)
		position[i] = 0;

	for (int i = 0; i < count; i++) {

		long dim = dims[i];
		long pos = poss[i];

		assert(dim < N);
		assert((0 <= pos) && (pos < out_dims[dim]));

		position[dim] = pos;
	}

	md_copy_block(N, position, out_dims, out_data, in_dims, in_data, CFL_SIZE); 

	unmap_cfl(N, in_dims, in_data);
	unmap_cfl(N, out_dims, out_data);

	xfree(dims);
	xfree(poss);

	return 0;
}


