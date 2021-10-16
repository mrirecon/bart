/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2012-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
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


static const char help_str[] = "Resizes an array along dimensions to sizes by truncating or zero-padding.";


int main_resize(int argc, char* argv[argc])
{
	long count = 0;
	unsigned int* dims = NULL;
	unsigned int* sizes = NULL;

	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_TUPLE(true, &count, 2, OPT_UINT, sizeof(*dims), &dims, "dim", OPT_UINT, sizeof(*sizes), &sizes, "size"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	bool center = false;

	const struct opt_s opts[] = {

		OPT_SET('c', &center, "center"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();


	unsigned int N = DIMS;

	long in_dims[N];
	long out_dims[N];

	void* in_data = load_cfl(in_file, N, in_dims);
	md_copy_dims(N, out_dims, in_dims);
	
	for (int i = 0; i < count; i++) {

		unsigned int dim = dims[i];
		unsigned int size = sizes[i];

		assert(dim < N);
		assert(size >= 1);

		out_dims[dim] = size;
	}

	void* out_data = create_cfl(out_file, N, out_dims);

	(center ? md_resize_center : md_resize)(N, out_dims, out_data, in_dims, in_data, CFL_SIZE);

	unmap_cfl(N, in_dims, in_data);
	unmap_cfl(N, out_dims, out_data);
	xfree(dims);
	xfree(sizes);

	return 0;
}

/* A simple example to understand bart resize usage using python interface
* # python code
* x = np.array([[1, 2, 3], [4, 5, 6]]) #intialize the array
* print('x =\n',x)
* x_bart_centered  = np.real(bart(1,'resize -c 0 6 1 5', x)) # bart resize with centering
* print('x_bart_centered = \n',x_bart_centered)
* x_bart_not_centered  = np.real(bart(1,'resize 0 6 1 5', x))  # bart resize without centering
* print('x_bart_not_centered = \n',x_bart_not_centered)
* # the output looks like this 
* x =
* [[1 2 3]
* [4 5 6]]
* x_bart_centered = 
* [[0. 0. 0. 0. 0.]
* [0. 0. 0. 0. 0.]
* [0. 1. 2. 3. 0.]
* [0. 4. 5. 6. 0.]
* [0. 0. 0. 0. 0.]
* [0. 0. 0. 0. 0.]]
* x_bart_not_centered = 
* [[1. 2. 3. 0. 0.]
* [4. 5. 6. 0. 0.]
* [0. 0. 0. 0. 0.]
* [0. 0. 0. 0. 0.]
* [0. 0. 0. 0. 0.]
* [0. 0. 0. 0. 0.]]
*/

