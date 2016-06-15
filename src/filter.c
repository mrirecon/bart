/* Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>

#include "num/multind.h"
#include "num/casorati.h"
#include "num/filter.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"



#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


static const char usage_str[] = "<input> <output>";
static const char help_str[] = "Apply filter.\n";


int main_filter(int argc, char* argv[])
{
	int len = -1;
	int dim = -1;

	const struct opt_s opts[] = {

		OPT_INT('m', &dim, "dim", "median filter along dimension dim"),
		OPT_INT('l', &len, "len", "length of filter"),
	};

	cmdline(&argc, argv, 2, 2, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();


	long in_dims[DIMS];
	
	complex float* in_data = load_cfl(argv[1], DIMS, in_dims);

	assert(dim >= 0);
	assert(dim < DIMS);
	assert(len > 0);
	assert(len <= in_dims[dim]);

	long tmp_dims[DIMS + 1];
	md_copy_dims(DIMS, tmp_dims, in_dims);
	tmp_dims[DIMS] = 1;

	long tmp2_strs[DIMS + 1];
	md_calc_strides(DIMS + 1, tmp2_strs, tmp_dims, CFL_SIZE);

	long tmp_strs[DIMS + 1];
	md_calc_strides(DIMS, tmp_strs, tmp_dims, CFL_SIZE);

	tmp_dims[DIMS] = len;
	tmp_dims[dim] = in_dims[dim] - len + 1;
	tmp_strs[DIMS] = tmp_strs[dim];

	long out_dims[DIMS];
	md_copy_dims(DIMS, out_dims, tmp_dims);

	complex float* out_data = create_cfl(argv[2], DIMS, out_dims);

	md_medianz2(DIMS + 1, DIMS, tmp_dims, tmp2_strs, out_data, tmp_strs, in_data);

	unmap_cfl(DIMS, in_dims, in_data);
	unmap_cfl(DIMS, out_dims, out_data);
	exit(0);
}


