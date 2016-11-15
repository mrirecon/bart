/* Copyright 2013, 2016. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012, 2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2016 Jon Tamir <jtamir@eecs.berkeley.edu>
 */

#include <stdbool.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif

static const char usage_str[] = "<input1> [<input2>] <output>";
static const char help_str[] =
		"Multiply <input1> and <input2> and accumulate in <output>.\n"
		"If <input2> is not specified, assume all-ones.";


int main_fmac(int argc, char* argv[])
{
	bool clear = true;
	bool conj = false;
	long squash = 0;

	const struct opt_s opts[] = {

		OPT_CLEAR('A', &clear, "add to existing output (instead of overwriting)"),
		OPT_SET('C', &conj, "conjugate input2"),
		OPT_LONG('s', &squash, "b", "squash dimensions selected by bitmask b"),
	};

	cmdline(&argc, argv, 2, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);
	int num_args = argc - 1;

	num_init();


	int N = DIMS;

	long dims1[N];
	long dims2[N];

	complex float* data1 = load_cfl(argv[1], N, dims1);

	complex float* data2 = NULL;
	
	if (3 == num_args)
		data2 = load_cfl(argv[2], N, dims2);
	else {

		md_singleton_dims(N, dims2);
		data2 = md_alloc(N, dims2, CFL_SIZE);
		md_zfill(N, dims2, data2, 1.);
	}

	long dims[N];
	md_merge_dims(N, dims, dims1, dims2);

	long dimso[N];
	md_select_dims(N, ~squash, dimso, dims);

	complex float* out = create_cfl((3 == num_args) ? argv[3] : argv[2], N, dimso);

	if (clear) {

		md_clear(N, dimso, out, CFL_SIZE);
	}

	long str1[N];
	long str2[N];
	long stro[N];

	md_calc_strides(N, str1, dims1, CFL_SIZE);
	md_calc_strides(N, str2, dims2, CFL_SIZE);
	md_calc_strides(N, stro, dimso, CFL_SIZE);

	(conj ? md_zfmacc2 : md_zfmac2)(N, dims, stro, out, str1, data1, str2, data2);

	unmap_cfl(N, dims1, data1);
	unmap_cfl(N, dimso, out);

	if (3 == num_args)
		unmap_cfl(N, dims2, data2);
	else
		md_free(data2);

	exit(0);
}


