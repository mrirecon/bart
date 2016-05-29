/* Copyright 2013. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012, 2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif

static const char usage_str[] = "<input1> <input2> <output>";
static const char help_str[] = "Multiply and accumulate.";


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

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);


	int N = DIMS;

	long dims1[N];
	long dims2[N];

	complex float* data1 = load_cfl(argv[1], N, dims1);
	complex float* data2 = load_cfl(argv[2], N, dims2);

        long out_dims[N];
        md_calc_fmac3_dims(N, squash, dims1, dims2, out_dims);

	complex float* out_data = create_cfl(argv[3], N, out_dims);

	if (clear) {
		md_clear(N, out_dims, out_data, CFL_SIZE);
	}

	conj ? md_fmacc3(N, dims1, data1, dims2, data2, out_dims, out_data) : md_fmac3(N, dims1, data1, dims2, data2, out_dims, out_data);

	unmap_cfl(N,    dims1,    data1);
	unmap_cfl(N,    dims2,    data2);
	unmap_cfl(N, out_dims, out_data);
	exit(0);
}


