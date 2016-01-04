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

static const char* usage_str = "<input1> <input2> <output>";
static const char* help_str = "Multiply and accumulate.";


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

	long dims[N];

	for (int i = 0; i < N; i++) {
		
		assert((dims1[i] == dims2[i]) || (1 == dims1[i]) || (1 == dims2[i]));
	
		dims[i] = (1 == dims1[i]) ? dims2[i] : dims1[i];
	}

	long dimso[N];
	md_select_dims(N, ~squash, dimso, dims);
	complex float* out = create_cfl(argv[3], N, dimso);

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
	unmap_cfl(N, dims2, data2);
	unmap_cfl(N, dimso, out);
	exit(0);
}


