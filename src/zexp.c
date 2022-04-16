/* Copyright 2016. The Regents of the University of California.
 * Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2016 Jon Tamir <jtamir@eecs.berkeley.edu>
 * 2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"



#ifndef DIMS
#define DIMS 16
#endif


static const char help_str[] = "Point-wise complex exponential.";


int main_zexp(int argc, char* argv[argc])
{
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	bool img = false;

	const struct opt_s opts[] = {

		OPT_SET('i', &img, "imaginary"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	
	num_init();

	long dims[DIMS];
	
	complex float* in_data = load_cfl(in_file, DIMS, dims);
	complex float* out_data = create_cfl(out_file, DIMS, dims);

	(img ? md_zexpj : md_zexp)(DIMS, dims, out_data, in_data);

	unmap_cfl(DIMS, dims, in_data);
	unmap_cfl(DIMS, dims, out_data);

	return 0;
}


