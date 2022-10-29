/* Copyright 2015. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors: 
 * 2012-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/wavelet.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/io.h"
#include "misc/opts.h"
#include "misc/misc.h"

#ifndef DIMS
#define DIMS 16
#endif


static const char help_str[] = "Perform a wavelet (cdf97) transform.";




int main_cdf97(int argc, char* argv[argc])
{
	unsigned long flags = 0;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_ULONG(true, &flags, "bitmask"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	bool inv = false;

	const struct opt_s opts[] = {

		OPT_SET('i', &inv, "inverse"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long dims[DIMS];
	complex float* idata = load_cfl(in_file, DIMS, dims);
	complex float* odata = create_cfl(out_file, DIMS, dims);

	md_copy(DIMS, dims, odata, idata, CFL_SIZE);

	unmap_cfl(DIMS, dims, idata);

	if (inv) {

		md_iresortz(DIMS, dims, flags, odata);
		md_icdf97z(DIMS, dims, flags, odata);

	} else {

		md_cdf97z(DIMS, dims, flags, odata);
		md_resortz(DIMS, dims, flags, odata);
	}

	unmap_cfl(DIMS, dims, odata);

	return 0;
}





