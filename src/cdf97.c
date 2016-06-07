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


static const char usage_str[] = "bitmask <input> <output>";
static const char help_str[] = "Perform a wavelet (cdf97) transform.\n";




int main_cdf97(int argc, char* argv[])
{
	bool inv = false;

	const struct opt_s opts[] = {

		OPT_SET('i', &inv, "inverse"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	unsigned int flags = atoi(argv[1]);

	long dims[DIMS];
	complex float* idata = load_cfl(argv[2], DIMS, dims);
	complex float* odata = create_cfl(argv[3], DIMS, dims);

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
	exit(0);
}





