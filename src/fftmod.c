/* Copyright 2013. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012, 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <complex.h>

#include "num/multind.h"
#include "num/fft.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif

static const char usage_str[] = "bitmask <input> <output>";
static const char help_str[] =	"Apply 1 -1 modulation along dimensions selected by the {bitmask}.\n";




int main_fftmod(int argc, char* argv[])
{
	bool inv = false;

	const struct opt_s opts[] = {

		OPT_SET('b', &inv, "(deprecated)"),
		OPT_SET('i', &inv, "\tinverse"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();


	unsigned long flags = labs(atol(argv[1]));

	int N = DIMS;
	long dims[N];

	complex float* idata = load_cfl(argv[2], N, dims);
	complex float* odata = create_cfl(argv[3], N, dims);

	(inv ? ifftmod : fftmod)(N, dims, flags, odata, idata);

	unmap_cfl(N, dims, idata);
	unmap_cfl(N, dims, odata);
	exit(0);
}


