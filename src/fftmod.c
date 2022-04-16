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

static const char help_str[] =	"Apply 1 -1 modulation along dimensions selected by the {bitmask}.";




int main_fftmod(int argc, char* argv[argc])
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

		OPT_SET('b', &inv, "(deprecated)"),
		OPT_SET('i', &inv, "inverse"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	int N = DIMS;
	long dims[N];

	complex float* idata = load_cfl(in_file, N, dims);
	complex float* odata = create_cfl(out_file, N, dims);

	(inv ? ifftmod : fftmod)(N, dims, flags, odata, idata);

	unmap_cfl(N, dims, idata);
	unmap_cfl(N, dims, odata);

	return 0;
}


