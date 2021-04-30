/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012, 2015 Martin Uecker <uecker@eecs.berkeley.edu>
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
static const char help_str[] =	"Apply fftshift along dimensions selected by the {bitmask}.\n";




int main_fftshift(int argc, char* argv[argc])
{
	bool b = false;

	const struct opt_s opts[] = {

		OPT_SET('b', &b, "apply ifftshift"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	unsigned long flags = labs(atol(argv[1]));

	int N = DIMS;
	long dims[N];

	complex float* idata = load_cfl(argv[2], N, dims);
	complex float* odata = create_cfl(argv[3], N, dims);

	(b ? ifftshift : fftshift)(N, dims, flags, odata, idata);

	unmap_cfl(N, dims, idata);
	unmap_cfl(N, dims, odata);
	return 0;
}


