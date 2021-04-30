/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012, 2014 Martin Uecker <uecker@eecs.berkeley.edu>
 */
 
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/opts.h"
#include "misc/utils.h"


#ifndef DIMS
#define DIMS 16
#endif





static const char usage_str[] = "flags <input> <output>";
static const char help_str[] = "Normalize along selected dimensions.\n";



int main_normalize(int argc, char* argv[argc])
{
	bool l1 = false;

	const struct opt_s opts[] = {

		OPT_SET('b', &l1, "l1"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	int N = DIMS;
	long dims[N];
	complex float* data = load_cfl(argv[2], N, dims);

	int flags = atoi(argv[1]);

	assert(flags >= 0);

	complex float* out = create_cfl(argv[3], N, dims);
	md_copy(N, dims, out, data, CFL_SIZE);

	(l1 ? normalizel1 : normalize)(N, flags, dims, out);

	unmap_cfl(N, dims, out);
	return 0;
}


