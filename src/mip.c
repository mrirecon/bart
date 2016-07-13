/* Copyright 2014-2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2014-2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <unistd.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/opts.h"
#include "misc/misc.h"


#ifndef DIMS
#define DIMS 16
#endif



static const char usage_str[] = "bitmask <input> <output>";
static const char help_str[] = "Maximum (minimum) intensity projection (MIP) along dimensions specified by bitmask.\n"; 


int main_mip(int argc, char* argv[argc])
{

	bool do_abs = false;
	bool mIP = false;

	const struct opt_s opts[] = {

		OPT_SET('m', &mIP, "minimum" ),
		OPT_SET('a', &do_abs, "do absolute value first" ),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();


	unsigned int flags = atoi(argv[1]);

	long idims[DIMS];
	complex float* in = load_cfl(argv[2], DIMS, idims);

	long odims[DIMS];
	md_select_dims(DIMS, ~flags, odims, idims);

	complex float* out = create_cfl(argv[3], DIMS, odims);

	complex float* tmp = md_alloc(DIMS, idims, CFL_SIZE);

	if (do_abs)
		md_zabs(DIMS, idims, tmp, in);
	else
		md_copy(DIMS, idims, tmp, in, CFL_SIZE);

	long istr[DIMS];
	long ostr[DIMS];

	md_calc_strides(DIMS, istr, idims, CFL_SIZE);
	md_calc_strides(DIMS, ostr, odims, CFL_SIZE);

	md_clear(DIMS, odims, out, CFL_SIZE);
	md_max2(DIMS, idims, ostr, (float*)out, ostr, (const float*)out, istr, (const float*)tmp);

	if (mIP) {

		// need result of max in output
		md_min2(DIMS, idims, ostr, (float*)out, ostr, (const float*)out, istr, (const float*)tmp);
	}

	md_free(tmp);

	unmap_cfl(DIMS, idims, in);
	unmap_cfl(DIMS, odims, out);

	exit(0);
}
