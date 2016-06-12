/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2013-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <stdbool.h>

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/fft.h"

#include "calib/walsh.h"


static const char usage_str[] = "<input> <output>";
static const char help_str[] = "Estimate coil sensitivities using walsh method (use with ecaltwo).";


int main_walsh(int argc, char* argv[])
{
	long bsize[3] = { 20, 20, 20 };
	long calsize[3] = { 24, 24, 24 };

	const struct opt_s opts[] = {

		OPT_VEC3('r', &calsize, "cal_size", "Limits the size of the calibration region."),
		OPT_VEC3('R', &calsize, "", "()"),
		OPT_VEC3('b', &bsize, "block_size", "Block size."),
		OPT_VEC3('B', &bsize, "", "()"),
	};

	cmdline(&argc, argv, 2, 2, usage_str, help_str, ARRAY_SIZE(opts), opts);


	long dims[DIMS];

	complex float* in_data = load_cfl(argv[1], DIMS, dims);

	assert((dims[0] == 1) || (calsize[0] < dims[0]));
	assert((dims[1] == 1) || (calsize[1] < dims[1]));
	assert((dims[2] == 1) || (calsize[2] < dims[2]));
	assert(1 == dims[MAPS_DIM]);

	long caldims[DIMS];
	complex float* cal_data = extract_calib(caldims, calsize, dims, in_data, false);
	unmap_cfl(DIMS, dims, in_data);

	debug_printf(DP_INFO, "Calibration region %ldx%ldx%ld\n", caldims[0], caldims[1], caldims[2]);

	dims[COIL_DIM] = dims[COIL_DIM] * (dims[COIL_DIM] + 1) / 2;
	complex float* out_data = create_cfl(argv[2], DIMS, dims);

	walsh(bsize, dims, out_data, caldims, cal_data);

	debug_printf(DP_INFO, "Done.\n");

	md_free(cal_data);
	unmap_cfl(DIMS, dims, out_data);
	exit(0);
}


