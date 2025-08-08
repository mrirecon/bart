/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013-2016 Martin Uecker
 */

#include <complex.h>
#include <stdbool.h>

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/mri2.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/fft.h"

#include "calib/walsh.h"


static const char help_str[] = "Estimate coil sensitivities using walsh method (use with ecaltwo).";


int main_walsh(int argc, char* argv[argc])
{
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	long bsize[3] = { 20, 20, 20 };
	long calsize[3] = { 24, 24, 24 };

	const struct opt_s opts[] = {

		OPT_VEC3('r', &calsize, "cal_size", "Limits the size of the calibration region."),
		OPT_VEC3('R', &calsize, "", "()"),
		OPT_VEC3('b', &bsize, "block_size", "Block size."),
		OPT_VEC3('B', &bsize, "", "()"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);


	long dims[DIMS];

	complex float* in_data = load_cfl(in_file, DIMS, dims);

	for (int i = 0; i < 2; i++)
		if ((dims[i] != 1) && (calsize[i] >= dims[i]))
			error("Incorrect size of dimension %d", i);

	if (1 != dims[MAPS_DIM])
		error("Maps dimension must have size one");

	long caldims[DIMS];
	complex float* cal_data = extract_calib(caldims, calsize, dims, in_data, false);
	unmap_cfl(DIMS, dims, in_data);

	debug_printf(DP_INFO, "Calibration region %ldx%ldx%ld\n", caldims[0], caldims[1], caldims[2]);

	dims[COIL_DIM] = dims[COIL_DIM] * (dims[COIL_DIM] + 1) / 2;

	complex float* out_data = create_cfl(out_file, DIMS, dims);

	walsh(bsize, dims, out_data, caldims, cal_data);

	debug_printf(DP_INFO, "Done.\n");

	md_free(cal_data);

	unmap_cfl(DIMS, dims, out_data);

	return 0;
}

