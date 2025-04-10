/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012 Martin Uecker
 */

#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <stdio.h>

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/mri2.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "num/multind.h"

#include "calib/direct.h"


static const char help_str[] =
	"Estimates coil sensitivities from the k-space center using\n"
	"a direct method (McKenzie et al.). The size of the fully-sampled\n"
	"calibration region is automatically determined but limited by\n"
	"{cal_size} (e.g. in the readout direction).";



int main_caldir(int argc, char* argv[argc])
{
	int calsize_ro = 0;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INT(true, &calsize_ro, "cal_size"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = {};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	long dims[DIMS];

	complex float* in_data = load_cfl(in_file, DIMS, dims);

	long calsize[3] = { calsize_ro, calsize_ro, calsize_ro };

	assert((dims[0] == 1) || (calsize_ro < dims[0]));
	assert(1 == dims[4]);
	
	complex float* out_data = create_cfl(out_file, DIMS, dims);


	long caldims[DIMS];
	complex float* cal_data = extract_calib(caldims, calsize, dims, in_data, false);

	debug_printf(DP_DEBUG1, "Calibration region %ldx%ldx%ld\n", caldims[0], caldims[1], caldims[2]);

	direct_calib(dims, out_data, caldims, cal_data);

	debug_printf(DP_DEBUG1, "Done.\n");

	md_free(cal_data);

	unmap_cfl(DIMS, dims, out_data);
	unmap_cfl(DIMS, dims, in_data);

	return 0;
}


