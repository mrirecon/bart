/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2012 Martin Uecker
 * uecker@eecs.berkeley.edu
 */

#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <stdio.h>

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"

#include "num/multind.h"
#include "num/fft.h"

#include "calib/direct.h"


static const char usage_str[] = "cal_size <input> <output>";
static const char help_str[] =
	"Estimates coil sensitivities from the k-space center using\n"
	"a direct method (McKenzie et al.). The size of the fully-sampled\n"
	"calibration region is automatically determined but limited by\n"
	"{cal_size} (e.g. in the readout direction).\n";



int main_caldir(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 3, usage_str, help_str);

	long dims[DIMS];

	complex float* in_data = load_cfl(argv[2], DIMS, dims);

	int calsize_ro = atoi(argv[1]);
	long calsize[3] = { calsize_ro, calsize_ro, calsize_ro };

	assert((dims[0] == 1) || (calsize_ro < dims[0]));
	assert(1 == dims[4]);
	
	complex float* out_data = create_cfl(argv[3], DIMS, dims);


	long caldims[DIMS];
	complex float* cal_data = extract_calib(caldims, calsize, dims, in_data, false);

	printf("Calibration region %ldx%ldx%ld\n", caldims[0], caldims[1], caldims[2]);

	direct_calib(dims, out_data, caldims, cal_data);

	printf("Done.\n");

	md_free(cal_data);

	unmap_cfl(DIMS, dims, (void*)out_data);
	unmap_cfl(DIMS, dims, (void*)in_data);

	exit(0);
}


