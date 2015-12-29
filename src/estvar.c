/* Copyright 2015. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2015 Siddharth Iyer <sid8795@gmail.com>
 * 2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <stdbool.h>
#include <stdio.h>

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/opts.h"

#include "num/flpmath.h"
#include "num/multind.h"

#include "misc/debug.h"

#include "calib/estvar.h"


static const char* usage_str = "<kspace>";
static const char* help_str = "Estimate the noise variance assuming white Gaussian noise.";


int main_estvar(int argc, char* argv[])
{
	long calsize_dims[3]  = { 24, 24, 24};
	long kernel_dims[3]   = {  6,  6,  6};

	const struct opt_s opts[] = {

		{ 'k', true, opt_vec3, &kernel_dims, " ksize\tkernel size" },
		{ 'K', true, opt_vec3, &kernel_dims, NULL },
		{ 'r', true, opt_vec3, &calsize_dims, " cal_size\tLimits the size of the calibration region." },
		{ 'R', true, opt_vec3, &calsize_dims, NULL },
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	int N = DIMS;
	long kspace_dims[N];

	complex float* kspace = load_cfl(argv[1], N, kspace_dims);

	for (int idx = 0; idx < 3; idx++) {

		kernel_dims[idx]  = (kspace_dims[idx] == 1) ? 1 : kernel_dims[idx];
		calsize_dims[idx] = (kspace_dims[idx] == 1) ? 1 : calsize_dims[idx];
	}

	float variance = estvar_kspace(N, kernel_dims, calsize_dims, kspace_dims, kspace);

	unmap_cfl(N, kspace_dims, kspace);

	printf("Estimated noise variance: %f\n", variance);

	exit(0);
}
