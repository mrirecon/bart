/* Copyright 2015. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <stdbool.h>

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/opts.h"

#include "num/flpmath.h"
#include "num/multind.h"

#include "calib/calmat.h"


static const char usage_str[] = "<kspace> <calibration matrix>";
static const char help_str[] = "Compute calibration matrix.";





int main_calmat(int argc, char* argv[])
{
	long calsize[3] = { 24, 24, 24 };
	long kdims[3] = { 5, 5, 5 };
	bool calcen = false;

	const struct opt_s opts[] = {

		OPT_VEC3('k', &kdims, "ksize", "kernel size"),
		OPT_VEC3('K', &kdims, "", "()"),
		OPT_VEC3('r', &calsize, "cal_size", "Limits the size of the calibration region."),
		OPT_VEC3('R', &calsize, "", "()"),
		OPT_SET('C', &calcen, "()"),
	};

	cmdline(&argc, argv, 2, 2, usage_str, help_str, ARRAY_SIZE(opts), opts);


	int N = DIMS;
	long ksp_dims[N];

	complex float* in_data = load_cfl(argv[1], N, ksp_dims);


	assert(1 == ksp_dims[MAPS_DIM]);



	long cal_dims[N];
	complex float* cal_data = NULL;

	if (!calcen) {

		cal_data = extract_calib(cal_dims, calsize, ksp_dims, in_data, false);

	} else {

		for (int i = 0; i < 3; i++)
			cal_dims[i] = (calsize[i] < ksp_dims[i]) ? calsize[i] : ksp_dims[i];

		for (int i = 3; i < N; i++)
			cal_dims[i] = ksp_dims[i];

		cal_data = md_alloc(N, cal_dims, CFL_SIZE);
		md_resize_center(N, cal_dims, cal_data, ksp_dims, in_data, CFL_SIZE);
	 }

	 for (int i = 0; i < 3; i++)
		 if (1 == ksp_dims[i])
			 kdims[i] = 1;



	 for (unsigned int i = 0; i < 3; i++)
		 if ((1 == cal_dims[i]) && (1 != ksp_dims[i]))
			error("Calibration region not found!\n");


	// FIXME: we should scale the data

	unmap_cfl(N, ksp_dims, in_data);


	long calmat_dims[N];
	md_singleton_dims(N, calmat_dims);
	complex float* cm = calibration_matrix(calmat_dims, kdims, cal_dims, cal_data);
	md_free(cal_data);

	complex float* out_data = create_cfl(argv[2], N, calmat_dims);
	md_copy(N, calmat_dims, out_data, cm, CFL_SIZE);
	md_free(cm);

	unmap_cfl(N, calmat_dims, out_data);

	exit(0);
}


