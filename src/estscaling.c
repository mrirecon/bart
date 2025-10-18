/* Copyright 2024. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2024 Moritz Blumenthal
 */

#include <math.h>
#include <complex.h>

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/mri.h"
#include "num/multind.h"

#include "sense/optcom.h"

#include "noncart/nufft.h"

#define CFL_SIZE sizeof(complex float)


static const char help_str[] = "Estimate scaling from k-space center.\n";

#define DIMS 16

int main_estscaling(int argc, char* argv[argc])
{
	const char* kspace_file = NULL;
	const char* scaling_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &kspace_file, "k-space"),
		ARG_OUTFILE(true, &scaling_file, "scaling"),
	};

	bool invert = false;
	long img_vec[3] = { 0, 0, 0 };
	float p = -1.;

	const struct opt_s opts[] = {

		OPT_SET('i', &invert, "invert scaling (to directly multiply k-space data)"),
		OPT_VEC3('x', &img_vec, "x:y:z", "image dimensions"),
		OPTL_FLOAT('p', "percentile", &p, "p", "use p-percentile for scaling"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	long ksp_dims[DIMS];
	complex float* ksp = load_cfl(kspace_file, DIMS, ksp_dims);

	long scaling_dims[DIMS];
	md_select_dims(DIMS, BATCH_FLAG, scaling_dims, ksp_dims);

	complex float* scaling = create_cfl(scaling_file, DIMS, scaling_dims);

	long slc_dims[DIMS];
	md_select_dims(DIMS, ~BATCH_FLAG, slc_dims, ksp_dims);

	for (long i = 0; i < ksp_dims[BATCH_DIM]; i++)
		scaling[i] = estimate_scaling(slc_dims, NULL, ksp + i * md_calc_size(DIMS, slc_dims), p);

	unmap_cfl(DIMS, ksp_dims, ksp);

	if (0 != md_calc_size(3, img_vec)) {

		float rescale = sqrtf(md_calc_size(3, img_vec) / (float)md_calc_size(3, ksp_dims));

		for (long i = 0; i < ksp_dims[BATCH_DIM]; i++)
			scaling[i] = scaling[i] / rescale;
	}

	if (invert) {

		for (long i = 0; i < ksp_dims[BATCH_DIM]; i++)
			scaling[i] = 1. / scaling[i];
	}

	unmap_cfl(DIMS, scaling_dims, scaling);

	return 0;
}

