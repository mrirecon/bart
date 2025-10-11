/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2025 Martin Heide
 */

#include <stdbool.h>
#include <complex.h>

#include "num/multind.h"
#include "num/init.h"
#include "num/flpmath.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "simu/grid.h"


static const char help_str[] = "Compute sampling grid for x-space / k-space (and time).";

int main_grid(int argc, char* argv[argc])
{
	const char* out_file = NULL;
	const char* traj_file = NULL;

	struct grid_opts go = grid_opts_init;

	struct arg_s args[] = {

		ARG_OUTFILE(true, &out_file, "grid"),
	};

	long sdims[3] = { -1, -1, -1 };
	long timedim = -1;

	const struct opt_s opts[] = {

		OPT_SET('k', &go.kspace, "Compute k-space grid"),
		OPT_INFILE('t', &traj_file, "Trajectory file", "Sampling trajectory for k-space\n"),
		OPT_VECN('D', sdims, "Size of x-space / k-space"),
		OPT_LONG('T', &timedim, "T", "Number of time points"),
		OPTL_FLVEC3(0, "b1", &go.b0, "f1:f2:f3", "First basis vector"),
		OPTL_FLVEC3(0, "b2", &go.b1, "f1:f2:f3", "Second basis vector"),
		OPTL_FLVEC3(0, "b3", &go.b2, "f1:f2:f3", "Third basis vector"),
		OPTL_FLOAT(0, "bt", &go.bt, "f1", "Time step."),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	if ((NULL != traj_file) && ((-1 != sdims[0]) || (-1 != sdims[1]) || (-1 != sdims[2])))
		error("Trajectory file and spatial/Fourier dims cannot be provided together.\n");

	complex float* traj = NULL;
	long tdims[DIMS];

	go.dims[TIME_DIM] = (0 >= timedim) ? 1 : timedim;

	if (NULL == traj_file) {

		go.dims[0] = (0 >= sdims[0]) ? 128 : sdims[0];
		go.dims[1] = (0 >= sdims[1]) ? 128 : sdims[1];
		go.dims[2] = (0 >= sdims[2]) ? 1 : sdims[2];

	} else {

		traj = load_cfl(traj_file, DIMS, tdims);

		go.dims[0] = 1;
		go.dims[1] = tdims[1];
		go.dims[2] = tdims[2];

		// if the user provides a 4d traj which has already a time we continue with those dims.
		if ((4 == tdims[0]) && (-1 == timedim))
			go.dims[TIME_DIM] = tdims[TIME_DIM];
	}

	if (0. == go.bt)
		go.bt = 1.;

	if ((0. == go.b0[0]) && (0. == go.b0[1]) && (0. == go.b0[2]))
		go.b0[0] = 0.5;

	if ((0. == go.b1[0]) && (0. == go.b1[1]) && (0. == go.b1[2]))
		go.b1[1] = 0.5;

	if ((0. == go.b2[0]) && (0. == go.b2[1]) && (0. == go.b2[2]))
		go.b2[2] = 0.5;

	long gdims[DIMS];

	float* grid = compute_grid(DIMS, gdims, &go, tdims, traj);

	unmap_cfl(DIMS, tdims, traj);

	complex float* optr = create_cfl(out_file, DIMS, gdims);

	md_zcmpl_real(DIMS, gdims, optr, grid);

	md_free(grid);

	unmap_cfl(DIMS, gdims, optr);

	return 0;
}

