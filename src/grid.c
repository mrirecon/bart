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

static const char help_str[] = "Compute sampling grid for domains x-space/ k-space x time.";

int main_grid(int argc, char* argv[argc])
{
	const char* out_file = NULL;
	const char* traj_file = NULL;

	struct grid_opts go = grid_opts_defaults;

	struct arg_s args[] = {

		ARG_OUTFILE(true, &out_file, "grid"),
	};

	long sdims[3] = {-1, -1, -1};
	long timedim = 1;

	const struct opt_s opts[] = {

		OPT_SET('k', &go.kspace, "Compute for k-space."),
		OPT_INFILE('t', &traj_file, "trajectory file", "Sampling trajectory for k-space of shape (3, Nx, N).\n"),
		OPT_VECN('D', sdims, "Dimensions x-space/ k-space per basis vector."),
		OPT_LONG('T', &timedim, "dt", "Dimension time domain.\n"),
		OPTL_FLVEC3(0, "b1", &go.b0, "f1:f2:f3", "First basis vector."), //
		OPTL_FLVEC3(0, "b2", &go.b1, "f1:f2:f3", "Second basis vector."), //
		OPTL_FLVEC3(0, "b3", &go.b2, "f1:f2:f3", "Third basis vector."), //
		OPTL_FLOAT(0, "bt", &go.bt, "f1", "Time direction."), //
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	if (NULL != traj_file && (-1 != sdims[0] || -1 != sdims[1] || -1 != sdims[2]))
		error("trajectory file *and* spatial/fourier dims cannot be provided together.\n");

	complex float* traj = NULL;
	long tdims[DIMS];

	if (NULL == traj_file) {

		go.dims[0] = (0 >= sdims[0]) ? 128 : sdims[0];
		go.dims[1] = (0 >= sdims[1]) ? 128 : sdims[1];
		go.dims[2] = (0 >= sdims[2]) ? 1 : sdims[2];

	} else {

		traj = load_cfl(traj_file, DIMS, tdims);

		go.dims[0] = 1;
		go.dims[1] = tdims[1];
		go.dims[2] = tdims[2];
	}

	if (0 == go.bt)
		go.bt = 1;

	go.dims[TIME_DIM] = (0 >= timedim) ? 1 : timedim;

	if ((0 == go.b0[0]) && (0 == go.b0[1]) && (0 == go.b0[2])) {

		go.b0[0] = 0.5;
		go.b0[1] = 0;
		go.b0[2] = 0;
	}

	if ((0 == go.b1[0]) && (0 == go.b1[1]) && (0 == go.b1[2])) {

		go.b1[0] = 0;
		go.b1[1] = 0.5;
		go.b1[2] = 0;
	}

	if ((0 == go.b2[0]) && (0 == go.b2[1]) && (0 == go.b2[2])) {

		go.b2[0] = 0;
		go.b2[1] = 0;
		go.b2[2] = 0.5;
	}

	long gdims[DIMS];

	float* grid = compute_grid(DIMS, gdims, &go, tdims, traj);

	unmap_cfl(DIMS, tdims, traj);


	complex float* optr = create_cfl(out_file, DIMS, gdims);

	md_clear(DIMS, gdims, optr, CFL_SIZE);

	long pos[DIMS];
	md_set_dims(DIMS, pos, 0);

	long strs[DIMS], ostrs[DIMS];
	md_calc_strides(DIMS, strs, gdims, FL_SIZE);
	md_calc_strides(DIMS, ostrs, gdims, CFL_SIZE);

	do {
		MD_ACCESS(DIMS, ostrs, pos, optr) = MD_ACCESS(DIMS, strs, pos, grid) + 0.i;

	} while(md_next(DIMS, gdims, ~0UL, pos));

	md_free(grid);

	unmap_cfl(DIMS, gdims, optr);

	return 0;
}

