/* Copyright 2024-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2024-2025 Markus Huemer <huemer@tugraz.at>
 * 2025 Daniel Mackner <daniel.mackner@tugraz.at>
 */

#include <stdbool.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/opts.h"

#include "noncart/radial.h"

#ifndef DIMS
#define DIMS 16
#endif

static const char help_str[] =
		"Correct delays for a given trajectory.\n";


int main_trajcor(int argc, char* argv[argc])
{
	const char* inputtraj_file = NULL;
	const char* outputtraj_file = NULL;

	float gdelays[3] = { 0., 0., 0. };

	struct arg_s args[] = {
		ARG_INFILE(true, &inputtraj_file, "traj"),
		ARG_OUTFILE(true, &outputtraj_file, "output traj"),
	};

	const struct opt_s opts[] = {
		OPT_FLVEC3('q', &gdelays, "delays", "gradient delays: y, x, xy"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();


	int N = DIMS;

	long dimstraj[N];

	complex float* traj = load_cfl(inputtraj_file, N, dimstraj);

	if (!traj_is_radial(DIMS, dimstraj, traj))
		error("Trajectory is not radial with same dk!\n");

	complex float* outtraj = create_cfl(outputtraj_file, N, dimstraj);


	// to be consistend with traj tool this is flipped
	// as x is the second entry in the traj and y is the first
	complex float mat[3][3] = {
		{ gdelays[1], gdelays[2], 0. },
		{ gdelays[2], gdelays[0], 0. },
		{ 0., 0., 0. },
	};


	long dir_dims[N];
	md_select_dims(N, ~MD_BIT(1), dir_dims, dimstraj);

	complex float* dir = md_alloc_sameplace(N, dir_dims, CFL_SIZE, traj);
	traj_radial_direction(N, dir_dims, dir, dimstraj, traj);

	long tdir_dims[N];
	md_transpose_dims(N, 0, 1, tdir_dims, dir_dims);

	long mat_dims[N];
	md_singleton_dims(N, mat_dims);
	md_max_dims(N, MD_BIT(0) | MD_BIT(1), mat_dims, dir_dims, tdir_dims);

	complex float* offset = md_alloc_sameplace(N, dir_dims, CFL_SIZE, traj);
	md_ztenmul(N, dir_dims, offset, mat_dims, &(mat[0][0]), tdir_dims, dir);

	md_free(dir);
	md_zadd2(N, dimstraj, MD_STRIDES(N, dimstraj, CFL_SIZE), outtraj, MD_STRIDES(N, dimstraj, CFL_SIZE), traj, MD_STRIDES(N, dir_dims, CFL_SIZE), offset);

	md_free(offset);

	unmap_cfl(N, dimstraj, traj);
	unmap_cfl(N, dimstraj, outtraj);

	return 0;
}


