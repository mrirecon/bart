/* Copyright 2025. Uecker Lab. University Medical Center Göttingen.
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

#include "simu/sens.h"
#include "simu/grid.h"
#include "simu/shape.h"

static const char help_str[] = "Compute coil sensitivitity maps in x-space or k-space.";

int main_coils(int argc, char* argv[argc])
{
	const char* out_file = NULL;
	const char* traj_file = NULL;

	struct grid_opts gopts = grid_opts_defaults;
	struct coil_opts copts = coil_opts_defaults;

	long ncoils = 0;

	bool legacy_fov = false;

	struct arg_s args[] = {

		ARG_OUTFILE(true, &out_file, "sens"),
	};

	// FIXME: ulong overflow for coil # 63 in OPT_ULONG
	const struct opt_s opts[] = {

		OPT_SET('k', &copts.kspace, "Compute for k-space."),
		OPT_SET('L', &legacy_fov, "(Legacy: Coil Coefficients for FOV [-1,1].)"),
		OPT_LONG('n', &ncoils, "", "Select first n coil channels."),
		OPT_ULONG('b', &copts.flags, "", "Bitmask for selecting subset of channels.\n"),
		OPT_INFILE('t', &traj_file, "grid", "grid on which sensitivity maps should be evaluated.\n"),
		OPTL_SELECT(0, "H2D8C", enum coil_type, &copts.ctype, HEAD_2D_8CH, "2D head coil 8 channels (default)."),
		OPTL_SELECT(0, "H3D64C", enum coil_type, &copts.ctype, HEAD_3D_64CH, "3D head coil 64 channels."),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	if (legacy_fov) {

		gopts.veclen[0] = 1;
		gopts.veclen[1] = 1;
		gopts.b0[0] = 1;
		gopts.b1[1] = 1;
	}

	if (0 < ncoils && 64 > ncoils)
		copts.flags = MD_BIT(ncoils) - 1;

	long gdims[DIMS];
	float* grid = NULL;

	switch (copts.ctype) {

	case HEAD_2D_8CH:

		cnstr_H2D8CH(DIMS, &copts, legacy_fov);

		break;

	case HEAD_3D_64CH:

		cnstr_H3D64CH(DIMS, &copts, legacy_fov);

		break;

	case COIL_NONE:

		assert(0);

		break;
	}

	if (NULL != traj_file) {

		complex float* cgrid = load_cfl(traj_file, DIMS, gdims);

		grid = md_alloc(DIMS, gdims, FL_SIZE);
		md_real(DIMS, gdims, grid, cgrid);

		unmap_cfl(DIMS, gdims, cgrid);

	} else {

		gopts.kspace = copts.kspace;

		if (gopts.kspace) {

			// case trigonometric polynomials
			struct tri_poly* t = copts.data;
			md_copy_dims(DIMS, gdims, t->cpdims);
			grid = md_alloc(DIMS, gdims, FL_SIZE);
			md_copy(DIMS, gdims, grid, t->cpos, FL_SIZE);

		} else {

			grid = compute_grid(DIMS, gdims, &gopts, NULL, NULL);
		}
	}

	long odims[DIMS];
	md_singleton_dims(DIMS, odims);

	for (int i = 0; i < 3; i++)
		odims[i] = gdims[i+1];

	odims[COIL_DIM] = copts.N;

	complex float* optr = create_cfl(out_file, DIMS, odims);
	complex double* sens = md_alloc(DIMS, odims, CDL_SIZE);

	sample_coils(DIMS, odims, sens, gdims, grid, &copts);

	long ostrs[DIMS], sstrs[DIMS], pos[DIMS];

	md_set_dims(DIMS, pos, 0);
	md_calc_strides(DIMS, ostrs, odims, CFL_SIZE);
	md_calc_strides(DIMS, sstrs, odims, CDL_SIZE);

	do {
		MD_ACCESS(DIMS, ostrs, pos, optr) = MD_ACCESS(DIMS, sstrs, pos, sens);

	} while(md_next(DIMS, odims, ~0UL, pos));

	md_free(grid);
	copts.dstr(&copts);
	md_free(sens);
	unmap_cfl(DIMS, odims, optr);

	return 0;
}

