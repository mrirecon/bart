/* Copyright 2024-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Nick Scholand
 *
 * Reference:
 *
 * Seiberlich N, Breuer F, Blaimer M, Jakob P, and Griswold M.
 * Self-calibrating GRAPPA operator gridding for radial and spiral trajectories.
 * Magn Reson Med 2007;59:930-935. https://doi.org/10.1002/mrm.21565
 *
 * Notes:
 * Implemented following the code in the MRFingerprintingRecon.jl julia package. Release ????.
 * (Add DOI here), https://github.com/MagneticResonanceImaging/MRFingerprintingRecon.jl
 *
 * ToDo:
 * - Benchmark and speed up GROG gridding: Allocations and/or matrix exponential increase computational cost markedly
 * - Move SVD based pseudo-inverse from stack to heap -> increase number of spokes for calibration
 */

#include <stdbool.h>
#include <complex.h>
#include <string.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/io.h"
#include "misc/mri.h"

#include "calib/grog.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


static void grog_calib2(int calib_spokes, int D, const long lnG_dims[D], complex float* lnG, const long tdims[D], const complex float* traj, const long ddims[D], const complex float* data)
{
	unsigned long tflags = md_nontriv_dims(DIMS, tdims);
	unsigned long dflags = md_nontriv_dims(DIMS, ddims);

	// if we have multiple data for some position, we only use the first

	long ddims1[D];
	md_select_dims(D, ~(dflags & ~tflags) | COIL_FLAG, ddims1, ddims);

	complex float *data1 = md_alloc(D, ddims1, CFL_SIZE);

	long pos[D];
	for (int i = 0; i < D; i++)
		pos[i] = 0;

	md_copy_block(D, pos, ddims1, data1, ddims, data, CFL_SIZE);

	long tdims2[5];
	md_copy_dims(5, tdims2, tdims);
	tdims2[PHS2_DIM] *= md_calc_size(D - 5, tdims + 5);

	long ddims2[D];
	md_singleton_dims(D - 5, ddims2 + 5);
	md_copy_dims(5, ddims2, ddims1);
	ddims2[PHS2_DIM] *= md_calc_size(D - 5, ddims1 + 5);

	complex float *data2 = md_alloc(D, ddims2, CFL_SIZE);

	md_reshape(D, ~(READ_FLAG|COIL_FLAG), ddims2, data2, ddims1, data1, CFL_SIZE);

	long ddims3[5];
	md_copy_dims(5, ddims3, ddims2);

	// truncate number of spokes

	if (0 < calib_spokes) {

		tdims2[PHS2_DIM] = MIN(tdims2[PHS2_DIM], calib_spokes);
		ddims3[PHS2_DIM] = MIN(ddims3[PHS2_DIM], calib_spokes);
	}

	assert(ddims3[PHS2_DIM] == tdims2[PHS2_DIM]);

	md_copy_block(5, pos, ddims3, data1, ddims2, data2, CFL_SIZE);

	grog_calib(5, lnG_dims, lnG, tdims2, traj, ddims3, data1);

	md_free(data1);
	md_free(data2);
}


static void grog_grid2(int D, const long tdims[D], const complex float* traj_shift, const long ddims[D], complex float* data_grid, const complex float* data, const long lnG_dims[D], complex float* lnG)
{
	unsigned long tflags = md_nontriv_dims(D, tdims);
	unsigned long dflags = md_nontriv_dims(D, ddims);

	// loop over dimensions

	unsigned long loop_flags = tflags & dflags & ~(PHS1_FLAG|PHS2_FLAG);

	if (0UL == loop_flags)
		return grog_grid(D, tdims, traj_shift, ddims, data_grid, data, lnG_dims, lnG);

	long tdims1[D];
	long ddims1[D];
	md_select_dims(D, ~loop_flags, tdims1, tdims);
	md_select_dims(D, ~loop_flags, ddims1, ddims);

	complex float* shift1 = md_alloc(D, tdims1, CFL_SIZE);
	complex float* data1 = md_alloc(D, ddims1, CFL_SIZE);
	complex float* data_grid1 = md_alloc(D, ddims1, CFL_SIZE);

	long pos[D];
	for (int i = 0; i < D; i++)
		pos[i] = 0;

	do {
		md_copy_block(D, pos, ddims1, data1, ddims, data, CFL_SIZE);
		md_copy_block(D, pos, tdims1, shift1, tdims, traj_shift, CFL_SIZE);

		grog_grid(D, tdims1, shift1, ddims1, data_grid1, data1, lnG_dims, lnG);

		md_copy_block(D, pos, ddims, data_grid, ddims1, data_grid1, CFL_SIZE);

	} while (md_next(D, ddims, loop_flags, pos));

	md_free(shift1);
	md_free(data1);
	md_free(data_grid1);
}


static const char help_str[] = "GROG calibration and gridding of radial data.\n";


int main_grog(int argc, char* argv[argc])
{
	const char* traj_file = NULL;
	const char* data_file = NULL;
	const char* grid_traj_file = NULL;
	const char* grid_data_file = NULL;

	int calib_spokes = -1;

	struct arg_s args[] = {

		ARG_INFILE(true, &traj_file, "radial trajectory"),
		ARG_INFILE(true, &data_file, "radial data"),
		ARG_INFILE(true, &grid_traj_file, "gridded trajectory"),
		ARG_OUTFILE(true, &grid_data_file, "gridded data"),
	};

	const struct opt_s opts[] = {

		OPTL_INT('s', "calib-spokes", &calib_spokes, "num", "Number of spokes for GROG calibration"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long tdims[DIMS];
	const complex float* traj = load_cfl(traj_file, DIMS, tdims);

	long ddims[DIMS];
	const complex float* data = load_cfl(data_file, DIMS, ddims);

	debug_printf(DP_DEBUG2, "tdims:\t");
	debug_print_dims(DP_DEBUG2, DIMS, tdims);

	debug_printf(DP_DEBUG2, "ddims:\t");
	debug_print_dims(DP_DEBUG2, DIMS, ddims);

	unsigned long tflags = md_nontriv_dims(DIMS, tdims);
	unsigned long dflags = md_nontriv_dims(DIMS, ddims);

	if (!md_check_compat(DIMS - 1, ~0UL, tdims + 1, ddims + 1))
		error("Incompatible dimensions\n");

	if (1UL != (tflags & ~dflags))
		error("Incompatible dimensions\n");


	// Calibration of GROG kernels

	long lnG_dims[DIMS];
	md_select_dims(DIMS, COIL_FLAG, lnG_dims, ddims);
	lnG_dims[READ_DIM] = tdims[READ_DIM]; // Number of dimensions
	lnG_dims[MAPS_DIM] = ddims[COIL_DIM];

	complex float* lnG = md_alloc(DIMS, lnG_dims, CFL_SIZE);

	double calib_start = timestamp();

	grog_calib2(calib_spokes, DIMS, lnG_dims, lnG, tdims, traj, ddims, data);

	double calib_end = timestamp();

	debug_printf(DP_DEBUG1, "Time for calibration: %f\n", calib_end - calib_start);

	// Shifting of Data

	long tdims2[DIMS];
	const complex float* traj_grid = load_cfl(grid_traj_file, DIMS, tdims2);

	if (!md_check_compat(DIMS, 0UL, tdims, tdims2))
		error("Incompatible trajectory.\n");

	complex float* data_grid = create_cfl(grid_data_file, DIMS, ddims);

	complex float* shift = md_alloc(DIMS, tdims, CFL_SIZE);

	double grid_start = timestamp();

	md_zsub(DIMS, tdims, shift, traj_grid, traj);

	grog_grid2(DIMS, tdims, shift, ddims, data_grid, data, lnG_dims, lnG);

	md_free(shift);

	double grid_end = timestamp();

	debug_printf(DP_DEBUG1, "Time for gridding: %f\n", grid_end - grid_start);

	md_free(lnG);

	unmap_cfl(DIMS, tdims, traj);
	unmap_cfl(DIMS, ddims, data);

	unmap_cfl(DIMS, tdims, traj_grid);
	unmap_cfl(DIMS, ddims, data_grid);

	return 0;
}

