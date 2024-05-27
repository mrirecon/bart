/* Copyright 2024. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Nick Scholand
 *
 * Reference:
 * 	Seiberlich, N., Breuer, F., Blaimer, M., Jakob, P. and Griswold, M. (2008),
 * 	Self-calibrating GRAPPA operator gridding for radial and spiral trajectories.
 * 	Magn. Reson. Med., 59: 930-935. https://doi.org/10.1002/mrm.21565
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
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/io.h"
#include "misc/mri.h"

#include "noncart/grog.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


static const char help_str[] =
	"GROG calibration and gridding of radial data.\n";


int main_grog(int argc, char* argv[argc])
{
	const char* traj_file = NULL;
	const char* data_file = NULL;
	const char* grid_traj_file = NULL;
	const char* grid_data_file = NULL;
	const char* measure_file = NULL;

	int calib_spokes = -1;

	struct arg_s args[] = {

		ARG_INFILE(true, &traj_file, "radial trajectory"),
		ARG_INFILE(true, &data_file, "radial data"),
		ARG_OUTFILE(true, &grid_traj_file, "gridded trajectory"),
		ARG_OUTFILE(true, &grid_data_file, "gridded data"),
	};

	const struct opt_s opts[] = {

		OPTL_INT('s', "calib-spokes", &calib_spokes, "num", "Number of spokes for GROG calibration"),
		OPTL_OUTFILE(0, "measure-time", &measure_file, "file", "Output file with timing of 1. calibration and 2. gridding."),
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

	// ------------------------------------------
	//	Prepare data for calibration
	// ------------------------------------------

	// Only data from unique set of projection angles -> Extract first time step
	// in case of equal dims it becomes a simple copy operation
	long ddims_slice[DIMS];
	md_copy_dims(DIMS, ddims_slice, ddims);

	if (tdims[TIME_DIM] < ddims[TIME_DIM])
		ddims_slice[TIME_DIM] = tdims[TIME_DIM];

	complex float* data_slice = md_alloc(DIMS, ddims_slice, CFL_SIZE);

	long pos[DIMS] = { [0 ... DIMS - 1] = 0 };
	md_copy_block(DIMS, pos, ddims_slice, data_slice, ddims, data, CFL_SIZE);

	// Combine data from unique projections projections:
	// Scenario: DIM > COIL_DIM and traj[DIM] == data[DIM]
	unsigned long flags = 0;
	long joined_dim = 1;
	long info_dim = PHS2_DIM; // Split dimension with unique projection angle INFOrmation can repeat in TIME_DIM
	int c = 0;

	for (int i = PHS2_DIM; i < DIMS; i++) {

		if ((1 != tdims[i]) && (tdims[i] == ddims[i])) {

			flags = MD_SET(flags, (unsigned long)i);

			joined_dim *= tdims[i];

			if ((PHS2_DIM != i)) {

				assert(0 == c); // Constraint to only one repeating dimension
				c++;

				info_dim = i;
			}
		}
	}

	debug_printf(DP_DEBUG2, "Joined Dim: %d,\tFlag: %d,\t info_dim: %ld\n", joined_dim, flags, info_dim);

	// Split dimension including unique projections should not be larger than TIME_DIM
	// See grog.mk for examples which combination sof input dimensions are supported
	assert(TIME_DIM >= info_dim);

	// Combine Trajectory
	long tdims2[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, tdims2, tdims);
	tdims2[PHS2_DIM] = joined_dim;

	complex float* traj2 = md_alloc(DIMS, tdims2, CFL_SIZE);
	md_reshape(DIMS, flags, tdims2, traj2, tdims, traj, CFL_SIZE);

	// Combine Data
	long ddims2[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|COIL_FLAG, ddims2, ddims_slice);
	ddims2[PHS2_DIM] = joined_dim;

	complex float* data2 = md_alloc(DIMS, ddims2, CFL_SIZE);

	md_reshape(DIMS, flags, ddims2, data2, ddims_slice, data_slice, CFL_SIZE);

	md_free(data_slice);

	// Reduce to selected number of spokes
	long tdims_calib[DIMS];
	md_copy_dims(DIMS, tdims_calib, tdims2);

	long ddims_calib[DIMS];
	md_copy_dims(DIMS, ddims_calib, ddims2);

	if (-1 != calib_spokes) {

		tdims_calib[PHS2_DIM] = calib_spokes;
		ddims_calib[PHS2_DIM] = calib_spokes;
	}

	complex float* traj_calib = md_alloc(DIMS, tdims_calib, CFL_SIZE);
	complex float* data_calib = md_alloc(DIMS, ddims_calib, CFL_SIZE);

	long pos2[DIMS] = { [0 ... DIMS - 1] = 0 };

	md_copy_block(DIMS, pos2, tdims_calib, traj_calib, tdims2, traj2, CFL_SIZE);
	md_copy_block(DIMS, pos2, ddims_calib, data_calib, ddims2, data2, CFL_SIZE);

	md_free(traj2);
	md_free(data2);

	// ------------------------------------------
	//	1. Calibration of GROG kernels
	// ------------------------------------------

	long lnG_dims[DIMS];
	md_select_dims(DIMS, COIL_FLAG, lnG_dims, ddims);
	lnG_dims[READ_DIM] = tdims[READ_DIM]; // Number of dimensions
	lnG_dims[MAPS_DIM] = ddims[COIL_DIM]; // Number of coils

	complex float* lnG = md_alloc(DIMS, lnG_dims, CFL_SIZE);

	double calib_start = timestamp();
	grog_calib(DIMS, lnG_dims, lnG, tdims_calib, traj_calib, ddims_calib, data_calib);
	double calib_end = timestamp();

	md_free(traj_calib);
	md_free(data_calib);

	// ------------------------------------------
	//	Prepare data for Gridding
	// ------------------------------------------

	// If info_dim is between COIL_DIM and TIME_DIM, copy will not work.
	// Therefore, reordering of the dimensions is required ensuring info_dim and PHS2_DIM are at the
	// end of the data block
	long tdimsT[DIMS];
	long ddimsT[DIMS];

	if (PHS2_DIM != info_dim) {

		md_transpose_dims(DIMS, PHS2_DIM, TIME_DIM, tdimsT, tdims);
		md_transpose_dims(DIMS, PHS2_DIM, TIME_DIM, ddimsT, ddims);

	} else {

		md_copy_dims(DIMS, tdimsT, tdims);
		md_copy_dims(DIMS, ddimsT, ddims);
	}

	complex float* trajT = md_alloc(DIMS, tdimsT, CFL_SIZE);
	complex float* dataT = md_alloc(DIMS, ddimsT, CFL_SIZE);

	if (PHS2_DIM != info_dim) {

		md_transpose(DIMS, PHS2_DIM, TIME_DIM, tdimsT, trajT, tdims, traj, CFL_SIZE);
		md_transpose(DIMS, PHS2_DIM, TIME_DIM, ddimsT, dataT, ddims, data, CFL_SIZE);

	} else {

		md_copy(DIMS, tdimsT, trajT, traj, CFL_SIZE);
		md_copy(DIMS, ddimsT, dataT, data, CFL_SIZE);
	}

	// Join spokes of info_dim and PHS2_DIM (latter is was transposed to TIME_DIM)
	long tdims_rs[DIMS];
	long ddims_rs[DIMS];

	if (PHS2_DIM == info_dim) {

		md_copy_dims(DIMS, tdims_rs, tdimsT);
		md_copy_dims(DIMS, ddims_rs, ddimsT);

	 } else if (TIME_DIM == info_dim) {

		md_select_dims(DIMS, ~(PHS2_FLAG | TIME_FLAG), tdims_rs, tdimsT);
		tdims_rs[TIME_DIM] = joined_dim;

		md_select_dims(DIMS, ~(PHS2_FLAG | TIME_FLAG), ddims_rs, ddimsT);
		ddims_rs[TIME_DIM] = joined_dim;

	} else {

		md_select_dims(DIMS, ~(MD_BIT(info_dim) | TIME_FLAG), tdims_rs, tdimsT);
		tdims_rs[TIME_DIM] = joined_dim;

		md_select_dims(DIMS, ~(MD_BIT(info_dim) | TIME_FLAG), ddims_rs, ddimsT);
		ddims_rs[TIME_DIM] = joined_dim;
	}

	complex float* traj_rs = md_alloc(DIMS, tdims_rs, CFL_SIZE);
	complex float* data_rs = md_alloc(DIMS, ddims_rs, CFL_SIZE);

	if (TIME_DIM != info_dim) {

		md_reshape(DIMS, MD_BIT(info_dim)|TIME_FLAG, tdims_rs, traj_rs, tdimsT, trajT, CFL_SIZE);
		md_reshape(DIMS, MD_BIT(info_dim)|TIME_FLAG, ddims_rs, data_rs, ddimsT, dataT, CFL_SIZE);

	} else {

		md_reshape(DIMS, MD_BIT(info_dim)|PHS2_FLAG, tdims_rs, traj_rs, tdimsT, trajT, CFL_SIZE);
		md_reshape(DIMS, MD_BIT(info_dim)|PHS2_FLAG, ddims_rs, data_rs, ddimsT, dataT, CFL_SIZE);
	}

	// Move combined spokes from TIME_DIM back to PHS2_DIM
	long tdims_rs2[DIMS];
	long ddims_rs2[DIMS];

	if (PHS2_DIM != info_dim) {

		md_transpose_dims(DIMS, PHS2_DIM, TIME_DIM, tdims_rs2, tdims_rs);
		md_transpose_dims(DIMS, PHS2_DIM, TIME_DIM, ddims_rs2, ddims_rs);

	} else {

		md_copy_dims(DIMS, tdims_rs2, tdims_rs);
		md_copy_dims(DIMS, ddims_rs2, ddims_rs);
	}

	complex float* traj_rs2 = md_alloc(DIMS, tdims_rs2, CFL_SIZE);
	complex float* data_rs2 = md_alloc(DIMS, ddims_rs2, CFL_SIZE);

	if (PHS2_DIM != info_dim) {

		md_transpose(DIMS, PHS2_DIM, TIME_DIM, tdims_rs2, traj_rs2, tdims_rs, traj_rs, CFL_SIZE);
		md_transpose(DIMS, PHS2_DIM, TIME_DIM, ddims_rs2, data_rs2, ddims_rs, data_rs, CFL_SIZE);

	} else {

		md_copy(DIMS, tdims_rs2, traj_rs2, traj_rs, CFL_SIZE);
		md_copy(DIMS, ddims_rs2, data_rs2, data_rs, CFL_SIZE);
	}

	// Copies of *_rs2 files for storing data calculated in multiple threads
	complex float* traj_rs2_grid = md_calloc(DIMS, tdims_rs2, CFL_SIZE);
	complex float* data_rs2_grid = md_calloc(DIMS, ddims_rs2, CFL_SIZE);

	// ------------------------------------------
	// 	2. Shifting of Data
	// ------------------------------------------

	double grid_start = timestamp();
	grog_grid(DIMS, tdims_rs2, traj_rs2_grid, traj_rs2, ddims_rs2, data_rs2_grid, data_rs2, lnG_dims, lnG);
	double grid_end = timestamp();

	md_free(lnG);

	// ------------------------------------------
	//	Reshape data to original format
	// ------------------------------------------

	if (PHS2_DIM != info_dim) {

		md_transpose(DIMS, PHS2_DIM, TIME_DIM, tdims_rs, traj_rs, tdims_rs2, traj_rs2_grid, CFL_SIZE);
		md_transpose(DIMS, PHS2_DIM, TIME_DIM, ddims_rs, data_rs, ddims_rs2, data_rs2_grid, CFL_SIZE);

	} else {

		md_copy(DIMS, tdims_rs, traj_rs, traj_rs2_grid, CFL_SIZE);
		md_copy(DIMS, ddims_rs, data_rs, data_rs2_grid, CFL_SIZE);
	}

	if (TIME_DIM != info_dim) {

		md_reshape(DIMS, MD_BIT(info_dim)|TIME_FLAG, tdimsT, trajT, tdims_rs, traj_rs, CFL_SIZE);
		md_reshape(DIMS, MD_BIT(info_dim)|TIME_FLAG, ddimsT, dataT, ddims_rs, data_rs, CFL_SIZE);

	} else {

		md_reshape(DIMS, MD_BIT(info_dim)|PHS2_FLAG, tdims_rs, traj_rs, tdimsT, trajT, CFL_SIZE);
		md_reshape(DIMS, MD_BIT(info_dim)|PHS2_FLAG, ddims_rs, data_rs, ddimsT, dataT, CFL_SIZE);
	}

	// Write to ouput
	complex float* traj_grid = create_cfl(grid_traj_file, DIMS, tdims);
	complex float* data_grid = create_cfl(grid_data_file, DIMS, ddims);

	if (PHS2_DIM != info_dim) {

		md_transpose(DIMS, PHS2_DIM, TIME_DIM, tdims, traj_grid, tdimsT, trajT, CFL_SIZE);
		md_transpose(DIMS, PHS2_DIM, TIME_DIM, ddims, data_grid, ddimsT, dataT, CFL_SIZE);

	} else {

		md_copy(DIMS, tdims, traj_grid, trajT, CFL_SIZE);
		md_copy(DIMS, ddims, data_grid, dataT, CFL_SIZE);
	}

	md_free(trajT);
	md_free(dataT);

	md_free(traj_rs);
	md_free(data_rs);

	md_free(traj_rs2);
	md_free(data_rs2);

	md_free(traj_rs2_grid);
	md_free(data_rs2_grid);

	unmap_cfl(DIMS, tdims, traj_grid);
	unmap_cfl(DIMS, ddims, data_grid);

	// 3. Benchmark calibration and gridding times

	long time_dims[DIMS] = { [0 ... DIMS - 1] = 1 };

	if (NULL != measure_file) {

		time_dims[READ_DIM] = 2;

		complex float* time = create_cfl(measure_file, DIMS, time_dims);

		time[0] = (float) (calib_end - calib_start);
		time[1] = (float) (grid_end - grid_start);

		unmap_cfl(DIMS, time_dims, time);
	}

	return 0;
}


