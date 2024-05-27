/* Copyright 2024. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Nick Scholand
 *
 * Implementation of:
 * 	Seiberlich, N., Breuer, F., Blaimer, M., Jakob, P. and Griswold, M. (2008),
 * 	Self-calibrating GRAPPA operator gridding for radial and spiral trajectories.
 * 	Magn. Reson. Med., 59: 930-935. https://doi.org/10.1002/mrm.21565
 *
 * Notes:
 * Implemented following the code in the MRFingerprintingRecon.jl julia package. Release ????.
 * (Add DOI here), https://github.com/MagneticResonanceImaging/MRFingerprintingRecon.jl
 */

#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"
#include "num/linalg.h"
#include "num/matexp.h"

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mri.h"

#include "grog.h"

// Calculate log(GROG operator along spokes) = vtheta
// 	-> Compare Eq. 8 in Seiberlich et al. MRM, 2008.
static void estimate_vtheta(int D, const long vtheta_dims[D], complex float* vtheta, const long ddims[D], const complex float* data)
{
	// Allocate the two datasets s(theta, r) and s(theta, r+1)
	long ddims_shift[DIMS];
	md_copy_dims(DIMS, ddims_shift, ddims);
	ddims_shift[PHS1_DIM] -= 1;

	long pos_shift[DIMS] = { [0 ... DIMS - 1] = 0 };

	complex float* shift1 = md_alloc(DIMS, ddims_shift, CFL_SIZE);
	complex float* shift2 = md_alloc(DIMS, ddims_shift, CFL_SIZE);

	// s(theta, r)
	md_copy_block(DIMS, pos_shift, ddims_shift, shift1, ddims, data, CFL_SIZE);

	// s(theta, r+1)
	pos_shift[PHS1_DIM] = 1;
	md_copy_block(DIMS, pos_shift, ddims_shift, shift2, ddims, data, CFL_SIZE);

	// Iterate through spokes solving Eq. 3

#pragma omp parallel for
	for (int spoke = 0; spoke < ddims[PHS2_DIM]; spoke++) {

		long pos[DIMS] = { [0 ... DIMS - 1] = 0 };
		pos[PHS2_DIM] = spoke;

		long tmp_dims[DIMS];
		md_select_dims(DIMS, ~PHS2_FLAG, tmp_dims, ddims_shift);
		complex float* tmp = md_alloc(DIMS, tmp_dims, CFL_SIZE);

		long pinv_dims[DIMS];
		md_transpose_dims(DIMS, PHS1_DIM, COIL_DIM, pinv_dims, tmp_dims);
		complex float* pinv = md_alloc(DIMS, pinv_dims, CFL_SIZE);

		md_copy_block(DIMS, pos, tmp_dims, tmp, ddims_shift, shift1, CFL_SIZE);

		// pseudo inverse: pinv(s(theta, r))
		mat_pinv_svd(tmp_dims[COIL_DIM], tmp_dims[PHS1_DIM],
			MD_CAST_ARRAY2(complex float, DIMS, pinv_dims, pinv, PHS1_DIM, COIL_DIM),
			MD_CAST_ARRAY2(complex float, DIMS, tmp_dims, tmp, PHS1_DIM, COIL_DIM));

		md_copy_block(DIMS, pos, tmp_dims, tmp, ddims_shift, shift2, CFL_SIZE);

		long Gtheta_dims[DIMS];
		md_select_dims(DIMS, COIL_FLAG, Gtheta_dims, ddims);
		Gtheta_dims[MAPS_DIM] = ddims[COIL_DIM];

		complex float* Gtheta = md_alloc(DIMS, Gtheta_dims, CFL_SIZE);
		complex float* Gtheta_tmp = md_alloc(DIMS, Gtheta_dims, CFL_SIZE);

		// s(theta, r+1) * pinv(s(theta, r))
		mat_mul(Gtheta_dims[MAPS_DIM], tmp_dims[PHS1_DIM], Gtheta_dims[COIL_DIM],
			MD_CAST_ARRAY2(complex float, DIMS, Gtheta_dims, Gtheta_tmp, COIL_DIM, MAPS_DIM),
			MD_CAST_ARRAY2(const complex float, DIMS, tmp_dims, tmp, PHS1_DIM, COIL_DIM),
			MD_CAST_ARRAY2(const complex float, DIMS, pinv_dims, pinv, PHS1_DIM, COIL_DIM));

		// Matrix logarithm!
		mat_logm(Gtheta_dims[MAPS_DIM],
			MD_CAST_ARRAY2(complex float, DIMS, Gtheta_dims, Gtheta, COIL_DIM, MAPS_DIM),
			MD_CAST_ARRAY2(complex float, DIMS, Gtheta_dims, Gtheta_tmp, COIL_DIM, MAPS_DIM));

		md_copy_block(DIMS, pos, vtheta_dims, vtheta, Gtheta_dims, Gtheta, CFL_SIZE);

		md_free(tmp);
		md_free(pinv);
		md_free(Gtheta);
		md_free(Gtheta_tmp);
	}

	md_free(shift1);
	md_free(shift2);
}

// Estimate pseudo inverse of distance matrix
static void get_pseudo_dist(int D, const long pinv_dims[D], complex float* pinv, const long tdims[D], const complex float* traj)
{
	// Estimate distance matrix nm
	// Assumption: RADIAL trajectories <- only single sample in PHS1_DIM is chosen

	long nmdims[DIMS];
	md_select_dims(DIMS, READ_FLAG|PHS2_FLAG, nmdims, tdims);

	long pos_sample[DIMS] = { [0 ... DIMS - 1] = 0 };

	complex float* sample = md_alloc(DIMS, nmdims, CFL_SIZE);

	md_copy_block(DIMS, pos_sample, nmdims, sample, tdims, traj, CFL_SIZE);

	pos_sample[PHS1_DIM] = 1;

	complex float* nm = md_alloc(DIMS, nmdims, CFL_SIZE);

	md_copy_block(DIMS, pos_sample, nmdims, nm, tdims, traj, CFL_SIZE);

	md_zaxpy(DIMS, nmdims, nm, -1., sample);

	md_free(sample);

	// Allocation on stack for more than 1 MB is not be possible
	assert(1. > 8. * (float)(pinv_dims[READ_DIM] * pinv_dims[READ_DIM]) / 1.e6);

	// Estimate pseudo inverse of nm: pinv(s(theta, r))
	mat_pinv_svd(nmdims[PHS2_DIM], nmdims[READ_DIM],
		MD_CAST_ARRAY2(complex float, DIMS, pinv_dims, pinv, READ_DIM, PHS2_DIM),
		MD_CAST_ARRAY2(complex float, DIMS, nmdims, nm, READ_DIM, PHS2_DIM));

	md_free(nm);
}


// Calculate lnG = log(GROG operator along axis) following Eq. 8
static void estimate_lnG(int D, const long lnG_dims[D], complex float* lnG, const long vtheta_dims[D], const complex float* vtheta, const long pinv_dims[D], const complex float* pinv)
{
	// transpose dims for tenmul operation
	long pinv_dimsT[DIMS];
	md_transpose_dims(DIMS, READ_DIM, PHS2_DIM, pinv_dimsT, pinv_dims);

	complex float* pinvT = md_alloc(DIMS, pinv_dimsT, CFL_SIZE);
	md_transpose(DIMS, READ_DIM, PHS2_DIM, pinv_dimsT, pinvT, pinv_dims, pinv, CFL_SIZE);

#pragma omp parallel for collapse(2)
	for (int i = 0; i < lnG_dims[COIL_DIM]; i++) {
		for (int j = 0; j < lnG_dims[MAPS_DIM]; j++) {

			long pos[DIMS] = { [0 ... DIMS - 1] = 0 };
			pos[COIL_DIM] = i;
			pos[MAPS_DIM] = j;

			// Slice coil of vtheta
			long tmp_dims[DIMS];
			md_select_dims(DIMS, ~(COIL_FLAG|MAPS_FLAG), tmp_dims, vtheta_dims);

			complex float* tmp = md_alloc(DIMS, tmp_dims, CFL_SIZE);

			md_copy_block(DIMS, pos, tmp_dims, tmp, vtheta_dims, vtheta, CFL_SIZE);

			long tmp2_dims[DIMS];
			md_select_dims(DIMS, READ_FLAG, tmp2_dims, pinv_dimsT); //pinv_dimsT[READ_DIM] == tdim[READ_DIM]

			complex float* tmp2 = md_alloc(DIMS, tmp2_dims, CFL_SIZE);

			// pinv(s(theta, r)) * vtheta
			md_ztenmul(DIMS, tmp2_dims, tmp2, pinv_dimsT, pinvT, tmp_dims, tmp);

			md_copy_block(DIMS, pos, lnG_dims, lnG, tmp2_dims, tmp2, CFL_SIZE);

			md_free(tmp);
			md_free(tmp2);
		}
	}

	md_free(pinvT);
}


// Calibration
// Idea: Shifting a sample along a spoke with GROG operator G_theta to its neighbour.
// Due to radial trajectories this can be exploited to learn the
// operators acting on the individual axes: G_x, G_y, and G_z
void grog_calib(int D, const long lnG_dims[D], complex float* lnG, const long tdims[D], const complex float* traj, const long ddims[D], const complex float* data)
{
	debug_printf(DP_DEBUG2, "tdims:\t");
	debug_print_dims(DP_DEBUG2, DIMS, tdims);

	debug_printf(DP_DEBUG2, "ddims:\t");
	debug_print_dims(DP_DEBUG2, DIMS, ddims);

	// STEP 1. Calculate log(GROG operator along spokes) = vtheta

	long vtheta_dims[DIMS];
	md_select_dims(DIMS, ~PHS1_FLAG, vtheta_dims, ddims);
	vtheta_dims[MAPS_DIM] = ddims[COIL_DIM]; // Number of coils

	complex float* vtheta = md_alloc(DIMS, vtheta_dims, CFL_SIZE);

	estimate_vtheta(DIMS, vtheta_dims, vtheta, ddims, data);

	// STEP 2. Find pseudo inverse of distance matrix nm
	// Here, the distance matrix is reused for all samples along
	// a spoke. This takes the assumption of a RADIAL readout.

	long tmp_dims[DIMS];
	md_select_dims(DIMS, READ_FLAG|PHS2_FLAG, tmp_dims, tdims);

	long pinv_dims[DIMS];
	md_transpose_dims(DIMS, READ_DIM, PHS2_DIM, pinv_dims, tmp_dims);

	complex float* pinv = md_alloc(DIMS, pinv_dims, CFL_SIZE);

	get_pseudo_dist(DIMS, pinv_dims, pinv, tdims, traj);

	// STEP 3. Calculate lnG = log(GROG operator along axis) following Eq. 8

	estimate_lnG(DIMS, lnG_dims, lnG, vtheta_dims, vtheta, pinv_dims, pinv);

	md_free(vtheta);
	md_free(pinv);

	debug_printf(DP_DEBUG2, "Finished GROG calibration.\n");
}

static void estimate_Gshift(int D, int axis, const long lnG_dims[D], complex float* G_shift, complex float* lnG_axis, complex float* lnG, float shift)
{
	long pos[DIMS] = { [0 ... DIMS - 1] = 0 };
	pos[READ_DIM] = axis;

	long single_lnG_dims[DIMS];
	md_select_dims(DIMS, ~READ_FLAG, single_lnG_dims, lnG_dims);

	md_clear(DIMS, single_lnG_dims, lnG_axis, CFL_SIZE);
	md_copy_block(DIMS, pos, single_lnG_dims, lnG_axis, lnG_dims, lnG, CFL_SIZE);

	md_zsmul(DIMS, single_lnG_dims, lnG_axis, lnG_axis, shift);

	// Matrix exponential to find operator G from ln(G)
	md_clear(DIMS, single_lnG_dims, G_shift, CFL_SIZE);

	zmat_exp(single_lnG_dims[COIL_DIM], 1.,
		MD_CAST_ARRAY2(complex float, DIMS, single_lnG_dims, G_shift, COIL_DIM, MAPS_DIM),
		MD_CAST_ARRAY2(const complex float, DIMS, single_lnG_dims, lnG_axis, COIL_DIM, MAPS_DIM));
}


// Gridding, following Eq. 2
void grog_grid(int D, const long tdims[D], complex float* traj_grid, const complex float* traj, const long ddims[D], complex float* data_grid, const complex float* data, const long lnG_dims[D], complex float* lnG)
{
	debug_printf(DP_DEBUG2, "tdims:\t");
	debug_print_dims(DP_DEBUG2, DIMS, tdims);

	debug_printf(DP_DEBUG2, "ddims:\t");
	debug_print_dims(DP_DEBUG2, DIMS, ddims);

	debug_printf(DP_DEBUG2, "lnG_dims:\t");
	debug_print_dims(DP_DEBUG2, DIMS, lnG_dims);

	long tstrs[DIMS];
	md_calc_strides(DIMS, tstrs, tdims, CFL_SIZE);

	long lnG_strs[DIMS];
	md_calc_strides(DIMS, lnG_strs, lnG_dims, CFL_SIZE);

#pragma omp parallel for collapse(2)
	for (int s = 0; s < ddims[PHS2_DIM]; s++) {		// Spoke
		for (int r = 0; r < ddims[PHS1_DIM]; r++) {	// Readout sample

			long pos_dataframe[DIMS] = { [0 ... DIMS - 1] = 0 };
			pos_dataframe[PHS1_DIM] = r;
			pos_dataframe[PHS2_DIM] = s;

			// Allocate data of data HERE avoids even more allocations within the most inner dimension-loop

			long tmp_data_dims[DIMS];
			md_select_dims(DIMS, ~(PHS1_FLAG|PHS2_FLAG), tmp_data_dims, ddims);

			long tmp_data_dimsT[DIMS];
			md_transpose_dims(DIMS, COIL_DIM, MAPS_DIM, tmp_data_dimsT, tmp_data_dims);

			long tmp_traj_dims[DIMS];
			md_select_dims(DIMS, ~(READ_FLAG|PHS1_FLAG|PHS2_FLAG), tmp_traj_dims, tdims);

			long single_lnG_dims[DIMS];
			md_select_dims(DIMS, ~READ_FLAG, single_lnG_dims, lnG_dims);

			complex float* tmp_traj = md_alloc(DIMS, tmp_traj_dims, CFL_SIZE); // Storage for coordinate in trajectory (Required for multi-dim support)

			complex float* lnG_axis = md_alloc(DIMS, single_lnG_dims, CFL_SIZE); //GROG operator
			complex float* G_shift = md_alloc(DIMS, single_lnG_dims, CFL_SIZE);

			complex float* tmp_data = md_alloc(DIMS, tmp_data_dims, CFL_SIZE);
			complex float* tmp_data2 = md_alloc(DIMS, tmp_data_dims, CFL_SIZE);
			complex float* tmp_dataT = md_alloc(DIMS, tmp_data_dimsT, CFL_SIZE); //for tenmul operation

			md_clear(DIMS, tmp_data_dims, tmp_data, CFL_SIZE);
			md_copy_block(DIMS, pos_dataframe, tmp_data_dims, tmp_data, ddims, data, CFL_SIZE);

			// Iterate through different dimensions and apply operator to s(kx, ky, kz)
			// Theoretically, the order should not matter, but practically it does
			// due to noise,....
			// Nevertheless, order allows you to change the order of the applied operators
			int order[3] = { 0, 1, 2 };

			long pos[DIMS] = { [0 ... DIMS - 1] = 0 };
			md_copy_dims(DIMS, pos, pos_dataframe);

			for (int dd = 0; dd < tdims[READ_DIM]; dd++) { // dimension

				int d = order[dd];
				pos[READ_DIM] = d;

				long ind_dataframe = md_calc_offset(DIMS, tstrs, pos) / (long)CFL_SIZE;

				float coord = crealf(traj[ind_dataframe]);
				float coord_r = round(coord);
				float shift = coord_r - coord;

				md_zfill(DIMS, tmp_traj_dims, tmp_traj, coord_r);
				md_copy_block(DIMS, pos, tdims, traj_grid, tmp_traj_dims, tmp_traj, CFL_SIZE);

				// Find shift operator for the specific sampling point (d, r, s)
				estimate_Gshift(DIMS, d, lnG_dims, G_shift, lnG_axis, lnG, shift);

				// Transform data with calculated shift operator
				md_clear(DIMS, tmp_data_dims, tmp_data2, CFL_SIZE);
				md_transpose(DIMS, COIL_DIM, MAPS_DIM, tmp_data_dimsT, tmp_dataT, tmp_data_dims, tmp_data, CFL_SIZE);
				md_ztenmul(DIMS, tmp_data_dims, tmp_data2, single_lnG_dims, G_shift, tmp_data_dimsT, tmp_dataT);

				md_copy(DIMS, tmp_data_dims, tmp_data, tmp_data2, CFL_SIZE);
			}

			md_copy_block(DIMS, pos_dataframe, ddims, data_grid, tmp_data_dims, tmp_data, CFL_SIZE);

			md_free(lnG_axis);
			md_free(G_shift);
			md_free(tmp_traj);
			md_free(tmp_data);
			md_free(tmp_data2);
			md_free(tmp_dataT);
		}
	}

	debug_printf(DP_DEBUG2, "Finished GROG gridding.\n");
}

