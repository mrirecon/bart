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
	long ddims_shift[D];
	md_copy_dims(D, ddims_shift, ddims);
	ddims_shift[PHS1_DIM] -= 1;

	long pos_shift[D];
	for (int i = 0; i < D; i++)
		pos_shift[i] = 0;

	complex float* shift1 = md_alloc(D, ddims_shift, CFL_SIZE);
	complex float* shift2 = md_alloc(D, ddims_shift, CFL_SIZE);

	// s(theta, r)
	md_copy_block(D, pos_shift, ddims_shift, shift1, ddims, data, CFL_SIZE);

	// s(theta, r+1)
	pos_shift[PHS1_DIM] = 1;
	md_copy_block(D, pos_shift, ddims_shift, shift2, ddims, data, CFL_SIZE);

	// Iterate through spokes solving Eq. 3

#pragma omp parallel for
	for (int spoke = 0; spoke < ddims[PHS2_DIM]; spoke++) {

		long pos[D];
		for (int i = 0; i < D; i++)
			pos[i] = 0;

		pos[PHS2_DIM] = spoke;

		long tmp_dims[D];
		md_select_dims(D, ~PHS2_FLAG, tmp_dims, ddims_shift);
		complex float* tmp = md_alloc(D, tmp_dims, CFL_SIZE);

		long pinv_dims[D];
		md_transpose_dims(D, PHS1_DIM, COIL_DIM, pinv_dims, tmp_dims);
		complex float* pinv = md_alloc(D, pinv_dims, CFL_SIZE);

		md_copy_block(D, pos, tmp_dims, tmp, ddims_shift, shift1, CFL_SIZE);

		// pseudo inverse: pinv(s(theta, r))
		mat_pinv_svd(tmp_dims[COIL_DIM], tmp_dims[PHS1_DIM],
			MD_CAST_ARRAY2(complex float, D, pinv_dims, pinv, PHS1_DIM, COIL_DIM),
			MD_CAST_ARRAY2(complex float, D, tmp_dims, tmp, PHS1_DIM, COIL_DIM));

		md_copy_block(D, pos, tmp_dims, tmp, ddims_shift, shift2, CFL_SIZE);

		long Gtheta_dims[D];
		md_select_dims(D, COIL_FLAG, Gtheta_dims, ddims);
		Gtheta_dims[MAPS_DIM] = ddims[COIL_DIM];

		complex float* Gtheta = md_alloc(D, Gtheta_dims, CFL_SIZE);
		complex float* Gtheta_tmp = md_alloc(D, Gtheta_dims, CFL_SIZE);

		// s(theta, r+1) * pinv(s(theta, r))
		mat_mul(Gtheta_dims[MAPS_DIM], tmp_dims[PHS1_DIM], Gtheta_dims[COIL_DIM],
			MD_CAST_ARRAY2(complex float, D, Gtheta_dims, Gtheta_tmp, COIL_DIM, MAPS_DIM),
			MD_CAST_ARRAY2(const complex float, D, tmp_dims, tmp, PHS1_DIM, COIL_DIM),
			MD_CAST_ARRAY2(const complex float, D, pinv_dims, pinv, PHS1_DIM, COIL_DIM));

		// Matrix logarithm!
		mat_logm(Gtheta_dims[MAPS_DIM],
			MD_CAST_ARRAY2(complex float, D, Gtheta_dims, Gtheta, COIL_DIM, MAPS_DIM),
			MD_CAST_ARRAY2(complex float, D, Gtheta_dims, Gtheta_tmp, COIL_DIM, MAPS_DIM));

		md_copy_block(D, pos, vtheta_dims, vtheta, Gtheta_dims, Gtheta, CFL_SIZE);

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

	long nmdims[D];
	md_select_dims(D, READ_FLAG|PHS2_FLAG, nmdims, tdims);

	long pos_sample[D];
	for (int i = 0; i < D; i++)
		pos_sample[i] = 0;

	complex float* sample = md_alloc(D, nmdims, CFL_SIZE);

	md_copy_block(D, pos_sample, nmdims, sample, tdims, traj, CFL_SIZE);

	pos_sample[PHS1_DIM] = 1;

	complex float* nm = md_alloc(D, nmdims, CFL_SIZE);

	md_copy_block(D, pos_sample, nmdims, nm, tdims, traj, CFL_SIZE);

	md_zaxpy(D, nmdims, nm, -1., sample);

	md_free(sample);

	// Estimate pseudo inverse of nm: pinv(s(theta, r))
	mat_pinv_svd(nmdims[PHS2_DIM], nmdims[READ_DIM],
		MD_CAST_ARRAY2(complex float, D, pinv_dims, pinv, READ_DIM, PHS2_DIM),
		MD_CAST_ARRAY2(complex float, D, nmdims, nm, READ_DIM, PHS2_DIM));

	md_free(nm);
}


// Calculate lnG = log(GROG operator along axis) following Eq. 8
static void estimate_lnG(int D, const long lnG_dims[D], complex float* lnG, const long vtheta_dims[D], const complex float* vtheta, const long pinv_dims[D], const complex float* pinv)
{
	// transpose dims for tenmul operation
	long pinv_dimsT[D];
	md_transpose_dims(D, READ_DIM, PHS2_DIM, pinv_dimsT, pinv_dims);

	complex float* pinvT = md_alloc(D, pinv_dimsT, CFL_SIZE);
	md_transpose(D, READ_DIM, PHS2_DIM, pinv_dimsT, pinvT, pinv_dims, pinv, CFL_SIZE);

#pragma omp parallel for collapse(2)
	for (int i = 0; i < lnG_dims[COIL_DIM]; i++) {
		for (int j = 0; j < lnG_dims[MAPS_DIM]; j++) {

			long pos[D];
			for (int i = 0; i < D; i++)
				pos[i] = 0;

			pos[COIL_DIM] = i;
			pos[MAPS_DIM] = j;

			// Slice coil of vtheta
			long tmp_dims[D];
			md_select_dims(D, ~(COIL_FLAG|MAPS_FLAG), tmp_dims, vtheta_dims);

			complex float* tmp = md_alloc(D, tmp_dims, CFL_SIZE);

			md_copy_block(D, pos, tmp_dims, tmp, vtheta_dims, vtheta, CFL_SIZE);

			long tmp2_dims[D];
			md_select_dims(D, READ_FLAG, tmp2_dims, pinv_dimsT); //pinv_dimsT[READ_DIM] == tdim[READ_DIM]

			complex float* tmp2 = md_alloc(D, tmp2_dims, CFL_SIZE);

			// pinv(s(theta, r)) * vtheta
			md_ztenmul(D, tmp2_dims, tmp2, pinv_dimsT, pinvT, tmp_dims, tmp);

			md_copy_block(D, pos, lnG_dims, lnG, tmp2_dims, tmp2, CFL_SIZE);

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
	debug_print_dims(DP_DEBUG2, D, tdims);

	debug_printf(DP_DEBUG2, "ddims:\t");
	debug_print_dims(DP_DEBUG2, D, ddims);

	// STEP 1. Calculate log(GROG operator along spokes) = vtheta

	long vtheta_dims[D];
	md_select_dims(D, ~PHS1_FLAG, vtheta_dims, ddims);
	vtheta_dims[MAPS_DIM] = ddims[COIL_DIM]; // Number of coils

	complex float* vtheta = md_alloc(D, vtheta_dims, CFL_SIZE);

	estimate_vtheta(D, vtheta_dims, vtheta, ddims, data);

	// STEP 2. Find pseudo inverse of distance matrix nm
	// Here, the distance matrix is reused for all samples along
	// a spoke. This takes the assumption of a RADIAL readout.

	long tmp_dims[D];
	md_select_dims(D, READ_FLAG|PHS2_FLAG, tmp_dims, tdims);

	long pinv_dims[D];
	md_transpose_dims(D, READ_DIM, PHS2_DIM, pinv_dims, tmp_dims);

	complex float* pinv = md_alloc(D, pinv_dims, CFL_SIZE);

	get_pseudo_dist(D, pinv_dims, pinv, tdims, traj);

	// STEP 3. Calculate lnG = log(GROG operator along axis) following Eq. 8

	estimate_lnG(D, lnG_dims, lnG, vtheta_dims, vtheta, pinv_dims, pinv);

	md_free(vtheta);
	md_free(pinv);

	debug_printf(DP_DEBUG2, "Finished GROG calibration.\n");
}


static void apply_Gshift(int D, const long dims[D], complex float* data,
		complex float* lnG_axis, float shift[3])
{
	int C = dims[COIL_DIM];

	assert(1 == dims[READ_DIM]);
	assert(1 == dims[MAPS_DIM]);

	long single_lnG_dims[D];
	md_singleton_dims(D, single_lnG_dims);

	single_lnG_dims[COIL_DIM] = C;
	single_lnG_dims[MAPS_DIM] = C;

	long dimsT[D];
	md_transpose_dims(D, COIL_DIM, MAPS_DIM, dimsT, dims);

	complex float* tmp = md_alloc(D, dims, CFL_SIZE); //for tenmul operation

	// Iterate through different dimensions and apply operator to s(kx, ky, kz)
	// Theoretically, the order does matter but, well, GROG

	for (int d = 0; d < 3; d++) { // dimension

		// Find shift operator for the specific sampling point (d, r, s)

		complex float lnG[C][C];
		for (int i = 0; i < C; i++)
			for (int j = 0; j < C; j++)
				lnG[i][j] = lnG_axis[(i * C + j) * 3 + d] * shift[d];

		// Matrix exponential to find operator G from ln(G)

		complex float G_shift[C][C];
		zmat_exp(C, 1., G_shift, lnG);

		// Transform data with calculated shift operator

		md_transpose(D, COIL_DIM, MAPS_DIM, dimsT, tmp, dims, data, CFL_SIZE);
		md_ztenmul(D, dims, data, single_lnG_dims, &G_shift[0][0], dimsT, tmp);
	}

	md_free(tmp);
}




// Gridding, following Eq. 2
void grog_grid(int D, const long tdims[D], complex float* traj_grid, const complex float* traj, const long ddims[D], complex float* data_grid, const complex float* data, const long lnG_dims[D], complex float* lnG)
{
	assert(3 == tdims[READ_DIM]);
	assert(1 == ddims[READ_DIM]);
	assert(1 == ddims[MAPS_DIM]);

	int C = ddims[COIL_DIM];
	assert(3 == lnG_dims[READ_DIM]);
	assert(C == lnG_dims[COIL_DIM]);
	assert(C == lnG_dims[MAPS_DIM]);
	assert(3L * C * C == md_calc_size(D, lnG_dims));

	long tstrs[D];
	md_calc_strides(D, tstrs, tdims, CFL_SIZE);

	md_zround(D, tdims, traj_grid, traj);

#pragma omp parallel for collapse(2)
	for (int s = 0; s < ddims[PHS2_DIM]; s++) {		// Spoke
		for (int r = 0; r < ddims[PHS1_DIM]; r++) {	// Readout sample

			long pos[D];
			for (int i = 0; i < D; i++)
				pos[i] = 0;

			pos[PHS1_DIM] = r;
			pos[PHS2_DIM] = s;

			// Allocate data of data HERE avoids even more allocations within the most inner dimension-loop
			long tmp_data_dims[D];
			md_select_dims(D, ~(PHS1_FLAG|PHS2_FLAG), tmp_data_dims, ddims);

			complex float* tmp_data = md_alloc(D, tmp_data_dims, CFL_SIZE);

			md_clear(D, tmp_data_dims, tmp_data, CFL_SIZE);
			md_copy_block(D, pos, tmp_data_dims, tmp_data, ddims, data, CFL_SIZE);

			float shift[3];

			for (int d = 0; d < 3; d++) { // dimension

				pos[READ_DIM] = d;

				long ind_dataframe = md_calc_offset(D, tstrs, pos) / (long)CFL_SIZE;
				shift[d] = crealf(traj_grid[ind_dataframe] - traj[ind_dataframe]);
			}

			apply_Gshift(D, tmp_data_dims, tmp_data, lnG, shift);

			pos[READ_DIM] = 0;
			md_copy_block(D, pos, ddims, data_grid, tmp_data_dims, tmp_data, CFL_SIZE);

			md_free(tmp_data);
		}
	}

	debug_printf(DP_DEBUG2, "Finished GROG gridding.\n");
}

