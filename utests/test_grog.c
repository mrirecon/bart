
#include <complex.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/matexp.h"

#include "misc/mri.h"

#include "simu/phantom.h"
#include "simu/sens.c"

#include "noncart/traj.h"

#include "calib/grog.h"

#include "utest.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


static bool test_grog(void)
{
	enum { N = 16 };

	long X = 15; // Samples
	long Y = 6; // Spokes
	long C = 4; // Coils
	int OV = 2;

	// oversampling
	X *= OV;

	long tdims[DIMS] = { 3, X, Y, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long ddims[DIMS] = { 1, X, Y, C, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long lnG_dims[DIMS] = { 3, 1, 1, C, C, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	// 1. Generate test samples on a radial non-Cartesian trajectory

	complex float* traj = md_alloc(DIMS, tdims, CFL_SIZE);

	struct traj_conf conf = traj_defaults;
	conf.radial = true;
	conf.golden = true;
	conf.tiny_gold = 1;

	double base_angle[DIMS] = { 0. };
	calc_base_angles(base_angle, 5, 1, conf);

	int p = 0;
	long pos[DIMS] = { 0 };

	do {
		int i = pos[PHS1_DIM];
		int sample = i;
		double read = (float)sample + 0.5 - (float)X / 2.;
		double angle = 0.;
		double angle2 = 0.;

		for (int d = 1; d < DIMS; d++)
			angle += pos[d] * base_angle[d];

		float read_dir[3];
		traj_read_dir(read_dir, angle, angle2);

		traj[p * 3 + 0] = (read * read_dir[0]) / (float)OV;
		traj[p * 3 + 1] = (read * read_dir[1]) / (float)OV;
		traj[p * 3 + 2] = (read * read_dir[2]) / (float)OV;

		p++;

	} while (md_next(DIMS, tdims, ~1UL, pos));

	// 2. Generate test data

	complex float* data = md_alloc(DIMS, ddims, CFL_SIZE);

	struct coil_opts copts = coil_opts_defaults;
	copts.ctype = HEAD_2D_8CH;

	long tstrs[DIMS] = { 0 };
	md_calc_strides(DIMS, tstrs, tdims, CFL_SIZE);

	calc_phantom(ddims, data, false, true, tstrs, traj, &copts);


	// 3. Run GROG kernel calibration

	complex float* lnG = md_alloc(DIMS, lnG_dims, CFL_SIZE);

	grog_calib(DIMS, lnG_dims, lnG, tdims, traj, ddims, data);

	// 4. Test calibration accuracy by shifting sample to its neighbor along spoke and compare both

	long shift_dims[DIMS];
	md_copy_dims(DIMS, shift_dims, ddims);
	shift_dims[PHS1_DIM] -= 1;

	complex float* shift = md_alloc(DIMS, shift_dims, CFL_SIZE);
	complex float* ref = md_alloc(DIMS, shift_dims, CFL_SIZE);

	long pos_shift[DIMS] = { [0 ... DIMS - 1] = 0 };
	md_copy_block(DIMS, pos_shift, shift_dims, shift, ddims, data, CFL_SIZE);

	pos_shift[PHS1_DIM] = 1;
	md_copy_block(DIMS, pos_shift, shift_dims, ref, ddims, data, CFL_SIZE);

	long lnG_strs[DIMS] = { 0 };
	md_calc_strides(DIMS, lnG_strs, lnG_dims, CFL_SIZE);

	complex float* diff = md_alloc(DIMS, tdims, CFL_SIZE);

	// #pragma omp parallel for collapse(2)
	for (int s = 0; s < shift_dims[PHS2_DIM]; s++)
		for (int r = 0; r < shift_dims[PHS1_DIM]; r++) {

			// Shift sample at position r to position r+1

			long pos_dataframe[DIMS] = { [0 ... DIMS - 1] = 0 };
			pos_dataframe[PHS1_DIM] = r;
			pos_dataframe[PHS2_DIM] = s;

			long tmp_data_dims[DIMS];
			md_select_dims(DIMS, ~(PHS1_FLAG|PHS2_FLAG), tmp_data_dims, shift_dims);

			complex float* tmp_data = md_alloc(DIMS, tmp_data_dims, CFL_SIZE);
			complex float* tmp_data2 = md_alloc(DIMS, tmp_data_dims, CFL_SIZE);

			md_copy_block(DIMS, pos_dataframe, tmp_data_dims, tmp_data, shift_dims, shift, CFL_SIZE);

			// Transposed data storage for tenmul operation
			long tmp_data_dimsT[DIMS];
			md_transpose_dims(DIMS, COIL_DIM, MAPS_DIM, tmp_data_dimsT, tmp_data_dims);
			complex float* tmp_dataT = md_alloc(DIMS, tmp_data_dimsT, CFL_SIZE);

			long pos[DIMS] = { [0 ... DIMS - 1] = 0 };
			md_copy_dims(DIMS, pos, pos_dataframe);

			long tmp_op_dims[DIMS];
			md_select_dims(DIMS, ~READ_FLAG, tmp_op_dims, lnG_dims);

			complex float* tmp_op = md_alloc(DIMS, tmp_op_dims, CFL_SIZE);
			complex float* tmp_op2 = md_alloc(DIMS, tmp_op_dims, CFL_SIZE);

			int order[3] = { 0, 1, 2 };

			for (int d0 = 0; d0 < tdims[READ_DIM]; d0++) {

				int d = order[d0];

				pos[READ_DIM] = d;
				pos[PHS1_DIM] = r;
				long ind_sample = md_calc_offset(DIMS, tstrs, pos) / (long)CFL_SIZE;

				complex float coord = traj[ind_sample];

				pos[PHS1_DIM]++;
				long ind_sample2 = md_calc_offset(DIMS, tstrs, pos) / (long)CFL_SIZE;

				complex float coord_r = traj[ind_sample2];

				complex float nm = coord_r - coord;

				diff[ind_sample] = nm;

				// Calculate nm operator for sampling point
				long pos_op[DIMS] = { [0 ... DIMS - 1] = 0 };
				pos_op[READ_DIM] = d;

				md_clear(DIMS, tmp_op_dims, tmp_op, CFL_SIZE);
				md_copy_block(DIMS, pos_op, tmp_op_dims, tmp_op, lnG_dims, lnG, CFL_SIZE);

				md_zsmul(DIMS, tmp_op_dims, tmp_op, tmp_op, nm);

				md_clear(DIMS, tmp_op_dims, tmp_op2, CFL_SIZE);
				zmat_exp(tmp_op_dims[COIL_DIM], 1.,
					MD_CAST_ARRAY2(complex float, DIMS, tmp_op_dims, tmp_op2, COIL_DIM, MAPS_DIM),
					MD_CAST_ARRAY2(const complex float, DIMS, tmp_op_dims, tmp_op, COIL_DIM, MAPS_DIM));


				// Transform data with calculated nm operator
				md_clear(DIMS, tmp_data_dims, tmp_data2, CFL_SIZE);
				md_transpose(DIMS, COIL_DIM, MAPS_DIM, tmp_data_dimsT, tmp_dataT, tmp_data_dims, tmp_data, CFL_SIZE);
				md_ztenmul(DIMS, tmp_data_dims, tmp_data2, tmp_op_dims, tmp_op2, tmp_data_dimsT, tmp_dataT);

				md_copy(DIMS, tmp_data_dims, tmp_data, tmp_data2, CFL_SIZE);
			}

			md_copy_block(DIMS, pos_dataframe, shift_dims, shift, tmp_data_dims, tmp_data, CFL_SIZE);

			md_free(tmp_op);
			md_free(tmp_op2);
			md_free(tmp_data);
			md_free(tmp_data2);
			md_free(tmp_dataT);
		}

	float err = md_znrmse(DIMS, shift_dims, ref, shift);

	// debug_printf(DP_INFO, "error:\t%f\n", err);

	md_free(traj);
	md_free(data);
	md_free(lnG);
	md_free(shift);
	md_free(ref);
	md_free(diff);

	UT_RETURN_ASSERT(0.3 > err);
}

UT_REGISTER_TEST(test_grog);

