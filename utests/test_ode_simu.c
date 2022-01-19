/* Copyright 2022. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 *	Nick Scholand
 */

#include <math.h>
#include <stdio.h>

#include "simu/bloch.h"
#include "simu/pulse.h"
#include "simu/simulation.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "misc/mri.h"
#include "misc/misc.h"

#include "utest.h"

// Validate the partial derivatives estimated with the SA with
// the difference quotient method for estimating gradients
static bool test_ode_bloch_simulation_gradients(void)
{
	float e = 1.E-3;
	float tol = 1.E-4;


	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = IRBSSFP;
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.0015;
	sim_data.seq.rep_num = 500;
	sim_data.seq.spin_num = 1;
        sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = sim_data.seq.te;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 1. / WATER_T1;
	sim_data.voxel.r2 = 1. / WATER_T2;
	sim_data.voxel.m0 = 1.;
	sim_data.voxel.w = 0;
	sim_data.voxel.b1 = 1.;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = 45.;
	sim_data.pulse.rf_end = 0.0009;

	sim_data.grad = simdata_grad_defaults;
	sim_data.tmp = simdata_tmp_defaults;

	float mxy_ref_sig[sim_data.seq.rep_num][3];
	float sa_r1_ref_sig[sim_data.seq.rep_num][3];
	float sa_r2_ref_sig[sim_data.seq.rep_num][3];
	float sa_m0_ref_sig[sim_data.seq.rep_num][3];
	float sa_b1_ref_sig[sim_data.seq.rep_num][3];

	bloch_simulation(&sim_data, mxy_ref_sig, sa_r1_ref_sig, sa_r2_ref_sig, sa_m0_ref_sig, sa_b1_ref_sig);

	/* ------------ R1 Partial Derivative Test -------------- */

	float mxy_tmp_sig[sim_data.seq.rep_num][3];
	float sa_r1_tmp_sig[sim_data.seq.rep_num][3];
	float sa_r2_tmp_sig[sim_data.seq.rep_num][3];
	float sa_m0_tmp_sig[sim_data.seq.rep_num][3];
	float sa_b1_tmp_sig[sim_data.seq.rep_num][3];

	struct sim_data data_r1 = sim_data;

	data_r1.voxel.r1 += e;

	bloch_simulation(&data_r1, mxy_tmp_sig, sa_r1_tmp_sig, sa_r2_tmp_sig, sa_m0_tmp_sig, sa_b1_tmp_sig);

	float err = 0;

	for (int i = 0; i < sim_data.seq.rep_num; i++)
		for (int j = 0; j < 3; j++) {

			err = fabsf(e * sa_r1_ref_sig[i][j] - (mxy_tmp_sig[i][j] - mxy_ref_sig[i][j]));

			if (tol < err) {

				printf("Error T1: (%d,%d)\t=>\t%f\n", i, j, err);
				return false;
			}
		}

	/* ------------ R2 Partial Derivative Test -------------- */

	struct sim_data data_r2 = sim_data;

	data_r2.voxel.r2 += e;

	bloch_simulation(&data_r2, mxy_tmp_sig, sa_r1_tmp_sig, sa_r2_tmp_sig, sa_m0_tmp_sig, sa_b1_tmp_sig);

	for (int i = 0; i < sim_data.seq.rep_num; i++)
		for (int j = 0; j < 3; j++) {

			err = fabsf(e * sa_r2_ref_sig[i][j] - (mxy_tmp_sig[i][j] - mxy_ref_sig[i][j]));

			if (tol < err) {

				printf("Error T2: (%d,%d)\t=>\t%f\n", i, j, err);
				return false;
			}
		}

	/* ------------ M0 Partial Derivative Test -------------- */

	struct sim_data data_m0 = sim_data;

	data_m0.voxel.m0 += e;

	bloch_simulation(&data_m0, mxy_tmp_sig, sa_r1_tmp_sig, sa_r2_tmp_sig, sa_m0_tmp_sig, sa_b1_tmp_sig);

	for (int i = 0; i < sim_data.seq.rep_num; i++)
		for (int j = 0; j < 3; j++) {

			err = fabsf(e * sa_m0_ref_sig[i][j] - (mxy_tmp_sig[i][j] - mxy_ref_sig[i][j]));

			if (tol < err) {

				printf("Error M0: (%d,%d)\t=>\t%f\n", i, j, err);
				return false;
			}
		}

	/* ------------ B1 Partial Derivative Test -------------- */

	struct sim_data data_b1 = sim_data;

	data_b1.voxel.b1 += e;

	bloch_simulation(&data_b1, mxy_tmp_sig, sa_r1_tmp_sig, sa_r2_tmp_sig, sa_m0_tmp_sig, sa_b1_tmp_sig);

	for (int i = 0; i < sim_data.seq.rep_num; i++)
		for (int j = 0; j < 3; j++) {

			err = fabsf(e * sa_b1_ref_sig[i][j] - (mxy_tmp_sig[i][j] - mxy_ref_sig[i][j]));

			if (tol < err) {

				printf("Error B1: (%d,%d)\t=>\t%f\n", i, j, err);
				return false;
			}
		}

	return true;
}

UT_REGISTER_TEST(test_ode_bloch_simulation_gradients);


/* Compare the simulated IR bSSFP signal with the analytical model
 * Assumptions: 1. TR << T_{1,2}
 *              2. T_RF \approx 0
 *
 * References:
 * Schmitt, P. , Griswold, M. A., Jakob, P. M., Kotas, M. , Gulani, V. , Flentje, M. and Haase, A. (2004),
 * Inversion recovery TrueFISP: Quantification of T1, T2, and spin density.
 * Magn. Reson. Med., 51: 661-667. doi:10.1002/mrm.20058
 *
 * Ehses, P. , Seiberlich, N. , Ma, D. , Breuer, F. A., Jakob, P. M., Griswold, M. A. and Gulani, V. (2013),
 * IR TrueFISP with a golden‐ratio‐based radial readout: Fast quantification of T1, T2, and proton density.
 * Magn Reson Med, 69: 71-81. doi:10.1002/mrm.24225
 */
static bool test_ode_irbssfp_simulation(void)
{
	float angle = 45.;
	float repetition = 100;

	float fa = angle * M_PI / 180.;

	float t1n = WATER_T1;
	float t2n = WATER_T2;
	float m0n = 1.;

	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = IRBSSFP;
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.0015;
	sim_data.seq.rep_num = repetition;
	sim_data.seq.spin_num = 1;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = sim_data.seq.te;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 1./t1n;
	sim_data.voxel.r2 = 1./t2n;
	sim_data.voxel.m0 = m0n;
	sim_data.voxel.w = 0;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = angle;

        // Choose close to hard-pulse approximation -> same assumptions as analytical model
	sim_data.pulse.rf_end = 0.00001;

	sim_data.grad = simdata_grad_defaults;
	sim_data.tmp = simdata_tmp_defaults;

	float mxy_sig[sim_data.seq.rep_num][3];
	float sa_r1_sig[sim_data.seq.rep_num][3];
	float sa_r2_sig[sim_data.seq.rep_num][3];
	float sa_m0_sig[sim_data.seq.rep_num][3];
	float sa_b1_sig[sim_data.seq.rep_num][3];

	bloch_simulation(&sim_data, mxy_sig, sa_r1_sig, sa_r2_sig, sa_m0_sig, sa_b1_sig);


	// Analytical Model
	float t1s = 1 / ((cosf(fa/2.)*cosf(fa/2.))/t1n + (sinf(fa/2.)*sinf(fa/2.))/t2n);
	float s0 = m0n * sinf(fa/2.);
	float stst = m0n * sinf(fa) / ((t1n/t2n + 1) - cosf(fa) * (t1n/t2n-1));
	float inv = 1 + s0 / stst;


        // Model Comparison
	float out_simu = 0.;
	float out_theory = 0.;
	float err = 0.;

	for (int z = 0; z < repetition; z++) {

                //Does NOT include phase information!
                // + data.tr through alpha/2 preparation
		out_theory = fabs(stst * (1. - inv * expf(-((float)(z+1) * sim_data.seq.tr)/t1s)));

		out_simu = cabsf(mxy_sig[z][1] + mxy_sig[z][0]*I);

		err = fabsf(out_simu - out_theory);

		if (10E-4 < err) {

			debug_printf(DP_ERROR, "err: %f,\t out_simu: %f,\t out_theory: %f\n", err, out_simu, out_theory);
			debug_printf(DP_ERROR, "Error in sequence test\n see: -> test_simulation() in test_ode_simu.c\n");

			return 0;
		}
	}
	return 1;
}

UT_REGISTER_TEST(test_ode_irbssfp_simulation);