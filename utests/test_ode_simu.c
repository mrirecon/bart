/* Copyright 2022. TU Graz. Institute of Biomedical Imaging.
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
#include "simu/epg.h"

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
	sim_data.seq.seq_type = SEQ_IRBSSFP;
	sim_data.seq.tr = 0.004;
	sim_data.seq.te = 0.002;
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
	sim_data.pulse.rf_end = 0.001;

	sim_data.grad = simdata_grad_defaults;
	sim_data.tmp = simdata_tmp_defaults;
        sim_data.other = simdata_other_defaults;

	int R = sim_data.seq.rep_num;

	float mxy_ref_sig[R][3];
	float sa_r1_ref_sig[R][3];
	float sa_r2_ref_sig[R][3];
	float sa_m0_ref_sig[R][3];
	float sa_b1_ref_sig[R][3];

	bloch_simulation(&sim_data, R, &mxy_ref_sig, &sa_r1_ref_sig, &sa_r2_ref_sig, &sa_m0_ref_sig, &sa_b1_ref_sig);

	/* ------------ R1 Partial Derivative Test -------------- */

	float mxy_tmp_sig[R][3];
	float sa_r1_tmp_sig[R][3];
	float sa_r2_tmp_sig[R][3];
	float sa_m0_tmp_sig[R][3];
	float sa_b1_tmp_sig[R][3];

	struct sim_data data_r1 = sim_data;

	data_r1.voxel.r1 += e;

	bloch_simulation(&data_r1, R, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig);

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

	bloch_simulation(&data_r2, R, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig);

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

	bloch_simulation(&data_m0, R, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig);

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

	bloch_simulation(&data_b1, R, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig);

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


static bool test_stm_bloch_simulation_gradients(void)
{
	float e = 1.E-3;
	float tol = 1.E-4;


	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
        sim_data.seq.type = SIM_STM;
	sim_data.seq.seq_type = SEQ_IRBSSFP;
	sim_data.seq.tr = 0.004;
	sim_data.seq.te = 0.002;
	sim_data.seq.rep_num = 500;
	sim_data.seq.spin_num = 1;
        sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = sim_data.seq.te;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 1. / WATER_T1;
	sim_data.voxel.r2 = 1. / WATER_T2;
	sim_data.voxel.m0 = 1.;
	sim_data.voxel.w = 0.;
	sim_data.voxel.b1 = 1.;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = 45.;
	sim_data.pulse.rf_end = 0.001;

	sim_data.grad = simdata_grad_defaults;
	sim_data.tmp = simdata_tmp_defaults;
        sim_data.other = simdata_other_defaults;

	int R = sim_data.seq.rep_num;

	float mxy_ref_sig[R][3];
	float sa_r1_ref_sig[R][3];
	float sa_r2_ref_sig[R][3];
	float sa_m0_ref_sig[R][3];
	float sa_b1_ref_sig[R][3];

	bloch_simulation(&sim_data, R, &mxy_ref_sig, &sa_r1_ref_sig, &sa_r2_ref_sig, &sa_m0_ref_sig, &sa_b1_ref_sig);

	/* ------------ R1 Partial Derivative Test -------------- */

	float mxy_tmp_sig[R][3];
	float sa_r1_tmp_sig[R][3];
	float sa_r2_tmp_sig[R][3];
	float sa_m0_tmp_sig[R][3];
	float sa_b1_tmp_sig[R][3];

	struct sim_data data_r1 = sim_data;

	data_r1.voxel.r1 += e;

	bloch_simulation(&data_r1, R, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig);

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

	bloch_simulation(&data_r2, R, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig);

	for (int i = 0; i < R; i++) {

		for (int j = 0; j < 3; j++) {

			err = fabsf(e * sa_r2_ref_sig[i][j] - (mxy_tmp_sig[i][j] - mxy_ref_sig[i][j]));

			if (tol < err) {

				printf("Error T2: (%d,%d)\t=>\t%f\n", i, j, err);
				return false;
			}
		}
	}

	/* ------------ M0 Partial Derivative Test -------------- */

	struct sim_data data_m0 = sim_data;

	data_m0.voxel.m0 += e;

	bloch_simulation(&data_m0, R, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig);

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

	bloch_simulation(&data_b1, R, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig);

	for (int i = 0; i < R; i++) {

		for (int j = 0; j < 3; j++) {

			err = fabsf(e * sa_b1_ref_sig[i][j] - (mxy_tmp_sig[i][j] - mxy_ref_sig[i][j]));

			if (tol < err) {

				printf("Error B1: (%d,%d)\t=>\t%f\n", i, j, err);
				return false;
			}
		}
	}

	return true;
}

UT_REGISTER_TEST(test_stm_bloch_simulation_gradients);


/* ODE Simulation
 * Compare the simulated IR bSSFP signal with the analytical model
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
	sim_data.seq.seq_type = SEQ_IRBSSFP;
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.0015;
	sim_data.seq.rep_num = repetition;
	sim_data.seq.spin_num = 1;
        sim_data.seq.perfect_inversion = true;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = sim_data.seq.te;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 1. / t1n;
	sim_data.voxel.r2 = 1. / t2n;
	sim_data.voxel.m0 = m0n;
	sim_data.voxel.w = 0;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = angle;

        // Choose close to hard-pulse approximation -> same assumptions as analytical model
	sim_data.pulse.rf_end = 0.00001;

	sim_data.grad = simdata_grad_defaults;
	sim_data.tmp = simdata_tmp_defaults;
        sim_data.other = simdata_other_defaults;

	int R = sim_data.seq.rep_num;

	float mxy_sig[R][3];
	float sa_r1_sig[R][3];
	float sa_r2_sig[R][3];
	float sa_m0_sig[R][3];
	float sa_b1_sig[R][3];

	bloch_simulation(&sim_data, R, &mxy_sig, &sa_r1_sig, &sa_r2_sig, &sa_m0_sig, &sa_b1_sig);


	// Analytical Model
	float t1s = 1. / ((cosf(fa / 2.) * cosf(fa / 2.)) / t1n + (sinf(fa / 2.) * sinf(fa / 2.)) / t2n);
	float s0 = m0n * sinf(fa / 2.);
	float stst = m0n * sinf(fa) / ((t1n / t2n + 1.) - cosf(fa) * (t1n / t2n - 1.));
	float inv = 1. + s0 / stst;


        // Model Comparison
	float out_simu = 0.;
	float out_theory = 0.;
	float err = 0.;

	for (int z = 0; z < repetition; z++) {

                //Does NOT include phase information!
                // + data.tr through alpha/2 preparation
		out_theory = fabs(stst * (1. - inv * expf(-((float)(z + 1.) * sim_data.seq.tr) / t1s)));

		out_simu = cabsf(mxy_sig[z][0] + mxy_sig[z][1] * I);

		err = fabsf(out_simu - out_theory);

		if (10E-4 < err) {

			debug_printf(DP_ERROR, "err: %f,\t out_simu: %f,\t out_theory: %f\n", err, out_simu, out_theory);
			debug_printf(DP_ERROR, "Error in sequence test\n see: -> test_ode_irbssfp_simulation() in test_ode_simu.c\n");

			return 0;
		}
	}
	return 1;
}

UT_REGISTER_TEST(test_ode_irbssfp_simulation);



/* ROT Simulation
 *
 * Compare the simulated IR bSSFP signal with the analytical model
 * Assumptions: 1. TR << T_{1,2}
 *              2. T_RF \approx 0
 *
 * References:
 *
 * Schmitt P, Griswold MA, Jakob PM, Kotas M, Gulani V, Flentje M, Haase A.
 * Inversion recovery TrueFISP: Quantification of T1, T2, and spin density.
 * Magn Reson Med 2004;51:661-667.
 *
 * Ehses P, Seiberlich N, Ma D, Breuer FA, Jakob PM, Griswold MA, Gulan, V.
 * IR TrueFISP with a golden‐ratio‐based radial readout: Fast quantification of T1, T2, and proton density.
 * Magn Reson Med 2013;69:71-81.
 *
 */
static bool test_rot_irbssfp_simulation(void)
{
	float angle = 45.;
	float repetition = 100;

	float fa = angle * M_PI / 180.;

	float t1n = WATER_T1;
	float t2n = WATER_T2;
	float m0n = 1.;

	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;

        sim_data.seq.type = SIM_ROT;
	sim_data.seq.seq_type = SEQ_IRBSSFP;
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.0015;
	sim_data.seq.rep_num = repetition;
	sim_data.seq.spin_num = 1;
        sim_data.seq.perfect_inversion = true;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = sim_data.seq.te;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 1. / t1n;
	sim_data.voxel.r2 = 1. / t2n;
	sim_data.voxel.m0 = m0n;
	sim_data.voxel.w = 0;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = angle;

        // Choose close to hard-pulse approximation -> same assumptions as analytical model
	sim_data.pulse.rf_end = 0.;

	sim_data.grad = simdata_grad_defaults;

	sim_data.tmp = simdata_tmp_defaults;

        sim_data.other = simdata_other_defaults;

        sim_data.other.sampling_rate = 10E5;

	int N = sim_data.seq.rep_num;

	float mxy_sig[N][3];
	float sa_r1_sig[N][3];
	float sa_r2_sig[N][3];
	float sa_m0_sig[N][3];
	float sa_b1_sig[N][3];

	bloch_simulation(&sim_data, N, &mxy_sig, &sa_r1_sig, &sa_r2_sig, &sa_m0_sig, &sa_b1_sig);


	// Analytical Model

	float t1s = 1. / ((cosf(fa / 2.) * cosf(fa / 2.)) / t1n + (sinf(fa / 2.) * sinf(fa / 2.)) / t2n);
	float s0 = m0n * sinf(fa / 2.);
	float stst = m0n * sinf(fa) / ((t1n / t2n + 1.) - cosf(fa) * (t1n / t2n - 1.));
	float inv = 1. + s0 / stst;


        // Model Comparison

	float out_simu = 0.;
	float out_theory = 0.;
	float err = 0.;

	for (int z = 0; z < repetition; z++) {

                //Does NOT include phase information!
                // + data.tr through alpha/2 preparation

		out_theory = fabs(stst * (1. - inv * expf(-((float)(z + 1.) * sim_data.seq.tr) / t1s)));

		out_simu = cabsf(mxy_sig[z][0] + mxy_sig[z][1] * I);

		err = fabsf(out_simu - out_theory);

		if (10E-4 < err) {

			debug_printf(DP_ERROR, "err: %f,\t out_simu: %f,\t out_theory: %f\n", err, out_simu, out_theory);
			debug_printf(DP_ERROR, "Error in sequence test\n see: -> test_rot_irbssfp_simulation() in test_ode_simu.c\n");

			return false;
		}
	}

	return true;
}

UT_REGISTER_TEST(test_rot_irbssfp_simulation);



/* Simulation Comparison IR FLASH
 *      ODE <-> STM
 * Stages of complexity:
 *      1. Inversion
 *      2. bSSFP
 *      3. Relaxation
 *      4. Slice-selection gradient moment != 0
 *      5. Off-Resonance
 */
static bool test_stm_ode_bssfp_comparison(void)
{
	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = SEQ_IRBSSFP;
	sim_data.seq.tr = 0.0045;
	sim_data.seq.te = 0.00225;
	sim_data.seq.rep_num = 500;
	sim_data.seq.spin_num = 1;
        sim_data.seq.perfect_inversion = true;
	sim_data.seq.inversion_pulse_length = 0.01;
        sim_data.seq.inversion_spoiler = 0.005;
	sim_data.seq.prep_pulse_length = sim_data.seq.te;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 1. / WATER_T1;
	sim_data.voxel.r2 = 1. / WATER_T2;
	sim_data.voxel.m0 = 1;
	sim_data.voxel.w = 0;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = 45.;
        sim_data.pulse.rf_end = 0.001;

	sim_data.grad = simdata_grad_defaults;
	sim_data.tmp = simdata_tmp_defaults;
        sim_data.other = simdata_other_defaults;

        sim_data.grad.mom_sl = 0.25 * 2. * M_PI * 1000.;	// [rad/s]
        sim_data.voxel.w =  0.25 * 2. * M_PI * 1000.;	// [rad/s]

	struct sim_data sim_ode = sim_data;

	int R = sim_ode.seq.rep_num;

	float mxy_ode[R][3];
	float sa_r1_ode[R][3];
	float sa_r2_ode[R][3];
	float sa_m0_ode[R][3];
	float sa_b1_ode[R][3];

	bloch_simulation(&sim_ode, R, &mxy_ode, &sa_r1_ode, &sa_r2_ode, &sa_m0_ode, &sa_b1_ode);

        sim_data.seq.type = SIM_STM;

	assert(R == sim_data.seq.rep_num);

	float mxy_matexp[R][3];
	float sa_r1_matexp[R][3];
	float sa_r2_matexp[R][3];
	float sa_m0_matexp[R][3];
	float sa_b1_matexp[R][3];

	bloch_simulation(&sim_data, R, &mxy_matexp, &sa_r1_matexp, &sa_r2_matexp, &sa_m0_matexp, &sa_b1_matexp);

	float tol = 10E-3;
	float err;

	for (int rep = 0; rep < sim_data.seq.rep_num; rep++) {

#if 0
                        bart_printf("ODE M[%d] x: %f,\t%f,\t%f\n", rep, mxy_ode[rep][0], mxy_ode[rep][1], mxy_ode[rep][2]);
                        bart_printf("STM M[%d] x: %f,\t%f,\t%f\n\n", rep, mxy_matexp[rep][0], mxy_matexp[rep][1], mxy_matexp[rep][2]);
#endif

        	for (int dim = 0; dim < 3; dim++) {

			err = fabsf(mxy_matexp[rep][dim] - mxy_ode[rep][dim]);
			if (tol < err)
				return 0;

			err = fabsf(sa_r1_matexp[rep][dim] - sa_r1_ode[rep][dim]);
			if (tol < err)
				return 0;

			err = fabsf(sa_r2_matexp[rep][dim] - sa_r2_ode[rep][dim]);
			if (tol < err)
				return 0;

			err = fabsf(sa_m0_matexp[rep][dim] - sa_m0_ode[rep][dim]);
			if (tol < err)
				return 0;

			err = fabsf(sa_b1_matexp[rep][dim] - sa_b1_ode[rep][dim]);
			if (tol < err)
				return 0;
		}
        }
	return 1;
}

UT_REGISTER_TEST(test_stm_ode_bssfp_comparison);


/* Simulation Comparison IR FLASH
 *      ODE <-> STM
 * Stages of complexity:
 *      1. Inversion
 *      2. FLASH
 *      3. Relaxation
 *      4. Slice-selection gradient moment != 0
 *      5. Off-Resonance
 */
static bool test_stm_ode_flash_comparison(void)
{
	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = SEQ_IRFLASH;
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.0017;
	sim_data.seq.rep_num = 500;
	sim_data.seq.spin_num = 1;
        sim_data.seq.perfect_inversion = true;
	sim_data.seq.inversion_pulse_length = 0.01;
        sim_data.seq.inversion_spoiler = 0.005;
	sim_data.seq.prep_pulse_length = sim_data.seq.te;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 1. / WATER_T1;
	sim_data.voxel.r2 = 1. / WATER_T2;
	sim_data.voxel.m0 = 1;
	sim_data.voxel.w = 0;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = 8.;
        sim_data.pulse.rf_end = 0.001;

	sim_data.grad = simdata_grad_defaults;
	sim_data.tmp = simdata_tmp_defaults;
        sim_data.other = simdata_other_defaults;

        sim_data.grad.mom_sl = 0.25 * 2. * M_PI * 1000.;	// [rad/s]
        sim_data.voxel.w =  0.25 * 2. * M_PI * 1000.;	// [rad/s]

	struct sim_data sim_ode = sim_data;

	int R = sim_ode.seq.rep_num;

	float mxy_ode[R][3];
	float sa_r1_ode[R][3];
	float sa_r2_ode[R][3];
	float sa_m0_ode[R][3];
	float sa_b1_ode[R][3];

	assert(R == sim_data.seq.rep_num);

	bloch_simulation(&sim_ode, R, &mxy_ode, &sa_r1_ode, &sa_r2_ode, &sa_m0_ode, &sa_b1_ode);

        sim_data.seq.type = SIM_STM;

	float mxy_matexp[R][3];
	float sa_r1_matexp[R][3];
	float sa_r2_matexp[R][3];
	float sa_m0_matexp[R ][3];
	float sa_b1_matexp[R][3];

	bloch_simulation(&sim_data, R, &mxy_matexp, &sa_r1_matexp, &sa_r2_matexp, &sa_m0_matexp, &sa_b1_matexp);

	float tol = 10E-3;
	float err;

	for (int rep = 0; rep < sim_data.seq.rep_num; rep++) {

#if 0
                        bart_printf("ODE M[%d] x: %f,\t%f,\t%f\n", rep, mxy_ode[rep][0], mxy_ode[rep][1], mxy_ode[rep][2]);
                        bart_printf("STM M[%d] x: %f,\t%f,\t%f\n\n", rep, mxy_matexp[rep][0], mxy_matexp[rep][1], mxy_matexp[rep][2]);
#endif

        	for (int dim = 0; dim < 3; dim++) {

			err = fabsf(mxy_matexp[rep][dim] - mxy_ode[rep][dim]);
			if (tol < err)
				return 0;

			err = fabsf(sa_r1_matexp[rep][dim] - sa_r1_ode[rep][dim]);
			if (tol < err)
				return 0;

			err = fabsf(sa_r2_matexp[rep][dim] - sa_r2_ode[rep][dim]);
			if (tol < err)
				return 0;

			err = fabsf(sa_m0_matexp[rep][dim] - sa_m0_ode[rep][dim]);
			if (tol < err)
				return 0;

			err = fabsf(sa_b1_matexp[rep][dim] - sa_b1_ode[rep][dim]);
			if (tol < err)
				return 0;
		}
        }
	return 1;
}

UT_REGISTER_TEST(test_stm_ode_flash_comparison);



// Test off-resonance effect in ODE simulation
//      - Set off-resonance so that magnetization is rotated by 90 degree within TE
//      - for w == 0 -> Mxy = 0+1*I
//      - Goal: Mxy = 1+0*I
static bool test_ode_simu_offresonance(void)
{
	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = SEQ_FLASH;	// Does not have preparation phase
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.001;
	sim_data.seq.rep_num = 1;
	sim_data.seq.spin_num = 1;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = 0.;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 0.;	// Turn off relaxation
	sim_data.voxel.r2 = 0.;	// Turn off relaxation
	sim_data.voxel.m0 = 1.;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = 90.;
	sim_data.pulse.rf_end = 10E-9;	// Close to Hard-Pulses

        sim_data.grad = simdata_grad_defaults;
	sim_data.tmp = simdata_tmp_defaults;
        sim_data.other = simdata_other_defaults;

	sim_data.voxel.w =  0.25 * 2. * M_PI * 1000.;	// [rad/s]

	int R = sim_data.seq.rep_num;

	float mxySig_ode[R][3];
	float saR1Sig_ode[R][3];
	float saR2Sig_ode[R][3];
	float saDensSig_ode[R][3];
	float sa_b1_ode[R][3];

	bloch_simulation(&sim_data, R, &mxySig_ode, &saR1Sig_ode, &saR2Sig_ode, &saDensSig_ode, &sa_b1_ode);

#if 0
	bart_printf("M\n x: %f+i*%f,\ty: %f+i*%f,\tz: %f+i*%f\n", mxySig_ode[0][0], mxySig_ode[0][0]
								, mxySig_ode[0][1], mxySig_ode[0][1]
								, mxySig_ode[0][2], mxySig_ode[0][2] );

	bart_printf("Err\n x: %f,\ty: %f,\tz: %f\n",	fabs(mxySig_ode[0][0] - 1.),
							fabs(mxySig_ode[0][1] - 0.),
							fabs(mxySig_ode[0][2] - 0.) );
#endif
	float tol = 10E-5;

	UT_ASSERT(	(fabs(mxySig_ode[0][0] - 1.) < tol)
                        && (fabs(mxySig_ode[0][1] - 0.) < tol)
                        && (fabs(mxySig_ode[0][2] - 0.) < tol) );

	return true;
}

UT_REGISTER_TEST(test_ode_simu_offresonance);


// Test off-resonance effect in STM simulation
//      - Set off-resonance so that magnetization is rotated by 90 degree within TE
//      - for w == 0 -> Mxy = 0+1*I
//      - Goal: Mxy = 1+0*I
static bool test_stm_simu_offresonance(void)
{
	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
        sim_data.seq.type = SIM_STM;
	sim_data.seq.seq_type = SEQ_FLASH;	// Does not have preparation phase
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.001;
	sim_data.seq.rep_num = 1;
	sim_data.seq.spin_num = 1;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = 0.;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 0.;	// Turn off relaxation
	sim_data.voxel.r2 = 0.;	// Turn off relaxation
	sim_data.voxel.m0 = 1.;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = 90.;
	sim_data.pulse.rf_end = 10E-9;	// Close to Hard-Pulses

        sim_data.grad = simdata_grad_defaults;
	sim_data.tmp = simdata_tmp_defaults;
        sim_data.other = simdata_other_defaults;

	sim_data.voxel.w =  0.25 * 2. * M_PI * 1000.;	// [rad/s]

	int R = sim_data.seq.rep_num;

	float mxySig_ode[R][3];
	float saR1Sig_ode[R][3];
	float saR2Sig_ode[R][3];
	float saDensSig_ode[R][3];
	float sa_b1_ode[R][3];

	bloch_simulation(&sim_data, R, &mxySig_ode, &saR1Sig_ode, &saR2Sig_ode, &saDensSig_ode, &sa_b1_ode);

#if 0
	bart_printf("M\n x: %f+i*%f,\ty: %f+i*%f,\tz: %f+i*%f\n", mxySig_ode[0][0], mxySig_ode[0][0]
								, mxySig_ode[0][1], mxySig_ode[0][1]
								, mxySig_ode[0][2], mxySig_ode[0][2] );

	bart_printf("Err\n x: %f,\ty: %f,\tz: %f\n",	fabs(mxySig_ode[0][0] - 1.),
							fabs(mxySig_ode[0][1] - 0.),
							fabs(mxySig_ode[0][2] - 0.) );
#endif
	float tol = 10E-5;

	UT_ASSERT(	(fabs(mxySig_ode[0][0] - 1.) < tol)
                        && (fabs(mxySig_ode[0][1] - 0.) < tol)
                        && (fabs(mxySig_ode[0][2] - 0.) < tol) );

	return true;
}

UT_REGISTER_TEST(test_stm_simu_offresonance);


// Test gradient during relaxation
//      - Simulate gradient dephasing by PI/2 between RF_end and TE of first repetition
//      - This should turn the magnetization from 0,1,0 to 1,0,0
static bool test_ode_simu_gradient(void)
{
	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = SEQ_FLASH;	// Does not have preparation phase
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.001;
	sim_data.seq.rep_num = 1;
	sim_data.seq.spin_num = 1;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = 0.;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 0.;	// Turn off relaxation
	sim_data.voxel.r2 = 0.;	// Turn off relaxation
	sim_data.voxel.m0 = 1.;
	sim_data.voxel.w = 0.;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = 90.;
	sim_data.pulse.rf_end = 10E-9;	// Close to Hard-Pulses

	sim_data.grad = simdata_grad_defaults;
	sim_data.grad.mom =  0.25 * 2. * M_PI * 1000.;	// [rad/s]

	sim_data.tmp = simdata_tmp_defaults;
        sim_data.other = simdata_other_defaults;

	int R = sim_data.seq.rep_num;

	float mxySig_ode[R][3];
	float saR1Sig_ode[R][3];
	float saR2Sig_ode[R][3];
	float saDensSig_ode[R][3];
	float sa_b1_ode[R][3];

	bloch_simulation(&sim_data, R, &mxySig_ode, &saR1Sig_ode, &saR2Sig_ode, &saDensSig_ode, &sa_b1_ode);

#if 0
	bart_printf("M\n x: %f+i*%f,\ty: %f+i*%f,\tz: %f+i*%f\n", mxySig_ode[sim_data.seq.rep_num-1][0], mxySig_ode[sim_data.seq.rep_num-1][0]
								, mxySig_ode[sim_data.seq.rep_num-1][1], mxySig_ode[sim_data.seq.rep_num-1][1]
								, mxySig_ode[sim_data.seq.rep_num-1][2], mxySig_ode[sim_data.seq.rep_num-1][2] );

	bart_printf("Err\n x: %f,\ty: %f,\tz: %f\n",	fabs(mxySig_ode[sim_data.seq.rep_num-1][0] - 1.),
							fabs(mxySig_ode[sim_data.seq.rep_num-1][1] - 0.),
							fabs(mxySig_ode[sim_data.seq.rep_num-1][2] - 0.) );

#endif
	float tol = 10E-5;

	UT_ASSERT(	(fabs(mxySig_ode[R - 1][0] - 1.) < tol)
                        && (fabs(mxySig_ode[R - 1][1] - 0.) < tol)
                        && (fabs(mxySig_ode[R - 1][2] - 0.) < tol) );

	return true;
}

UT_REGISTER_TEST(test_ode_simu_gradient);


// Test gradient during relaxation STM
//      - Simulate gradient dephasing by PI/2 between RF_end and TE of first repetition
//      - This should turn the magnetization from 0,1,0 to 1,0,0
static bool test_stm_simu_gradient(void)
{
	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
        sim_data.seq.type = SIM_STM;
	sim_data.seq.seq_type = SEQ_FLASH;	// Does not have preparation phase
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.001;
	sim_data.seq.rep_num = 1;
	sim_data.seq.spin_num = 1;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = 0.;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 0.;	// Turn off relaxation
	sim_data.voxel.r2 = 0.;	// Turn off relaxation
	sim_data.voxel.m0 = 1.;
	sim_data.voxel.w = 0.;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = 90.;
	sim_data.pulse.rf_end = 10E-9;	// Close to Hard-Pulses

	sim_data.grad = simdata_grad_defaults;
	sim_data.grad.mom =  0.25 * 2. * M_PI * 1000.;	// [rad/s]

	sim_data.tmp = simdata_tmp_defaults;
        sim_data.other = simdata_other_defaults;

	int R = sim_data.seq.rep_num;

	float mxySig_ode[R][3];
	float saR1Sig_ode[R][3];
	float saR2Sig_ode[R][3];
	float saDensSig_ode[R][3];
	float sa_b1_ode[R][3];

	bloch_simulation(&sim_data, R, &mxySig_ode, &saR1Sig_ode, &saR2Sig_ode, &saDensSig_ode, &sa_b1_ode);

#if 0
	bart_printf("M\n x: %f+i*%f,\ty: %f+i*%f,\tz: %f+i*%f\n", mxySig_ode[sim_data.seq.rep_num-1][0], mxySig_ode[sim_data.seq.rep_num-1][0]
								, mxySig_ode[sim_data.seq.rep_num-1][1], mxySig_ode[sim_data.seq.rep_num-1][1]
								, mxySig_ode[sim_data.seq.rep_num-1][2], mxySig_ode[sim_data.seq.rep_num-1][2] );

	bart_printf("Err\n x: %f,\ty: %f,\tz: %f\n",	fabs(mxySig_ode[sim_data.seq.rep_num-1][0] - 1.),
							fabs(mxySig_ode[sim_data.seq.rep_num-1][1] - 0.),
							fabs(mxySig_ode[sim_data.seq.rep_num-1][2] - 0.) );

#endif
	float tol = 10E-5;

	UT_ASSERT(	(fabs(mxySig_ode[R - 1][0] - 1.) < tol)
                        && (fabs(mxySig_ode[R - 1][1] - 0.) < tol)
                        && (fabs(mxySig_ode[R - 1][2] - 0.) < tol) );

	return true;
}

UT_REGISTER_TEST(test_stm_simu_gradient);


// Shaihan J Malik, Alessandro Sbrizzi, Hans Hoogduin, and Joseph V Hajnal
// Equivalence of EPG and Isochromat-based simulation of MR signals
// Proc. Intl. Soc. Mag. Reson. Med. 24 (2016), No. 3196
static void ode_fourier_modes(int N, struct sim_data* data, complex float fn[N], float angle)
{
	complex float m_plus[N];

	int t = data->seq.rep_num - 1;

	// Perform ODE simulations for isochromates

	for (int i = 0; i < N; i++) {

		struct sim_data sim_ode = *data;

		sim_ode.voxel.w = angle * i / N;

		int R = sim_ode.seq.rep_num;

		float mxySig_ode[R][3];
		float saR1Sig_ode[R][3];
		float saR2Sig_ode[R][3];
		float saDensSig_ode[R][3];
		float sa_b1_ode[R][3];

		bloch_simulation(&sim_ode, R, &mxySig_ode, &saR1Sig_ode, &saR2Sig_ode, &saDensSig_ode, &sa_b1_ode);

		// Save M+
		m_plus[i] = mxySig_ode[t][0] + mxySig_ode[t][1] * I;
	}

#if 0	// Print out values
	for (int i = 0; i < N; i++) {
		bart_printf("M+\n w/TE: %f, Mxy: %f+%f*I\n", 2 * M_PI / N * i, crealf(m_plus[i]), cimagf(m_plus[i]));
		bart_printf("|Mxy|: %f\n", cabsf(m_plus[i]));
	}
#endif

	// Estimate Fn based on DFT

	for (int j = 0; j < N; j++) {

		fn[j] = 0.;

		for (int m = 0; m < N; m++)
			fn[j] += m_plus[m] * cexpf(-2. * M_PI * I * (-(float)N / 2. + j) * (float)m / (float)N);

		fn[j] /= N; // Scale to compensate for Fn \prop N
	}
}

// Show relation between EPG and isochromate simulations
// Idea: use isochromates off-resonance distributions to loop through Fourier coefficients
static bool test_ode_epg_relation(void)
{
	// General simulation details

	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = SEQ_FLASH;
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.001;
	sim_data.seq.rep_num = 1;
	sim_data.seq.spin_num = 1;
	sim_data.seq.inversion_pulse_length = 0.;
        sim_data.seq.perfect_inversion = true;
	sim_data.seq.prep_pulse_length = 0.;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 0.;	// Turn off relaxation
	sim_data.voxel.r2 = 0.;	// Turn off relaxation
	sim_data.voxel.m0 = 1.;
	sim_data.voxel.w = 0.;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = 8.;
	sim_data.pulse.rf_end = 10E-9;	// Close to Hard-Pulses

	sim_data.grad = simdata_grad_defaults;
	sim_data.tmp = simdata_tmp_defaults;
        sim_data.other = simdata_other_defaults;

	// Estimate Fourier modes from ODE simulation

	int N = 10; //number of isochromates

	complex float fn[N];

	float angles[4] = {0, 1, 2, 3};	// [rotations/ms]

	complex float test_modes[4] = { 0. };

	for (int i = 0; i < 4; i++) {

		ode_fourier_modes(N, &sim_data, fn, angles[i] * (2. * M_PI * 1000.));	// [rad/s]

		test_modes[i] = fn[N/2+i];
	}

	// Compute F(n=0) mode with EPG

	int T = sim_data.seq.rep_num;
	int M = 2*T;

	complex float signal[T];
	complex float states[3][M][T]; // 3 -> dims: Fn,F-n,Zn; M: k-states; T: repetition

	flash_epg_der(T, M, signal, states, NULL, NULL, sim_data.pulse.flipangle, sim_data.seq.tr, 1000000., 1000000., 1., sim_data.voxel.w, 0L);

#if 0
	for (int i = 0; i < M; i++) {

		bart_printf("EPG: Fn: k: %d,\t%f+%f*I\n", i, crealf(states[0][i][T-1]), cimagf(states[0][i][T-1])); // 0 -> Fn

		bart_printf("\nTest Modes:\t%f+%f*I\n", crealf(test_modes[i]), cimagf(test_modes[i]));
	}

	bart_printf("\nSignal(EPG):\t %f+%f*i\n\n", crealf(signal[T-1]), cimagf(signal[T-1]));

	bart_printf("Err\n k0: %f,\tk1: %f\n",	(fabs(cimagf(states[0][0][T-1]) - cimagf(test_modes[0]))),
							(fabs(cimagf(states[0][1][T-1]) - cimagf(test_modes[1]))) );
#endif

	float tol = 10E-5;

	// "* -1" for EPG rotations defined COUNTER clockwise, while Bloch equations rotate ation CLOCKWISE around x
	// FIXME: Redefine EPG clockwise
	UT_ASSERT(	(fabs(cimagf(-1. * states[0][0][T-1]) - cimagf(test_modes[0])) < tol)
                        && (fabs(cimagf(states[0][1][T-1]) - cimagf(test_modes[1])) < tol) );

	return true;
}

UT_REGISTER_TEST(test_ode_epg_relation);


/* HARD PULSE Simulation
 *
 * Compare the simulated IR bSSFP signal with the analytical model
 *
 * Assumptions: 1. TR << T_{1,2}
 *              2. T_RF \approx 0
 */
static bool test_hp_irbssfp_simulation(void)
{
	float angle = 45.;
	float repetition = 100;

	float fa = angle * M_PI / 180.;

	float t1n = WATER_T1;
	float t2n = WATER_T2;
	float m0n = 1.;

	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = SEQ_IRBSSFP;
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.0015;
	sim_data.seq.rep_num = repetition;
	sim_data.seq.spin_num = 1;
        sim_data.seq.perfect_inversion = true;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = sim_data.seq.te;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 1. / t1n;
	sim_data.voxel.r2 = 1. / t2n;
	sim_data.voxel.m0 = m0n;
	sim_data.voxel.w = 0;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = angle;

        // Choose HARD-PULSE approximation
        // -> same assumptions as analytical model
	sim_data.pulse.rf_end = 0.;

	sim_data.grad = simdata_grad_defaults;
	sim_data.tmp = simdata_tmp_defaults;
        sim_data.other = simdata_other_defaults;

	int R = sim_data.seq.rep_num;

	float mxy_sig[R][3];
	float sa_r1_sig[R][3];
	float sa_r2_sig[R][3];
	float sa_m0_sig[R][3];
	float sa_b1_sig[R][3];

	bloch_simulation(&sim_data, R, &mxy_sig, &sa_r1_sig, &sa_r2_sig, &sa_m0_sig, &sa_b1_sig);


	// Analytical Model
	float t1s = 1 / ((cosf(fa / 2.) * cosf(fa / 2.)) / t1n + (sinf(fa / 2.) * sinf(fa / 2.)) / t2n);
	float s0 = m0n * sinf(fa / 2.);
	float stst = m0n * sinf(fa) / ((t1n / t2n + 1) - cosf(fa) * (t1n / t2n - 1));
	float inv = 1 + s0 / stst;


        // Model Comparison
	float out_simu = 0.;
	float out_theory = 0.;
	float err = 0.;

	for (int z = 0; z < repetition; z++) {

                //Does NOT include phase information!
                // + data.tr through alpha/2 preparation
		out_theory = fabs(stst * (1. - inv * expf(-((float)(z + 1) * sim_data.seq.tr) / t1s)));

		out_simu = cabsf(mxy_sig[z][1] + mxy_sig[z][0] * I);

		err = fabsf(out_simu - out_theory);

		if (10E-4 < err) {

			debug_printf(DP_ERROR, "err: %f,\t out_simu: %f,\t out_theory: %f\n", err, out_simu, out_theory);
			debug_printf(DP_ERROR, "Error in sequence test\n see: -> test_simulation() in test_ode_simu.c\n");

			return 0;
		}
	}
	return 1;
}

UT_REGISTER_TEST(test_hp_irbssfp_simulation);


// Test off-resonance effect in HARD PULSE simulation
//      - Set off-resonance so that magnetization is rotated by 90 degree within TE
//      - for w == 0 -> Mxy = 0+1*I
//	- Rotation through off-resonance clockwise
//      - Goal: Mxy = 1+0*I
static bool test_hp_simu_offresonance(void)
{
	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = SEQ_FLASH;	// Does not have preparation phase
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.001;
	sim_data.seq.rep_num = 1;
	sim_data.seq.spin_num = 1;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = 0.;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 0.;	// Turn off relaxation
	sim_data.voxel.r2 = 0.;	// Turn off relaxation
	sim_data.voxel.m0 = 1.;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = 90.;
	sim_data.pulse.rf_end = 0.;	// Here: HARD PULSE!

        sim_data.grad = simdata_grad_defaults;
	sim_data.tmp = simdata_tmp_defaults;
        sim_data.other = simdata_other_defaults;

	sim_data.voxel.w =  0.25 * 2. * M_PI * 1000.;	// [rad/s]

	int R = sim_data.seq.rep_num;

	float mxySig_ode[R][3];
	float saR1Sig_ode[R][3];
	float saR2Sig_ode[R][3];
	float saDensSig_ode[R][3];
	float sa_b1_ode[R][3];

	bloch_simulation(&sim_data, R, &mxySig_ode, &saR1Sig_ode, &saR2Sig_ode, &saDensSig_ode, &sa_b1_ode);

#if 0
	bart_printf("M\n x: %f+i*%f,\ty: %f+i*%f,\tz: %f+i*%f\n", crealf(mxySig_ode[0][0]), cimagf(mxySig_ode[0][0])
								, crealf(mxySig_ode[0][1]), cimagf(mxySig_ode[0][1])
								, crealf(mxySig_ode[0][2]), cimagf(mxySig_ode[0][2]) );

	bart_printf("Err\n x: %f,\ty: %f,\tz: %f\n",	fabs(mxySig_ode[0][0] + 1.),
							fabs(mxySig_ode[0][1] - 0.),
							fabs(mxySig_ode[0][2] - 0.) );
#endif
	float tol = 10E-5;

	UT_ASSERT(	(fabs(mxySig_ode[0][0] - 1.) < tol)
                        && (fabs(mxySig_ode[0][1] - 0.) < tol)
                        && (fabs(mxySig_ode[0][2] - 0.) < tol) );

	return true;
}

UT_REGISTER_TEST(test_hp_simu_offresonance);


// Test gradient during relaxation in HARD PULSE simulation
//      - Simulate gradient dephasing by PI/2 between RF_end and TE of first repetition
//      - This should turn the magnetization from 0,1,0 to 1,0,0
static bool test_hp_simu_gradient(void)
{
	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = SEQ_FLASH;	// Does not have preparation phase
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.001;
	sim_data.seq.rep_num = 1;
	sim_data.seq.spin_num = 1;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = 0.;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 0.;	// Turn off relaxation
	sim_data.voxel.r2 = 0.;	// Turn off relaxation
	sim_data.voxel.m0 = 1.;
	sim_data.voxel.w = 0.;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = 90.;
	sim_data.pulse.rf_end = 0.;	// Here: HARD PULSE!

	sim_data.grad = simdata_grad_defaults;
	sim_data.grad.mom =  0.25 * 2. * M_PI * 1000.;	// [rad/s]

	sim_data.tmp = simdata_tmp_defaults;
        sim_data.other = simdata_other_defaults;

	int R = sim_data.seq.rep_num;

	float mxySig_ode[R][3];
	float saR1Sig_ode[R][3];
	float saR2Sig_ode[R][3];
	float saDensSig_ode[R][3];
	float sa_b1_ode[R][3];

	bloch_simulation(&sim_data, R, &mxySig_ode, &saR1Sig_ode, &saR2Sig_ode, &saDensSig_ode, &sa_b1_ode);

#if 0
	bart_printf("M\n x: %f+i*%f,\ty: %f+i*%f,\tz: %f+i*%f\n", crealf(mxySig_ode[0][0]), cimagf(mxySig_ode[0][0])
								, crealf(mxySig_ode[0][1]), cimagf(mxySig_ode[0][1])
								, crealf(mxySig_ode[0][2]), cimagf(mxySig_ode[0][2]) );

	bart_printf("Err\n x: %f,\ty: %f,\tz: %f\n",	fabs(mxySig_ode[sim_data.seq.rep_num-1][0] + 1.),
							fabs(mxySig_ode[sim_data.seq.rep_num-1][1] - 0.),
							fabs(mxySig_ode[sim_data.seq.rep_num-1][2] - 0.) );

#endif
	float tol = 10E-5;

	UT_ASSERT(	(fabs(mxySig_ode[sim_data.seq.rep_num-1][0] - 1.) < tol)
                        && (fabs(mxySig_ode[sim_data.seq.rep_num-1][1] - 0.) < tol)
                        && (fabs(mxySig_ode[sim_data.seq.rep_num-1][2] - 0.) < tol) );

	return true;
}

UT_REGISTER_TEST(test_hp_simu_gradient);


// Test refocussing of z-gradient moment in ODE simulation
// Idea:        Without balanced z moments the magnetization state changes
//              for different repetitions. In the refocused case it
//              is almost unaffected.
static bool test_ode_z_gradient_refocus(void)
{
	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
        sim_data.seq.type = SIM_ODE;
	sim_data.seq.seq_type = SEQ_BSSFP;
	sim_data.seq.tr = 0.004;
	sim_data.seq.te = 0.002;
	sim_data.seq.rep_num = 2;
	sim_data.seq.spin_num = 1;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = sim_data.seq.te;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 0.;
	sim_data.voxel.r2 = 0.;
	sim_data.voxel.m0 = 1.;
	sim_data.voxel.w = 0;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = 45.;

	sim_data.pulse.rf_end = 0.001;

	sim_data.grad = simdata_grad_defaults;
        sim_data.grad.mom_sl = 0.5 * 2. * M_PI * 1000.;	// [rad/s]

	sim_data.tmp = simdata_tmp_defaults;
        sim_data.other = simdata_other_defaults;

	int R = sim_data.seq.rep_num;

	float mxy_sig[R][3];
	float sa_r1_sig[R][3];
	float sa_r2_sig[R][3];
	float sa_m0_sig[R][3];
	float sa_b1_sig[R][3];

	bloch_simulation(&sim_data, R, &mxy_sig, &sa_r1_sig, &sa_r2_sig, &sa_m0_sig, &sa_b1_sig);

        sim_data.grad.mom_sl = 0.5 * 2. * M_PI * 1000.;	// [rad/s]

        float mxy_sig2[R][3];
	float sa_r1_sig2[R][3];
	float sa_r2_sig2[R][3];
	float sa_m0_sig2[R][3];
	float sa_b1_sig2[R][3];

	bloch_simulation(&sim_data, R, &mxy_sig2, &sa_r1_sig2, &sa_r2_sig2, &sa_m0_sig2, &sa_b1_sig2);

#if 0
	bart_printf("M1 -> x: %f,\ty: %f,\tz: %f\n", mxy_sig[0][0], mxy_sig[0][1], mxy_sig[0][2]);
	bart_printf("M2 -> x: %f,\ty: %f,\tz: %f\n", mxy_sig[1][0], mxy_sig[1][1], mxy_sig[1][2]);
#endif

	float tol = 1.E-2;

	UT_ASSERT(      (fabsf(mxy_sig[0][0]-mxy_sig[1][0]) < tol)
                        && (fabsf(mxy_sig[0][1]-mxy_sig[1][1]) < tol)
                        && (fabsf(mxy_sig[0][2]-mxy_sig[1][2]) < tol) );

	return true;
}

UT_REGISTER_TEST(test_ode_z_gradient_refocus);


// Test refocussing of z-gradient moment in STM simulation
// Idea:        Without balanced z moments the magnetization state changes
//              for different repetitions. In the refocused case it
//              is almost unaffected.
static bool test_stm_z_gradient_refocus(void)
{
	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
        sim_data.seq.type = SIM_STM;
	sim_data.seq.seq_type = SEQ_BSSFP;
	sim_data.seq.tr = 0.004;
	sim_data.seq.te = 0.002;
	sim_data.seq.rep_num = 2;
	sim_data.seq.spin_num = 1;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = sim_data.seq.te;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 0.;
	sim_data.voxel.r2 = 0.;
	sim_data.voxel.m0 = 1.;
	sim_data.voxel.w = 0;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = 45.;

	sim_data.pulse.rf_end = 0.001;

	sim_data.grad = simdata_grad_defaults;
        sim_data.grad.mom_sl = 0.5 * 2. * M_PI * 1000.;	// [rad/s]

	sim_data.tmp = simdata_tmp_defaults;
        sim_data.other = simdata_other_defaults;

	int R = sim_data.seq.rep_num;

	float mxy_sig[R][3];
	float sa_r1_sig[R][3];
	float sa_r2_sig[R][3];
	float sa_m0_sig[R][3];
	float sa_b1_sig[R][3];

	bloch_simulation(&sim_data, R, &mxy_sig, &sa_r1_sig, &sa_r2_sig, &sa_m0_sig, &sa_b1_sig);

#if 0
	bart_printf("M1 -> x: %f,\ty: %f,\tz: %f\n", mxy_sig[0][0], mxy_sig[0][1], mxy_sig[0][2]);
	bart_printf("M2 -> x: %f,\ty: %f,\tz: %f\n", mxy_sig[1][0], mxy_sig[1][1], mxy_sig[1][2]);
#endif

	float tol = 1.E-2;

	UT_ASSERT(   (fabsf(mxy_sig[0][0] - mxy_sig[1][0]) < tol)
		  && (fabsf(mxy_sig[0][1] - mxy_sig[1][1]) < tol)
		  && (fabsf(mxy_sig[0][2] - mxy_sig[1][2]) < tol));

	return true;
}

UT_REGISTER_TEST(test_stm_z_gradient_refocus);


// Test inversion pulse
static bool test_ode_inversion(void)
{
        enum { N = 3 };              // Number of dimensions (x, y, z)
	enum { P = 4 };              // Number of parameters with estimated derivative (Mxy, R1, NULL, R2, B1)

        struct sim_data data;

        data.seq = simdata_seq_defaults;
        data.seq.seq_type = SEQ_FLASH;
        data.seq.tr = 0.001;
        data.seq.te = 0.001;
        data.seq.rep_num = 1;
        data.seq.spin_num = 1;
        data.seq.inversion_pulse_length = 0.01;
        data.seq.inversion_spoiler = 0.005;

        data.voxel = simdata_voxel_defaults;
        data.voxel.r1 = 0.;
        data.voxel.r2 = 0.;
        data.voxel.m0 = 1;
        data.voxel.w = 0;

        data.pulse = simdata_pulse_defaults;
        data.pulse.flipangle = 0.;
        data.pulse.rf_end = 0.01;

        data.grad = simdata_grad_defaults;
        data.tmp = simdata_tmp_defaults;
        data.other = simdata_other_defaults;

        float xp[P][N] = { { 0., 0., 1. }, { 0. }, { 0. }, { 0. } };

        float h = 10E-5;
        float tol = 0.005; // >99.5% inversion efficiency

        inversion(&data, h, tol, N, P, xp, 0., 0.005);

        // bart_printf("%f, %f, %f\n", xp[0][0], xp[0][1], xp[0][2]);

        UT_ASSERT(fabs(xp[0][2] + 1.) < tol);

	return 1;
}

UT_REGISTER_TEST(test_ode_inversion);


// Test STM matrix creation.
// Test with
//      - RF pulse: sim_data.seq.pulse_applied = true;
//      - Relaxation: sim_data.voxel.r1 and sim_data.voxel.r2 != 0;
//      - z-Gradient: sim_data.grad.gb[2] = 2. * M_PI * 1000.;
static bool test_stm_matrix_creation(void)
{
	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = SEQ_FLASH;
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.0017;
        sim_data.seq.pulse_applied = true;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 10.;
	sim_data.voxel.r2 = 1.;
	sim_data.voxel.m0 = 1.;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = 90.;

        sim_data.grad = simdata_grad_defaults;
        sim_data.grad.gb[2] = 2. * M_PI * 1000.;	// [rad/s]

	sim_data.tmp = simdata_tmp_defaults;
        sim_data.other = simdata_other_defaults;

        // Prepare RF pulse to have correct flip angle
        sinc_pulse_init(&sim_data.pulse, sim_data.pulse.rf_start, sim_data.pulse.rf_end, sim_data.pulse.flipangle, sim_data.pulse.phase, sim_data.pulse.bwtp, sim_data.pulse.alpha);


        // Create STM for Bloch equation only

        float t0 = 0.;
        float t1 = 0.001;

        int N1 = 4;
        float out1[4][4] = { { 0. }, { 0. }, { 0. }, { 0. } };

        float m1[4] = { 0. };
        m1[2] = -1.;
        m1[3] = 1.;

        mat_exp_simu(&sim_data, N1, t0, t1, out1);
        apply_sim_matrix(N1, m1, out1);


        // Create STM for Bloch + SA with dR1, dM0, dR2

        int N2 = 10;
        float out2[10][10] = {  { 0. }, { 0. }, { 0. }, { 0. },
                                { 0. }, { 0. }, { 0. }, { 0. },
                                { 0. }, { 0. } };

        float m2[10] = { 0. };
        m2[2] = -1.;
        m2[9] = 1.;

        mat_exp_simu(&sim_data, N2, t0, t1, out2);
        apply_sim_matrix(N2, m2, out2);


        // Potential output
#if 0
        bart_printf("\tM1\tM2\n");

        for (int i = 0; i < 4; i++) {
                bart_printf("%f \t %f\n", m1[i], (3 == i) ? m2[9] : m2[i]);
        }
#endif

        // Compare signal part of STM matrices estimated above by its effect on the magnetization

        float tol = 1.E-5;

	UT_ASSERT(	(fabs(m1[0] - m2[0]) < tol)
                        && (fabs(m1[1] - m2[1]) < tol)
                        && (fabs(m1[2] - m2[2]) < tol)
                        && (fabs(m1[3] - m2[9]) < tol) );

	return true;
}

UT_REGISTER_TEST(test_stm_matrix_creation);

