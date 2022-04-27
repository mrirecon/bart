/* Copyright 2022. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 *	Nick Scholand
 */

#include <stdio.h>
#include <memory.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/opts.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/linalg.h"
#include "num/init.h"
#include "num/ode.h"

#include "simu/bloch.h"
#include "simu/pulse.h"

#include "simulation.h"


void debug_sim(struct sim_data* data)
{
        debug_printf(DP_WARN, "Simulation-Debug-Output\n\n");
        debug_printf(DP_WARN, "Voxel-Parameter:\n");
        debug_printf(DP_INFO, "\tR1:%f\n", data->voxel.r1);
        debug_printf(DP_INFO, "\tR2:%f\n", data->voxel.r2);
        debug_printf(DP_INFO, "\tM0:%f\n", data->voxel.m0);
        debug_printf(DP_INFO, "\tw:%f\n", data->voxel.w);
        debug_printf(DP_INFO, "\tB1:%f\n\n", data->voxel.b1);

        debug_printf(DP_WARN, "Seq-Parameter:\n");
        debug_printf(DP_INFO, "\tSequence:%d\n", data->seq.seq_type);
        debug_printf(DP_INFO, "\tTR:%f\n", data->seq.tr);
        debug_printf(DP_INFO, "\tTE:%f\n", data->seq.te);
        debug_printf(DP_INFO, "\t#Rep:%d\n", data->seq.rep_num);
        debug_printf(DP_INFO, "\t#Spins:%d\n", data->seq.spin_num);
        debug_printf(DP_INFO, "\tIPL:%f\n", data->seq.inversion_pulse_length);
        debug_printf(DP_INFO, "\tPPL:%f\n", data->seq.prep_pulse_length);
        debug_printf(DP_INFO, "\tPulse Applied?:%d\n\n", data->seq.pulse_applied);

        debug_printf(DP_WARN, "Gradient-Parameter:\n");
        debug_printf(DP_INFO, "\tMoment:%f\n", data->grad.mom);
        debug_printf(DP_INFO, "\tMoment SL:%f\n\n", data->grad.mom_sl);

        debug_printf(DP_WARN, "Pulse-Parameter:\n");
        debug_printf(DP_INFO, "\tRF-Start:%f\n", data->pulse.rf_start);
        debug_printf(DP_INFO, "\tRF-End:%f\n", data->pulse.rf_end);
        debug_printf(DP_INFO, "\tFlipangle:%f\n", data->pulse.flipangle);
        debug_printf(DP_INFO, "\tPhase:%f\n", data->pulse.phase);
        debug_printf(DP_INFO, "\tBWTP:%f\n", data->pulse.bwtp);
        debug_printf(DP_INFO, "\tNL:%f\n", data->pulse.nl);
        debug_printf(DP_INFO, "\tNR:%f\n", data->pulse.nr);
        debug_printf(DP_INFO, "\tN:%f\n", data->pulse.n);
        debug_printf(DP_INFO, "\tt0:%f\n", data->pulse.t0);
        debug_printf(DP_INFO, "\tAlpha:%f\n", data->pulse.alpha);
        debug_printf(DP_INFO, "\tA:%f\n\n", data->pulse.A);
}


const struct simdata_voxel simdata_voxel_defaults = {

	.r1 = 0.,
	.r2 = 0.,
	.m0 = 1.,
	.w = 0.,
	.b1 = 1.,
};


const struct simdata_seq simdata_seq_defaults = {

	.seq_type = BSSFP,
	.tr = 0.004,
	.te = 0.002,
	.rep_num = 1,
	.spin_num = 1,

        .perfect_inversion = false,
	.inversion_pulse_length = 0.01,
        .inversion_spoiler = 0.,

	.prep_pulse_length = 0.001,

        .pulse_applied = false,
};


const struct simdata_tmp simdata_tmp_defaults = {

        .rep_counter = 0,
	.spin_counter = 0,
	.t = 0.,
	.w1 = 0.,
	.r2spoil = 0.,
};


const struct simdata_grad simdata_grad_defaults = {

	.gb = { 0., 0., 0. },
	.gb_eff = { 0., 0., 0.},
	.mom = 0.,
	.mom_sl = 0.,
};


static void bloch_pdy2(void* _data, float* out, float t, const float* in)
{
	struct sim_data* data = _data;
	(void)t;

	bloch_pdy((float(*)[3])out, in, data->voxel.r1, data->voxel.r2+data->tmp.r2spoil, data->grad.gb_eff);
}


static void bloch_pdp2(void* _data, float* out, float t, const float* in)
{
	struct sim_data* data = _data;
	(void)t;

	bloch_b1_pdp((float(*)[3])out, in, data->voxel.r1, data->voxel.r2+data->tmp.r2spoil, data->grad.gb_eff, data->pulse.phase, data->tmp.w1);
}


/* ------------ Bloch Equation -------------- */

static void bloch_simu_fun(void* _data, float* out, float t, const float* in)
{
	struct sim_data* data = _data;

	if (data->seq.pulse_applied) {

                // Hyperbolic Secant pulse
                if (true == data->pulse.hs.on) {

                        data->tmp.w1 = pulse_hypsec_am(&data->pulse.hs, t);

                        data->pulse.phase = pulse_hypsec_phase(&data->pulse.hs, t);
                }
                //Windowed Sinc pulse
                else {
                        data->tmp.w1 = pulse_sinc(&data->pulse, t);
                }

		data->grad.gb_eff[0] = cosf(data->pulse.phase) * data->tmp.w1 * data->voxel.b1 + data->grad.gb[0];
		data->grad.gb_eff[1] = sinf(data->pulse.phase) * data->tmp.w1 * data->voxel.b1 + data->grad.gb[1];

	} else {

		data->tmp.w1 = 0.;
		data->grad.gb_eff[0] = data->grad.gb[0];
		data->grad.gb_eff[1] = data->grad.gb[1];
	}

	// Units: [gb] = rad/s
	data->grad.gb_eff[2] = data->grad.gb[2] + data->voxel.w;

	bloch_ode(out, in, data->voxel.r1, data->voxel.r2+data->tmp.r2spoil, data->grad.gb_eff);
}

/* ------------ Read-Out -------------- */

static void adc_corr(int N, int P, float out[P][N], float in[P][N], float angle)
{
	for (int i = 0; i < P; i++)
		rotz(out[i], in[i], angle);
}


static long vector_position(int d, int r, int rep_max, int s, int spin_max)
{
        return d * spin_max * rep_max + r * spin_max + s;

}


static void collect_signal(struct sim_data* data, int N, int P, float* mxy, float* sa_r1, float* sa_r2, float* sa_b1, float xp[P + 1][N])
{
	float tmp[4][3] = { { 0. }, { 0. }, { 0. }, { 0. } };

	adc_corr(N, P, tmp, xp, -data->pulse.phase);

        long ind = 0;

	for (int i = 0; i < N; i++) {

                ind = vector_position(i, data->tmp.rep_counter, data->seq.rep_num, data->tmp.spin_counter, data->seq.spin_num);

		if (NULL != mxy)
			mxy[ind] = tmp[0][i];

		if (NULL != sa_r1)
			sa_r1[ind] = tmp[1][i];

		if (NULL != sa_r2)
			sa_r2[ind] = tmp[2][i];

		if (NULL != sa_b1)
			sa_b1[ind] = tmp[3][i];
	}
}


static void sum_up_signal(struct sim_data* data, float *mxy,  float *sa_r1, float *sa_r2, float *sa_b1,
                        float (*mxy_sig)[3], float (*sa_r1_sig)[3], float (*sa_r2_sig)[3], float (*sa_m0_sig)[3], float (*sa_b1_sig)[3])
{
        float sum_mxy;
	float sum_sa_r1;
	float sum_sa_r2;
	float sum_sa_b1;

        long ind = 0;

        // Dimensions; [x, y, z]
	for (int dim = 0; dim < 3; dim++) {

		sum_mxy = 0.;
		sum_sa_r1 = 0.;
		sum_sa_r2 = 0.;
		sum_sa_b1 = 0.;

                //Repetitions
		for (int r = 0; r < data->seq.rep_num; r++) {

                        // Spins
			for (int spin = 0; spin < data->seq.spin_num; spin++) {

                                ind = vector_position(dim, r, data->seq.rep_num, spin, data->seq.spin_num);

				sum_mxy += mxy[ind];
				sum_sa_r1 += sa_r1[ind];
				sum_sa_r2 += sa_r2[ind];
				sum_sa_b1 += sa_b1[ind];
			}

                        // Mean
                        mxy_sig[r][dim] = sum_mxy * data->voxel.m0 / (float)data->seq.spin_num;
                        sa_r1_sig[r][dim] = sum_sa_r1 * data->voxel.m0 / (float)data->seq.spin_num;
                        sa_r2_sig[r][dim] = sum_sa_r2 * data->voxel.m0 / (float)data->seq.spin_num;
                        sa_b1_sig[r][dim] = sum_sa_b1 * data->voxel.m0 / (float)data->seq.spin_num;
                        sa_m0_sig[r][dim] = sum_mxy / (float)data->seq.spin_num;

                        sum_mxy = 0.;
                        sum_sa_r1 = 0.;
                        sum_sa_r2 = 0.;
                        sum_sa_b1 = 0.;
		}
	}
}

/* ------------ RF-Pulse -------------- */

// Single hard pulse without discrete sampling
static void hard_pulse(struct sim_data* data, int N, int P, float xp[P][N])
{
	data->grad.gb[2] = (data->grad.mom_sl + data->voxel.w);	//[rad/s]

        for (int i = 0; i < P; i++)
                bloch_excitation2(xp[i], xp[i], data->pulse.flipangle/180.*M_PI, data->pulse.phase);

        data->grad.gb[2] = 0.;
}


static void ode_pulse(struct sim_data* data, float h, float tol, int N, int P, float xp[P][N])
{
	// Turn on potential slice-selection gradient
	data->grad.gb[2] = data->grad.mom_sl; // [rad/s]

        // Choose P-1 because ODE interface treats signal seperat and P only describes the number of parameters
	ode_direct_sa(h, tol, N, P-1, xp, data->pulse.rf_start, data->pulse.rf_end, data,  bloch_simu_fun, bloch_pdy2, bloch_pdp2);

	data->grad.gb[2] = 0.;
}


void start_rf_pulse(struct sim_data* data, float h, float tol, int N, int P, float xp[P][N])
{
	data->seq.pulse_applied = true;

	if (0. == data->pulse.rf_end)
		hard_pulse(data, N, P, xp);
	else
		ode_pulse(data, h, tol, N, P, xp);
}


/* ------------ Relaxation -------------- */

static void hard_relaxation(struct sim_data* data, int N, int P, float xp[P][N], float st, float end)
{
	float xp2[3] = { 0. };

	data->grad.gb[2] = (data->grad.mom + data->voxel.w); // [rad/s]

	for (int i = 0; i < P; i++) {

		xp2[0] = xp[i][0];
		xp2[1] = xp[i][1];
		xp2[2] = xp[i][2];

		bloch_relaxation(xp[i], end-st, xp2, data->voxel.r1, data->voxel.r2+data->tmp.r2spoil, data->grad.gb);
	}

	data->grad.gb[2] = 0.;
}


static void ode_relaxation(struct sim_data* data, float h, float tol, int N, int P, float xp[P][N], float st, float end)
{
	data->seq.pulse_applied = false;

	data->grad.gb[2] = data->grad.mom; // [rad/s], offresonance w appears in Bloch equation and can be skipped here

        // Choose P-1 because ODE interface treats signal seperat and P only describes the number of parameters
	ode_direct_sa(h, tol, N, P-1, xp, st, end, data, bloch_simu_fun, bloch_pdy2, bloch_pdp2);

	data->grad.gb[2] = 0.;
}


static void relaxation2(struct sim_data* data, float h, float tol, int N, int P, float xp[P][N], float st, float end)
{
	data->seq.pulse_applied = false;

        if (0. == data->pulse.rf_end)
		hard_relaxation(data, N, P, xp, st, end);
	else
		ode_relaxation(data, h, tol, N, P, xp, st, end);
}


/* ------------ Structural Elements -------------- */

static void prepare_sim(struct sim_data* data)
{
        if (0. != data->pulse.rf_end)
        	sinc_pulse_create(&data->pulse, data->pulse.rf_start, data->pulse.rf_end, data->pulse.flipangle, data->pulse.phase, data->pulse.bwtp, data->pulse.alpha);
}


static void run_sim(struct sim_data* data, float* mxy, float* sa_r1, float* sa_r2, float* sa_b1, float h, float tol, int N, int P, float xp[P][N], bool get_signal)
{
	start_rf_pulse(data, h, tol, N, P, xp);

	relaxation2(data, h, tol, N, P, xp, data->pulse.rf_end, data->seq.te);

        if (get_signal)
		collect_signal(data, N, P, mxy, sa_r1, sa_r2, sa_b1, xp);

#if 1	// Smooth spoiling for FLASH sequences
	if (    (FLASH == data->seq.seq_type) ||
                (IRFLASH == data->seq.seq_type))
		data->tmp.r2spoil = 10000.;
#endif

        // Balance z-gradient for bSSFP type sequences
        if ((BSSFP == data->seq.seq_type) || (IRBSSFP == data->seq.seq_type)) {

                relaxation2(data, h, tol, N, P, xp, data->seq.te, data->seq.tr-data->pulse.rf_end);

                data->grad.mom = -data->grad.mom_sl;

                relaxation2(data, h, tol, N, P, xp, data->seq.tr-data->pulse.rf_end, data->seq.tr);

                data->grad.mom = 0.;
        }
        else
        	relaxation2(data, h, tol, N, P, xp, data->seq.te, data->seq.tr);

	data->tmp.r2spoil = 0.;	// effects spoiled sequences only
}


/* ------------ Sequence Specific Blocks -------------- */

static void inversion(struct sim_data* data, float h, float tol, int N, int P, float xp[P][N], float st, float end)
{
	struct sim_data inv_data = *data;

        // Perfect inversion
        for (int p = 0; p < P; p++)
                bloch_excitation2(xp[p], xp[p], M_PI, 0.);

        relaxation2(&inv_data, h, tol, N, P, xp, st, end);
}


static void alpha_half_preparation(struct sim_data* data, float h, float tol, int N, int P, float xp[P][N])
{
	struct sim_data prep_data = *data;

        prep_data.pulse.flipangle = data->pulse.flipangle / 2.;
        prep_data.pulse.phase = M_PI;
        prep_data.seq.te = data->seq.prep_pulse_length;
        prep_data.seq.tr = data->seq.prep_pulse_length;

        assert(prep_data.pulse.rf_end <= prep_data.seq.prep_pulse_length);

        prepare_sim(&prep_data);

        run_sim(&prep_data, NULL, NULL, NULL, NULL, h, tol, N, P, xp, false);
}


/* ------------ Main Simulation -------------- */

void bloch_simulation(struct sim_data* data, float (*mxy_sig)[3], float (*sa_r1_sig)[3], float (*sa_r2_sig)[3], float (*sa_m0_sig)[3], float (*sa_b1_sig)[3])
{
	float tol = 10E-6;      // Tolerance of ODE solver

        enum { N = 3 };         // Number of dimensions (x, y, z)
	enum { P = 4 };         // Number of parameters with estimated derivative (Mxy, R1, R2, B1)


        long storage_size = data->seq.spin_num * data->seq.rep_num * 3 * sizeof(float);

	float* mxy = malloc(storage_size);
	float* sa_r1 = malloc(storage_size);
	float* sa_r2 = malloc(storage_size);
	float* sa_b1 = malloc(storage_size);


	float w_backup = data->voxel.w;
	float zgradient_max = data->grad.mom_sl;

	for (data->tmp.spin_counter = 0; data->tmp.spin_counter < data->seq.spin_num; data->tmp.spin_counter++) {


                // Full Symmetric slice profile
		//      - Calculate slice profile by looping over spins with z-gradient
		if (1 != data->seq.spin_num) {

                        // Ensures central spin on main lope is set
			assert(1 == data->seq.spin_num % 2);

			data->grad.mom_sl = zgradient_max/(data->seq.spin_num-1) * (data->tmp.spin_counter - (int)(data->seq.spin_num/2));
		}

		float xp[P][N] = { { 0., 0., 1. }, { 0. }, { 0. }, { 0. } };

		float h = 0.0001;

		data->voxel.w = w_backup;

		data->pulse.phase = 0;

                data->tmp.t = 0;
                data->tmp.rep_counter = 0;


                // Apply perfect inversion

                if (    (IRBSSFP == data->seq.seq_type) ||
                        (IRFLASH == data->seq.seq_type))
                        inversion(data, h, tol, N, P, xp, 0., data->seq.inversion_pulse_length);


                // Alpha/2 and TR/2 signal preparation

                if (    (BSSFP == data->seq.seq_type) ||
                        (IRBSSFP == data->seq.seq_type))
                        alpha_half_preparation(data, h, tol, N, P, xp);


                prepare_sim(data);


                // Loop over Pulse Blocks

                data->tmp.t = 0;

                while (data->tmp.rep_counter < data->seq.rep_num) {

                        // Change phase of bSSFP sequence in each repetition block
                        if (    (BSSFP == data->seq.seq_type) ||
                                (IRBSSFP == data->seq.seq_type))
                                data->pulse.phase = M_PI * (float)(data->tmp.rep_counter);

                        run_sim(data, mxy, sa_r1, sa_r2, sa_b1, h, tol, N, P, xp, true);

                        data->tmp.rep_counter++;
                }
	}


	// Sum up magnetization

        sum_up_signal(data, mxy, sa_r1, sa_r2, sa_b1, mxy_sig, sa_r1_sig, sa_r2_sig, sa_m0_sig, sa_b1_sig);

	free(mxy);
	free(sa_r1);
	free(sa_r2);
	free(sa_b1);
}