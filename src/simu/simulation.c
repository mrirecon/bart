/* Copyright 2022. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 *	Nick Scholand
 */

#include <complex.h>
#include <math.h>
#include <stdbool.h>

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/linalg.h"
#include "num/ode.h"

#include "simu/bloch.h"
#include "simu/pulse.h"

#include "simulation.h"


void debug_sim(struct sim_data* data)
{
        debug_printf(DP_INFO, "Simulation-Debug-Output\n\n");
        debug_printf(DP_INFO, "Voxel-Parameter:\n");
        debug_printf(DP_INFO, "\tR1:%f\n", data->voxel.r1);
        debug_printf(DP_INFO, "\tR2:%f\n", data->voxel.r2);
        debug_printf(DP_INFO, "\tM0:%f\n", data->voxel.m0);
        debug_printf(DP_INFO, "\tw:%f\n", data->voxel.w);
        debug_printf(DP_INFO, "\tB1:%f\n\n", data->voxel.b1);

        debug_printf(DP_INFO, "Seq-Parameter:\n");
        debug_printf(DP_INFO, "\tSimulation Type:%d\n", data->seq.type);
        debug_printf(DP_INFO, "\tSequence:%d\n", data->seq.seq_type);
        debug_printf(DP_INFO, "\tTR:%f\n", data->seq.tr);
        debug_printf(DP_INFO, "\tTE:%f\n", data->seq.te);
        debug_printf(DP_INFO, "\t#Rep:%d\n", data->seq.rep_num);
        debug_printf(DP_INFO, "\t#Spins:%d\n", data->seq.spin_num);
        debug_printf(DP_INFO, "\tIPL:%f\n", data->seq.inversion_pulse_length);
        debug_printf(DP_INFO, "\tISP:%f\n", data->seq.inversion_spoiler);
        debug_printf(DP_INFO, "\tPPL:%f\n", data->seq.prep_pulse_length);
        debug_printf(DP_INFO, "\tAveraged Spokes:%d\n", data->seq.averaged_spokes);
        debug_printf(DP_INFO, "\tSlice Thickness:%f m\n", data->seq.slice_thickness);
	debug_printf(DP_INFO, "\tNominal Slice Thickness:%f m\n", data->seq.nom_slice_thickness);
        debug_printf(DP_INFO, "\tPulse Applied?:%d\n\n", data->seq.pulse_applied);

        debug_printf(DP_INFO, "Gradient-Parameter:\n");
        debug_printf(DP_INFO, "\tMoment:%f\n", data->grad.mom);
        debug_printf(DP_INFO, "\tSlice-Selection Gradient Strength:%f T/m\n", data->grad.sl_gradient_strength);
        debug_printf(DP_INFO, "\tMoment SL:%f\n\n", data->grad.mom_sl);

        debug_printf(DP_INFO, "Pulse-Parameter:\n");
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

        debug_printf(DP_INFO, "Inversion Pulse-Parameter:\n");
        debug_printf(DP_INFO, "\tA0:%f\n", data->pulse.hs.a0);
        debug_printf(DP_INFO, "\tBeta:%f\n", data->pulse.hs.beta);
        debug_printf(DP_INFO, "\tMu:%f\n", data->pulse.hs.mu);
        debug_printf(DP_INFO, "\tDuration:%f\n", data->pulse.hs.duration);
        debug_printf(DP_INFO, "\tON?:%d\n", data->pulse.hs.on);

        debug_printf(DP_INFO, "Other Parameter:\n");
        debug_printf(DP_INFO, "\tODE Tolerance:%f\n", data->other.ode_tol);
        debug_printf(DP_INFO, "\tPulse Sampling Rate:%f Hz\n", data->other.sampling_rate);
}


const struct simdata_voxel simdata_voxel_defaults = {

	.r1 = 0.,
	.r2 = 0.,
	.m0 = 1.,
	.w = 0.,
	.b1 = 1.,
};


const struct simdata_seq simdata_seq_defaults = {

        .type = SIM_ODE,
	.seq_type = SEQ_BSSFP,
	.tr = 0.004,
	.te = 0.002,
	.rep_num = 1,
	.spin_num = 1,

        .perfect_inversion = false,
	.inversion_pulse_length = 0.01,
        .inversion_spoiler = 0.,

	.prep_pulse_length = 0.002,

        .averaged_spokes = 1,
        .slice_thickness = 0.,
	.nom_slice_thickness = 0.001,

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
        .sl_gradient_strength = 0.,
	.mom = 0.,
	.mom_sl = 0.,
};


const struct simdata_other simdata_other_defaults = {

	.ode_tol = 1e-5,
	.sampling_rate = 1e+6,
};




/* --------- Matrix Operations --------- */


static void vm_mul_transpose(int N, float out[N], const float matrix[N][N], const float in[N])
{
	for (int i = 0; i < N; i++) {

		out[i] = 0.;

		for (int j = 0; j < N; j++)
			out[i] += matrix[j][i] * in[j];
	}
}


static void mm_mul(int N, float out[N][N], const float in1[N][N], const float in2[N][N])
{
	for (int i = 0; i < N; i++) {

		for (int j = 0; j < N; j++) {

			out[i][j] = 0.;

			for (int k = 0; k < N; k++)
				out[i][j] += in1[i][k] * in2[k][j];
		}
	}
}


/* ------------ Bloch Equations -------------- */

static void set_gradients(struct sim_data* data, float t)
{
	if (data->seq.pulse_applied) {

                if (data->pulse.hs.on) {

			// Hyperbolic Secant pulse

                        data->tmp.w1 = pulse_hypsec_am(&data->pulse.hs, t);

                        data->pulse.phase = pulse_hypsec_phase(&data->pulse.hs, t);

                } else {

			// Windowed Sinc pulse

                        data->tmp.w1 = pulse_sinc(&data->pulse, t);
                }

                // Definition from Bernstein et al., Handbook of MRI Pulse Sequences, p. 26f
                // dM/dt = M x (e_x*B_1*sin(phase)-e_y*B_1*sin(phase) +e_z* B_0)) - ...
		data->grad.gb_eff[0] = cosf(data->pulse.phase) * data->tmp.w1 * data->voxel.b1 + data->grad.gb[0];
		data->grad.gb_eff[1] = -sinf(data->pulse.phase) * data->tmp.w1 * data->voxel.b1 + data->grad.gb[1];

	} else {

		data->tmp.w1 = 0.;
		data->grad.gb_eff[0] = data->grad.gb[0];
		data->grad.gb_eff[1] = data->grad.gb[1];
	}

	// Units: [gb] = rad/s
	data->grad.gb_eff[2] = data->grad.gb[2];
}


/* --------- ODE Simulation --------- */

static void bloch_simu_ode_fun(void* _data, float* out, float t, const float* in)
{
        struct sim_data* data = _data;

        set_gradients(data, t);

	bloch_ode(out, in, data->voxel.r1, data->voxel.r2+data->tmp.r2spoil, data->grad.gb_eff);
}


static void bloch_pdy2(void* _data, float* out, float t, const float* in)
{
	struct sim_data* data = _data;
	(void)t;

	bloch_pdy((float(*)[3])out, in, data->voxel.r1, data->voxel.r2 + data->tmp.r2spoil, data->grad.gb_eff);
}


static void bloch_pdp2(void* _data, float* out, float t, const float* in)
{
	struct sim_data* data = _data;
	(void)t;

	bloch_b1_pdp((float(*)[3])out, in, data->voxel.r1, data->voxel.r2 + data->tmp.r2spoil, data->grad.gb_eff, data->pulse.phase, data->tmp.w1);
}


/* ---------  State-Transition Matrix Simulation --------- */


static void bloch_simu_stm_fun(void* _data, float* out, float t, const float* in)
{
        struct ode_matrix_simu_s* ode_data = _data;
	struct sim_data* data = ode_data->sim_data;

        unsigned int N = ode_data->N;

        set_gradients(data, t);

	float matrix_time[N][N];

        if (4 == N) // M
                bloch_matrix_ode(matrix_time, data->voxel.r1, data->voxel.r2+data->tmp.r2spoil, data->grad.gb_eff);

        else if (10 == N) // M, dR1, dR2, dM0
                bloch_matrix_ode_sa(matrix_time, data->voxel.r1, data->voxel.r2+data->tmp.r2spoil, data->grad.gb_eff);

        else if (13 == N) // M, dR1, dR2, dM0, dB1
	        bloch_matrix_ode_sa2(matrix_time, data->voxel.r1, data->voxel.r2+data->tmp.r2spoil, data->grad.gb_eff, data->pulse.phase, data->tmp.w1);

        else
                error("Please choose correct dimension for STM matrix!\n");


        for (unsigned int i = 0; i < N; i++) {

		out[i] = 0.;

		for (unsigned int j = 0; j < N; j++)
			out[i] += matrix_time[i][j] * in[j];
	}
}


void ode_matrix_interval_simu(struct sim_data* _data, float h, float tol, unsigned int N, float out[N], float st, float end)
{
        struct ode_matrix_simu_s data = { N, _data };

	ode_interval(h, tol, N, out, st, end, &data, bloch_simu_stm_fun);
}


void mat_exp_simu(struct sim_data* data, int N, float st, float end, float out[N][N])
{
	assert(end >= st);

	// compute F(t) := T{exp(tA)}
	// F(0) = id
	// d/dt F = A

	float h = (end - st) / 100.;
	float tol = 1.E-6;

	for (int i = 0; i < N; i++) {

		for (int j = 0; j < N; j++)
			out[i][j] = (i == j) ? 1. : 0.;

		ode_matrix_interval_simu(data, h, tol, N, out[i], st, end);
	}
}


static void create_sim_matrix(struct sim_data* data, int N, float matrix[N][N], float st, float end)
{
	if (data->seq.pulse_applied)
		sinc_pulse_init(&data->pulse, data->pulse.rf_start, data->pulse.rf_end, data->pulse.flipangle, data->pulse.phase, data->pulse.bwtp, data->pulse.alpha);

	mat_exp_simu(data, N, st, end, matrix);
}

void apply_sim_matrix(int N, float m[N], float matrix[N][N])
{
	float tmp[N];

	for (int i = 0; i < N; i++)
		tmp[i] = m[i];

	vm_mul_transpose(N, m, matrix, tmp);
}


/* ------------ Read-Out -------------- */

static void adc_corr(int P, float out[P][3], float in[P][3], float angle)
{
	for (int i = 0; i < P; i++)
		rotz(out[i], in[i], angle);
}



static void collect_signal(struct sim_data* data, int P, int R, int S, float (*m)[R][S][3], float (*sa_r1)[R][S][3], float (*sa_r2)[R][S][3], float (*sa_b1)[R][S][3], float xp[P][3])
{
	float tmp[4][3] = { { 0. }, { 0. }, { 0. }, { 0. } };

	adc_corr(P, tmp, xp, -data->pulse.phase);

	for (int i = 0; i < 3; i++) {

		int r = data->tmp.rep_counter;
		int s = data->tmp.spin_counter;

		if (NULL != m)
			(*m)[r][s][i] = tmp[0][i];

		if (NULL != sa_r1)
			(*sa_r1)[r][s][i] = tmp[1][i];

		if (NULL != sa_r2)
			(*sa_r2)[r][s][i] = tmp[2][i];

		if (NULL != sa_b1)
			(*sa_b1)[r][s][i] = tmp[3][i];
	}
}


static void sum_up_signal(float m0, int R, int S, int A, float D, float (*m)[R * A][S][3], float (*sa_r1)[R * A][S][3], float (*sa_r2)[R * A][S][3], float (*sa_b1)[R * A][S][3],
                        float (*m_state)[R][3], float (*sa_r1_state)[R][3], float (*sa_r2_state)[R][3], float (*sa_m0_state)[R][3], float (*sa_b1_state)[R][3])
{
	float norm = m0 / ((float)A * D);

	for (int r = 0; r < R; r++) {

		for (int dim = 0; dim < 3; dim++) {

			float sum_m = 0.;
			float sum_sa_r1 = 0.;
			float sum_sa_r2 = 0.;
			float sum_sa_b1 = 0.;

			for (int a = 0; a < A; a++) {

				for (int spin = 0; spin < S; spin++) {

					sum_m += (*m)[r * A + a][spin][dim];
					sum_sa_r1 += (*sa_r1)[r * A + a][spin][dim];
					sum_sa_r2 += (*sa_r2)[r * A + a][spin][dim];
					sum_sa_b1 += (*sa_b1)[r * A + a][spin][dim];
				}
			}

                        // Mean
                        (*m_state)[r][dim] = sum_m * norm;
                        (*sa_r1_state)[r][dim] = sum_sa_r1 * norm;
                        (*sa_r2_state)[r][dim] = sum_sa_r2 * norm;
                        (*sa_b1_state)[r][dim] = sum_sa_b1 * norm;
                        (*sa_m0_state)[r][dim] = sum_m / ((float)A * D);
		}
	}
}

/* ------------ RF-Pulse -------------- */

// Single hard pulse without discrete sampling
static void hard_pulse(struct sim_data* data, int N, int P, float xp[P][N])
{
        for (int i = 0; i < P; i++)
                bloch_excitation2(xp[i], xp[i], data->pulse.flipangle / 180. * M_PI, data->pulse.phase);
}


// Homogeneously discretized pulse with rotational matrices
static void rot_pulse(struct sim_data* data, int N, int P, float xp[P][N])
{
        if (0. == data->pulse.rf_end) {

                hard_pulse(data, N, P, xp);

	} else {

                assert(0. < data->other.sampling_rate);

                float sample_time = 1. / data->other.sampling_rate;

                assert((data->pulse.rf_end-data->pulse.rf_start) > sample_time);

                float t_im = data->pulse.rf_start + sample_time / 2.;

                float xp2[3] = { 0. };
                float xp3[3] = { 0. };

                float w1 = 0;

                while (t_im <= data->pulse.rf_end) {

                        // RF-pulse strength of current interval

                        w1 = pulse_sinc(&data->pulse, t_im);

                        for (int i = 0; i < P; i++) {

                                xp2[0] = xp[i][0];
                                xp2[1] = xp[i][1];
                                xp2[2] = xp[i][2];

                                bloch_excitation2(xp3, xp2, w1 * sample_time, data->pulse.phase);

                                bloch_relaxation(xp[i], sample_time, xp3, data->voxel.r1, data->voxel.r2, data->grad.gb);
                        }

                        t_im += sample_time;
                }
        }
}


void rf_pulse(struct sim_data* data, float h, float tol, int N, int P, float xp[P][N], float stm_matrix[P * N][P * N])
{
	data->seq.pulse_applied = true;

        // Single hard pulse is special case of homogeneously sampled sinc pulse
        if (0. == data->pulse.rf_end)
                data->seq.type = SIM_ROT;

        // Define effective z Gradient = Slice-selection gradient + off-resonance [rad/s]
	data->grad.gb[2] = data->grad.mom_sl + data->voxel.w;

        switch (data->seq.type) {

        case SIM_ROT:

                rot_pulse(data, N, P, xp);
                break;

        case SIM_ODE:

                // Choose P-1 because ODE interface treats signal separate and P only describes the number of parameters
		ode_direct_sa(h, tol, N, P - 1, xp, data->pulse.rf_start, data->pulse.rf_end, data,  bloch_simu_ode_fun, bloch_pdy2, bloch_pdp2);
                break;

        case SIM_STM:

                create_sim_matrix(data, P * N, stm_matrix, data->pulse.rf_start, data->pulse.rf_end);
                break;
        }

        data->grad.gb[2] = 0.;
}


/* ------------ Relaxation -------------- */

static void hard_relaxation(struct sim_data* data, int N, int P, float xp[P][N], float st, float end)
{
	float xp2[3] = { 0. };

	for (int i = 0; i < P; i++) {

		xp2[0] = xp[i][0];
		xp2[1] = xp[i][1];
		xp2[2] = xp[i][2];

		bloch_relaxation(xp[i], end - st, xp2, data->voxel.r1, data->voxel.r2 + data->tmp.r2spoil, data->grad.gb);
	}
}


static void relaxation2(struct sim_data* data, float h, float tol, int N, int P, float xp[P][N], float st, float end, float stm_matrix[P * N][P * N])
{
	data->seq.pulse_applied = false;

        // Single hard pulse is special case of homogeneously sampled sinc pulse
        if (0. == data->pulse.rf_end)
                data->seq.type = SIM_ROT;

        // Define effective z Gradient =Gradient Moments + off-resonance [rad/s]
        data->grad.gb[2] = data->grad.mom + data->voxel.w;

        switch (data->seq.type) {

        case SIM_ROT:

                hard_relaxation(data, N, P, xp, st, end);
                break;

        case SIM_ODE:

                // Choose P-1 because ODE interface treats signal separate and P only describes the number of parameters
                ode_direct_sa(h, tol, N, P - 1, xp, st, end, data, bloch_simu_ode_fun, bloch_pdy2, bloch_pdp2);
                break;

        case SIM_STM:

                create_sim_matrix(data, P * N, stm_matrix, st, end);
                break;
        }

        data->grad.gb[2] = 0.;
}


/* ------------ Conversion ODE -> STM -------------- */


static void stm2ode(int N, int P, float out[P][N], float in[P * N + 1])
{
        for (int p = 0; p < P; p++)
                for(int n = 0; n < N; n++)
                        out[p][n] = in[p * N + n];
}

static void ode2stm(int N, int P, float out[P * N + 1], float in[P][N])
{
        for (int p = 0; p < P; p++)
                for(int n = 0; n < N; n++)
                        out[p * N + n] = in[p][n];

        out[P * N] = 1.;
}


/* ------------ Structural Elements -------------- */

static void prepare_sim(struct sim_data* data, int N, int P, float (*mte)[P * N + 1][P * N + 1], float (*mtr)[P * N + 1][P * N + 1])
{
        switch (data->seq.type) {

        case SIM_ROT:
        case SIM_ODE:

                if (0. != data->pulse.rf_end)
			sinc_pulse_init(&data->pulse, data->pulse.rf_start, data->pulse.rf_end, data->pulse.flipangle, data->pulse.phase, data->pulse.bwtp, data->pulse.alpha);

                break;

        case SIM_STM: ;

                int M = P * N + 1;

                // Matrix: 0 -> T_RF
                float mrf[M][M];
                rf_pulse(data, 0., 0., M, 1, NULL, mrf);

                // Matrix: T_RF -> TE
                float mrel[M][M];


                if (0 != data->grad.mom_sl) {

			float rewind_end = 1.5 * data->pulse.rf_end; // FIXME: Why?

                        if (data->seq.te < rewind_end) {

				rewind_end = data->pulse.rf_end;

				debug_printf(DP_WARN, "Slice-Selection Gradient rewinder does not fit between RF_end and TE!\n");

				relaxation2(data, 0., 0., M, 1, NULL, rewind_end, data->seq.te, mrel);

                        } else {
                                // Slice-Rewinder
				float tmp[M][M];
				float tmp2[M][M];

                                data->grad.mom = -data->grad.mom_sl;
                                relaxation2(data, 0, 0, M, 1, NULL, data->pulse.rf_end, rewind_end, tmp);
                                data->grad.mom = 0.; // [rad/s]

                                relaxation2(data, 0, 0, M, 1, NULL, rewind_end, data->seq.te, tmp2);

                                mm_mul(M, mrel, tmp, tmp2);
			}

		} else {

	                relaxation2(data, 0., 0., M, 1, NULL, data->pulse.rf_end, data->seq.te, mrel);
		}

                // Join matrices: 0 -> TE
		mm_mul(M, *mte, mrf, mrel);

                // Smooth spoiling for FLASH sequences

                if (NULL != mtr) {

                        if (   (SEQ_FLASH == data->seq.seq_type)
                            || (SEQ_IRFLASH == data->seq.seq_type)) {

                                data->tmp.r2spoil = 10000.;
			}

                        // Balance z-gradient for bSSFP type sequences

                        if (   (SEQ_BSSFP == data->seq.seq_type)
                            || (SEQ_IRBSSFP == data->seq.seq_type)) {

				float tmp[M][M];
		                float tmp2[M][M];

                                // Matrix: TE -> TR-1/2*T_RF
                                relaxation2(data, 0., 0., M, 1, NULL, data->seq.te, data->seq.tr-0.5*data->pulse.rf_end, tmp);

                                // Matrix: TR-1/2*T_RF -> TR
                                data->grad.mom = -data->grad.mom_sl;
                                relaxation2(data, 0., 0., M, 1, NULL, data->seq.tr-0.5*data->pulse.rf_end, data->seq.tr, tmp2);
                                data->grad.mom = 0.;

                                // Join matrices: TE -> TR
                                mm_mul(M, *mtr, tmp, tmp2);

                        } else {

                                relaxation2(data, 0., 0., M, 1, NULL, data->seq.te, data->seq.tr, *mtr);
                        }

                        data->tmp.r2spoil = 0.;	// effects spoiled sequences only
                }

                break;
        }
}


static void run_sim(struct sim_data* data, int R, int S, float (*mxy)[R][S][3], float (*sa_r1)[R][S][3], float (*sa_r2)[R][S][3], float (*sa_b1)[R][S][3],
                        float h, float tol, int N, int P, float xp[P][N],
                        float xstm[P * N + 1], float mte[P * N + 1][P * N + 1], float mtr[P * N + 1][P * N + 1])
{
        switch (data->seq.type) {

        case SIM_ROT:
        case SIM_ODE:

                rf_pulse(data, h, tol, N, P, xp, NULL);

                // Slice-Rewinder if time is long enough

		float rewind_end = data->pulse.rf_end;

                if (0 != data->grad.mom_sl) {

			rewind_end = 1.5 * data->pulse.rf_end; // FIXME: Why?

                        if (data->seq.te < rewind_end) {

                                debug_printf(DP_WARN, "Slice-Selection Gradient rewinder does not fit between RF_end and TE!\n");

				rewind_end = data->pulse.rf_end;

                        } else {

                                data->grad.mom = -data->grad.mom_sl;
                                relaxation2(data, h, tol, N, P, xp, data->pulse.rf_end, rewind_end, NULL);
                                data->grad.mom = 0.; // [rad/s]
			}
		}

		relaxation2(data, h, tol, N, P, xp, rewind_end, data->seq.te, NULL);


		collect_signal(data, P, R, S, mxy, sa_r1, sa_r2, sa_b1, xp);


                // Smooth spoiling for FLASH sequences

                if (   (SEQ_FLASH == data->seq.seq_type)
                    || (SEQ_IRFLASH == data->seq.seq_type)) {

                        data->tmp.r2spoil = 10000.;
		}


                // Balance z-gradient for bSSFP type sequences

                if (   (SEQ_BSSFP == data->seq.seq_type)
                    || (SEQ_IRBSSFP == data->seq.seq_type)) {

                        relaxation2(data, h, tol, N, P, xp, data->seq.te, data->seq.tr - 0.5 * data->pulse.rf_end, NULL);

                        data->grad.mom = -data->grad.mom_sl;
                        relaxation2(data, h, tol, N, P, xp, data->seq.tr - 0.5 * data->pulse.rf_end, data->seq.tr, NULL);
                        data->grad.mom = 0.;

                } else {

                        relaxation2(data, h, tol, N, P, xp, data->seq.te, data->seq.tr, NULL);
                }

                data->tmp.r2spoil = 0.;	// effects spoiled sequences only

                break;

        case SIM_STM:

                // Evolution: 0 -> TE
                apply_sim_matrix(N * P + 1, xstm, mte);

                // Save data
                stm2ode(N, P, xp, xstm);

                collect_signal(data, P, R, S, mxy, sa_r1, sa_r2, sa_b1, xp);

                // Evolution: TE -> TR
                apply_sim_matrix(N * P + 1, xstm, mtr);

                break;
        }
}


/* ------------ Sequence Specific Blocks -------------- */

void inversion(const struct sim_data* data, float h, float tol, int N, int P, float xp[P][N], float st, float end)
{
	struct sim_data inv_data = *data;

        // Non Slice-Selective as default
        // FIXME: Add slice-selective inversion too
        inv_data.grad.mom_sl = 0.;

        // Enforce ODE: Way more efficient here!
        inv_data.seq.type = SIM_ODE;

        if (data->seq.perfect_inversion) {

                // Apply perfect inversion

                for (int p = 0; p < P; p++)
                        bloch_excitation2(xp[p], xp[p], M_PI, 0.);

                relaxation2(&inv_data, h, tol, N, P, xp, st, end, NULL);

        } else {
                // Hyperbolic Secant inversion

                inv_data.pulse.hs = hs_pulse_defaults;
                inv_data.pulse.hs.on = true;
                inv_data.pulse.hs.duration = data->seq.inversion_pulse_length;
                inv_data.pulse.rf_end = data->seq.inversion_pulse_length;

                rf_pulse(&inv_data, h, tol, N, P, xp, NULL);

                // Spoiler gradients
                inv_data.tmp.r2spoil = 10000.;
                relaxation2(&inv_data, h, tol, N, P, xp, st, end, NULL);
        }
}


static void alpha_half_preparation(const struct sim_data* data, float h, float tol, int N, int P, float xp[P][N])
{
	struct sim_data prep_data = *data;

        // Enforce ODE: Way more efficient here!
        prep_data.seq.type = SIM_ODE;

        prep_data.pulse.flipangle = data->pulse.flipangle / 2.;
        prep_data.pulse.phase = M_PI;
        prep_data.seq.te = 1.5 * prep_data.pulse.rf_end;
        prep_data.seq.tr = data->seq.prep_pulse_length;

        // Ensure enough time for slice-seelction gradient rephaser
        assert(2 * prep_data.pulse.rf_end <= prep_data.seq.prep_pulse_length);

        prepare_sim(&prep_data, N, P, NULL, NULL);

	int R = data->seq.rep_num;
	int S = data->seq.spin_num;

        run_sim(&prep_data, R, S, NULL, NULL, NULL, NULL, h, tol, N, P, xp, NULL, NULL, NULL);
}


/* ------------ Main Simulation -------------- */

void bloch_simulation(const struct sim_data* _data, int R, float (*m_state)[R][3], float (*sa_r1_state)[R][3], float (*sa_r2_state)[R][3], float (*sa_m0_state)[R][3], float (*sa_b1_state)[R][3])
{
	// FIXME: split config + variable part

	struct sim_data data = *_data;  // Lose information of complex pointer variables

        float tol = _data->other.ode_tol;      // Tolerance of ODE solver

        enum { N = 3 };         // Number of dimensions (x, y, z)
	enum { P = 4 };         // Number of parameters with estimated derivative (M, DR1, DR2, DB1)

        assert(0 < P);

        enum { M = N * P + 1 };     // STM based on single vector and additional +1 for linearized system matrix

	assert(R == data.seq.rep_num);

	int A = data.seq.averaged_spokes;

        // Unit: [M0] = 1 Magnetization / mm
        //      -> Default slice thickness of a single isochromat set to 1 mm
        //              FIXME: Slice Profile in relative units of theoretical slice thickness?
        // debug_printf(DP_INFO, "Theoretical Slice Thickness: %f\n", 2 * M_PI / ((data.pulse.rf_end - data.pulse.rf_start) / (2. + (data.pulse.nl - 1.) + (data.pulse.nr - 1.)) * data.grad.sl_gradient_strength * GAMMA_H1));
        float default_slice_thickness = 0.001; // [m]

	data.seq.rep_num *= A;
	data.seq.averaged_spokes = 1;

	int S = data.seq.spin_num;

	float (*mxy)[R * A][S][3] = xmalloc(sizeof *mxy);
	float (*sa_r1)[R * A][S][3] = xmalloc(sizeof *sa_r1);
	float (*sa_r2)[R * A][S][3] = xmalloc(sizeof *sa_r2);
	float (*sa_b1)[R * A][S][3] = xmalloc(sizeof *sa_b1);

	for (data.tmp.spin_counter = 0; data.tmp.spin_counter < S; data.tmp.spin_counter++) {

                float h = 0.0001;


		if (1 != S) {

                        // Calculate slice profile by looping over spins with z-gradient
                        //      Full Symmetric slice profile

                        // Ensures central spin on main lope is set
			assert(1 == S % 2);
                        assert(0. != _data->seq.slice_thickness);

			data.grad.mom_sl = (_data->grad.sl_gradient_strength * _data->seq.slice_thickness * GAMMA_H1) / (S - 1) * (data.tmp.spin_counter - (int)(S / 2.));

                } else {

                        data.seq.slice_thickness = default_slice_thickness;

                        // Define z-Position with slice thickness for single spin
                        data.grad.mom_sl = _data->grad.sl_gradient_strength * _data->seq.slice_thickness * GAMMA_H1;
                }

                data.pulse.flipangle = _data->pulse.flipangle;

                // Initialize ODE
		float xp[P][N];

                for (int p = 0; p < P; p++)
                        for (int n = 0; n < N; n++)
                                xp[p][n] = 0.;

                xp[0][2] = 1.;

                // Initialize STM
                float xstm[M] = { 0. };

                xstm[2] = 1.;
                xstm[M - 1] = 1.;


                // Reset parameters
		data.voxel.w = _data->voxel.w;
		data.pulse.phase = 0;
                data.tmp.t = 0;
                data.tmp.rep_counter = 0;

                // Apply perfect inversion

                if (   (SEQ_IRBSSFP == data.seq.seq_type)
		    || (SEQ_IRFLASH == data.seq.seq_type))
                        inversion(&data, h, tol, N, P, xp, 0., data.seq.inversion_spoiler);

                // Alpha/2 and TR/2 signal preparation

                if (   (SEQ_BSSFP == data.seq.seq_type)
                    || (SEQ_IRBSSFP == data.seq.seq_type))
                        alpha_half_preparation(&data, h, tol, N, P, xp);

                float mtr[M][M];
                float mte[2][M][M];

                ode2stm(N, P, xstm, xp);

                // STM requires two matrices for RFPhase=0 and RFPhase=PI
                // Therefore mte and mte2 need to be estimated
		//
                if (   (SEQ_BSSFP == data.seq.seq_type)
                    || (SEQ_IRBSSFP == data.seq.seq_type)) {

                        data.pulse.phase = M_PI;
                        prepare_sim(&data, N, P, &mte[1], NULL);
                        data.pulse.phase = 0.;
                }

		prepare_sim(&data, N, P, &mte[0], &mtr);


                // Loop over Pulse Blocks

                data.tmp.t = 0;

                while (data.tmp.rep_counter < data.seq.rep_num) {

			bool odd = false;

                        // Change phase of bSSFP sequence in each repetition block
                        if (   (SEQ_BSSFP == data.seq.seq_type)
                            || (SEQ_IRBSSFP == data.seq.seq_type)) {

                                data.pulse.phase = M_PI * (float)(data.tmp.rep_counter);

				odd = (1 == data.tmp.rep_counter % 2);
                        }

			run_sim(&data, R * A, S, mxy, sa_r1, sa_r2, sa_b1, h, tol, N, P, xp, xstm, mte[odd], mtr);

                        data.tmp.rep_counter++;
                }
	}


	// Sum up magnetization

        // Scale signal with density
        //      - Relative to default slice thickness to keep strength of simulation higher
        float D = (float)S / (data.seq.slice_thickness / default_slice_thickness);

        sum_up_signal(data.voxel.m0, data.seq.rep_num / A, data.seq.spin_num, A, D, mxy, sa_r1, sa_r2, sa_b1, m_state, sa_r1_state, sa_r2_state, sa_m0_state, sa_b1_state);

	xfree(mxy);
	xfree(sa_r1);
	xfree(sa_r2);
	xfree(sa_b1);
}


