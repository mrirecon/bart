/* Copyright 2022. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 *	Nick Scholand
 *	Martin Juschitz
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
#include "num/matexp.h"
#include "num/ode.h"

#include "simu/bloch.h"
#include "simu/pulse.h"

#include "simulation.h"


void debug_sim(struct sim_data* data)
{
        debug_printf(DP_INFO, "Simulation-Debug-Output\n\n");
        debug_printf(DP_INFO, "Voxel-Parameter:\n");
        debug_printf(DP_INFO, "\tNumber of Pools:%d\n\n", data->voxel.P);
        debug_printf(DP_INFO, "\tR1:%f\n\n", data->voxel.r1[0]);
        debug_printf(DP_INFO, "\tR2:%f\n\n", data->voxel.r2[0]);
        debug_printf(DP_INFO, "\tM0:%f\n", data->voxel.m0[0]);
        debug_printf(DP_INFO, "\tOm:%f\n\n", data->voxel.Om[0]);
        debug_printf(DP_INFO, "\tR2_2:%f \tR2_3:%f\t R2_4:%f\t R2_5:%f\n\n", data->voxel.r2[1], data->voxel.r2[2],data->voxel.r2[3],data->voxel.r2[4]);
        debug_printf(DP_INFO, "\tR1_2:%f \tR1_3:%f\t R1_4:%f\t R1_5:%f\n\n", data->voxel.r1[1], data->voxel.r1[2],data->voxel.r1[3],data->voxel.r1[4]);
        debug_printf(DP_INFO, "\tM0_2:%f \tM0_3:%f \tM0_4:%f \tM0_5:%f\n", data->voxel.m0[1], data->voxel.m0[2], data->voxel.m0[3], data->voxel.m0[4]);
        debug_printf(DP_INFO, "\tOm_2:%f \tOm_3:%f \tOm_4:%f \tOm_5:%f\n", data->voxel.Om[1], data->voxel.Om[2], data->voxel.Om[3], data->voxel.Om[4]);
        debug_printf(DP_INFO, "\tk[0]:%f\n\n", data->voxel.k[0]);
        debug_printf(DP_INFO, "\tk[1]:%f\n\n", data->voxel.k[1]);
        debug_printf(DP_INFO, "\tk[2]:%f\n\n", data->voxel.k[2]);
        debug_printf(DP_INFO, "\tk[3]:%f\n\n", data->voxel.k[3]);
        debug_printf(DP_INFO, "\tw:%f\n", data->voxel.w);
        debug_printf(DP_INFO, "\tB1:%f\n\n", data->voxel.b1);

        debug_printf(DP_INFO, "Seq-Parameter:\n");
        debug_printf(DP_INFO, "\tSimulation Type:%d\n", data->seq.type);
        debug_printf(DP_INFO, "\tSequence:%d\n", data->seq.seq_type);
        debug_printf(DP_INFO, "\tModel:%d\n\n", data->seq.model);
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
        debug_printf(DP_INFO, "\tFlipangle:%f\n", CAST_UP(&data->pulse.sinc)->flipangle);
        debug_printf(DP_INFO, "\tPhase:%f\n", data->pulse.phase);
        debug_printf(DP_INFO, "\tBWTP:%f\n", data->pulse.sinc.bwtp);
        debug_printf(DP_INFO, "\tAlpha:%f\n", data->pulse.sinc.alpha);
        debug_printf(DP_INFO, "\tA:%f\n\n", data->pulse.sinc.A);

        debug_printf(DP_INFO, "Inversion Pulse-Parameter:\n");
        debug_printf(DP_INFO, "\tA0:%f\n", data->pulse.hs.a0);
        debug_printf(DP_INFO, "\tBeta:%f\n", data->pulse.hs.beta);
        debug_printf(DP_INFO, "\tMu:%f\n", data->pulse.hs.mu);
        debug_printf(DP_INFO, "\tDuration:%f\n", CAST_UP(&data->pulse.hs)->duration);

        debug_printf(DP_INFO, "Other Parameter:\n");
        debug_printf(DP_INFO, "\tODE Tolerance:%f\n", data->other.ode_tol);
        debug_printf(DP_INFO, "\tPulse Sampling Rate:%f Hz\n", data->other.sampling_rate);
}


const struct simdata_voxel simdata_voxel_defaults = {

	.P = 1,

	.r1 = { 0., 0., 0., 0., 0. },
	.r2 = { 0., 0., 0., 0., 0. },
	.m0 = { 1., 1., 1., 1., 1. },
	.Om = { 0., 0., 0., 0., 0. },

	.k = { 0., 0., 0., 0. },

	.w = 0.,
	.b1 = 1.,
};


const struct simdata_seq simdata_seq_defaults = {

        .type = SIM_ODE,
	.seq_type = SEQ_BSSFP,
	.model = MODEL_BLOCH,

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



const struct simdata_grad simdata_grad_defaults = {

	.gb = { 0., 0., 0. },
        .sl_gradient_strength = 0.,
	.mom = 0.,
	.mom_sl = 0.,
};


const struct simdata_other simdata_other_defaults = {

	.ode_tol = 1e-5,
	.stm_tol = 1e-6,
	.sampling_rate = 1e+6,
};

const struct simdata_cest simdata_cest_defaults = {

	.n_pulses = 1,

	.t_d = 0.,
	.t_pp = 0.0065,
	.gamma = 42.5764,
	.b1_amp = 1,
	.b0 = 3.,

	.off_start = 5.,
	.off_stop = -5.,

	.ref_scan = false,
	.ref_scan_ppm = -300.,
};


/* pulses */

const struct simdata_pulse simdata_pulse_defaults = {

	.type = PULSE_SINC,

	.rf_start = 0.,
	.rf_end = 0.001,
};

void pulse_init(struct simdata_pulse* pulse, float rf_start, float rf_end, float angle /*[deg]*/, float phase, float bwtp, float alpha)
{
	pulse->rf_start = rf_start;	// [s]
	pulse->rf_end = rf_end;		// [s]

	pulse->sinc = pulse_sinc_defaults;
	pulse_sinc_init(&pulse->sinc, rf_end - rf_start, angle, phase, bwtp, alpha);
}


/* ------------ Bloch Equations -------------- */

static void compute_fields(struct sim_data* data, float gb_eff[3], float t)
{
	// Units: [gb] = rad/s
	gb_eff[0] = data->grad.gb[0];
	gb_eff[1] = data->grad.gb[1];
	gb_eff[2] = data->grad.gb[2];

	complex float w1 = 0.;

	if (data->seq.pulse_applied) {

		struct pulse* ps = NULL;

		switch (data->pulse.type) {

		case PULSE_SINC:
			ps = CAST_UP(&data->pulse.sinc);
			break;

		case PULSE_HS:
			ps = CAST_UP(&data->pulse.hs);
			break;

		case PULSE_REC:
			ps = CAST_UP(&data->pulse.rect);
			break;
		}

		w1 = cexpf(1.i * data->pulse.phase) * pulse_eval(ps, t);

                // Definition from Bernstein et al., Handbook of MRI Pulse Sequences, p. 26f
                // dM/dt = M x (e_x*B_1*sin(phase)-e_y*B_1*sin(phase) +e_z* B_0)) - ...
		gb_eff[0] = crealf(w1);
		gb_eff[1] = -cimagf(w1);
	}
}


/* ---------  State-Transition Matrix Simulation --------- */


static void bloch_simu_stm_fun(struct sim_data* data, float r2spoil, int N, float* out, float t, const float* in)
{
	float gb_eff[3];
	compute_fields(data, gb_eff, t);

	complex float w1 = gb_eff[0] - 1.i * gb_eff[1];

	gb_eff[0] *= data->voxel.b1;
	gb_eff[1] *= data->voxel.b1;

	float matrix_time[N][N];
	int N_pools_b1 = 15 * data->voxel.P * data->voxel.P + 1;
	int N_pools = N_pools_b1 - 3 * data->voxel.P;

	float r2[data->voxel.P];

	for (int i = 0; i < data->voxel.P; i++)
		r2[i] = data->voxel.r2[i] + r2spoil;

	if (N == 4) { // M

		bloch_matrix_ode(matrix_time, data->voxel.r1[0], r2[0], gb_eff);

	} else if (N == 10) { // M, dR1, dR2, dM0

		bloch_matrix_ode_sa(matrix_time, data->voxel.r1[0], r2[0], gb_eff);

	} else if (N == 13) { // M, dR1, dR2, dM0, dB1

		bloch_matrix_ode_sa2(matrix_time, data->voxel.r1[0], r2[0], gb_eff, w1);

	} else if (N == N_pools) { // M,  dR1, dR2, dk, dOm, dM0

		assert(MODEL_BMC == data->seq.model);
		bloch_mcc_matrix_ode_sa(data->voxel.P, matrix_time, data->voxel.r1, r2, data->voxel.k, data->voxel.m0, data->voxel.Om, gb_eff);

	} else if (N == N_pools_b1) { // M, dR1, dR2, dB1, dk, dOm, dM0

		assert(MODEL_BMC == data->seq.model);

		bloch_mcc_matrix_ode_sa2(data->voxel.P, matrix_time, data->voxel.r1, r2, data->voxel.k, data->voxel.m0, data->voxel.Om, gb_eff, w1);

	} else {

		assert(0);
	}

	matf_vecmul(N, N, out, matrix_time, in);
}

void mat_exp_simu(struct sim_data* data, float r2spoil, int N, float st, float end, float out[N][N])
{
	NESTED(void, call, (float* out, float t, const float* in))
	{
		bloch_simu_stm_fun(data, r2spoil, N, out, t, in);
	};

	mat_to_exp(N, st, end, out, data->other.stm_tol, call);
}

static void create_sim_matrix(struct sim_data* data, int N, float matrix[N][N], float st, float end, float r2spoil)
{
	if (data->seq.pulse_applied)
		pulse_init(&data->pulse, data->pulse.rf_start, data->pulse.rf_end,
				CAST_UP(&data->pulse.sinc)->flipangle, data->pulse.phase,
				data->pulse.sinc.bwtp, data->pulse.sinc.alpha);

	mat_exp_simu(data, r2spoil, N, st, end, matrix);
}


void apply_sim_matrix(int N, float m[N], float matrix[N][N])
{
	float tmp[N];
	vecf_copy(N, tmp, m);

	float mT[N][N];
	matf_transpose(N, N, mT, matrix);
	matf_vecmul(N, N, m, mT, tmp);
}


/* ------------ Read-Out -------------- */

static void adc_corr(int P, int pools, float out[P][pools][3], const float in[P][3 * pools], float angle)
{
	for (int i = 0; i < P; i++) {
		for (int p = 0; p < pools; p++) {

			float in2[3];

			for (int j = 0; j < 3; j++)
				in2[j] = in[i][p * 3 + j];

			rotz(out[i][p], in2, angle);
		}
	}
}



static void collect_signal(struct sim_data* data, int P, int pools, float (*m)[pools][3], float (*sa_r1)[pools][3], float (*sa_r2)[pools][3], float (*sa_b1)[1][3], float (*sa_m0)[pools][3], float (*sa_k)[pools][3], float (*sa_Om)[pools][3], float xp[P][pools * 3])
{
        float tmp[P][pools][3];

        adc_corr(P, pools, tmp, xp, -data->pulse.phase);

	// Keep all entries for m
	// Only keep x,y,z components with respect to water pool for SA params
	for (int p = 0; p < pools; p++) {
		for (int i = 0; i < 3; i++) {

			if (NULL != m)
				(*m)[p][i] = tmp[0][p][i];

			if (NULL != sa_r1)
				(*sa_r1)[p][i] = tmp[1 + p][0][i];

			if (NULL != sa_r2)
				(*sa_r2)[p][i] = tmp[1 + pools + p][0][i];

			if ( (0 == p) && (NULL != sa_b1) )
				(*sa_b1)[p][i] = tmp[1 + 2 * pools][p][i];

			if ( (1 < pools) && (NULL != sa_m0) ) // For 1 == pools -> sa_m0 == m, see sum_up_signal()
				(*sa_m0)[p][i] = tmp[2 + 2 * pools + p][0][i];

			// pools - 1 instances of k and Om
			if (p < pools - 1) {

				if (NULL != sa_k)
					(*sa_k)[p][i] = tmp[2 + 3 * pools + p][0][i];

				if (NULL != sa_Om)
					(*sa_Om)[p][i] = tmp[1 + 4 * pools + p][0][i];
			}
		}
	}
}


static void sum_up_signal(float m0, int R, int S, int A, float D, int pools,
			float (*m)[R * A][S][pools][3], float (*sa_r1)[R * A][S][pools][3], float (*sa_r2)[R * A][S][pools][3], float (*sa_b1)[R * A][S][1][3],
			float (*sa_m0)[R * A][S][pools][3], float (*sa_k)[R * A][S][pools][3], float (*sa_om)[R * A][S][pools][3],
                        float (*m_state)[R][pools][3], float (*sa_r1_state)[R][pools][3], float (*sa_r2_state)[R][pools][3], float (*sa_m0_state)[R][pools][3], float (*sa_b1_state)[R][1][3], float (*sa_k_state)[R][pools][3], float (*sa_om_state)[R][pools][3])
{
	float norm = m0 / ((float)A * D);

        if (pools > 1)
                norm = 1. / ((float)A * D);

	for (int p = 0; p < pools; p++) {

		for (int r = 0; r < R; r++) {

			for (int dim = 0; dim < 3; dim++) {

				float sum_m = 0.;
				float sum_sa_r1 = 0.;
				float sum_sa_r2 = 0.;
				float sum_sa_b1 = 0.;
				float sum_sa_m0 = 0.;
				float sum_sa_k = 0.;
				float sum_sa_om = 0.;

				for (int a = 0; a < A; a++) {

					for (int spin = 0; spin < S; spin++) {

						sum_m += (*m)[r * A + a][spin][p][dim];
						sum_sa_r1 += (*sa_r1)[r * A + a][spin][p][dim];
						sum_sa_r2 += (*sa_r2)[r * A + a][spin][p][dim];

						if (p == 0)
							sum_sa_b1 += (*sa_b1)[r * A + a][spin][p][dim];

						sum_sa_k += (*sa_k)[r * A + a][spin][p][dim];
						sum_sa_om += (*sa_om)[r * A + a][spin][p][dim];
						sum_sa_m0 += (*sa_m0)[r * A + a][spin][p][dim];
					}
				}

				// Mean
				(*m_state)[r][p][dim] = sum_m * norm;
				(*sa_r1_state)[r][p][dim] = sum_sa_r1 * norm;
				(*sa_r2_state)[r][p][dim] = sum_sa_r2 * norm;

				if (0 == p)
					(*sa_b1_state)[r][p][dim] = sum_sa_b1 * norm;

				if (1 == pools)
					(*sa_m0_state)[r][p][dim] = sum_m / ((float)A * D);
				else
					(*sa_m0_state)[r][p][dim] = sum_sa_m0 * norm;

				if (NULL != sa_k_state)
					(*sa_k_state)[r][p][dim] = sum_sa_k * norm;

				if (NULL != sa_om_state)
					(*sa_om_state)[r][p][dim] = sum_sa_om * norm;
			}
		}
	}
}

/* ------------ RF-Pulse -------------- */

// Single hard pulse without discrete sampling
static void hard_pulse(struct sim_data* data, int N, int P, float xp[P][N])
{
        for (int i = 0; i < P; i++)
                bloch_excitation2(xp[i], xp[i], DEG2RAD(CAST_UP(&data->pulse.sinc)->flipangle), data->pulse.phase);
}


// Homogeneously discretized pulse with rotational matrices
static void rot_pulse(struct sim_data* data, int N, int P, float xp[P][N])
{
        if (0. == data->pulse.rf_end) {

                hard_pulse(data, N, P, xp);

	} else {

                assert(0. < data->other.sampling_rate);

                float sample_time = 1. / data->other.sampling_rate;

                assert((data->pulse.rf_end - data->pulse.rf_start) > sample_time);

                float t_im = data->pulse.rf_start + sample_time / 2.;

                for (; t_im <= data->pulse.rf_end; t_im += sample_time) {

                        // RF-pulse strength of current interval

                        float w1 = crealf(pulse_eval(CAST_UP(&data->pulse.sinc), t_im));

                        for (int i = 0; i < P; i++) {

				float xp3[3] = { 0. };

                                bloch_excitation2(xp3, xp[i], w1 * sample_time, data->pulse.phase);

                                bloch_relaxation(xp[i], sample_time, xp3, data->voxel.r1[0], data->voxel.r2[0], data->grad.gb);
                        }
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

	__block complex float w1;	// clang workaround (needs to be outside switch)
					//
        switch (data->seq.type) {

        case SIM_ROT:

                rot_pulse(data, N, P, xp);
                break;

        case SIM_ODE: {

		float gb_eff[3];
		void *gb_eff_p = gb_eff;	// clang workaround

		NESTED(void, call_fun, (float* out, float t, const float* in))
		{
			float *gb_eff = gb_eff_p;
			compute_fields(data, gb_eff, t);

			w1 = gb_eff[0] - 1.i * gb_eff[1];

			gb_eff[0] *= data->voxel.b1;
			gb_eff[1] *= data->voxel.b1;


			float r2[data->voxel.P];

			if (MODEL_BMC == data->seq.model) {

				for (int i = 0; i < data->voxel.P; i++)
					r2[i] = data->voxel.r2[i];

				bloch_mcconnell_ode(data->voxel.P, out, in, data->voxel.r1, r2, data->voxel.k, data->voxel.m0, data->voxel.Om, gb_eff);

			} else {

				bloch_ode(out, in, data->voxel.r1[0], data->voxel.r2[0], gb_eff);
			}
		};

		NESTED(void, call_pdy2, (float* out, float t, const float* in))
		{
			float *gb_eff = gb_eff_p;
			(void)t;

			if (MODEL_BMC == data->seq.model) {

				bloch_mcc_pdy(data->voxel.P, (float(*)[N])out, in, data->voxel.r1, data->voxel.r2, data->voxel.k, data->voxel.m0, data->voxel.Om, gb_eff);

			} else {

				bloch_pdy((float(*)[3])out, in, data->voxel.r1[0], data->voxel.r2[0], gb_eff);
			}
		};

		NESTED(void, call_pdp2, (float* out, float t, const float* in))
		{
			float *gb_eff = gb_eff_p;
			(void)t;

			if (MODEL_BMC == data->seq.model) {

				bloch_mcc_b1_pdp(data->voxel.P, (float(*)[N])out, in, data->voxel.r1, data->voxel.r2, data->voxel.k, data->voxel.m0, gb_eff, w1);

			} else {

				bloch_b1_pdp((float(*)[3])out, in, data->voxel.r1[0], data->voxel.r2[0], gb_eff, w1);
			}
		};

		// Choose P-1 because ODE interface treats signal separate and P only describes the number of parameters
		ode_direct_sa(h, tol, N, P - 1, xp, data->pulse.rf_start, data->pulse.rf_end, call_fun, call_pdy2, call_pdp2);

	}	break;

        case SIM_STM:

                create_sim_matrix(data, P * N, stm_matrix, data->pulse.rf_start, data->pulse.rf_end, 0.);
                break;
        }

        data->grad.gb[2] = 0.;
}


/* ------------ Relaxation -------------- */

static void hard_relaxation(struct sim_data* data, int N, int P, float xp[P][N], float st, float end, float r2spoil)
{
	assert(0. <= (end - st));

	float xp2[3];


	for (int i = 0; i < P; i++) {

		vecf_copy(3, xp2, xp[i]);
		bloch_relaxation(xp[i], end - st, xp2, data->voxel.r1[0], data->voxel.r2[0] + r2spoil, data->grad.gb);
	}
}


void relaxation2(struct sim_data* data, float h, float tol, int N, int P, float xp[P][N], float st, float end, float stm_matrix[P * N][P * N], float r2spoil)
{
	data->seq.pulse_applied = false;

        // Single hard pulse is special case of homogeneously sampled sinc pulse
        if (0. == data->pulse.rf_end)
                data->seq.type = SIM_ROT;

        // Define effective z Gradient =Gradient Moments + off-resonance [rad/s]
        data->grad.gb[2] = data->grad.mom + data->voxel.w;
		
	__block complex float w1;	// clang workaround (needs to be outside switch)

        switch (data->seq.type) {

        case SIM_ROT:

                hard_relaxation(data, N, P, xp, st, end, r2spoil);
                break;

        case SIM_ODE: {

		float gb_eff[3];
		void *gb_eff_p = gb_eff;	// clang workaround

		NESTED(void, call_fun, (float* out, float t, const float* in))
		{
			float *gb_eff = gb_eff_p;
			compute_fields(data, gb_eff, t);

			w1 = gb_eff[0] - 1.i * gb_eff[1];

			gb_eff[0] *= data->voxel.b1;
			gb_eff[1] *= data->voxel.b1;

			float r2[data->voxel.P];

			if (MODEL_BMC == data->seq.model) {

				for (int i = 0; i < data->voxel.P; i++)
					r2[i] = data->voxel.r2[i] + r2spoil;

				bloch_mcconnell_ode(data->voxel.P, out, in, data->voxel.r1, r2, data->voxel.k, data->voxel.m0, data->voxel.Om, gb_eff);
			} else {

				bloch_ode(out, in, data->voxel.r1[0], data->voxel.r2[0] + r2spoil, gb_eff);
			}
		};

		NESTED(void, call_pdy2, (float* out, float t, const float* in))
		{
			float *gb_eff = gb_eff_p;
			(void)t;

			float r2[data->voxel.P];

			if (MODEL_BMC == data->seq.model) {

				for (int i = 0; i < data->voxel.P; i++)
					r2[i] = data->voxel.r2[i] + r2spoil;

				bloch_mcc_pdy(data->voxel.P, (float(*)[N])out, in, data->voxel.r1, r2, data->voxel.k, data->voxel.m0, data->voxel.Om, gb_eff);

			} else {

				bloch_pdy((float(*)[3])out, in, data->voxel.r1[0], data->voxel.r2[0] + r2spoil, gb_eff);
			}
		};

		NESTED(void, call_pdp2, (float* out, float t, const float* in))
		{
			float *gb_eff = gb_eff_p;
			(void)t;

			float r2[data->voxel.P];

			if (MODEL_BMC == data->seq.model) {

				for (int i = 0; i < data->voxel.P; i++)
					r2[i] = data->voxel.r2[i] + r2spoil;

				bloch_mcc_b1_pdp(data->voxel.P, (float(*)[N])out, in, data->voxel.r1, r2, data->voxel.k, data->voxel.m0, gb_eff, w1);

			} else {

				bloch_b1_pdp((float(*)[3])out, in, data->voxel.r1[0], data->voxel.r2[0] + r2spoil, gb_eff, w1);
			}
		};

		// Choose P-1 because ODE interface treats signal separate and P only describes the number of parameters
		ode_direct_sa(h, tol, N, P - 1, xp, st, end, call_fun, call_pdy2, call_pdp2);

	}	break;

        case SIM_STM:

                create_sim_matrix(data, P * N, stm_matrix, st, end, r2spoil);
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
			pulse_init(&data->pulse, data->pulse.rf_start, data->pulse.rf_end,
				CAST_UP(&data->pulse.sinc)->flipangle, data->pulse.phase,
				data->pulse.sinc.bwtp, data->pulse.sinc.alpha);

                break;

        case SIM_STM:

                int M = P * N + 1;

                // Matrix: 0 -> T_RF
                float mrf[M][M];
                rf_pulse(data, 0., 0., M, 1, NULL, mrf);

                // Matrix: T_RF -> TE
                float mrel[M][M];

		//assert(0. == data->grad.mom);

                if ((0 != data->grad.mom_sl) && (data->seq.te != data->pulse.rf_end)) {

			// Slice-Rewinder

			// Time-independent gradient integral
			data->grad.mom = -data->grad.mom_sl * (0.5 * data->pulse.rf_end) / (data->seq.te - data->pulse.rf_end);

			relaxation2(data, 0, 0, M, 1, NULL, data->pulse.rf_end, data->seq.te, mrel, 0.);

			data->grad.mom = 0.; // [rad/s]

		} else {

	                relaxation2(data, 0., 0., M, 1, NULL, data->pulse.rf_end, data->seq.te, mrel, 0.);
		}

                // Join matrices: 0 -> TE
		matf_mul(M, M, M, *mte, mrf, mrel);

                // Smooth spoiling for FLASH sequences

                if (NULL != mtr) {

			float r2spoil = 0.;

                        if (   (SEQ_FLASH == data->seq.seq_type)
                            || (SEQ_IRFLASH == data->seq.seq_type)) {

                                r2spoil = 10000.;
			}

                        // Balance z-gradient for bSSFP type sequences

			//assert(0. == data->grad.mom);

			// Matrix: TE -> TR
                        if ((   (SEQ_BSSFP == data->seq.seq_type)
                            || (SEQ_IRBSSFP == data->seq.seq_type))
			    && (data->seq.te != data->seq.tr)) {

				// Time-independent gradient integral
                                data->grad.mom = -data->grad.mom_sl * (0.5 * data->pulse.rf_end) / (data->seq.tr - data->seq.te);

                                relaxation2(data, 0., 0., M, 1, NULL, data->seq.te, data->seq.tr, *mtr, r2spoil);

                                data->grad.mom = 0.;

                        } else {

                                relaxation2(data, 0., 0., M, 1, NULL, data->seq.te, data->seq.tr, *mtr, r2spoil);
                        }
                }

                break;
        }
}


static void run_sim(struct sim_data* data, int pools,
			float (*mxy)[pools][3], float (*sa_r1)[pools][3], float (*sa_r2)[pools][3], float (*sa_b1)[1][3],
			float (*sa_m0)[pools][3], float (*sa_k)[pools][3], float (*sa_Om)[pools][3],
                        float h, float tol, int N, int P, float xp[P][N],
                        float xstm[P * N + 1], float mte[P * N + 1][P * N + 1], float mtr[P * N + 1][P * N + 1])
{
	float r2spoil = 0.;

        switch (data->seq.type) {

        case SIM_ROT:
        case SIM_ODE:

                rf_pulse(data, h, tol, N, P, xp, NULL);

                // Slice-Rewinder

                if ((0 != data->grad.mom_sl) && (data->seq.te != data->pulse.rf_end)) {

			// Time-independent gradient integral
			data->grad.mom = -data->grad.mom_sl * (0.5 * data->pulse.rf_end) / (data->seq.te - data->pulse.rf_end);

			relaxation2(data, h, tol, N, P, xp, data->pulse.rf_end, data->seq.te, NULL, 0.);

			data->grad.mom = 0.; // [rad/s]

		} else {

			relaxation2(data, h, tol, N, P, xp, data->pulse.rf_end, data->seq.te, NULL, 0.);
		}

		collect_signal(data, P, pools, mxy, sa_r1, sa_r2, sa_b1, sa_m0, sa_k, sa_Om, xp);

                // Smooth spoiling for FLASH sequences

                if (   (SEQ_FLASH == data->seq.seq_type)
                    || (SEQ_IRFLASH == data->seq.seq_type)) {

                        r2spoil = 10000.;
		}


                // Balance z-gradient for bSSFP type sequences

                if (   (   (SEQ_BSSFP == data->seq.seq_type)
 			|| (SEQ_IRBSSFP == data->seq.seq_type))
		    && (data->seq.te != data->seq.tr)) {

			// Time-independent gradient integral
                        data->grad.mom = -data->grad.mom_sl * (0.5 * data->pulse.rf_end) / (data->seq.tr - data->seq.te);

                        relaxation2(data, h, tol, N, P, xp, data->seq.te, data->seq.tr, NULL, r2spoil);

                        data->grad.mom = 0.;

                } else {

                        relaxation2(data, h, tol, N, P, xp, data->seq.te, data->seq.tr, NULL, r2spoil);
                }

		break;

        case SIM_STM:

                // Evolution: 0 -> TE
		//assert(xstm != mte);
		assert(NULL != xstm);
                apply_sim_matrix(N * P + 1, xstm, mte);

                // Save data
                stm2ode(N, P, xp, xstm);

                collect_signal(data, P, pools, mxy, sa_r1, sa_r2, sa_b1, sa_m0, sa_k, sa_Om, xp);

                // Evolution: TE -> TR
		//assert(xstm != mtr);
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

                relaxation2(&inv_data, h, tol, N, P, xp, st, end, NULL, 0.);

        } else {
                // Hyperbolic Secant inversion
		inv_data.pulse.type = PULSE_HS;

                inv_data.pulse.hs = pulse_hypsec_defaults;
                CAST_UP(&inv_data.pulse.hs)->duration = data->seq.inversion_pulse_length;
                inv_data.pulse.rf_end = data->seq.inversion_pulse_length;

                rf_pulse(&inv_data, h, tol, N, P, xp, NULL);

                // Spoiler gradients
                relaxation2(&inv_data, h, tol, N, P, xp, st, end, NULL, 10000.);
        }
}


static void alpha_half_preparation(const struct sim_data* data, int pools, float h, float tol, int N, int P, float xp[P][N])
{
	struct sim_data prep_data = *data;

	assert(0. <= data->seq.prep_pulse_length);

	// Enforce ODE: Way more efficient here!
	prep_data.seq.type = SIM_ODE;
        CAST_UP(&prep_data.pulse.sinc)->flipangle = CAST_UP(&data->pulse.sinc)->flipangle / 2.;
	prep_data.pulse.phase = M_PI;
	prep_data.seq.te = (data->pulse.rf_end + data->seq.prep_pulse_length) / 2.;
	prep_data.seq.tr = data->seq.prep_pulse_length;

	if (0. < data->seq.prep_pulse_length) {

		prepare_sim(&prep_data, N, P, NULL, NULL);

		run_sim(&prep_data, pools, NULL, NULL, NULL, NULL, NULL, NULL, NULL, h, tol, N, P, xp, NULL, NULL, NULL);

	} else { // Perfect preparation

		for (int p = 0; p < P; p++)
                        bloch_excitation2(xp[p], xp[p], DEG2RAD(CAST_UP(&prep_data.pulse.sinc)->flipangle), prep_data.pulse.phase);
	}
}


/* ------------ CEST -------------- */

static void calc_off_res(struct sim_data* data, int N, float off_res[N])
{
	float om_larmor = 2 * M_PI * data->cest.b0 * data->cest.gamma;
	float incr = fabsf(data->cest.off_start - data->cest.off_stop) / (N - 1);

	for (int i = 0; i < N; i++)
		off_res[i] = (data->cest.off_start - i * incr) * om_larmor;
}


static void reset_xp(int N, int P, float xp[N][P], float m0[P])
{
	for (int p = 0; p < P; p++)
		for (int n = 0; n < N; n++)
			xp[p][n] = 0.;

	for (int p = 0; p < P; p++) {

		xp[0][2 + p * 3] = m0[p];
		xp[2 + 2 * P + p][2 + p * 3] = 1.;
	}
}


static void cest_seq(struct sim_data* data, float h, float tol, int N, int P, float xp[P][N], float offset)
{	
	// FIX ME : Allow different pulse types
	data->pulse.type = PULSE_REC;
	data->pulse.rect = pulse_rect_defaults;
	data->pulse.rect.A = data->cest.b1_amp * 2. * M_PI * data->cest.gamma;

	//debug_printf(DP_INFO, "offset [ppm] : %f\n", offset / (2. * M_PI * data->cest.b0 * data->cest.gamma));

	for (int p = 0; p < data->cest.n_pulses; p++) {

		data->voxel.w = offset;
		rf_pulse(data, h, tol, N, P, xp, NULL);
		data->voxel.w = 0.;

		if ( (data->cest.n_pulses - 1 > p) && (0. < data->cest.t_d) )
			relaxation2(data, h, tol, N, P, xp, 0, data->cest.t_d, NULL, 0.);
	}

	relaxation2(data, h, tol, N, P, xp, 0, data->cest.t_pp, NULL, 10000.);
}


/* ------------ Main Simulation -------------- */

void bloch_simulation2(const struct sim_data* _data, int R, int pools, float (*m_state)[R][pools][3],
			float (*sa_r1_state)[R][pools][3], float (*sa_r2_state)[R][pools][3], float (*sa_m0_state)[R][pools][3],
			float (*sa_b1_state)[R][1][3], float (*sa_k_state)[R][pools][3], float (*sa_om_state)[R][pools][3])
{
	// FIXME: split config + variable part

	struct sim_data data = *_data;  // Lose information of complex pointer variables

        float tol = _data->other.ode_tol;      // Tolerance of ODE solver

	// Dimensions and parameters according to number of pools
	int N = 3 * pools;	// Number of dimensions (x, y, z) * #pools
	int P = (1 == pools) ? 4 : 5 * pools;		// Number of parameters with estimated derivative (M, DR1, DR2, DB1, DM0, Dk, DOm)

        assert(0 < P);

        int M = N * P + 1;     // STM based on single vector and additional +1 for linearized system matrix

	assert(R == data.seq.rep_num);

	int A = data.seq.averaged_spokes;

        // Unit: [M0] = 1 Magnetization / mm
        //      -> Default slice thickness of a single isochromat set to 1 mm
        //              FIXME: Slice Profile in relative units of theoretical slice thickness?
        // debug_printf(DP_INFO, "Theoretical Slice Thickness: %f\n", 2 * M_PI / ((data.pulse.rf_end - data.pulse.rf_start) / (2. + (data.pulse.nl - 1.) + (data.pulse.nr - 1.)) * data.grad.sl_gradient_strength * GAMMA_H1));
        float default_slice_thickness = 0.001; // [m]

	data.seq.rep_num *= A;
	data.seq.averaged_spokes = 1;

	float off_res[data.seq.rep_num];
	float ref_scan = 0.;

	if (SEQ_CEST == data.seq.seq_type)
		calc_off_res(&data, data.seq.rep_num, off_res);

	int S = data.seq.spin_num;

	float (*Fmxy)[R * A][S][pools][3] = xmalloc(sizeof *Fmxy);
	float (*Fsa_r1)[R * A][S][pools][3] = xmalloc(sizeof *Fsa_r1);
	float (*Fsa_r2)[R * A][S][pools][3] = xmalloc(sizeof *Fsa_r2);
	float (*Fsa_b1)[R * A][S][1][3] = xmalloc(sizeof *Fsa_b1);

	float (*Fsa_m0)[R * A][S][pools][3] = xmalloc(sizeof *Fsa_m0);
	float (*Fsa_k)[R * A][S][pools][3] = xmalloc(sizeof *Fsa_k);
	float (*Fsa_Om)[R * A][S][pools][3] = xmalloc(sizeof *Fsa_Om);

	for (int s = 0; s < S; s++) {

                float h = 0.0001;


		if (1 != S) {

                        // Calculate slice profile by looping over spins with z-gradient
                        //      Full Symmetric slice profile

                        // Ensures central spin on main lope is set
			assert(1 == S % 2);
                        assert(0. != _data->seq.slice_thickness);

			// w_{gz} [rad/s] = gamma_H1 [rad/(sT)] * Gz [T/m] * z [m], FIXME: Change name of variable mom_sl
			data.grad.mom_sl = (_data->grad.sl_gradient_strength * _data->seq.slice_thickness * GAMMA_H1) / (S - 1) * (s - (int)(S / 2.));

                } else {

                        data.seq.slice_thickness = default_slice_thickness;

                        // Define z-Position with slice thickness for single spin
                        data.grad.mom_sl = _data->grad.sl_gradient_strength * _data->seq.slice_thickness * GAMMA_H1;
                }


                // Initialize ODE
		float xp[P][N];

                for (int p = 0; p < P; p++)
                        for (int n = 0; n < N; n++)
                                xp[p][n] = 0.;

                xp[0][2] = 1.;

                // Initialize STM
                float xstm[M];

		for (int n = 0; n < M; n++)
			xstm[n] = 0.;

                xstm[2] = 1.;
                xstm[M - 1] = 1.;

                if (MODEL_BMC == data.seq.model) {

			for (int p = 0; p < data.voxel.P; p++) {

				xp[0][2 + p * 3] = data.voxel.m0[p];

				// Sensitivities to m0[0] and m0[1] need 1 as initialization
				xp[2 + 2 * data.voxel.P + p][2 + p * 3] = 1.;
				xstm[2 + p * 3] = data.voxel.m0[p];
			}
                }

                // Reset parameters
		data.voxel.w = _data->voxel.w;
		data.pulse.phase = 0;

                // Apply inversion

                if (   (SEQ_IRBSSFP == data.seq.seq_type)
		    || (SEQ_IRFLASH == data.seq.seq_type))
                        inversion(&data, h, tol, N, P, xp, 0., data.seq.inversion_spoiler);

                // Alpha/2 and TR/2 signal preparation

                if (   (SEQ_BSSFP == data.seq.seq_type)
                    || (SEQ_IRBSSFP == data.seq.seq_type))
                        alpha_half_preparation(&data, pools, h, tol, N, P, xp);

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

		if ((SEQ_CEST == data.seq.seq_type ) && data.cest.ref_scan) {

			cest_seq(&data, h, tol, N, P, xp, data.cest.ref_scan_ppm * 2 * M_PI * data.cest.b0 * data.cest.gamma);

			ref_scan = xp[0][2];
			reset_xp(N, data.voxel.P, xp, data.voxel.m0);
		}

                // Loop over Pulse Blocks

                for (int r = 0; r < data.seq.rep_num; r++) {

			auto mxy = &((*Fmxy)[r][s]);
			auto sa_r1 = &((*Fsa_r1)[r][s]);
			auto sa_r2 = &((*Fsa_r2)[r][s]);
			auto sa_b1 = &((*Fsa_b1)[r][s]);
			auto sa_m0 = &((*Fsa_m0)[r][s]);
			auto sa_k = &((*Fsa_k)[r][s]);
			auto sa_Om = &((*Fsa_Om)[r][s]);

			bool odd = false;

                        // Change phase of bSSFP sequence in each repetition block
                        if (   (SEQ_BSSFP == data.seq.seq_type)
                            || (SEQ_IRBSSFP == data.seq.seq_type)) {

                                data.pulse.phase = M_PI * r;

				odd = (1 == r % 2);
                        }

			if (SEQ_CEST == data.seq.seq_type) {

				assert(MODEL_BMC == data.seq.model);
				assert(SIM_ODE == data.seq.type);

				cest_seq(&data, h, tol, N, P, xp, off_res[r]);

				if (data.cest.ref_scan)
					xp[0][2] = xp[0][2] / ref_scan;

				collect_signal(&data, P, pools, mxy, sa_r1, sa_r2, sa_b1, sa_m0, sa_k, sa_Om, xp);
				reset_xp(N, data.voxel.P, xp, data.voxel.m0);

			} else {

				run_sim(&data, pools, mxy, sa_r1, sa_r2, sa_b1, sa_m0, sa_k, sa_Om, h, tol, N, P, xp, xstm, mte[odd], mtr);
			}
                }
	}


	// Sum up magnetization

        // Scale signal with density
        //      - Relative to default slice thickness to keep strength of simulation higher
        float D = (float)S / (data.seq.slice_thickness / default_slice_thickness);

        sum_up_signal(data.voxel.m0[0], data.seq.rep_num / A, data.seq.spin_num, A, D, pools,
			Fmxy, Fsa_r1, Fsa_r2, Fsa_b1, Fsa_m0, Fsa_k, Fsa_Om,
			m_state, sa_r1_state, sa_r2_state, sa_m0_state, sa_b1_state, sa_k_state, sa_om_state);

	xfree(Fmxy);
	xfree(Fsa_r1);
	xfree(Fsa_r2);
	xfree(Fsa_b1);
	xfree(Fsa_m0);
        xfree(Fsa_k);
        xfree(Fsa_Om);
}

// Wrapper for single pool simulation
void bloch_simulation(const struct sim_data* data, int R, float (*m)[R][3], float (*sa_r1)[R][3], float (*sa_r2)[R][3], float (*sa_m0)[R][3],	float (*sa_b1)[R][3])
{
	bloch_simulation2(data, R, 1, (void*)m, (void*)sa_r1, (void*)sa_r2, (void*)sa_m0, (void*)sa_b1, NULL, NULL);
}

