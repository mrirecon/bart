/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2021 Volkert Roeloffs
 *
 * References:
 * Hennig, J., Weigel, M., & Scheffler, K. (2004). Calculation of flip angles
 * for echo trains with predefined amplitudes with the extended phase graph
 * (EPG)‐algorithm: principles and applications to hyperecho and TRAPS
 * sequences. Magnetic Resonance in Medicine 51(1), 68-80.
 *
 * Mathias Weigel, Extended phase graphs: Dephasing, RF pulses, and echoes
 * - pure and simple J. Magn. Reson. Imaging 2015;41:266–295
 *
 * Brian Hargreaves EPG simulator: http://web.stanford.edu/~bah/software/epg/
 *
 * code follows the logic of Eric Hughes' C++ implementation
 * Eric Hughes EPG simulator: https://github.com/EricHughesABC/EPG
 *
 * and computes partial derivatives similiar to:
 * K. Layton, M. Morelande, D. Wright, P. Farrell, B. Moran, L A. Johnston.
 * Modelling and Estimation of Multicomponent T2 Distributions, IEEE TMI, 2013
 */

/* order of derivatives: T1, T2, B1, \phi */

#include <stdbool.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"
#include "num/linalg.h"
#include "num/multind.h"


#include "epg.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


void create_relax_matrix_der(complex float ee[3][3], complex float dee[4][3][3], float T1, float T2, float offres, float tau)
{
	complex float phasor = cexpf(I * 2.0 * M_PI * offres * tau);

	ee[0][0] = cexpf(-tau / T2) * phasor;
	ee[1][1] = cexpf(-tau / T2) * conjf(phasor); 
	ee[2][2] = cexpf(-tau / T1);

	if (NULL != dee) {

		// T1
		dee[0][2][2] = tau / (T1 * T1) * ee[2][2];

		// T2
		dee[1][0][0] = tau / (T2 * T2) * ee[0][0];
		dee[1][1][1] = tau / (T2 * T2) * ee[1][1];

		// offres
		dee[3][0][0] =  I * 2.0 * M_PI * tau * ee[0][0];
		dee[3][1][1] = -I * 2.0 * M_PI * tau * ee[1][1];
	}
}


void create_relax_matrix(complex float ee[3][3], float T1, float T2, float offres, float tau)
{
	create_relax_matrix_der(ee, NULL, T1, T2, offres, tau);
}

void create_rf_pulse_matrix_der(complex float T[3][3], complex float dT[4][3][3], float alpha, float phi)
{
	float alpha_half = alpha / 2.0;
	float cah = cos(alpha_half);
	float cah2 = cah * cah;
	float sah = sin(alpha_half);
	float sah2 = sah * sah;
	float ca = cos(alpha);
	float sa = sin(alpha);
	complex float i_phi = 0.0 + 1.i * phi;
	complex float i_half = 0.0 + 1.i * 0.5;

	T[0][0] = cah2;
	T[0][1] = cexpf(2.0 * i_phi) * sah2;
	T[0][2] = -1.0i * cexpf(i_phi) * sa;

	T[1][0] = cexpf(-2.0 * i_phi) * sah2;
	T[1][1] = cah2;
	T[1][2] = 1.i * cexpf(-i_phi) * sa;

	T[2][0] = -i_half * cexpf(-i_phi) * sa;
	T[2][1] = i_half * cexpf(i_phi) * sa;
	T[2][2] = ca;

	if (NULL != dT) {
		
		// B1
		dT[2][0][0] = -cah * sah * alpha;
		dT[2][0][1] = cexpf(2.0 * i_phi) * sah * cah * alpha;
		dT[2][0][2] = -1.0i * cexpf(i_phi) * ca * alpha;

		dT[2][1][0] = cexpf(-2.0 * i_phi) * sah * cah * alpha;
		dT[2][1][1] = -cah * sah * alpha;
		dT[2][1][2] = 1.0i * cexpf(-i_phi) * ca * alpha;

		dT[2][2][0] = -i_half * cexpf(-i_phi) * ca * alpha;
		dT[2][2][1] = i_half * cexpf(i_phi) * ca * alpha;
		dT[2][2][2] = -sa * alpha;
	}
}


void create_rf_pulse_matrix(complex float T[3][3], float alpha, float phi)
{
	create_rf_pulse_matrix_der(T, NULL, alpha, phi);
}


void epg_pulse_der(complex float T[3][3], int num_cols, complex float states[3][num_cols],
	   	complex float dT[4][3][3], complex float dstates[4][3][num_cols])
{
	complex float states_next[3][num_cols];

	mat_mul(3, 3, num_cols, states_next, T, states);

	if ((NULL != dT) && (NULL != dstates)) {

		// derivatives	
		long dims_dstates[3] = { num_cols, 3, 4 };
		long dims_dT[3] = { 3, 3, 4 };
		long pos[3] = { 0 };

		complex float der[3][num_cols];
		long dims_der[3] = { num_cols, 3, 1 };

		complex float derdT[3][3];
		long dims_derdT[3] = { 3, 3, 1 };

		// loop over T1, T2, B1, offres
		// and compute derivatives according to product rule
		
		complex float tmp[3][num_cols];

		for (int i = 0; i < 4; i++) {

			pos[2] = i;
			md_copy_block(3, pos, dims_derdT, derdT, dims_dT, dT, CFL_SIZE);
			mat_mul(3, 3, num_cols, tmp, derdT, states);

			md_copy_block(3, pos, dims_der, der, dims_dstates, dstates, CFL_SIZE);
			mat_muladd(3, 3, num_cols, tmp, T, der);

			md_copy_block(3, pos, dims_dstates, dstates, dims_der, tmp, CFL_SIZE);
		}
	}

	mat_copy(3, num_cols, states, states_next);
}


void epg_pulse(complex float T[3][3], int num_cols, complex float states[3][num_cols])
{
	epg_pulse_der(T, num_cols, states, NULL, NULL);
}


void epg_relax_der(complex float ee[3][3], int num_cols, complex float states[3][num_cols],
		complex float dee[4][3][3], complex float dstates[4][3][num_cols])
{
	complex float states_next[3][num_cols];

	mat_mul(3, 3, num_cols, states_next, ee, states);

	states_next[2][0] = states_next[2][0] + 1.0 - ee[2][2]; // constant part

	if ((NULL != dee) && (NULL != dstates)) {

		// derivatives
		long dims_dstates[3] = { num_cols, 3, 4 };
		long dims_dee[3] = { 3, 3, 4 };
		long pos[3] = { 0 };

		complex float der[3][num_cols];
		long dims_der[3] = { num_cols, 3, 1 };

		complex float derdee[3][3];
		long dims_derdee[3] = { 3, 3, 1 };

		// loop over T1, T2, B1, phi
		// and compute derivatives according to product rule

		complex float tmp[3][num_cols];

		for (int i = 0; i < 4; i++) {

			pos[2] = i;
			md_copy_block(3, pos, dims_derdee, derdee, dims_dee, dee, CFL_SIZE);
			mat_mul(3, 3, num_cols, tmp, derdee, states);

			md_copy_block(3, pos, dims_der, der, dims_dstates, dstates, CFL_SIZE);
			mat_muladd(3, 3, num_cols, tmp, ee, der);

			if (0 == i) // for T1, include derivative of constant part
				tmp[2][0] += -dee[0][2][2];

			md_copy_block(3, pos, dims_dstates, dstates, dims_der, tmp, CFL_SIZE);
		}
	}

	mat_copy(3, num_cols, states, states_next);
}


void epg_relax(complex float ee[3][3], int num_cols, complex float states[3][num_cols])
{
	epg_relax_der(ee, num_cols, states, NULL, NULL);
}


void epg_grad_der( int num_cols, complex float states[3][num_cols],
	   	complex float dstates[4][3][num_cols])
{
	for (int i = 0; i< num_cols - 1; i++) {

		states[0][num_cols - 1 - i] = states[0][num_cols - 1 - i - 1];
		states[1][i] = states[1][i + 1];

		if (NULL != dstates) {
			for (int j = 0; j < 4; j++) {

				dstates[j][0][num_cols - 1 - i] = dstates[j][0][num_cols - 1 - i - 1];
				dstates[j][1][i] = dstates[j][1][i + 1];
			}
		}
	}

	states[1][num_cols - 1] = 0.0;
	states[0][0] = conjf(states[1][0]);

	if (NULL != dstates) {

		for (int j = 0; j < 4; j++) {

			dstates[j][1][num_cols - 1] = 0.0;
			dstates[j][0][0] = conjf(dstates[j][1][0]);
		}
	}
}


void epg_grad( int num_cols, complex float states[3][num_cols])
{
	epg_grad_der(num_cols, states, NULL);	
}


void epg_spoil_der(int num_cols, complex float states[3][num_cols], complex float dstates[4][3][num_cols])
{
	for (int i = 0; i < num_cols; i++) {

		states[0][i] = 0.0;
		states[1][i] = 0.0;

		if (NULL != dstates) {

			for (int j = 0; j < 4; j++) {

				dstates[j][0][i] = 0.0;
				dstates[j][1][i] = 0.0;
			}
		}
	}
}


void epg_spoil(int num_cols, complex float states[3][num_cols])
{
	epg_spoil_der(num_cols, states, NULL);
}


void epg_adc_der(int idx_current, int N, int M, complex float signal[N],
		 complex float states[3][M][N], complex float dsignal[4][N],
		 complex float dstates[4][3][M][N],
		 complex float state_current[3][M],
		 complex float dstate_current[4][3][M],
		 float adc_phase)
{
	complex float phasor_offset = cexpf(1.i * (adc_phase));

	// save signal
	signal[idx_current] = state_current[0][0] * phasor_offset;

	// save states
	if (NULL != states)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < M; k++)
				states[j][k][idx_current] = state_current[j][k] * phasor_offset;

	// save dsignal
	if (NULL != dsignal)
		for (int j = 0; j < 4; j++)
			dsignal[j][idx_current] = dstate_current[j][0][0] * phasor_offset;

	// save dstates
	if (NULL != dstates)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < M; k++)
				for (int m = 0; m < 4; m++)
					dstates[m][j][k][idx_current] = dstate_current[m][j][k] * phasor_offset;
}


void epg_adc(int idx_current, int N, int M, complex float signal[N],
		 complex float states[3][M][N],
		 complex float state_current[3][M],
		 float adc_phase)
{
	epg_adc_der(idx_current, N, M, signal, states, NULL, NULL, state_current, NULL, adc_phase);
}


void cpmg_epg_der(int N, int M, complex float signal[N], complex float states[3][M][N], complex float dsignal[4][N], complex float dstates[4][3][M][N], float rf_exc, float rf_ref, float TE, float T1, float T2, float B1, float offres)
{
	complex float state_current[3][M];
	memset(state_current, 0, sizeof state_current);

	complex float dstate_current[4][3][M];
	memset(dstate_current, 0, sizeof dstate_current);
	
	complex float ee[3][3] = { {0. } };
	complex float dee[4][3][3] = { { { 0. } } };
	complex float T_exc[3][3] = { { 0. } };
	complex float dT_exc[4][3][3] = { { { 0. } } };
	complex float T_ref[3][3] = { { 0. } };
	complex float dT_ref[4][3][3] = { { {0. } } };

	// create relaxation matrix and derivative
	create_relax_matrix_der(ee, dee, T1, T2, offres, TE / 2.0);

	// create 90(y) rotation matrix
	create_rf_pulse_matrix_der(T_exc, dT_exc, B1 * rf_exc * M_PI / 180.0, M_PI / 2.0);

	// create 180(x) rotation matrix
	create_rf_pulse_matrix_der(T_ref, dT_ref, B1 * rf_ref * M_PI / 180.0, 0.0);

	// initialize Z magnetization
	state_current[2][0] = 1.0;

	// apply 90 degree pulse
	epg_pulse_der(T_exc, M, state_current, dT_exc, dstate_current);

	// apply 180 pulses of CPMG sequence
	for (int i = 0; i < N; i++) {

		//apply relaxation for first half of half echo time
		epg_relax_der(ee, M, state_current, dee, dstate_current);
		
		//apply gradient
		epg_grad_der(M, state_current, dstate_current);

		//apply refocusing pulse
		epg_pulse_der(T_ref, M, state_current, dT_ref, dstate_current);

		//apply relaxation for second half of half echo time
		epg_relax_der(ee, M, state_current, dee, dstate_current);

		//apply gradient
		epg_grad_der(M, state_current, dstate_current);

		// sample signal
		epg_adc_der(i, N, M, signal, states, dsignal, dstates, state_current, dstate_current, 0.0);
	}
}


void fmssfp_epg(int N, int M, complex float signal[N], complex float states[3][M][N], float FA, float TR, float T1, float T2, float B1, float offres)
{
	if (NULL != states)
		memset(states, 0., 3 * sizeof *states);

	complex float state_current[3][M];
	memset(state_current, 0., sizeof state_current);

	complex float ee[3][3] = { {0. } };
	complex float T_exc[3][3] = { { 0. } };

	create_relax_matrix(ee, T1, T2, offres, TR / 2.0);

	// initialize Z magnetization
	state_current[2][0] = 1.0;

	// initialize phase
	float rf_phase = 0.0;

	// include prep loop
	for (int l = 0; l < 2; l++) {

		// loop over excitations

		for (int i = 0; i<N; i++) {
		
			rf_phase += M_PI;
			rf_phase += 2.0 * M_PI * (float)i / (float)N;
			rf_phase = fmodf(rf_phase, 2.0 * M_PI);

			// change phase of rf pulse
			create_rf_pulse_matrix(T_exc, B1 * FA * M_PI / 180.0,  M_PI / 2 - rf_phase);

			//apply excitation pulse
			epg_pulse(T_exc, M, state_current);

			//apply relaxation for first half of TR
			epg_relax(ee, M, state_current);

			//save signal after prep cycle
			if (1 == l)
				epg_adc(i, N, M, signal, states, state_current, rf_phase);

			//apply relaxation for second half of TR
			epg_relax(ee, M, state_current);
		}
	}
}

void fmssfp_epg_der(int N, int M, complex float signal[N], complex float states[3][M][N], complex float dsignal[4][N], complex float dstates[4][3][M][N], float FA, float TR, float T1, float T2, float B1, float offres)
{
	complex float state_current[3][M];
	memset(state_current, 0, sizeof state_current);

	complex float dstate_current[4][3][M];
	memset(dstate_current, 0, sizeof dstate_current);

	complex float ee[3][3] = { { 0. } };
	complex float dee[4][3][3] = { { { 0. } } };
	complex float T_exc[3][3] = { { 0. } };
	complex float dT_exc[4][3][3] = { { { 0. } } };

	create_relax_matrix_der(ee, dee, T1, T2, offres, TR / 2.0);

	// initialize Z magnetization
	state_current[2][0] = 1.0;

	// initialize phase
	float rf_phase = 0.0;

	// include prep loop
	for (int l = 0; l < 2; l++) {

		// loop over excitations
		for (int i = 0; i < N; i++) {

			rf_phase += M_PI;
			rf_phase += 2. * M_PI * (float)i / (float)N;
			rf_phase = fmodf(rf_phase, 2.0 * M_PI);

			// change phase of rf pulse
			create_rf_pulse_matrix_der(T_exc, dT_exc, B1 * FA * M_PI / 180.0, rf_phase);

			//apply excitation pulse
			epg_pulse_der(T_exc, M, state_current, dT_exc, dstate_current);

			//apply relaxation for first half of TR
			epg_relax_der(ee, M, state_current, dee, dstate_current);

			//save signal after prep cycle
			if (1 == l)
				epg_adc_der(i, N, M, signal, states, dsignal, dstates, state_current, dstate_current, M_PI / 2 - rf_phase);

			//apply relaxation for second half of TR
			epg_relax_der(ee, M, state_current, dee, dstate_current);
		}
	}
}

void hyperecho_epg(int N, int M, complex float signal[N], complex float states[3][M][N], float rf_exc, float rf_ref, float TE, float FA, float T1, float T2, float B1, float offres)
{
	assert(N % 2 == 1); // hyperecho requires symmetrie around central 180 deg 

	complex float state_current[3][M];
	memset(state_current, 0, sizeof state_current);

	complex float ee[3][3] = { { 0.0 } };
	complex float T_exc[3][3] = { { 0.0 } };
	complex float T_ref[3][3] = { { 0.0 } };
	complex float T_alpha[3][3] = { { 0.0 } };
	complex float T_malpha[3][3] = { { 0.0 } };

	// create relaxation matrix and derivative
	create_relax_matrix(ee, T1, T2, offres, TE / 2.0);

	// create 90(y) rotation matrix
	create_rf_pulse_matrix(T_exc, B1 * rf_exc * M_PI / 180.0, M_PI / 2.0);

	// create 180(x) rotation matrix
	create_rf_pulse_matrix(T_ref, B1 * rf_ref * M_PI / 180.0, 0.0);

	// create alpha(x) rotation matrix
	create_rf_pulse_matrix(T_alpha, B1 * FA * M_PI / 180.0, 0.0);

	// create alpha(-x) rotation matrix
	create_rf_pulse_matrix(T_malpha, B1 * FA * M_PI / 180.0, M_PI);

	// initialize Z magnetization
	state_current[2][0] = 1.0;

	// apply 90 degree pulse
	epg_pulse(T_exc, M, state_current);

	for (int i = 0; i < N; i++) {

		epg_relax(ee, M, state_current);
		epg_grad(M, state_current);

		if (i < N / 2)
			epg_pulse(T_alpha, M, state_current);
		else if (i == N / 2)
			epg_pulse(T_ref, M, state_current);
		else
			epg_pulse(T_malpha, M, state_current);

		epg_relax(ee, M, state_current);
		epg_grad(M, state_current);

		epg_adc(i, N, M, signal, states, state_current, 0.0);
	}
}

void flash_epg_der(int N, int M, complex float signal[N], complex float states[3][M][N], complex float dsignal[4][N], complex float dstates[4][3][M][N], float FA, float TR, float T1, float T2, float B1, float offres, int spoiling)
{
	// spoiling 0: ideal 1: conventional RF 2: random
	complex float state_current[3][M];
	memset(state_current, 0, sizeof state_current);

	complex float dstate_current[4][3][M];
	memset(dstate_current, 0, sizeof dstate_current);

	complex float ee[3][3] = { { 0.0 } };
	complex float dee[4][3][3] = { { { 0.0 } } };
	complex float T_exc[3][3] = { { 0.0 } };
	complex float dT_exc[4][3][3] = { { { 0.0 } } };

	create_relax_matrix_der(ee, dee, T1, T2, offres, TR);

	// initialize Z magnetization
	state_current[2][0] = 1.0;

	// initialize phase
	float rf_phase = 0.0;

	// loop over excitations
	for (int i = 0; i < N; i++) {
	
		if (1 == spoiling)
			rf_phase += i * 50.0 / 180.0 * M_PI;
		if (2 == spoiling)
			rf_phase += 2 * M_PI * rand() / RAND_MAX;

		rf_phase = fmodf(rf_phase, 2.0 * M_PI);

		// change phase of rf pulse
		create_rf_pulse_matrix_der(T_exc, dT_exc, B1 * FA * M_PI / 180.0, rf_phase);

		// apply relaxation for one full TR
		epg_relax_der(ee, M, state_current, dee, dstate_current);

		// apply excitation pulse
		epg_pulse_der(T_exc, M, state_current, dT_exc, dstate_current);

		// sample signal
		epg_adc_der(i, N, M, signal, states, dsignal, dstates, state_current, dstate_current, rf_phase);

		// dephase
		epg_grad_der(M, state_current, dstate_current);

		// spoil
		if (0 == spoiling) 
			epg_spoil_der(M, state_current, dstate_current);
	}
}


void hahnspinecho_epg(int N, int M, complex float signal[N], complex float states[3][M][N],
        float FA, float TE, float T1, float T2, float B1, float offres)
{
	if (2 < N)
		error("Hahn spin echo sequence only implemented for 1 or 2 echoes!");

	// 3 excitation pulses of FA, with timing | TE | TE - ADC - 2TE | TE - ADC
	if (NULL != states)
		memset(states, 0.0, 3 * sizeof(*states));

	complex float state_current[3][M];
	memset(state_current, 0, sizeof state_current);

	complex float ee[3][3] = { { 0.0 } };
	complex float T_exc[3][3] = { { 0.0 } };

	create_relax_matrix(ee, T1, T2, offres, TE);
	create_rf_pulse_matrix(T_exc, B1 * FA * M_PI / 180.0, 0.0);

	// start from equilibrium
	state_current[2][0] = 1.0;

	// ------------ start of sequence -------------

	epg_pulse(T_exc, M, state_current);		// first pulse

	epg_relax(ee, M, state_current);		// relaxation and dephasing for time TE
	epg_grad(M, state_current);

	epg_pulse(T_exc, M, state_current);		// second pulse

	epg_relax(ee, M, state_current);		// relaxation and dephasing for time TE
	epg_grad(M, state_current);

	epg_adc(0, N, M, signal, states, state_current, 0.0);	 // first signal

	if (N == 2) {

		epg_relax(ee, M, state_current);	// relaxation and dephasing for time 2*TE
		epg_relax(ee, M, state_current);
		epg_grad(M, state_current);
		epg_grad(M, state_current);

		epg_pulse(T_exc, M, state_current);	// second pulse

		epg_relax(ee, M, state_current);	// relaxation and dephasing for time TE
		epg_grad(M, state_current);

		epg_adc(1, N, M, signal, states, state_current, 0.0);// second signal
	}

	// ------------ end of sequence -------------
}


void bssfp_epg_der(int N, int M, complex float signal[N], complex float states[3][M][N], complex float dsignal[4][N], complex float dstates[4][3][M][N], float FA, float TR, float T1, float T2, float B1, float offres)
{
	complex float state_current[3][M];
	memset(state_current, 0, sizeof state_current);

	complex float dstate_current[4][3][M];
	memset(dstate_current, 0, sizeof dstate_current);

	complex float ee[3][3] = { { 0.0 } };
	complex float dee[4][3][3] = { { { 0.0 } } };
	complex float T_exc[3][3] = { { 0.0 } };
	complex float dT_exc[4][3][3] = { { {0.0 } } };
	complex float Tah_exc[3][3] = { { 0.0 } };
	complex float dTah_exc[4][3][3] = { { {0.0 } } };

	// initialize phase
	float rf_phase = 0.0;

	create_rf_pulse_matrix_der(Tah_exc, dTah_exc, B1 * FA / 2.0 * M_PI / 180.0, rf_phase);
	create_relax_matrix_der(ee, dee, T1, T2, offres, TR / 2.0);

	// initialize Z magnetization
	state_current[2][0] = 1.0;

	//apply alpha/2 excitation pulse
	epg_pulse_der(Tah_exc, M, state_current, dTah_exc, dstate_current);

	//apply relaxation for half of TR
	epg_relax_der(ee, M, state_current, dee, dstate_current);

	// loop over excitations
	for (int i = 0; i < N; i++) {

		rf_phase += M_PI;
		rf_phase = fmodf(rf_phase, 2.0 * M_PI);

		// change phase of rf pulse
		create_rf_pulse_matrix_der(T_exc, dT_exc, B1 * FA * M_PI / 180.0, rf_phase);

		//apply excitation pulse
		epg_pulse_der(T_exc, M, state_current, dT_exc, dstate_current);

		//apply relaxation for first half of TR
		epg_relax_der(ee, M, state_current, dee, dstate_current);

		//save signal after prep cycle
		epg_adc_der(i, N, M, signal, states, dsignal, dstates, state_current, dstate_current, M_PI / 2 - rf_phase);

		//apply relaxation for second half of TR
		epg_relax_der(ee, M, state_current, dee, dstate_current);
	}
}

