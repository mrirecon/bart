/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <assert.h>

#include "num/multind.h"

#include "misc/mri.h"
#include "misc/misc.h"

#include "seq/seq.h"

#include "kernel.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


void linearize_events(int N, struct seq_event ev[__VLA(N)], double* start_block, enum block mode, double tr, double raster)
{
	if ((0 >= N) || (0. > *start_block))
		return;

	for (int i = 0; i < N; i++) {

		ev[i].start += *start_block;
		ev[i].mid   += *start_block;
		ev[i].end   += *start_block;
	}

	*start_block += seq_block_end(N, ev, mode, tr, raster);
}


/*
 * Compute 0th moment on a raster after RF pulse.
 * 0th moment before excitation pulse is zero.
 */
void compute_moment0(int M, float moments[M][3], double dt, int N, const struct seq_event ev[N])
{
	for (int i = 0; i < M; i++) 
		for (int a = 0; a < 3; a++)
			moments[i][a] = 0.;

	double m_rf[3] = { };

	// last pulse
	int rf_idx = -1;

	if (0 < events_counter(SEQ_EVENT_PULSE, N, ev))
		rf_idx = events_idx(events_counter(SEQ_EVENT_PULSE, N, ev) - 1, SEQ_EVENT_PULSE, N, ev);

	double t_reset = -1;
	if ((0 < rf_idx) && (SEQ_RF_EXCITATION == ev[rf_idx].pulse.type)) {

		t_reset = ev[rf_idx].mid;
		moment_sum(m_rf, t_reset, N, ev);
	}


	for (int p = 0; p < M; p++) {

		assert((0 <= p) && (p <= M));

		double m[3] = { };
		if ((p + 0.5) * dt > t_reset) {

			moment_sum(m, (p + 0.5) * dt, N, ev);
			for (int a = 0; a < 3; a++) 
				moments[p][a] = m[a] - m_rf[a];
		} else {

			for (int a = 0; a < 3; a++) 
				moments[p][a] = m[a];
		}
	}
}


/*
 * Compute times and phase of adc samples. 
 */
void compute_adc_samples(int D, const long adc_dims[D], complex float* adc, int N, const struct seq_event ev[N])
{
	md_clear(D, adc_dims, adc, CFL_SIZE);

	long adc_strs[D];
	md_calc_strides(D, adc_strs, adc_dims, CFL_SIZE);

	int e = 0;

	for (int i = 0; i < N; i++) {

		if (SEQ_EVENT_ADC != ev[i].type)
			continue;

		double dwell = 1.E-9 * ev[i].adc.dwell_ns /  ev[i].adc.os;

		long pos[DIMS] = { };
		pos[TE_DIM] = e;

		do {
			double ts = ev[i].start + (pos[1] + 0.5) * dwell;

			MD_ACCESS(D, adc_strs, (pos[READ_DIM] = 0, pos), adc) = ts;
			MD_ACCESS(D, adc_strs, (pos[READ_DIM] = 1, pos), adc) = cexpf(1.i * DEG2RAD(ev[i].adc.phase + ev[i].adc.freq * 360. * (ts - ev[i].mid)));

		} while (md_next(D, adc_dims, PHS1_FLAG, pos));

		e++;
	}

	assert(e == adc_dims[TE_DIM]);
}


void gradients_support(int M, double gradients[M][6], int N, const struct seq_event ev[N])
{
	for (int i = 0; i < M; i++) 
		for (int a = 0; a < 6; a++)
			gradients[i][a] = 0.;

	int g = 0;

	for (int i = 0; i < N; i++) {

		if (SEQ_EVENT_GRADIENT != ev[i].type)
			continue;

		gradients[g][0] = ev[i].start;
		gradients[g][1] = ev[i].mid;
		gradients[g][2] = ev[i].end;
		gradients[g][3] = ev[i].grad.ampl[0];
		gradients[g][4] = ev[i].grad.ampl[1];
		gradients[g][5] = ev[i].grad.ampl[2];

		g++;
	}
}

