/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>
#include <assert.h>

#include "misc/misc.h"

#include "seq/gradient.h"
#include "seq/misc.h"

#include "seq_event.h"


int events_counter(enum seq_event_type type, int N, const struct seq_event ev[N])
{
	int ctr = 0;

	if (0 > N)
		return 0;

	for (int i = 0; i < N; i++)
		if (type == ev[i].type)
			ctr++;

	return ctr;
}

int events_idx(int n, enum seq_event_type type, int N, const struct seq_event ev[N])
{
	int count = 0;

	for (int i = 0; i < N; i++) {

		if (type != ev[i].type)
			continue;

		if (count == n)
			return i;

		count++;
	}

	return -1;
}

/*
 * setup a trapezoid consisting of two superimposed triangles
 *
 *    /\/\
 *   / /\ \
 */
int seq_grad_to_event(struct seq_event ev[2], double start, const struct grad_trapezoid* grad, double proj[3])
{
	if (!grad)
		return 0;

	ev[0].type = SEQ_EVENT_GRADIENT;
	ev[0].start = start;
	ev[0].mid = start + grad->rampup;
	ev[0].end = start + grad_duration(grad);

	for (int a = 0; a < 3; a++)
		ev[0].grad.ampl[a] = proj[a] * grad->ampl;

	ev[1].type = SEQ_EVENT_GRADIENT;
	ev[1].start = start + grad->rampup;
	ev[1].mid = start + grad_duration(grad);
	ev[1].end = start + grad_total_time(grad);

	for (int a = 0; a < 3; a++)
		ev[1].grad.ampl[a] = proj[a] * grad->ampl;

	return 2;
}


/*
 * compute TE for E echoes from list of events
 */
void events_get_te(int E, long te[__VLA(E)], int N, const struct seq_event ev[__VLA(N)])
{
	long rf_mid = -1;
	long adc_mid[E];

	int e = 0;
	for (int i = 0; i < N; i++) {

		if (SEQ_EVENT_PULSE == ev[i].type)
			rf_mid = ev[i].mid;

		if (SEQ_EVENT_ADC == ev[i].type) {

			adc_mid[e] = ev[i].mid;
			e++; 
		}
	}

	assert(e == E);
	assert(rf_mid >= 0);

	for (int i = 0; i < E; i++)
		te[i] = adc_mid[i] - rf_mid;
}

/*
 * compute total time from event list
 * gradients_only: if true, only consider gradient events
 * flat_end: if true, use end of flat-top of gradients
 */
double events_end_time(int N, const struct seq_event ev[__VLA(N)], int gradients_only, int flat_end)
{
	double end = 0.;

	for (int i = 0; i < N; i++) {

		if (gradients_only && (SEQ_EVENT_GRADIENT != ev[i].type))
			continue;

		end = ((SEQ_EVENT_GRADIENT == ev[i].type) && flat_end) ? MAX(end, ev[i].mid) : MAX(end, ev[i].end);
	}

	return end;
}

/*
 * compute 0th moment of gradient at time t for a
 * (potentially assymetric) triangle
 */
void moment(double m0[3], double t, const struct seq_event* ev)
{
	assert(SEQ_EVENT_GRADIENT == ev->type);

	for (int a = 0; a < 3; a++)
		m0[a] = 0.;

	if (ev->start > t)
		return;

	double s = ev->start;
	double e = ev->end;
	double c = ev->mid;

	for (int a = 0; a < 3; a++) {

		if (c > s) {

			double A = ev->grad.ampl[a] / (c - s);

			m0[a] += A * pow((MIN(c, t) - s), 2.) / 2.;
		}

		if (e > c) {

			double B = ev->grad.ampl[a] / (e - c);

			if (t > c) {

				m0[a] += B * pow((e - c), 2.) / 2.;

				if (e > t)
					m0[a] -= B * pow((e - t), 2.) / 2.;
			}
		}
	}
}


/*
 * sum moments for a list of events
 */
void moment_sum(double m0[3], double t, int N, const struct seq_event ev[N])
{
	for (int a = 0; a < 3; a++)
		m0[a] = 0.;

	double m_rf[3] = { 0., 0., 0. };

	for (int i = 0; i < N; i++) {

		// FIXME: I do no tunderstand this

		if ((SEQ_EVENT_PULSE == ev[i].type) && (t > ev[i].mid))
			moment_sum(m_rf, ev[i].mid, N, ev);

		if (SEQ_EVENT_GRADIENT != ev[i].type)
			continue;

		double m[3];
		moment(m, t, &ev[i]);

		for (int a = 0; a < 3; a++)
			m0[a] += m[a];
	}

	for (int a = 0; a < 3; a++)
		m0[a] -= m_rf[a];
}
