/* Copyright 2022. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 *	Nick Scholand
 */

#include <math.h>

#include "misc/misc.h"

#include "num/specfun.h"

#include "pulse.h"


const struct simdata_pulse simdata_pulse_defaults = {

	.rf_start = 0.,
	.rf_end = 0.01,
	.flipangle = 1.,
	.phase = 0.,
	.bwtp = 4.,
	.nl = 2.,
	.nr = 2.,
	.n = 2.,
	.t0 = 1.,
	.alpha = 0.46,
	.A = 1.,
};


// Sinc definition
static float sincf(float x)
{
	return (0. == x) ? 1. : (sinf(x) / x);
}


// Analytical definition of windowed sinc pulse
// 	! centered around 0
// 	-> Shift by half of pulse length to start pulse at t=0
float pulse_sinc(const struct simdata_pulse* pulse, float t)
{
	float mid = (pulse->rf_start + pulse->rf_end) / 2.;

	t -= mid;

	return pulse->A * ((1. - pulse->alpha) + pulse->alpha * cosf(M_PI * t / (pulse->n * pulse->t0)))
				* sincf(M_PI * t / pulse->t0);
}


// Analytical integral of windowed Sinc
static float sinc_antiderivative(const struct simdata_pulse* pulse, float t)
{
	float c = M_PI / pulse->n / pulse->t0;

	return 	pulse->A * pulse->t0 * (pulse->alpha * (Si(c * t * (pulse->n-1)) + Si(c * t * (pulse->n+1)))
			- 2 * (pulse->alpha-1) * Si(M_PI / pulse->t0 * t)) / 2 / M_PI;
}


float sinc_integral(const struct simdata_pulse* pulse)
{
	float shift = (pulse->rf_end - pulse->rf_start) / 2.;

	return sinc_antiderivative(pulse, shift) - sinc_antiderivative(pulse, -shift);
}


// Assume symmetric windowed sinc pulses
// 	- Ensure windowed sinc leads to 90 deg rotation if its integral is pi/2
void pulse_create(struct simdata_pulse* pulse, float rf_start, float rf_end, float angle /*[deg]*/, float phase, float bwtp, float alpha)
{
	pulse->rf_start = rf_start;
	pulse->rf_end = rf_end;
	pulse->flipangle = 90.;
	pulse->phase = phase;
	pulse->nl = bwtp / 2.;	// Symmetry condition: nl=nr
	pulse->nr = bwtp / 2.;
	pulse->n = MAX(pulse->nl, pulse->nr);
	pulse->t0 = (rf_end - rf_start) / (2. + (pulse->nl - 1.) + (pulse->nr - 1.));
	pulse->alpha = alpha;
	pulse->A = 1.;

	float integral = sinc_integral(pulse);

	float scaling = M_PI / 2. / integral;

	pulse->flipangle = angle;

	pulse->A = scaling / 90. * angle;
}
