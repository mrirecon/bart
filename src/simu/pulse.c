/* Copyright 2022. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 *	Nick Scholand
 */

#include <math.h>
#include <assert.h>

#include "misc/misc.h"

#include "num/specfun.h"

#include "pulse.h"


const struct simdata_pulse simdata_pulse_defaults = {

	.rf_start = 0.,
	.rf_end = 0.001,
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

/* sinc pulse */

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
void sinc_pulse_init(struct simdata_pulse* pulse, float rf_start, float rf_end, float angle /*[deg]*/, float phase, float bwtp, float alpha)
{
	pulse->rf_start = rf_start;	// [s]
	pulse->rf_end = rf_end;		// [s]
	pulse->flipangle = 90.;		// [deg]
	pulse->phase = phase;		// [rad]
	pulse->nl = bwtp / 2.;		// Symmetry condition: nl=nr
	pulse->nr = bwtp / 2.;
	pulse->n = MAX(pulse->nl, pulse->nr);
	pulse->t0 = (rf_end - rf_start) / (2. + (pulse->nl - 1.) + (pulse->nr - 1.)); // [s]
	pulse->alpha = alpha;
	pulse->A = 1.;

	float integral = sinc_integral(pulse);

	float scaling = M_PI / 2. / integral;

	pulse->flipangle = angle;

	pulse->A = scaling / 90. * angle;
}


/* Hyperbolic Secant Pulse
 *
 * Baum J, Tycko R, Pines A.
 * Broadband and adiabatic inversion of a two-level system by phase-modulated pulses.
 * Phys Rev A 1985;32:3435-3447.
 *
 * Bernstein MA, King KF, Zhou XJ.
 * Handbook of MRI Pulse Sequences.
 * Chapter 6
 */

const struct hs_pulse hs_pulse_defaults = {

	.a0 = 13000.,
	.beta = 800.,
	.mu = 4.9, /* sech(x)=0.01*/
	.duration = 0.01,
        .on = false,
};

static float sechf(float x)
{
	return 1. / coshf(x);
}

float pulse_hypsec_am(const struct hs_pulse* pulse, float t /*[s]*/)
{
        //Check adiabatic condition
        assert(pulse->a0 > sqrtf(pulse->mu)*pulse->beta);

        return pulse->a0 * sechf(pulse->beta * (t-pulse->duration / 2.));
}

float pulse_hypsec_fm(const struct hs_pulse* pulse, float t /*[s]*/)
{
	return -pulse->mu * pulse->beta * tanhf(pulse->beta * (t - pulse->duration / 2.));
}

float pulse_hypsec_phase(const struct hs_pulse* pulse, float t /*[s]*/)
{
        return pulse->mu * logf(sechf(pulse->beta * (t-pulse->duration / 2.)))
                + pulse->mu * logf(pulse->a0);
}

