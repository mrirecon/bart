/* Copyright 2022-2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>
#include <assert.h>
#include <complex.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/types.h"

#include "num/specfun.h"

#include "seq/pulse_library.h"

#include "pulse.h"

DEF_TYPEID(pulse_sinc);
DEF_TYPEID(pulse_sms);
DEF_TYPEID(pulse_rect);
DEF_TYPEID(pulse_hypsec);
DEF_TYPEID(pulse_arb);
DEF_TYPEID(pulse_gauss);

extern inline complex float pulse_eval(const struct pulse* p, float t);


/* windowed sinc pulse */

static float sinc_windowed(float alpha, float t, float n)
{
	return ((1. - alpha) + alpha * cosf(M_PI * t / n)) * sincf(M_PI * t);
}

/* analytical integral of windowed sinc */

static float sinc_windowed_antiderivative(float alpha, float t, float n)
{
	return (alpha * (Si(M_PI * t * (n + 1.) / n) + Si(M_PI * t * (n - 1.) / n))
			- 2. * (alpha - 1.) * Si(M_PI * t)) / (2. * M_PI);
}



// Analytical definition of windowed sinc pulse
// 	! centered around 0
// 	-> Shift by half of pulse length to start pulse at t=0
static complex float pulse_sinc(const struct pulse_sinc* ps, float t)
{
	float mid = CAST_UP(ps)->duration / 2.;
	float t0 = CAST_UP(ps)->duration / ps->bwtp;

	assert((0 <= t) && (t <= CAST_UP(ps)->duration));

	return ps->A * sinc_windowed(ps->alpha, (t - mid) / t0, ps->bwtp / 2.);
}

float pulse_sinc_integral(const struct pulse_sinc* ps)
{
	float mid = CAST_UP(ps)->duration / 2.;
	float t0 = CAST_UP(ps)->duration / ps->bwtp;

	return ps->A * t0 * (sinc_windowed_antiderivative(ps->alpha, +mid / t0, ps->bwtp / 2.)
			- sinc_windowed_antiderivative(ps->alpha, -mid / t0, ps->bwtp / 2.));
}


static complex float pulse_sinc_eval(const struct pulse* _ps, float t)
{
	auto ps = CAST_DOWN(pulse_sinc, _ps);

	return pulse_sinc(ps, t);
}


const struct pulse_sinc pulse_sinc_defaults = {

	.super.duration = 0.001,
	.super.flipangle = 1.,
	.super.eval = pulse_sinc_eval,
	.super.TYPEID = &TYPEID2(pulse_sinc),
	// .pulse.phase = 0.,

	.alpha = 0.46,
	.A = 1.,
	.bwtp = 4.,
};


// Assume symmetric windowed sinc pulses
// 	- Ensure windowed sinc leads to 90 deg rotation if its integral is pi/2
void pulse_sinc_init(struct pulse_sinc* ps, float duration, float angle /*[deg]*/, float phase, float bwtp, float alpha)
{
	ps->super.duration = duration;
	ps->super.flipangle = angle;
	ps->super.eval = pulse_sinc_eval;
	(void)phase;
//	ps->pulse.phase = phase;

	ps->bwtp = bwtp;
	ps->alpha = alpha;
	ps->A = 1.;

	ps->A = DEG2RAD(angle) / pulse_sinc_integral(ps);
}


/* SMS pulse: sinc pulse with phase modulation; mb_factor=1 equals sinc_pulse */

/* SMS-multiband phase modulation
 * */
static complex float pulse_sms_phase_modulation(const struct pulse_sms* ps, float t) 
{
	float phase = 0.;
	complex float mod = 0. + 0.i;

	for (int i = 0; i < ps->mb_factor; i++) {

		phase = (t - CAST_UP(ps)->duration / 2.) 
			* ((i - (ps->mb_factor - 1.) / 2.) * ps->SMS_dist)
			* (-2. * M_PI * (ps->bwtp / (ps->super.duration * ps->slice_th)))
			+ 2 * M_PI * (ps->mb_part * i) / ps->mb_factor;

		mod += cosf(phase) + 1.i * sinf(phase);
	}

	return mod / ps->mb_factor;
}	


static complex float pulse_sms(const struct pulse_sms* ps, float t)
{
	float mid = CAST_UP(ps)->duration / 2.;
	float t0 = CAST_UP(ps)->duration / ps->bwtp;

	assert((0 <= t) && (t <= CAST_UP(ps)->duration));

	float rf = ps->A * sinc_windowed(ps->alpha, (t - mid) / t0, ps->bwtp / 2.);
	complex float pm = 1. + 0.i;
	
	if (1 != ps->mb_factor)
		pm = pulse_sms_phase_modulation(ps, t);

	return rf * pm;
}

float pulse_sms_integral(const struct pulse_sms* ps)
{
	float mid = CAST_UP(ps)->duration / 2.;
	float t0 = CAST_UP(ps)->duration / ps->bwtp;

	return (ps->A / ps->mb_factor) * t0 * (sinc_windowed_antiderivative(ps->alpha, +mid / t0, ps->bwtp / 2.)
			- sinc_windowed_antiderivative(ps->alpha, -mid / t0, ps->bwtp / 2.));
}


static complex float pulse_sms_eval(const struct pulse* _ps, float t)
{
	auto ps = CAST_DOWN(pulse_sms, _ps);

	return pulse_sms(ps, t);
}


const struct pulse_sms pulse_sms_defaults = {

	.super.duration = 0.001,
	.super.flipangle = 1.,
	.super.eval = pulse_sms_eval,
	.super.TYPEID = &TYPEID2(pulse_sms),
	// .pulse.phase = 0.,

	.alpha = 0.46,
	.A = 1.,
	.bwtp = 4.,
	.mb_factor = 3,
	.mb_part = 0,
	.SMS_dist = 27.e-3,
	.slice_th = 6.e-3,
};


void pulse_sms_init(struct pulse_sms* ps, float duration, float angle /*[deg]*/, float /* phase */, float bwtp, float alpha, 
			int mb, int part, float dist, float th)
{
	ps->super.duration = duration;
	ps->super.flipangle = angle;
	ps->super.eval = pulse_sms_eval;

	ps->alpha = alpha;
	ps->bwtp = bwtp;
	ps->mb_factor = mb;
	ps->mb_part = part;
	ps->SMS_dist = dist;
	ps->slice_th = th;
	ps->A = 1.;

	ps->A = DEG2RAD(angle) / pulse_sms_integral(ps);
}


/* Rectangular pulse */

void pulse_rect_init(struct pulse_rect* pr, float duration, float angle /*[deg]*/, float phase)
{
	pr->super.duration = duration;
	pr->super.flipangle = angle;

	assert(0. == phase);
//	pulse->phase = phase;		// [rad]

	pr->A = angle / duration * M_PI / 180.;
}

static float pulse_rect(const struct pulse_rect* pr, float t)
{
	(void)t;
	return pr->A;
}

static complex float pulse_rect_eval(const struct pulse* _pr, float t)
{
	auto pr = CAST_DOWN(pulse_rect, _pr);

	return pulse_rect(pr, t);
}

const struct pulse_rect pulse_rect_defaults = {

	.super.duration = 0.001,
	.super.flipangle = 1.,
	.super.eval = pulse_rect_eval,
	.super.TYPEID = &TYPEID2(pulse_rect),

	.A = 1.,
};


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

static float sechf(float x)
{
	return 1. / coshf(x);
}

static float pulse_hypsec_am(const struct pulse_hypsec* hs, float t /*[s]*/)
{
        // Check adiabatic condition
        assert(hs->A > (sqrtf(hs->mu) * hs->beta));

        return hs->A * sechf(hs->beta * (t - CAST_UP(hs)->duration / 2.));
}

#if 0
static float pulse_hypsec_fm(const struct pulse_hypsec* hs, float t /*[s]*/)
{
	return -hs->mu * hs->beta * tanhf(hs->beta * (t - CAST_UP(hs)->duration / 2.));
}
#endif

float pulse_hypsec_phase(const struct pulse_hypsec* hs, float t /*[s]*/)
{
        return hs->mu * logf(sechf(hs->beta * (t - CAST_UP(hs)->duration / 2.)))
                + hs->mu * logf(hs->a0);
}

static complex float pulse_hypsec_eval(const struct pulse* _pr, float t)
{
	auto pr = CAST_DOWN(pulse_hypsec, _pr);

	return pulse_hypsec_am(pr, t) * cexp(1.i * pulse_hypsec_phase(pr, t));
}

const struct pulse_hypsec pulse_hypsec_defaults = {

	.super.duration = 0.01,
	.super.flipangle = 180.,
	.super.eval = pulse_hypsec_eval,
	.super.TYPEID = &TYPEID2(pulse_hypsec),
//	.pulse.phase = 0.,

	.a0 = 14.E-6,
	.beta = 800.,
	.mu = 4.9, /* sech(x)=0.01*/
	.A = 1.,
	.gamma =  GYRO,
};


void pulse_hypsec_init(float gamma, struct pulse_hypsec* pr)
{
	pr->gamma = gamma;
	pr->A = pr->a0 * 2 * M_PI * pr->gamma;
}


float pulse_hypsec_integral(const struct pulse_hypsec* hs)
{
	return 4. * hs->A / hs->beta * atanf(tanhf(hs->beta * CAST_UP(hs)->duration / 4.));
}



static complex float pulse_arb_eval(const struct pulse* _pa, float t)
{
	auto pa = CAST_DOWN(pulse_arb, _pa);

	assert((t >= 0.) && (t <= _pa->duration));

	int idx = (int)(t / _pa->duration * pa->samples);

	assert(idx < pa->samples);

	return pa->A * pa->values[idx];
}

const struct pulse_arb pulse_arb_oc_cest_sat_defaults = {

	.super.duration = 0.1,
	.super.flipangle = 1482.66, // B1rms = 1uT for oc_cest_sat_pulse
	.super.eval = pulse_arb_eval,
	.super.TYPEID = &TYPEID2(pulse_arb),

	.samples = 1000,
	.values = oc_cest_sat_pulse,
	.A = 1.,
	.gamma =  GYRO,
};


void pulse_arb_init(struct pulse_arb* pa, float gamma)
{
	pa->gamma = gamma;
	pa->A = DEG2RAD(pa->super.flipangle) / pulse_arb_integral(pa);

}


float pulse_arb_integral(const struct pulse_arb* pa)
{
	float sum = 0.;

	for (int i = 0; i < pa->samples; i++)
		sum += pa->A * cabsf(pa->values[i]);

	return sum * (pa->super.duration / pa->samples);
}


static float gauss_windowed(float alpha, float t, float n)
{
	return ((1. - alpha) + alpha * cosf(M_PI * t / n)) * expf(- M_PI * t * t);
}

static complex float pulse_gauss(const struct pulse_gauss* pg, float t)
{
	float mid = CAST_UP(pg)->duration / 2.;
	float t0 = CAST_UP(pg)->duration / pg->bwtp;

	assert((0 <= t) && (t <= CAST_UP(pg)->duration));

	return pg->A * gauss_windowed(pg->alpha, (t - mid) / t0, pg->bwtp / 2.);
}

float pulse_gauss_integral(const struct pulse_gauss* pg)
{
	const int N = 1000;
	float t0 = CAST_UP(pg)->duration / N;

	double sum = 0.;

	for (int i = 0; i < N; i++)
		sum += pulse_gauss(pg, i * t0);

	return pg->A * sum * t0;
}


static complex float pulse_gauss_eval(const struct pulse* _ps, float t)
{
	auto pg = CAST_DOWN(pulse_gauss, _ps);

	return pulse_gauss(pg, t);
}


const struct pulse_gauss pulse_gauss_defaults = {

	.super.duration = 0.025,
	.super.flipangle = 360.,
	.super.eval = pulse_gauss_eval,
	.super.TYPEID = &TYPEID2(pulse_gauss),
	// .pulse.phase = 0.,

	.alpha = 0.5,
	.A = 1.,
	.bwtp = .2,
};


void pulse_gauss_init(struct pulse_gauss* pg, float duration, float angle /*[deg]*/, float phase, float bwtp, float alpha)
{
	pg->super.duration = duration;
	pg->super.flipangle = angle;
	pg->super.eval = pulse_gauss_eval;
	(void)phase;

	pg->bwtp = bwtp;
	pg->alpha = alpha;
	pg->A = 1.;

	pg->A = DEG2RAD(angle) / pulse_gauss_integral(pg);
}

