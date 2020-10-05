/* Copyright 2019-2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <math.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/mri.h"

#include "signals.h"




const struct signal_model signal_TSE_defaults = {

	.t2 = 0.1,
	.m0 = 1.,
	.te = 0.01,
};

static float signal_TSE(const struct signal_model* data, int ind)
{
	float r2 = 1. / data->t2;
	float m0 = data->m0;
	float te = data->te;

	return m0 * expf(-ind * te * r2);
}

void TSE_model(const struct signal_model* data, int N, complex float out[N])
{
	for (int ind = 0; ind < N; ind++)
		out[ind] = signal_TSE(data, ind);
}


/*
 * Hybrid-state free precession in nuclear magnetic resonance. 
 * Jakob Assländer, Dmitry S. Novikov, Riccardo Lattanzi, Daniel K. Sodickson & Martijn A. Cloos.
 * Communications Physics. Volume 2, Article number: 73 (2019)
 */
const struct signal_model signal_hsfp_defaults = {

	.t1 = 0.781,
	.t2 = 0.065,
	.tr = 0.0045,
	.beta = -1.,
};



struct r0_a_sum {

	float r0;
	float a;
};

static struct r0_a_sum r0_a_sum(const struct signal_model* data, int N, const float pa[N], int ind)
{
	struct r0_a_sum sum = { 0., 0. };

	for (int i2 = 0; i2 < ind; i2++) {

		float x = fabsf(pa[i2]);

		sum.a += sinf(x) * sinf(x) / data->t2 + cosf(x) * cosf(x) / data->t1;
		sum.r0 += cosf(x) * expf(sum.a * data->tr);
	}

	sum.a = expf(-sum.a * data->tr);
	sum.r0 *= data->tr;

	return sum;
}


static float r0(const struct signal_model* data, int N, const float pa[N])
{
	struct r0_a_sum sum = r0_a_sum(data, N, pa, N);

	return data->beta / data->t1 * sum.a / (1. - data->beta * sum.a) * sum.r0;
}


static float signal_hsfp(const struct signal_model* data, float r0_val, int N, const float pa[N], int ind)
{
	struct r0_a_sum sum = r0_a_sum(data, N, pa, ind);

	return sum.a * (r0_val + 1. / data->t1 * sum.r0);
}


void hsfp_simu(const struct signal_model* data, int N, const float pa[N], complex float out[N])
{
	float r0_val = r0(data, N, pa);

	for (int ind = 0; ind < N; ind++)
		out[ind] = signal_hsfp(data, r0_val, N, pa, ind);
}


/*
 * Time saving in measurement of NMR and EPR relaxation times.
 * Look DC, Locker DR.  Rev Sci Instrum 1970;41:250–251.
 */
const struct signal_model signal_looklocker_defaults = {

	.t1 = 1.,
	.m0 = 1.,
	.tr = 0.0041,
	.fa = 8. * M_PI / 180.,
	.ir = true,
	.ir_ss = false,
};

static float signal_looklocker(const struct signal_model* data, int ind, float* m_start)
{
	float fa   = data->fa;
	float t1   = data->t1;
	float m0   = data->m0;
	float tr   = data->tr;
	bool  ir   = data->ir;
	bool ir_ss = data->ir_ss;

	float s0 = 0.; 

	float r1s = 1. / t1 - logf(cosf(fa)) / tr;
	float mss = m0 / (t1 * r1s);
	
	if (NULL == m_start) {

		if (ir_ss)
			s0 = -1. * mss;		
		else
			s0 = ir ? (-1. * m0) : m0;			
	} else {

		s0 = *m_start;
	}

	return mss - (mss - s0) * expf(-ind * tr * r1s);
}

void looklocker_model(const struct signal_model* data, int N, complex float out[N])
{
	for (int ind = 0; ind < N; ind++)
		out[ind] = signal_looklocker(data, ind, NULL);
}

/*
 * MOLLI
 */

extern void MOLLI_model(const struct signal_model* data, int N, int Hbeats, float time_T1relax, complex float out[N])
{
	assert(0 == (N % Hbeats));

	bool  ir   = data->ir;
	float m0   = data->m0;
	float s0   = ir ? (-1) * m0 : m0;
	float temp = s0;
	float r1   = 1.0 / data->t1;
	int cycle  = 0;

	for (int ind = 0; ind < N; ind++) {

		if((0 < ind) && (0 == ind % (N / Hbeats))) {

			temp = m0 + (out[ind-1] - m0) * expf(-1.0 * time_T1relax * r1);
			cycle++;
		}
			
		out[ind] = signal_looklocker(data, ind - cycle * (N / Hbeats), &temp);
	}
}

/*
 * Inversion recovery TrueFISP: Quantification of T1, T2, and spin density.
 * Schmitt, P. , Griswold, M. A., Jakob, P. M., Kotas, M. , Gulani, V. , Flentje, M. and Haase, A., 
 * Magn. Reson. Med., 51: 661-667. doi:10.1002/mrm.20058, (2004)
 */
const struct signal_model signal_IR_bSSFP_defaults = {

	.t1 = 1.,
	.t2 = 0.1,
	.m0 = 1.,
	.tr = 0.0045,
	.fa = 45. * M_PI / 180.,
};

static float signal_IR_bSSFP(const struct signal_model* data, int ind)
{
	float fa = data->fa;
	float t1 = data->t1;
	float t2 = data->t2;
	float m0 = data->m0;
	float tr = data->tr;

	float fa2 = fa / 2.;
	float s0 = m0 * sinf(fa2);
	float r1s = (cosf(fa2) * cosf(fa2)) / t1 + (sinf(fa2) * sinf(fa2)) / t2;
	float mss = m0 * sinf(fa) / ((t1 / t2 + 1.) - cosf(fa) * (t1 / t2 - 1.));

	return mss - (mss + s0) * expf(-ind * tr * r1s);
}

void IR_bSSFP_model(const struct signal_model* data, int N, complex float out[N])
{
	for (int ind = 0; ind < N; ind++)
		out[ind] = signal_IR_bSSFP(data, ind);
}


/*
 * multi gradient echo model (WFR2S)
 */
const struct signal_model signal_multi_grad_echo_defaults = {

	.m0 = 1.,
	.m0_water = .80,
	.m0_fat = .20,
	.t2star = .03, // s
	.off_reson = 100, // Hz
	.te = 3. * 1.E-3, // s
	.b0 = 3., // Tesla
};


complex float calc_fat_modulation(float b0, float TE)
{
	enum { FATPEAKS = 6 };
	float ppm[FATPEAKS] = { -3.80, -3.40, -2.60, -1.94, -0.39, +0.60 };
	float amp[FATPEAKS] = { 0.087, 0.693, 0.128, 0.004, 0.039, 0.048 };

	complex float out = 0.;

	for (int pind = 0; pind < FATPEAKS; pind++) {

		complex float phs = 2.i * M_PI * GYRO * b0 * ppm[pind] * TE;
		out += amp[pind] * cexpf(phs);
	}

	return out;
}


static complex float signal_multi_grad_echo(const struct signal_model* data, int ind)
{
	assert(data->m0 == data->m0_water + data->m0_fat);

	complex float TE = data->te * ind + 0.i;

	complex float cshift = calc_fat_modulation(data->b0, TE);

	float W = data->m0_water;
	float F = data->m0_fat;

	complex float z = -1. / data->t2star + 2.i * M_PI * data->off_reson;

	return (W + F * cshift) * cexpf(z * TE);
}

void multi_grad_echo_model(const struct signal_model* data, int N, complex float out[N])
{
	for (int ind = 0; ind < N; ind++)
		out[ind] = signal_multi_grad_echo(data, ind);
}



