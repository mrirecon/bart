/* Copyright 2019-2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <math.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "simu/tsegf.h"

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

	debug_printf(DP_WARN, "TSE model is deprecated and will be removed in future releases.\nPlease use the SE model!");

	for (int ind = 0; ind < N; ind++)
		out[ind] = signal_TSE(data, ind);
}


// TSE model using generating function formalism
// following Petrovic, Signal modeling for quantitative magnetic resonance imaging, Dissertation, 2020  and Sumpf et al., IEEE Trans Med Imaging, 2014

const struct signal_model signal_TSE_GEN_defaults = {

	.t1 = 0.781,
	.t2 = 0.1,
	.m0 = 1.,
	.fa = M_PI,
	.te = 0.01,
	.freq_samples = 4086,
};


void TSE_GEN_model(const struct signal_model* data, int N, complex float out[N])
{
	float para[4] = { data->m0, expf(-data->te / data->t1), expf(-data->te / data->t2), cosf(data->fa) };

	tse(N, out, data->freq_samples, para);	
}


const struct signal_model signal_SE_defaults = {

	.t1 = 0.781,
	.t2 = 0.1,
	.m0 = 1.,
	.te = 0.01,
	.ti = 0,
	.tr = 1000.,
	.ir = false,
	.single_repetition = false,
};

// Bernstein, M.A., King, K.F and Zhou, X.J. (2004) Handbook of MRI Pulse Sequences. Elsevier, Amsterdam.
static float signal_SE(const struct signal_model* data, int ind)
{
	float r1 = 1. / data->t1;
	float r2 = 1. / data->t2;
	float m0 = data->m0;
	float te = data->te;
	float tr = data->tr;
	float ti = data->ti;

	if (data->ir)
		// Chapter 14, Basic Pulse Sequences, 14.2 Inversion Recovery, p. 609
		// Add inversion module changing Mz to spin-echo signal model
		return m0 * (1. - 2. * expf( -r1 * ti * ((data->single_repetition) ? 1 : ind))) * (1. - 2. * expf(-(tr - te / 2.) * r1) + expf(-tr * r1)) * expf(-te * r2);
	else
		// Chapter 14, Basic Pulse Sequences, 14.3 Radiofrequency Spin Echo, p. 639
		return m0 * (1. - 2. * expf(-(tr - ((data->single_repetition) ? 1 : ind) * te / 2.) * r1) + expf(-tr * r1)) * expf(- ((data->single_repetition) ? 1 : ind) * te * r2);
}

void SE_model(const struct signal_model* data, int N, complex float out[N])
{
	for (int ind = 0; ind < N; ind++)
		out[ind] = signal_SE(data, ind);
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
	.ms = 0.5,
	.tr = 0.0041,
	.fa = 8. * M_PI / 180.,
	.ir = true,
	.ir_ss = false,
	.short_tr_LL_approx = false,
};

static float signal_looklocker2(const struct signal_model* data, float t1, float m0, int ind, float* m_start)
{
	float fa   = data->fa;
	float tr   = data->tr;
	bool  ir   = data->ir;
	bool ir_ss = data->ir_ss;

	float s0 = 0.; 

	float r1s = 1. / t1 - logf(cosf(fa)) / tr;

	//R Deichmann and A Haase.
	// Quantification of T1 values by SNAPSHOT-FLASH NMR imaging
	// Journal of Magnetic Resonance (1969), Volume 96, Issue 3, 1992.
	// https://doi.org/10.1016/0022-2364(92)90347-A
	// Eq. 5 -> 6
	float mss = m0 * ((data->short_tr_LL_approx) ? 1. / (t1 * r1s) : (1. - expf(-tr / t1)) / (1. - expf(-tr * r1s)));
	
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

static float signal_looklocker(const struct signal_model* data, int ind, float* m_start)
{
	float t1   = data->t1;
	float m0   = data->m0;
	return signal_looklocker2(data, t1, m0, ind, m_start);
}


void looklocker_model(const struct signal_model* data, int N, complex float out[N])
{
	for (int ind = 0; ind < N; ind++)
		out[ind] = signal_looklocker(data, ind, NULL);
}

void looklocker_model2(const struct signal_model* data, int N, complex float out[N])
{	
	float r1s   = 1. / data->t1;
	float m0    = data->m0;
	float tr    = data->tr;
	float mss   = data->ms;

	for (int ind = 0; ind < N; ind++)
		out[ind] =  mss - (mss + m0) * expf(-(ind + 0.5) * tr  * r1s);
}

/*
 * MOLLI
 */

extern void MOLLI_model(const struct signal_model* data, int N, complex float out[N])
{
	assert(0 == (N % data->Hbeats));

	bool  ir   = data->ir;
	float m0   = data->m0;
	float s0   = ir ? (-1) * m0 : m0;
	float temp = s0;
	float r1   = 1.0 / data->t1;
	int cycle  = 0;

	for (int ind = 0; ind < N; ind++) {

		if ((0 < ind) && (0 == ind % (N / data->Hbeats))) {

			temp = m0 + (crealf(out[ind - 1]) - m0) * expf(-1.0 * data->time_T1relax * r1);
			cycle++;
		}
			
		out[ind] = signal_looklocker(data, ind - cycle * (N / data->Hbeats), &temp);
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
 * multi gradient echo model (R2S)
 */
const struct signal_model signal_multi_grad_echo_defaults = {

	.m0 = 1.,
	.m0_water = 1.,
	.m0_fat = .0,
	.t2star = .03, // s
	.off_reson = 20, // Hz
	.te = 1.6 * 1.E-3, // s
	.b0 = 3., // Tesla
};


/*
 * multi gradient echo model (WFR2S)
 */
const struct signal_model signal_multi_grad_echo_fat = {

	.m0 = 1.,
	.m0_water = 0.8,
	.m0_fat = 0.2,
	.t2star = .05, // s
	.off_reson = 20, // Hz
	.te = 1.6 * 1.E-3, // s
	.b0 = 3., // Tesla
	.fat_spec = FAT_SPEC_1,
};



complex float calc_fat_modulation(float b0, float TE, enum fat_spec fs)
{
	enum { FATPEAKS = 6 };

	float ppm[FATPEAKS] = { 0. };
	float amp[FATPEAKS] = { 0. };

	switch (fs) {
	
	case FAT_SPEC_0:
		/* 
		 * ISMRM fat-water toolbox v1 (2012)
		 * Hernando D.
		 */
		ppm[0] = -3.80E-6; amp[0] = 0.087;
		ppm[1] = -3.40E-6; amp[1] = 0.693;
		ppm[2] = -2.60E-6; amp[2] = 0.128;
		ppm[3] = -1.94E-6; amp[3] = 0.004;
		ppm[4] = -0.39E-6; amp[4] = 0.039;
		ppm[5] = +0.60E-6; amp[5] = 0.048;

		break;

	case FAT_SPEC_1:
		/* 
		 * Hamilton G, Yokoo T, Bydder M, Cruite I, Schroeder ME, Sirlin CB, Middleton MS. 
		 * In vivo characterization of the liver fat 1H MR spectrum. 
		 * NMR Biomed 24:784-790 (2011)
		 */
		ppm[0] = -3.80E-6; amp[0] = 0.086;
		ppm[1] = -3.40E-6; amp[1] = 0.537;
		ppm[2] = -2.60E-6; amp[2] = 0.165;
		ppm[3] = -1.94E-6; amp[3] = 0.046;
		ppm[4] = -0.39E-6; amp[4] = 0.052;
		ppm[5] = +0.60E-6; amp[5] = 0.114;

		break;

	}

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

	float TE = data->te * ind;

	float W = data->m0_water;
	float F = data->m0_fat;
	complex float cshift = calc_fat_modulation(data->b0, TE, data->fat_spec);

	complex float z = -1. / data->t2 + 2.i * M_PI * data->off_reson;

	return (W + F * cshift) * cexpf(z * TE);
}



void multi_grad_echo_model(const struct signal_model* data, int N, complex float out[N])
{
	for (int ind = 0; ind < N; ind++)
		out[ind] = signal_multi_grad_echo(data, ind);
}


/*
 * IR Look-Locker multi gradient echo model
 */
const struct signal_model signal_ir_multi_grad_echo_fat_defaults = {

	.m0 = 1.,
	.m0_water = 0.8,
	.m0_fat = 0.2,
	.t2 = .05, // s
	.off_reson = 20, // Hz
	.te = 1.6 * 1.E-3, // s
	.b0 = 3., // Tesla
	.fat_spec = FAT_SPEC_1,

	.t1 = 1.2, // s
	.t1_fat = 0.3, // s
	.tr = 0.008, // s
	.fa = 6. * M_PI / 180.,
	.ir = true,

};

static complex float signal_ir_multi_grad_echo(const struct signal_model* data, int ind_TE, int ind_TI)
{
	assert(data->m0 == data->m0_water + data->m0_fat);

	float TE = data->te * ind_TE;

	float t1 = data->t1;
	float m0 = data->m0_water;
	float W  = signal_looklocker2(data, t1, m0, ind_TI, NULL);

	t1       = data->t1_fat;
	m0       = data->m0_fat;
	float F  = signal_looklocker2(data, t1, m0, ind_TI, NULL);

	complex float cshift = calc_fat_modulation(data->b0, TE, data->fat_spec);

	complex float z = -1. / data->t2 + 2.i * M_PI * data->off_reson;

	return (W + F * cshift) * cexpf(z * TE);
}

void ir_multi_grad_echo_model(const struct signal_model* data, int NE, int N, complex float out[N])
{
	int NI = N / NE;
	for (int ind_e = 0; ind_e < NE; ind_e++)
		for (int ind_i = 0; ind_i < NI; ind_i++)
			out[ind_i + NI * ind_e] = signal_ir_multi_grad_echo(data, ind_e, ind_i);
}




/*
 * Alsop DC, Detre JA, Golay X, Günther M, Hendrikse J, Hernandez-Garcia L, 
 * Lu H, MacIntosh BJ, Parkes LM, Smits M, van Osch MJ, Wang DJJ, Wong EC, 
 * Zaharchuk G. 
 * Recommended implementation of arterial spin-labeled perfusion MRI for clinical applications: 
 * A consensus of the ISMRM perfusion study group and the European consortium for ASL in dementia. 
 * Magn Reson Med 73:102–116 (2015) 
 * doi: 10.1002/mrm.25197 
 * 
 * Herscovitch P, Raichle ME. 
 * What is the correct value for the brain—blood partition coefficient for water? 
 * J Cereb Blood Flow Metab 5:65–69 (1985) 
 * 
 * Cerebral blood flow, blood volume and oxygen utilization. Normal values and effect of age
 * K. L. Leenders, D. Perani, A. A. Lammertsma, J. D. Heather, P. Buckingham, M. J. Healy, 
 * J. M. Gibbs, R. J. Wise, J. Hatazawa & S. Herold.
 * Brain: a journal of neurology, Volume 113, Issue 1, Pages 27–47 (February 1990)
 */
// Default: continuous (CASL) or pseudo-continuous ASL (pCASL)
const struct signal_model signal_buxton_defaults = {

	.m0 = 1.,
	.tr = 0.01,					// Time between PLDs in s
	.t1b = 1.65, 				// Relaxation time of blood in s at 3T
	.t1 = 1.4, 					// Relaxation time of tissue in s at 3T
	.f = 60., 					// Typical cerebal blood flow in gray matter in ml/100g/min
	.delta_t = 1.8,				// Arterial transit time (ATT) in s
	.lambda = 0.9,				// Brain-blood partition coefficient in ml/g for whole brain
	.acquisition_only = false, 	// Only signal during the acquisition is returned
	.pulsed = false, 			// Continuous ASL
	.tau = 1.8,					// Labeling duration in s
	.alpha = 0.85				// Labeling efficiency
};

// Default values for Pulsed ASL (PASL)
const struct signal_model signal_buxton_pulsed = {

	.m0 = 1.,
	.tr = 0.01,					// Time between PLDs in s
	.t1b = 1.65, 				// Relaxation time of blood in s at 3T
	.t1 = 1.4, 					// Relaxation time of tissue in s at 3T
	.f = 60., 					// Typical cerebal blood flow in gray matter in ml/100g/min
	.delta_t = 1.8,				// Arterial transit time (ATT) in s
	.lambda = 0.9, 				// Brain-blood partition coefficient in ml/g for whole brain
	.acquisition_only = false, 	// Only signal during the acquisition is returned
	.pulsed = true,				// Pulsed ASL
	.tau = 0.8, 				// Labeling duration in s
	.alpha = 0.98,				// Labeling efficiency
};

/*
 * Buxton RB, Frank LR, Wong EC, Siewert B, Warach S, Edelman RR. 
 * A general kinetic model for quantitative perfusion imaging with arterial spin labeling. 
 * Magn Reson Med 40:383–396 (1998) 
 */
static float signal_buxton(const struct signal_model* data, int ind)
{
	float m0 = data->m0;
	float lambda = data->lambda;
	float t1 = data->t1;
	float t1b = data->t1b;
	float tau = data->tau;
	float alpha = data->alpha;
	float delta_t = data->delta_t;
	float f = data->f / 6000; // ml/100g/min -> ml/g/s

	float t = ind * data->tr;

	// Return only the part of the signal during the acquisition

	if (data->acquisition_only)
		t += delta_t + tau;


	float m0_b = m0 / lambda;
	float t1_p = 1 / (1 / t1 + f / lambda);
	float k = 1 / t1b - 1 / t1_p;

	float q = 0;
	float delta_M = 0;

	if (data->pulsed) {

		if (t <= delta_t) {

			delta_M = 0;

		} else if (t < delta_t + tau) {

			q = (expf(k * t) * (expf(-k * delta_t) - expf(-k * t))) / (k * (t - delta_t));
			delta_M = 2 * m0_b * alpha * f * (t - delta_t) * expf(-t / t1b) * q;

		} else {

			q = (expf(k * t) * (expf(-k * delta_t) - expf(-k * (tau + delta_t)))) / (k * tau);
			delta_M = 2 * m0_b * alpha * f * tau * expf(-t / t1b) * q;
		}
	} else {

		if (t < delta_t) {

			delta_M = 0;

		} else if (t <= delta_t + tau) {

			q = 1 - expf(-(t - delta_t) / t1_p);
			delta_M = 2 * m0_b * alpha * f * t1_p * expf(-delta_t / t1b) * q;

		} else {

			q = 1 - expf(-tau / t1_p);
			delta_M = 2 * m0_b * alpha * f * t1_p * expf(-delta_t / t1b) * expf(-(t - tau - delta_t) / t1_p) * q;
		}
	}

	return delta_M;
}

void buxton_model(const struct signal_model* data, int N, complex float out[N])
{
	for (int ind = 0; ind < N; ind++)
		out[ind] = signal_buxton(data, ind);
}
