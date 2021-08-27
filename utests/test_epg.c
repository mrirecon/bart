#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <assert.h>
#include "misc/mri.h"
#include "num/flpmath.h"

#include "simu/epg.h"

#include "utest.h"

static bool test_cpmg_epg(void)
{
	int N = 4;
	int M = 2 * N;
	complex float signal[N];
	float B1 = 0.8;
	float rf_exc = 90.0 / B1;
	float rf_ref = 180.0 / B1;
	float T1 = 1.000;
	float T2 = 0.100;
	float TE = 0.010;
	float omega = 0.0;

	cpmg_epg_der(N, M, signal, NULL, NULL, NULL, rf_exc, rf_ref, TE, T1, T2, B1, omega);
	float monoexp = expf(-N * TE / T2);

	UT_ASSERT(1E-5 > (cabsf(signal[N - 1]) - monoexp));
 
	return true;
}

UT_REGISTER_TEST(test_cpmg_epg);


static bool test_CPMG_ideal_der(void)
{
	int N = 4;
	int M = 2 * N;
	complex float signal[N];
	complex float states[3][M][N];
	complex float dsignal[4][N];
	complex float dstates[4][3][M][N];
	float B1 = 0.8;
	float rf_exc = 90.0 / B1;
	float rf_ref = 180.0 / B1;
	float T1 = 1.000;
	float T2 = 0.100;
	float TE = 0.010;
	float omega = 0.0;

	cpmg_epg_der(N, M, signal, states, dsignal, dstates, rf_exc, rf_ref, TE, T1, T2, B1, omega);
	float signal_monoexp = expf(-N * TE / T2);
	float dsignal_dT2_monoexp = signal_monoexp * N * TE / (T2 * T2);

	UT_ASSERT(
		   (1E-5 > cabsf(signal[N-1] - signal_monoexp))
		&& (1E-5 > cabsf(dsignal[0][N-1] - (float)0.0)) // close-to-zero T1 dependency
		&& (1E-5 > cabsf(dsignal[1][N-1] - dsignal_dT2_monoexp))
	);
}

UT_REGISTER_TEST(test_CPMG_ideal_der);


static bool test_hyperecho_epg(void)
{
	int N = 23;
	int M = 2 * N;
	complex float signal[N];
	complex float states[3][M][N];
	float B1 = 0.8;
	float FA = 10.0;
	float rf_exc = 10.0 / B1;
	float rf_ref = 180.0 / B1;
	float T1 = 1000;
	float T2 = 100;
	float TE = 10;
	float omega = 0.0;

	float signal_monoexp = expf(-N * TE / T2);

	hyperecho_epg(N, M, signal, states, rf_exc, rf_ref, TE, FA, T1, T2, B1, omega);

	UT_ASSERT(1E-5 > (cabsf(signal[N - 1]) - signal_monoexp));

    return true;
}

UT_REGISTER_TEST(test_hyperecho_epg);
