/* Copyright 2018. Martin Uecker.
 * Copyright 2023. Insitute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */


#include <math.h>
#include <string.h>

#include "misc/mri.h"

#include "num/ode.h"
#include "num/linalg.h"

#include "simu/bloch.h"

#include "utest.h"


static void ode_direct_sa_wrap(float h, float tol, int N, int P, float x[P + 1][N],
	float st, float end, void* data,
	void (*der)(void* data, float* out, float t, const float* yn),
	void (*pdy)(void* data, float* out, float t, const float* yn),
	void (*pdp)(void* data, float* out, float t, const float* yn))
{
	NESTED(void, call_der, (float* out, float t, const float* in)) { der(data, out, t, in); };
	NESTED(void, call_pdy, (float* out, float t, const float* in)) { pdy(data, out, t, in); };
	NESTED(void, call_pdp, (float* out, float t, const float* in)) { pdp(data, out, t, in); };

	ode_direct_sa(h, tol, N, P, x, st, end, call_der, call_pdy, call_pdp);
}



struct bloch_s {

	float r1;
	float r2;
	float gb[3];
};

static void bloch_fun(void* _data, float* out, float t, const float* in)
{
	struct bloch_s* data = _data;
	(void)t;

	bloch_ode(out, in, data->r1, data->r2, data->gb);
}

static void bloch_pdy2(void* _data, float* out, float t, const float* in)
{
	struct bloch_s* data = _data;
	(void)t;

	bloch_pdy((float(*)[3])out, in, data->r1, data->r2, data->gb);
}

static void bloch_pdp2(void* _data, float* out, float t, const float* in)
{
	struct bloch_s* data = _data;
	(void)t;

	bloch_pdp((float(*)[3])out, in, data->r1, data->r2, data->gb);
}


static bool test_ode_bloch(void)
{
	struct bloch_s data = { 1. / WATER_T1, 1. / WATER_T2, { 0., 0., GAMMA_H1 * SKYRA_GRADIENT * 0.0001 } };

	float x[3] = { 1., 0., 0. };
	float x0[3] = { 1., 0., 0. };
	float x2[3];
	float h = 0.1;
	float tol = 0.000001;
	float end = 0.2;

	NESTED(void, call_fun, (float* out, float t, const float* in)) { bloch_fun(&data, out, t, in); };

	ode_interval(h, tol, 3, x, 0., end, call_fun);
	bloch_relaxation(x2, end, x0, data.r1, data.r2, data.gb);

	float err2 = 0.;

	for (int i = 0; i < 3; i++)
		err2 += powf(x[i] - x2[i], 2.);

#if __GNUC__ >= 10
	return (err2 < 1.E-6);
#else
	return (err2 < 1.E-7);
#endif
}

UT_REGISTER_TEST(test_ode_bloch);



static bool test_bloch_matrix(void)
{
	struct bloch_s data = { 1. / WATER_T1, 1. / WATER_T2, { 0., 0., GAMMA_H1 * SKYRA_GRADIENT * 0.0001 } };

	float m[4][4];
	bloch_matrix_ode(m, data.r1, data.r2, data.gb);

	float m0[4] = { 0.1, 0.2, 0.3, 1. };

	float out[3];
	bloch_ode(out, m0, data.r1, data.r2, data.gb);

	float out2[4];

	for (unsigned int i = 0; i < 4; i++) {

		out2[i] = 0.;

		for (unsigned int j = 0; j < 4; j++)
			out2[i] += m[i][j] * m0[j];
	}

	return    (0. == out[0] - out2[0])
	       && (0. == out[1] - out2[1])
	       && (0. == out[2] - out2[2])
	       && (0. == out2[3]);
}

UT_REGISTER_TEST(test_bloch_matrix);



static bool test_ode_matrix_bloch(void)
{
	struct bloch_s data = { 1. / WATER_T1, 1. / WATER_T2, { 0., 0., GAMMA_H1 * SKYRA_GRADIENT * 0.0001 } };

	float x[4] = { 1., 0., 0., 1. };
	float x0[3] = { 1., 0., 0. };
	float x2[3];
	float h = 0.1;
	float tol = 0.000001;
	float end = 0.2;

	float m[4][4];
	bloch_matrix_ode(m, data.r1, data.r2, data.gb);

	ode_matrix_interval(h, tol, 4, x, 0., end, m);
	bloch_relaxation(x2, end, x0, data.r1, data.r2, data.gb);

	float err2 = 0.;

	for (int i = 0; i < 3; i++)
		err2 += powf(x[i] - x2[i], 2.);

	return (err2 < 1.E-6);
}

UT_REGISTER_TEST(test_ode_matrix_bloch);



struct sa_data_s {

	int N;
	float* r;
};

static void sa_fun(void* _data, float* out, float t)
{
	struct sa_data_s* data = _data;

	for (int i = 0; i < data->N; i++)
		out[i] = expf(data->r[i] * t);
}

static void sa_der(void* _data, float* out, float t, const float* in)
{
	struct sa_data_s* data = _data;
	(void)t;

	for (int i = 0; i < data->N; i++)
		out[i] = data->r[i] * in[i];
}

static void sa_pdy(void* _data, float* out, float t, const float* in)
{
	struct sa_data_s* data = _data;
	(void)t; (void)in;

	for (int i = 0; i < data->N; i++)
		for (int j = 0; j < data->N; j++)
			out[i * data->N + j] = (i == j) ? data->r[i] : 0.;
}

static void sa_pdp(void* _data, float* out, float t, const float* in)
{
	struct sa_data_s* data = _data;
	(void)t;

	for (int i = 0; i < data->N; i++)
		for (int j = 0; j < data->N; j++)
			out[i * data->N + j] = (i == j) ? in[i] : 0.;
}

static bool test_ode_sa(void)
{
	float h = 0.1;
	float tol = 0.000001;
	float end = 0.2;

	struct sa_data_s data = { 1, (float[1]){ -WATER_T2 } };

	float xp[2][1] = { { 1. }, { 0. } };
	ode_direct_sa_wrap(h, tol, 1, 1, xp, 0., end, &data, sa_der, sa_pdy, sa_pdp);

	float x[1] = { 1. };
	sa_fun(&data, x, end);

	if (fabsf(x[0] - xp[0][0]) > 1.E-30)
		return false;

	float y[1];
	float q = 0.0001;
	data.r[0] += q;
	sa_fun(&data, y, end);

	float df = y[0] - x[0];
	float err = fabsf(df - xp[1][0] * q);

	return err < 1.E-9;
}

UT_REGISTER_TEST(test_ode_sa);


static bool test_ode_sa2(void)
{
	float h = 0.1;
	float tol = 0.000001;
	float end = 0.2;

	int N = 10;
	float r[N];

	for (int i = 0; i < N; i++)
		r[i] = -WATER_T2 / (1 + i);

	struct sa_data_s data = { N, r };

	float xp[N + 1][N];
	for (int i = 0; i < N; i++) {

		xp[0][i] = 1.;
		for (int j = 0; j < N; j++)
			xp[1 + j][i] = 0;
	}

	ode_direct_sa_wrap(h, tol, N, N, xp, 0., end, &data, sa_der, sa_pdy, sa_pdp);

	float x[N];
	memset(x, 0, sizeof x);		// -fanalyzer uninitialized
	sa_fun(&data, x, end);

	for (int i = 0; i < N; i++) {

		float err = fabsf(x[i] - xp[0][i]);

		if (err > 1.E-6)
			return false;
	}

	float y[N];
	float q = 0.0001;

	for (int j = 0; j < N; j++) {

		data.r[j] += q;
		sa_fun(&data, y, end);
		data.r[j] -= q;

		for (int i = 0; i < N; i++) {

			float df = y[i] - x[i];
			float sa = xp[1 + j][i] * q;
			float err = fabsf(df - sa);

			if (err > 1.E-7)
				return false;
		}
	}

	return true;
}

UT_REGISTER_TEST(test_ode_sa2);




static bool test_ode_sa_bloch(void)
{
	struct bloch_s data = { 1. / WATER_T1, 1. / WATER_T2, { 0., 0., GAMMA_H1 * SKYRA_GRADIENT * 0.0001 } };

	float xp[3][3] = { { 1., 0., 0. }, { 0. }, { 0. } };
	float x0[3] = { 1., 0., 0. };
	float x2[3];
	float x2r1[3];
	float x2r2[3];
	float h = 0.1;
	float tol = 0.000001;
	float end = 0.2;
	float q = 1.E-3;

	ode_direct_sa_wrap(h, tol, 3, 2, xp, 0., end, &data, bloch_fun, bloch_pdy2, bloch_pdp2);
	bloch_relaxation(x2, end, x0, data.r1, data.r2, data.gb);
	bloch_relaxation(x2r1, end, x0, data.r1 + q, data.r2, data.gb);
	bloch_relaxation(x2r2, end, x0, data.r1, data.r2 + q, data.gb);

	float err2 = 0.;

	for (int i = 0; i < 3; i++)
		err2 += powf(xp[0][i] - x2[i], 2.);

	if (err2 > 1.E-6)
		return false;

	for (int i = 0; i < 3; i++)
		err2 += powf(xp[0][i] - x2[i], 2.);


	for (int i = 0; i < 3; i++) {

		float err = fabsf(q * xp[1][i] - (x2r1[i] - x2[i]));

		if (err > 1.E-7)
			return false;
	}

	for (int i = 0; i < 3; i++) {

		float err = fabsf(q * xp[2][i] - (x2r2[i] - x2[i]));

		if (err > 1.E-7)
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_ode_sa_bloch);



static bool test_bloch_relaxation(void)
{
	float x0[3] = { 0., 1., 0. };
	float x[3] = { 0. };
	float ref[3] = { 0., 1. / M_E, 1. - 1. / M_E };

        struct bloch_s data = { 1., 1., { 0., 0., 0. } };

        bloch_relaxation(x, 1., x0, data.r1, data.r2, data.gb);

	float err2 = 0.;

	for (int i = 0; i < 3; i++)
		err2 += powf(x[i] - ref[i], 2.);

	UT_RETURN_ASSERT(err2 < 1.E-6);
}

UT_REGISTER_TEST(test_bloch_relaxation);




static bool test_bloch_excitation(void)
{
	float x0[3] = { 0., 0., 1. };
	float x[3] = { 0. };
	float ref[3] = { 0., 1., 0. };

        struct bloch_s data = { 1., 1., { M_PI / 2., 0., 0. } };

        bloch_excitation(x, 1., x0, data.r1, data.r2, data.gb);

	float err2 = 0.;

	for (int i = 0; i < 3; i++)
		err2 += powf(x[i] - ref[i], 2.);

	UT_RETURN_ASSERT(err2 < 1.E-6);
}

UT_REGISTER_TEST(test_bloch_excitation);



static bool test_bloch_excitation2_phase(void)
{
	float x0[3] = { 0., 0., 1. };
	float x[3] = { 0. };
	float ref[3] = { 1., 0., 0. };

	bloch_excitation2(x, x0, M_PI / 2., M_PI / 2.);

	float err2 = 0.;

	for (int i = 0; i < 3; i++)
		err2 += powf(x[i] - ref[i], 2.);

	UT_RETURN_ASSERT(err2 < 1.E-6);
}

UT_REGISTER_TEST(test_bloch_excitation2_phase);


// Test if bloch_b1_pdp performs as expected
static bool test_bloch_b1_pdp(void)
{
	float gb[3] = { 0. };

	float in[3] = { 0., 0., 1. };
	float out[3][3] = { { 0., 0., 1. }, { 0. }, { 0. } };

	// M_z == 1

	bloch_b1_pdp(out, in, 0., 0., gb, 1.);

	if (1. != out[2][1])
		return false;

	bloch_b1_pdp(out, in, 0., 0., gb, 1.i);

	if (1. != out[2][0])
		return false;

	bloch_b1_pdp(out, in, 0., 0., gb, 2.i);

	if (2. != out[2][0])
		return false;

	// M_y == 1

	in[1] = 1.;
	in[2] = 0.;

	bloch_b1_pdp(out, in, 0., 0., gb, 1.);

	if ((1. != out[0][2]) || (-1. != out[1][1]) || (-1. != out[2][2]))
		return false;


	// M_x == 1

	in[0] = 1.;
	in[1] = 0.;

	bloch_b1_pdp(out, in, 0., 0., gb, 1.);

	if ((1. != out[0][2]) || (-1. != out[1][0]))
		return false;

	bloch_b1_pdp(out, in, 0., 0., gb, 1.i);

	if ((1. != out[0][2]) || (-1. != out[1][0]) || (-1. != out[2][2]))
		return false;

	return true;

}

UT_REGISTER_TEST(test_bloch_b1_pdp);


// SA tests including B1

// dB1
static void bloch_wrap_pdp(void* _data, float* out, float t, const float* in)
{
	struct bloch_s* data = _data;
	(void)t;

	bloch_b1_pdp((float(*)[3])out, in, data->r1, data->r2, data->gb, M_PI / (2. * 0.2));
}

// dFA
static void bloch_wrap_pdp2(void* _data, float* out, float t, const float* in)
{
	struct bloch_s* data = _data;
	(void)t;

	bloch_b1_pdp((float(*)[3])out, in, data->r1, data->r2, data->gb, 1.);
}

// dFA + phase of PI/2
static void bloch_wrap_pdp3(void* _data, float* out, float t, const float* in)
{
	struct bloch_s* data = _data;
	(void)t;

	bloch_b1_pdp((float(*)[3])out, in, data->r1, data->r2, data->gb, +1.i);
}


// Test B1 gradient
static bool test_ode_sa_bloch_b1(void)
{
	int N = 3;
	int P = 3;

	float end = 0.2;

	// FA 90 degree:	a = gamma * b1 * time
	float fa = M_PI / (2. * end);

	struct bloch_s data = { 0. , 0., { fa, 0, 0. } };

	float xp[4][3] = { { 0., 0., 1. }, { 0. }, { 0. }, { 0. } };
	float x0[3] = { 0., 0., 1. };
	float x2[3] = { 0., 0., 0. };
	float x2b1[3] = { 0., 0., 0. };

	float h = 0.1;
	float tol = 0.000001;

	float q = 1.E-3;

	ode_direct_sa_wrap(h, tol, N, P, xp, 0., end, &data, bloch_fun, bloch_pdy2, bloch_wrap_pdp);

	bloch_excitation(x2, end, x0, data.r1, data.r2, data.gb);

	data.gb[0] += q;
	bloch_excitation(x2b1, end, x0, data.r1, data.r2, data.gb);

	for (int i = 0; i < N; i++) {

                // dm/dfa = dm/db1*db1/dfa = dm/db1*1/(nom.FA)
		float err = fabsf(q * xp[3][i] / fa - (x2b1[i] - x2[i]));

		if (err > 1.E-7)
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_ode_sa_bloch_b1);


// Test FA gradient
static bool test_ode_sa_bloch_fa(void)
{
	int N = 3;
	int P = 3;

	float end = 0.2;

	// FA 90 degree:	a = gamma * b1 * time
	float fa = M_PI / (2. * end);

	struct bloch_s data = { 0. , 0., { fa, 0., 0. } };

	float xp[4][3] = { { 0., 0., 1. }, { 0. }, { 0. }, { 0. } };
	float x0[3] = { 0., 0., 1. };
	float x2[3] = { 0., 0., 0. };
	float x2b1[3] = { 0., 0., 0. };

	float h = 0.1;
	float tol = 0.000001;

	float q = 1.E-3;

	ode_direct_sa_wrap(h, tol, N, P, xp, 0., end, &data, bloch_fun, bloch_pdy2, bloch_wrap_pdp2);

	bloch_excitation(x2, end, x0, data.r1, data.r2, data.gb);

	data.gb[0] += q;
	bloch_excitation(x2b1, end, x0, data.r1, data.r2, data.gb);

	for (int i = 0; i < N; i++) {

                // dm/dfa = dm/db1*db1/dfa = dm/db1*1/(nom.FA)
		float err = fabsf(q * xp[3][i] - (x2b1[i] - x2[i]));

		if (err > 1.E-7)
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_ode_sa_bloch_fa);


// Test FA gradient with included RF phase
static bool test_ode_sa_bloch_fa_with_phase(void)
{
	int N = 3;
	int P = 3;

	float end = 0.2;

	// FA 90 degree:	a = gamma * b1 * time
	float fa = M_PI / 2. / end;
	float phase = M_PI / 2.;

	struct bloch_s data = { 0. , 0., { cosf(phase) * fa, -sinf(phase) * fa, 0. } };

	float xp[4][3] = { { 0., 0., 1. }, { 0. }, { 0. }, { 0. } };
	float x0[3] = { 0., 0., 1. };
	float x2[3] = { 0., 0., 0. };
	float x2b1[3] = { 0., 0., 0. };

	float h = 0.1;
	float tol = 0.000001;

	float q = 1.E-3;

	// phase == M_PI/2. requires bloch_wrap_pdp3!
	ode_direct_sa_wrap(h, tol, N, P, xp, 0., end, &data, bloch_fun, bloch_pdy2, bloch_wrap_pdp3);

	bloch_excitation2(x2, x0, fa * end, phase);

	fa += q;
	bloch_excitation2(x2b1, x0, fa * end, phase);

	for (int i = 0; i < N; i++) {

                // dm/dfa = dm/db1*db1/dfa = dm/db1*1/(nom.FA)
		float err = fabsf(q * xp[3][i] - (x2b1[i] - x2[i]));

		if (err > 1.E-7)
			return false;
	}
	return true;
}

UT_REGISTER_TEST(test_ode_sa_bloch_fa_with_phase);





static bool test_bloch_mcconnel(void)
{
	int P = 1;

	float m1[1 + P * 3][1 + P * 3];
	float gb[3] = { 0., 0., 0. };
	float k[1] = { 0.2 };

	//float k[1][1] = { { 0. } };
	float Th[1] = { 1. };
	float Om[1] = { 0. };
	float r1[1] = { 1. };
	float r2[1] = { 0.1 };

	bloch_mcconnel_matrix_ode(P, m1, r1, r2, k, Th, Om, gb);

	float m2[4][4];

	bloch_matrix_ode(m2, r1[0], r2[0], gb);

	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			if (m1[i][j] != m2[i][j])
				return false;

	return true;
}


UT_REGISTER_TEST(test_bloch_mcconnel);



static bool test_bloch_mcconnel2(void)
{
	int P = 2;

	float m1[1 + P * 3][1 + P * 3];
	float gb[3] = { 0., 0., 0. };
	float k[2][2] = { { -0.2, 0.2 }, { 0.2, -0.2 } };
	float k2[1] = { 0.2 };
	float Th[2] = { 0.5, 0.5 };
	float Om[2] = { 0., 0. };
	float r1[2] = { 1., 1.1 };
	float r2[2] = { 0.1, 0.05 };
	bloch_mcconnel_matrix_ode(P, m1, r1, r2, k2, Th, Om, gb);

	float m2[4][4];
	bloch_matrix_ode(m2, r1[0], r2[0], gb);

	float m3[4][4];
	bloch_matrix_ode(m3, r1[1], r2[1], gb);

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			if (fabsf(m1[0 + i][0 + j] - (m2[i][j] + ((i == j) ? k[0][0] * Th[1] : 0.f))) > 1.E-3)
				return false;

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			if (fabsf(m1[3 + i][3 + j] - (m3[i][j] + ((i == j) ? k[1][1] * Th[0] : 0.f))) > 1.E-3)
				return false;

	return true;
}


UT_REGISTER_TEST(test_bloch_mcconnel2);



static bool test_ode_bloch_mcconnel(void)
{
	int P = 2;
	int N = 1 + P * 3;

	float m1[N][N];
	float gb[3] = { 0., 0., 0. };
	float k[1] = { 0.2 };
	float Th[2] = { 0.5, 0.5 };
	float Om[2] = { 0., 0.01 };
	float r1[2] = { 1., 1.1 };
	float r2[2] = { 0.1, 0.05 };

	bloch_mcconnel_matrix_ode(P, m1, r1, r2, k, Th, Om, gb);

	float end = 10.;

	float h = 0.01;
	float tol = 0.001;

	float x[N];

	for (int i = 0; i < N; i++)
		x[i] = 0.;

	x[N - 1] = 1.;

	ode_matrix_interval(h, tol, N, x, 0., end, m1);

	float x2[7] = { 0., 0., 0.5, 0., 0., 0.5, 1. };

	float err = 0.;

	for (int i = 0; i < N; i++)
		err += powf(x[i] - x2[i], 2.);

	UT_RETURN_ASSERT(err < 1.E-9);
}


UT_REGISTER_TEST(test_ode_bloch_mcconnel);


static bool test_bmc_ode_pools(void)
{
	int P = 5;
	float out_5[P * 3];
	float out_1[3] = { 0., 0., 0. };

	float in_5[P * 3];
	float in_1[3] = { 0.5, 0.2, 0.8 };
	float r1[P];
	float r2[P];
	float k[4] = { 0., 0., 0., 0. };

	for (int p = 0; p < P; p++) {

		in_5[0 + 3 * p] = 0.5;
		in_5[1 + 3 * p] = 0.2;
		in_5[2 + 3 * p] = 0.8;
		r1[p] = 1.2;
		r2[p] = 20.;
	}

	float gb[3] = { 0.2, 0.6, 0.9 };
	float m0[5] = { 1., 1., 1., 1., 1. };
	float Om[5] = { 0., 0., 0., 0., 0. };

	bloch_mcconnell_ode(P, out_5, in_5, r1, r2, k, m0, Om, gb);
	bloch_ode(out_1, in_1, r1[0], r2[0], gb);

	for (int p = 0; p < P; p++)
		for (int d = 0; d < 3; d++)
			if (fabsf(out_5[d + p * 3] - out_1[d]) > 1e-6)
				return false;

	return true;
}

UT_REGISTER_TEST(test_bmc_ode_pools);



static bool test_bloch_mcconnell_matrix(void)
{
	int P = 2;
	int N = 1 + P * 3;

	float gb[3] = { 0., 0.05, 0.06 };
	float k[1] = { 0.2 };
	float Th[2] = { 1., 1. };
	float Om[2] = { 0., 0.05 };
	float r1[2] = { 1., 1.1 };
	float r2[2] = { 0.1, 0.05 };
	float m_in[7] = { 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 1. };
	float m_in_2[6] = { 0.1, 0.2, 0.3, 0.1, 0.2, 0.3 };
	float m1[N][N];

	bloch_mcconnel_matrix_ode(P, m1, r1, r2, k, Th, Om, gb);

	float out1[N];
	float out2[N - 1];

	bloch_mcconnell_ode(P, out2, m_in_2, r1, r2, k, Th, Om, gb);

	matf_vecmul(N, N, out1, m1, m_in);

	for (int i = 0; i < N - 1; i++)
		if (out1[i] != out2[i])
			return false;

	return true;

}

UT_REGISTER_TEST(test_bloch_mcconnell_matrix);


// Test function for SA with case P = 2
static bool test_bloch_mcconnell_b1_pdp(void)
{
	int P = 2;
	float out_2_pool[P * 5 - 1][P * 3];
	float in[6] = { 0.1, 0.4, 1, 0.5, 0.7, 0.8 };

	float gb[3] = { 0.2, 0.6, 0.9 };
	float k[1] = { -2. };
	float m0[2] = { 0.93, 0.76 };
	float r1[2] = { 1., 1.1 };
	float r2[2] = { 0.6, 0.08 };
	complex float b1 = cexpf(+0.1i) * 1.1;

	for (int i = 0; i < P * 5 - 1; i++)
		for (int j = 0; j < P * 3; j++)
			out_2_pool[i][j] = 0.;

	// R1
	out_2_pool[0][2] = -(in[2] - m0[0]);
	// R1_2
	out_2_pool[1][5] = -(in[5] - m0[1]);
	// R2
	out_2_pool[2][0] = -in[0];
	out_2_pool[2][1] = -in[1];
	// R2_2
	out_2_pool[3][3] = -in[3];
	out_2_pool[3][4] = -in[4];
	// B1
	out_2_pool[4][0] = cimagf(b1) * in[2];
	out_2_pool[4][1] = crealf(b1) * in[2];
	out_2_pool[4][2] = -cimagf(b1) * in[0] - crealf(b1) * in[1];
	out_2_pool[4][3] = cimagf(b1) * in[5];
	out_2_pool[4][4] = crealf(b1) * in[5];
	out_2_pool[4][5] = -cimagf(b1) * in[3] - crealf(b1) * in[4];
	// k
	out_2_pool[7][0] = -in[0] * m0[1] + in[3] * m0[0];
	out_2_pool[7][1] = -in[1] * m0[1] + in[4] * m0[0];
	out_2_pool[7][2] = -in[2] * m0[1] + in[5] * m0[0];
	out_2_pool[7][3] = in[0] * m0[1] - (in[3] * m0[0]);
	out_2_pool[7][4] = in[1] * m0[1] - in[4] * m0[0];
	out_2_pool[7][5] = in[2] * m0[1] - in[5] * m0[0];
	// m0_1
	out_2_pool[5][0] = in[3] * k[0];
	out_2_pool[5][1] = in[4] * k[0];
	out_2_pool[5][2] = in[5] * k[0] + r1[0];
	out_2_pool[5][3] = -in[3] * k[0];
	out_2_pool[5][4] = -in[4] * k[0];
	out_2_pool[5][5] = -in[5] * k[0];
	// m0_2
	out_2_pool[6][0] = -in[0] * k[0];
	out_2_pool[6][1] = -in[1] * k[0];
	out_2_pool[6][2] = -in[2] * k[0];
	out_2_pool[6][3] = in[0] * k[0];
	out_2_pool[6][4] = in[1] * k[0];
	out_2_pool[6][5] = r1[1] + in[2] * k[0];
	// Om
	out_2_pool[8][3] = in[4];
	out_2_pool[8][4] = -in[3];

	float out_gen[9][P * 3];

	for (int i = 0; i < P * 5 - 1; i++)
		for (int j = 0; j < P * 3; j++)
			out_gen[i][j] = 0.;

	bloch_mcc_b1_pdp(P, out_gen, in, r1, r2, k, m0, gb, b1);

	for (int i = 0; i < P * 5 - 1; i++)
		for (int j = 0; j < P * 3; j++)
			if (fabsf(out_gen[i][j] - out_2_pool[i][j]) > 1.e-6)
				return false;

	return true;
}

UT_REGISTER_TEST(test_bloch_mcconnell_b1_pdp);


// Test function for SA with case P = 3
static bool test_bloch_mcconnell_b1_pdp_3(void)
{
	int P = 3;
	float out_3_pool[P * 5 - 1][P * 3];
	float in[9] = { 0.1, 0.4, 1., 0.5, 0.7, 0.8, -0.6, -0.9, 2. };

	float gb[3] = { 0.2, 0.7, 0.9 };
	float k[2] = { 2.3, 8.4 };
	float m0[3] = { 0.93, 0.76, 0.37 };
	float r1[3] = { 1., 1.1, 2.3 };
	float r2[3] = { 0.6, 0.08, 0.15 };
	complex float b1 = cexpf(+0.1i) * 1.1;

	for (int i = 0; i < P * 5 - 1; i++)
		for (int j = 0; j < P * 3; j++)
			out_3_pool[i][j] = 0.;

	// R1
	out_3_pool[0][2] = -(in[2] - m0[0]);
	// R1_2
	out_3_pool[1][5] = -(in[5] - m0[1]);
	// R1_3
	out_3_pool[2][8] = -(in[8] - m0[2]);
	// R2
	out_3_pool[3][0] = -in[0];
	out_3_pool[3][1] = -in[1];
	// R2_2
	out_3_pool[4][3] = -in[3];
	out_3_pool[4][4] = -in[4];
	// R2_3
	out_3_pool[5][6] = -in[6];
	out_3_pool[5][7] = -in[7];
	// B1
	out_3_pool[6][0] = cimagf(b1) * in[2];
	out_3_pool[6][1] = crealf(b1) * in[2];
	out_3_pool[6][2] = -cimagf(b1) * in[0] - crealf(b1) * in[1];
	out_3_pool[6][3] = cimagf(b1) * in[5];
	out_3_pool[6][4] = crealf(b1) * in[5];
	out_3_pool[6][5] = -cimagf(b1) * in[3] - crealf(b1) * in[4];
	out_3_pool[6][6] = cimagf(b1) * in[8];
	out_3_pool[6][7] = crealf(b1) * in[8];
	out_3_pool[6][8] = -cimagf(b1) * in[6] - crealf(b1) * in[7];

	// m0_1
	out_3_pool[7][0] = in[3] * k[0] + in[6] * k[1];
	out_3_pool[7][1] = in[4] * k[0] + in[7] * k[1];
	out_3_pool[7][2] = in[5] * k[0] + in[8] * k[1] + r1[0];
	out_3_pool[7][3] = -in[3] * k[0];
	out_3_pool[7][4] = -in[4] * k[0];
	out_3_pool[7][5] = -in[5] * k[0];
	out_3_pool[7][6] = - in[6] * k[1];
	out_3_pool[7][7] = - in[7] * k[1];
	out_3_pool[7][8] = - in[8] * k[1];
	// m0_2
	out_3_pool[8][0] = -in[0] * k[0];
	out_3_pool[8][1] = -in[1] * k[0];
	out_3_pool[8][2] = -in[2] * k[0];
	out_3_pool[8][3] = in[0] * k[0];
	out_3_pool[8][4] = in[1] * k[0];
	out_3_pool[8][5] = r1[1] + in[2] * k[0];
	// m0_3
	out_3_pool[9][0] = -in[0] * k[1];
	out_3_pool[9][1] = -in[1] * k[1];
	out_3_pool[9][2] = -in[2] * k[1];
	out_3_pool[9][6] = in[0] * k[1];
	out_3_pool[9][7] = in[1] * k[1];
	out_3_pool[9][8] = r1[2] + in[2] * k[1];
	// k1
	out_3_pool[10][0] = -in[0] * m0[1] + in[3] * m0[0];
	out_3_pool[10][1] = -in[1] * m0[1] + in[4] * m0[0];
	out_3_pool[10][2] = -in[2] * m0[1] + in[5] * m0[0];
	out_3_pool[10][3] = in[0] * m0[1] - in[3] * m0[0];
	out_3_pool[10][4] = in[1] * m0[1] - in[4] * m0[0];
	out_3_pool[10][5] = in[2] * m0[1] - in[5] * m0[0];
	// k2
	out_3_pool[11][0] = -in[0] * m0[2] + in[6] * m0[0];
	out_3_pool[11][1] = -in[1] * m0[2] + in[7] * m0[0];
	out_3_pool[11][2] = -in[2] * m0[2] + in[8] * m0[0];
	out_3_pool[11][6] = in[0] * m0[2] - in[6] * m0[0];
	out_3_pool[11][7] = in[1] * m0[2] - in[7] * m0[0];
	out_3_pool[11][8] = in[2] * m0[2] - in[8] * m0[0];
	// Om1
	out_3_pool[12][3] = in[4];
	out_3_pool[12][4] = -in[3];
	// Om2
	out_3_pool[13][6] = in[7];
	out_3_pool[13][7] = -in[6];


	float out_gen[P * 5 - 1][P * 3];

	for (int i = 0; i < P * 5 - 1; i++)
		for (int j = 0; j < P * 3; j++)
			out_gen[i][j] = 0.;

	bloch_mcc_b1_pdp(P, out_gen, in, r1, r2, k, m0, gb, b1);

	for (int i = 0; i < P * 5 - 1; i++)
		for (int j = 0; j < P * 3; j++)
			if (fabsf(out_gen[i][j] - out_3_pool[i][j]) > 1e-6)
				return false;

	return true;
}

UT_REGISTER_TEST(test_bloch_mcconnell_b1_pdp_3);

// Test STM function for SA with P = 2
static bool test_bloch_mcconnell_matrix_ode_sa(void)
{
	int P = 2;

	float m[61][61] = { { 0. } };
	float m2[61][61] = { { 0. } };
	float m_tmp[7][7] = { { 0. } };

	float gb[3] = { 0. };
	float Om[2] = { 0. };
	float m0[2] = { 0.93, 0.76 };
	float r1[2] = { 1., 1.1 };
	float r2[2] = { 0.6, 0.08 };
	complex float b1 = cexpf(+0.1i) * 1.1;
	float k[1] = { 0.2 };

	bloch_mcconnel_matrix_ode(P, m_tmp, r1, r2, k, m0, Om, gb);

	for (int p = 0; p < 10; p++)
		for (int i = 0; i < 6; i++)
			for (int j = 0; j < 6; j++)
				m[6 * p + i][6 * p + j]= m_tmp[i][j];

	m[2][60] = m0[0] * r1[0];
	m[5][60] = m0[1] * r1[1];

	// R1
	m[8][2] = -1.;
	m[8][60] = m0[0];
	// R1_2
	m[17][5] = -1.;
	m[17][60] = m0[1];
	// R2
	m[18][0] = -1.;
	m[19][1] = -1.;
	// R2_2
	m[27][3] = -1.;
	m[28][4] = -1.;
	// B1
	m[30][2] = cimagf(b1);
	m[31][2] = crealf(b1);
	m[32][0] = -cimagf(b1);
	m[32][1] = -crealf(b1);
	m[33][5] = cimagf(b1);
	m[34][5] = crealf(b1);
	m[35][3] = -cimagf(b1);
	m[35][4] = -crealf(b1);
	// M0
	for (int i = 0; i < 3; i++) {

		m[36 + i][i + 3] = k[0];
		m[39 + i][i + 3] = -k[0];
	}
	m[38][60] = r1[0];
	// M0_2
	for (int i = 0; i < 3; i++) {

		m[42 + i][i] = -k[0];
		m[45 + i][i] = k[0];
	}
	m[47][60] = r1[1];
	// k
	for (int i = 0; i < 3; i++) {

		m[48 + i][i] = -m0[1];
		m[48 + i][i + 3] = m0[0];
		m[51 + i][i] = m0[1];
		m[51 + i][i + 3] = -m0[0];
	}
	// Om
	m[57][4] = 1.;
	m[58][3] = -1.;

	bloch_mcc_matrix_ode_sa2(P, m2, r1, r2, k, m0, Om, gb, b1);

	for (int i = 0; i < 61; i++)
		for (int j = 0; j < 61; j++)
			if (m[i][j] != m2[i][j])
				return false;

	return true;
}

UT_REGISTER_TEST(test_bloch_mcconnell_matrix_ode_sa);

