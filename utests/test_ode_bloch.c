/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */


#include <math.h>

#include "num/ode.h"

#include "simu/bloch.h"

#include "utest.h"

#if 0
static void lorenz(float out[3], const float in[3], float sigma, float rho, float beta)
{
	out[0] = sigma * (in[1] - in[0]);
	out[1] = in[0] * (rho - in[2]) - in[1];
	out[2] = in[0] * in[1] - beta * in[2];
}


static void lorenz_fun(void* data, float* out, float t, const float* in)
{
	(void)data; (void)t;
	lorenz(out, in, 10., 28., 8. / 3.);
}
#endif


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

	ode_interval(h, tol, 3, x, 0., end, &data, bloch_fun);
	bloch_relaxation(x2, end, x0, data.r1, data.r2, data.gb);

	float err2 = 0.;

	for (int i = 0; i < 3; i++)
		err2 += powf(x[i] - x2[i], 2.);

	return (err2 < 1.E-7);
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
	ode_direct_sa(h, tol, 1, 1, xp, 0., end, &data, sa_der, sa_pdy, sa_pdp);

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


	ode_direct_sa(h, tol, N, N, xp, 0., end, &data, sa_der, sa_pdy, sa_pdp);

	float x[N];
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

	//		printf("%d %d-%e-%e-%e\n", j, i, err, df, sa);

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
	ode_direct_sa(h, tol, 3, 2, xp, 0., end, &data, bloch_fun, bloch_pdy2, bloch_pdp2);
	bloch_relaxation(x2, end, x0, data.r1, data.r2, data.gb);
	bloch_relaxation(x2r1, end, x0, data.r1 + q, data.r2, data.gb);
	bloch_relaxation(x2r2, end, x0, data.r1, data.r2 + q, data.gb);

	float err2 = 0.;

	for (int i = 0; i < 3; i++)
		err2 += powf(xp[0][i] - x2[i], 2.);

	if (err2 > 1.E-7)
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
