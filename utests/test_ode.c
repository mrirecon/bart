/* Copyright 2018-2023. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>

#include "misc/nested.h"

#include "num/ode.h"

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



static bool test_ode_matrix(void)
{
	float mat[2][2] = { { 0., +1. }, { -1., 0. } };

	float x[2] = { 1., 0. };
	float h = 10.;
	float tol = 1.E-4;

	ode_matrix_interval(h, tol, 2, x, 0., M_PI, mat);

	float err = pow(fabs(x[0] + 1.), 2.) + pow(fabs(x[1] - 0.), 2.);

	return (sqrtf(err) < 1.E-3);
}

UT_REGISTER_TEST(test_ode_matrix);


static bool test_ode_adjoint(void)
{
	float r1 = 1.;
	float r2 = 0.5;

	float x0[2] = { 1., 1. };
	float h = 0.1;
	float tol = 1.E-3;

	int N = 10;
	float t[N + 1];

	for (int i = 0; i <= N; i++)
		t[i] = i * (1. / N);

	float x[N + 1][2];
	float z[N + 1][2];

	NESTED(void, sys, (float out[2], float t, const float in[2]))
	{
		(void)t;

		out[0] = -r1 * in[0];
		out[1] = -r2 * in[1];
	};

	NESTED(void, cost, (float out[2], float t))
	{
		(void)t;

		for (int l = 0; l < 2; l++)
			out[l] = 1.;
	};

	ode_adjoint_sa(h, tol, N, t, 2, x, z, x0, sys, sys, cost);

	if (1.E-3 < powf(fabs(x[N][0] - expf(-r1)), 2.))
	       return false;

	if (1.E-3 < powf(fabs(x[N][1] - expf(-r2)), 2.))
		return false;

	float d1 = -1.;
	float d2 = -3.;

	float dp[2][2][2] = { 0 };
	dp[0][0][0] = d1;
	dp[1][1][1] = d2;

	float dj[2] = { 0., 0. };
	ode_adjoint_sa_eq_eval(N, 2, 2, dj, x, z, dp);

	int N2 = 1000;
	float djX = 0.;

	for (int i = 0; i < N2; i++)
		djX += ((d1 * i) / N2) * expf(i * (-(r1) / N2)) / N2;

	if (5.E-4 < fabs(dj[0] - djX))
	       return false;

	float djY = 0.;

	for (int i = 0; i < N2; i++)
		djY += ((d2 * i) / N2) * expf(i * (-(r2) / N2)) / N2;

	if (1.E-3 < fabs(dj[1] - djY))
		return false;

	return true;
}

UT_REGISTER_TEST(test_ode_adjoint);




static bool test_ode_matrix_adjoint(void)
{
	float r1 = 1.;
	float r2 = 0.5;
	float mat[2][2] = { { -r1, 0. }, { 0., -r2 } };

	float x0[2] = { 1., 1. };
	float h = 0.1;
	float tol = 1.E-3;

	int N = 35;	// 10 is ok for equal_eq
	float t[N + 1];

	for (int i = 0; i <= N; i++)
		t[i] = i * (1. / N);

	float x[N + 1][2];
	float z[N + 1][2];

	float sys[N][2][2];

	for (int i = 0; i < N; i++)
		for (int l = 0; l < 2; l++)
			for (int k = 0; k < 2; k++)
				sys[i][l][k] = mat[l][k];

	float cost[N][2];

	for (int i = 0; i < N; i++)
		for (int l = 0; l < 2; l++)
			cost[i][l] = 1.;
 
	ode_matrix_adjoint_sa(h, tol, N, t, 2, x, z, x0, sys, cost);

	if (1.E-3 < powf(fabs(x[N][0] - expf(-r1)), 2.))
	       return false;

	if (1.E-3 < powf(fabs(x[N][1] - expf(-r2)), 2.))
		return false;

	float d1 = -1.;
	float d2 = -3.;

	float dp[2][2][2] = { 0 };
	dp[0][0][0] = d1;
	dp[1][1][1] = d2;

	float dj[2] = { 0., 0. };
	ode_adjoint_sa_eq_eval(N, 2, 2, dj, x, z, dp);

	float dj2[2] = { 0., 0. };
	ode_adjoint_sa_eval(N, t, 2, 2, dj2, x, z, dp);


	int N2 = 1000;
	float djX = 0.;

	for (int i = 0; i < N2; i++)
		djX += ((d1 * i) / N2) * expf(i * (-(r1) / N2)) / N2;

	if (5.E-4 < fabs(dj[0] - djX))
	       return false;

	if (1.E-2 < fabs(dj2[0] - djX))
	       return false;

	float djY = 0.;

	for (int i = 0; i < N2; i++)
		djY += ((d2 * i) / N2) * expf(i * (-(r2) / N2)) / N2;

	if (1.E-3 < fabs(dj[1] - djY))
		return false;

	if (5.E-2 < fabs(dj2[1] - djY))
	       return false;

	return true;
}

UT_REGISTER_TEST(test_ode_matrix_adjoint);

