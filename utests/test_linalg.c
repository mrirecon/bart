/* Copyright 2016-2019. Martin Uecker.
 * Copyright 2021-2025. Graz University of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>

#include "num/linalg.h"

#include "utest.h"



static bool test_mat_det(void)
{
	const complex float M[3][3] = {
		{ -2., -1., 2. },
		{ 2., 1., 4. },
		{ -3, 3., -1. },
	};

	return 54. == mat_det(3, M);
}

UT_REGISTER_TEST(test_mat_det);


static bool test_mat_pinv(void)
{
	const complex float A[3][2] = {
		{ 1.i, 4. },
		{ 2.i, 5. },
		{ 3.i, 6. },
	};

	const complex float C[2][3] = {
		{ -0.00000 + 0.94444i, +0.00000 + 0.11111i, +0.00000 - 0.72222i },
		{ +0.44444 + 0.00000i, +0.11111 + 0.00000i, +-0.22222 - 0.00000i },
	};

	complex float B[2][3];

	mat_pinv_left(3, 2, B, A);

	float err = 0.;

	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 3; j++)
			err += powf(cabsf(C[i][j] - B[i][j]), 2.);

	return (err < 1.E-10);
}


UT_REGISTER_TEST(test_mat_pinv);

static bool test_mat_svd(void)
{
	complex float A[3][4] = {
		{+0.000000e+00,	+3.535534e-01,	+5.000000e-01,	+3.535534e-01},
		{+23.000000e-01,+3.535534e-01,	-2.185570e-08,	-3.535534e-01},
		{+0.000000e+00,	+0.000000e+00,	+0.000000e+00,	+0.000000e+00}
	};

	complex float C[3][4] = {
		{+0.000000e+00,	+3.535534e-01,	+5.000000e-01,	+3.535534e-01},
		{+23.000000e-01,+3.535534e-01,	-2.185570e-08,	-3.535534e-01},
		{+0.000000e+00,	+0.000000e+00,	+0.000000e+00,	+0.000000e+00}
	};

	complex float U[3][3];
	complex float VH[4][4];
	float S[3];

	mat_svd(3, 4, U, VH, S, A);

	complex float B[3][4];

	mat_svd_recov(3, 4, B, U, VH, S);

	float err = 0.;

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 4; j++)
			err += powf(cabsf(C[i][j] - B[i][j]), 2.);

	return (err < 1.E-10);
}


UT_REGISTER_TEST(test_mat_svd);


// Test complex valued matrices + right
static bool test_mat_pinv2(void)
{
	complex float A[3][2] = {
		{ 1.i, 4. },
		{ 2.i, 5. },
		{ 3.i, 6. },
	};

	const complex float C[2][3] = {
		{ -0.00000 + 0.94444i, +0.00000 + 0.11111i, +0.00000 - 0.72222i },
		{ +0.44444 + 0.00000i, +0.11111 + 0.00000i, +-0.22222 - 0.00000i },
	};

	complex float B[2][3];

	mat_pinv_svd(3, 2, B, A);

	float err = 0.;

	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 3; j++)
			err += powf(cabsf(C[i][j] - B[i][j]), 2.);

	return (err < 1.E-10);
}


UT_REGISTER_TEST(test_mat_pinv2);

// Test real valued matrices + left
static bool test_mat_pinv3(void)
{
	complex float A[3][4] = {
		{+0.000000e+00,	+3.535534e-01,	+5.000000e-01,	+3.535534e-01},
		{+23.000000e-01,+3.535534e-01,	-2.185570e-08,	-3.535534e-01},
		{+0.000000e+00,	+0.000000e+00,	+0.000000e+00,	+0.000000e+00}
	};

	const complex float C[4][3] = {
		{9.07366543e-09,  4.15162454e-01,  0.},
		{7.07106783e-01,  6.38183045e-02,  0.},
		{9.99999973e-01, -1.97253619e-09,  0.},
		{7.07106780e-01, -6.38183017e-02,  0.}
	};

	complex float B[4][3];

	mat_pinv_svd(3, 4, B, A);

	float err = 0.;

	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 3; j++)
			err += powf(cabsf(C[i][j] - B[i][j]), 2.);

	return (err < 1.E-10);
}

UT_REGISTER_TEST(test_mat_pinv3);


static bool test_mat_pinvT(void)
{
	complex float A[3][4] = {
		{+0.000000e+00,	+3.535534e-01,	+5.000000e-01,	+3.535534e-01},
		{+23.000000e-01,+3.535534e-01,	-2.185570e-08,	-3.535534e-01},
		{+0.000000e+00,	+0.000000e+00,	+0.000000e+00,	+0.000000e+00}
	};

	complex float B[4][3];
	complex float B2[3][4];
	complex float B3[4][3];
	complex float A2[4][3];

	mat_transpose(3, 4, A2, A);

	mat_pinv_svd(3, 4, B, A);

	mat_pinv_svd(4, 3, B2, A2);

	mat_transpose(3, 4, B3, B2);

	float err = 0.;

	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 3; j++)
			err += powf(cabsf(B[i][j] - B3[i][j]), 2.);

	return (err < 1.E-10);
}


UT_REGISTER_TEST(test_mat_pinvT);


static bool test_thomas_algorithm(void)
{
	const complex float A[7][3] = {

		{ 0. , 1., 0.5 },
		{ 0.5, 1., 0.5 },
		{ 0.5, 1., 0.5 },
		{ 0.5, 1., 0.5 },
		{ 0.5, 1., 0.5 },
		{ 0.5, 1., 0.5 },
		{ 0.5, 1., 0.  },
	};

	const complex float d[7] = { 1., 0., 0., 0., 0., 0., 0. };
	complex float x[7];

	thomas_algorithm(7, x, A, d);

	bool ok = true;

	for (int i = 0; i < 7; i++)
		ok &= (cabsf(x[i] - 0.25f * (7 - i) * powf(-1., i)) < 1e-6);

	return ok;
}

UT_REGISTER_TEST(test_thomas_algorithm);


static bool test_mat_schur(void)
{
	enum { N = 2 };

	complex float A[N][N] = {
		{ 1, 1 },
		{ -2, 3 }
	};

	complex float T[N][N];
	complex float Z[N][N];

	mat_schur(N, T, Z, A);

	complex float B[N][N];
	mat_schur_recov(N, B, T, Z);

	float err = 0.;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			err += powf(cabsf(B[i][j] - A[i][j]), 2.);

	// debug_printf(DP_INFO, "\nB:\n");
	// for (int i = 0; i < N; i++) {
	// 	for (int j = 0; j < N; j++)
	// 		debug_printf(DP_INFO, "%f+i*%f ", crealf(B[i][j]), cimagf(B[i][j]));
	// 	debug_printf(DP_INFO, "\n");
	// }
	// debug_printf(DP_INFO, "err: %f\n", err);

	return (err < 1.E-10);
}


UT_REGISTER_TEST(test_mat_schur);


static bool test_mat_ceig(void)
{
	enum { N = 2 };

	complex double A[N][N] = {
		{ 0., -1. },
		{ 1., 0. }
	};

	complex double EV[N];

	mat_ceig_double(N, EV, A);

	complex double ref[N] = {1.i, -1.i};

	double err = 0.;

	for (int i = 0; i < N; i++)
		err += powf(cabs(ref[i] - EV[i]), 2.);

	// debug_printf(DP_INFO, "\nEV:\n");
	// for (int i = 0; i < N; i++)
	// 	debug_printf(DP_INFO, "%f+i*%f\n", crealf(EV[i]), cimagf(EV[i]));
	// debug_printf(DP_INFO, "err: %f\n", err);

	return (err < 1.E-10);
}


UT_REGISTER_TEST(test_mat_ceig);


static bool test_mat_eig(void)
{
	enum { N = 2 };

	double A[N][N] = {
		{ 0., 1. },
		{ 1., 1. }
	};

	double EV[N];

	mat_eig_double(N, EV, A);

	double ref[N] = {-0.618033988749895, 1.618033988749895};

	double err = 0.;

	for (int i = 0; i < N; i++)
		err += powf(fabs(ref[i] - EV[i]), 2.);

	// debug_printf(DP_INFO, "\nEV:\n");
	// for (int i = 0; i < N; i++)
	// 	debug_printf(DP_INFO, "%1.15e\n", EV[i]);
	// debug_printf(DP_INFO, "err: %f\n", err);

	return (err < 1.E-10);
}


UT_REGISTER_TEST(test_mat_eig);


static bool test_mat_band_reorder(void)
{
	enum { N = 4 };

	// Lower form

	double A_low[3][N] = {
		{ 1., 2., 3., 4. },
		{ 5., 5., 5., 0. },
		{ 2., 2., 0., 0.}
	};

	double A[N][N];
	mat_band_reorder(N, 3, A, A_low, false);

	// Upper form

	double A_up[3][N] = {
		{ 0., 0., 2., 2. },
		{ 0., 5., 5., 5. },
		{ 1., 2., 3., 4. }
	};

	double A2[N][N];
	mat_band_reorder(N, 3, A2, A_up, true);

	// Check

	double A_ref[N][N] = {
		{ 1., 5., 2., 0. },
		{ 5., 2., 5., 2. },
		{ 2., 5., 3., 5. },
		{ 0., 2., 5., 4. }
	};

	double err = 0.;
	double err2 = 0.;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) {

			err += powf(fabs(A_ref[i][j] - A[i][j]), 2.);
			err2 += powf(fabs(A_ref[i][j] - A2[i][j]), 2.);
	}

	// debug_printf(DP_INFO, "err: %f\n", err);
	// debug_printf(DP_INFO, "err2: %f\n", err2);

	return ( (err < 1.E-10) && (err2 < 1.E-10) );
}


UT_REGISTER_TEST(test_mat_band_reorder);


static bool test_trimat_solve(void)
{
	enum { N = 4 };

	// Lower

	complex float A[N][N] = {
		{ 3. + 1i, 0., 0., 0. },
		{ 2., 1., 0., 0. },
		{ 1., 0., 1., 0. },
		{ 1., 1., 1., 1. }
	};

	complex float B[N] = { 4., 2., 4., 2. };
	complex float ref[N] = { 1.2 - 0.4i, -0.4 + 0.8i, 2.8 + 0.4i, -1.6 - 0.8i };

	solve_tri_matrix_vec(N, A, B, false);

	// Upper

	complex float A2[N][N] = {
		{ 3. + 1i, 3., 2., 1. },
		{ 0., 1., 3., 1i },
		{ 0., 0., 1., 1. },
		{ 0., 0., 0., 1. }
	};

	complex float B2[N] = { 4., 2., 4., 2. };
	complex float ref2[N] = { 3.6 + 0.8i, -4. - 2i, 2., 2. };

	solve_tri_matrix_vec(N, A2, B2, true);

	double err = 0.;
	double err2 = 0.;

	for (int i = 0; i < N; i++) {

		err += powf(cabsf(ref[i] - B[i]), 2.);
		err2 += powf(cabsf(ref2[i] - B2[i]), 2.);
	}
	// debug_printf(DP_INFO, "err: %f\n", err);
	// debug_printf(DP_INFO, "err2: %f\n", err2);

	return ( (err < 1.E-10) && (err2 < 1.E-10) );
}


UT_REGISTER_TEST(test_trimat_solve);


static bool test_trimat_solve_sylvester(void)
{
	enum { N = 4 };
	enum { M = 1 };

	complex float A[N][N] = {
		{ 3. + 1i, 3., 2., 1. },
		{ 0., 1., 3., 1i },
		{ 0., 0., 1., 1. },
		{ 0., 0., 0., 1. }
	};

	complex float B[M][M] = { { 1 } };
	complex float C[N][M] = { { 1. }, { 2. }, { 3. }, { 4. } };

	complex float ref[N][M] = {	{ -0.47058824+0.86764706i },
					{ 0.25      -1.i },
					{ 0.5 },
					{ 2. } };

	float scale = 1.;

	solve_tri_matrix_sylvester(N, M, &scale, A, B, C);

	// debug_printf(DP_INFO, "\nC:\n");
	// for (int i = 0; i < N; i++) {
	// 	for (int j = 0; j < M; j++)
	// 		debug_printf(DP_INFO, "%f+i*%f ", crealf(C[i][j]), cimagf(C[i][j]));
	// 	debug_printf(DP_INFO, "\n");
	// }
	// debug_printf(DP_INFO, "scale: %f\n", scale);

	double err = 0.;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			err += powf(cabsf(ref[i][j] - C[i][j]), 2.);

	// debug_printf(DP_INFO, "err: %f\n", err);

	return (err < 1.E-10);
}


UT_REGISTER_TEST(test_trimat_solve_sylvester);


static bool test_trimat_sqrt(void)
{
	enum { N = 4 };

	complex float A[N][N] = {
		{ 3. + 1i, 3., 2., 1. },
		{ 0., 1., 3., 1i },
		{ 0., 0., 1., 1. },
		{ 0., 0., 0., 1. }
	};

	complex float ref[N][N] = {
		{ 1.7553173+0.28484878i,  1.07729003-0.11137184i, 0.14411829+0.04573188j, 0.43589711-0.26401321i },
		{ 0., 1., 1.5, -0.375+0.5i },
		{ 0., 0., 1., 0.5 },
		{ 0., 0., 0., 1. }
	};

	// Single block -> only within-block interactions
	complex float B[N][N];
	sqrtm_tri_matrix(N, 32, B, A);

	// Two Blocks -> tests between-block interactions
	complex float B2[N][N];
	sqrtm_tri_matrix(N, 2, B2, A);

	float err = 0.;
	float err2 = 0.;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) {

			err += powf(cabsf(ref[i][j] - B[i][j]), 2.);
			err2 += powf(cabsf(ref[i][j] - B2[i][j]), 2.);
	}
	// debug_printf(DP_INFO, "err: %f\n", err);
	// debug_printf(DP_INFO, "err2: %f\n", err2);

	return ( (err < 1.E-10) && (err2 < 1.E-10) );
}


UT_REGISTER_TEST(test_trimat_sqrt);


static bool test_onenorm_power(void)
{
	enum { N = 4 };

	complex float A[N][N] = {
		{ 3. + 1i, 3., 2., 1. },
		{ 0., 1., 3., 1i },
		{ 0., 0., 1., 1. },
		{ 0., 0., 0., 1. }
	};

	float ref = 65.764732189829;
	float err = fabsf(ref - mat_onenorm_power(N, 4, A));

	// debug_printf(DP_INFO, "err: %f\n", err);

	return (err < 1.E-10);
}


UT_REGISTER_TEST(test_onenorm_power);


static bool test_trimat_logm(void)
{
	enum { N = 4 };

	complex float A[N][N] = {
		{ 3. + 1i, 3., 2., 1. },
		{ 0., 1., 3., 1i },
		{ 0., 0., 1., 1. },
		{ 0., 0., 0., 1. }
	};

	complex float ref[N][N] = {
		{ 1.15129255+0.32175055i,  1.57460139-0.30467486i, -0.84354899+0.28651276j, 1.08154031-0.9493378i },
		{ 0., 0., 3., -1.5+1i },
		{ 0., 0., 0., 1. },
		{ 0., 0., 0., 0. }
	};

	complex float B[N][N];
	logm_tri_matrix(N, B, A);

	float err = 0.;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			err += powf(cabsf(ref[i][j] - B[i][j]), 2.);

	// debug_printf(DP_INFO, "err: %f\n", err);

	return (err < 1.E-10);
}


UT_REGISTER_TEST(test_trimat_logm);


static bool test_trimat_logm2(void)
{
	enum { N = 6 };

	complex float A[N][N] = {
		{ 3. + 1.i, 3., 2., 1., 4., 6. },
		{ 0., 1., 3., 1.i, 2.i, 1. },
		{ 0., 0., 1., 1., 3., 5. },
		{ 0., 0., 0., 1., 6., 7.i },
		{ 0., 0., 0., 0., 4.i, 1. },
		{ 0., 0., 0., 0., 0., 1. }
	};

	complex float ref[N][N] = {
		{ 1.15129255+0.32175055i, 1.57460139-0.30467486i, -0.84354899+0.28651276i, 1.08154031-0.9493378i, 0.98541423-0.22099226i, 1.47210491-6.27495164i },
		{  0., 0., 3., -1.5+1.i, -0.48087229-0.115629i, -2.67539081+7.41406577i },
		{ 0., 0., 0., 1., 0.52448693-0.10292112i,  4.94493109-3.61735451i },
		{ 0., 0., 0., 0., 1.72831445-2.51152015i, -0.3396703 +8.15283896i },
		{ 0., 0., 0., 0., 1.38629436+1.57079633i,  0.28805241-0.41858669i },
		{ 0., 0., 0., 0., 0., 0. }
	};

	complex float B[N][N];
	logm_tri_matrix(N, B, A);

	float err = 0.;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			err += powf(cabsf(ref[i][j] - B[i][j]), 2.);

	// debug_printf(DP_INFO, "err: %f\n", err);

	return (err < 1.E-10);
}


UT_REGISTER_TEST(test_trimat_logm2);


static bool test_logm(void)
{
	enum { N = 4 };

	complex float A[N][N] = {
		{ 3.+1i, 3., 2., 1. },
		{ 2.i, 1., 3., 1.i },
		{ 1., 4.+4.i, 1., 1. },
		{ 1.i, 0., 0., 1. }
	};

	complex float ref[N][N] = {
		{ 1.07990257-0.02007447i, 0.95641065+0.09227379i, 0.32946696-0.14265677i, 0.53555843-0.13405487i },
		{ -0.43929099+0.19432415i, 1.58969964-1.12246402i, 0.65622748+0.88426883i, -0.13780588-0.11543179i },
		{ 0.81859337+0.58525132i, -0.76794456+1.58589704i, 1.17135624-1.18457638i, 0.28367556+0.54639508i },
		{ -0.04033954+0.55224787i, 0.07444528-0.25834293i, -0.08394852-0.09113472i, -0.04200947-0.22026408i }
	};

	complex float B[N][N];
	mat_logm(N, B, A);

	float err = 0.;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			err += powf(cabsf(ref[i][j] - B[i][j]), 2.);

	// debug_printf(DP_INFO, "err: %f\n", err);

	return (err < 1.E-10);
}


UT_REGISTER_TEST(test_logm);

