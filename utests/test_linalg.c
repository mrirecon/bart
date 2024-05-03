/* Copyright 2016-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <math.h>

#include "num/linalg.h"

#include "utest.h"



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