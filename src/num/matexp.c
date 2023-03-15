/* Copyright 2018-2023. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "num/ode.h"
#include "num/linalg.h"

#include "matexp.h"


void mat_exp(int N, float t, float out[N][N], const float in[N][N])
{
	// compute F(t) := exp(tA)

	// F(0) = id
	// d/dt F = A

	float h = t / 100.;
	float tol = 1.E-6;

	for (int i = 0; i < N; i++) {

		for (int j = 0; j < N; j++)
			out[i][j] = (i == j) ? 1. : 0.;

		ode_matrix_interval(h, tol, N, out[i], 0., t, in);
	}
}


void mat_to_exp(int N, float st, float en, float out[N][N], float tol,
		void CLOSURE_TYPE(f)(float* out, float t, const float* yn))
{
	float h = (en - st) / 100.;

	for (int i = 0; i < N; i++) {

		for (int j = 0; j < N; j++)
			out[i][j] = (i == j) ? 1. : 0.;

		ode_interval(h, tol, N, out[i], st, en, f);
	}
}

