/* Copyright 2017-2020. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Martin Uecker, Sebastian Rosenzweig
 */


#include <complex.h>
#include <math.h>

#include "num/linalg.h"

#include "qform.h"



float quadratic_form(const float qf[3], float phi)
{
	float x = cosf(phi);
	float y = sinf(phi);

	return x * x * qf[0] + 2. * x * y * qf[2] + y * y * qf[1];
}


void fit_quadratic_form(float qf[3], int N, const float phi[N], const float v[N])
{
	complex float lhs[3] = { 0., 0., 0. };
	complex float mat[3][3] = { { 0. } };

	for (int i = 0; i < N; i++) {

		float x = cosf(phi[i]);
		float y = sinf(phi[i]);

		lhs[0] += x * x * v[i];
		lhs[1] += y * y * v[i];
		lhs[2] += 2. * x * y * v[i];

		mat[0][0] +=      x * x *      x * x;
		mat[0][1] +=      x * x *      y * y;
		mat[0][2] +=      x * x * 2. * x * y;
		mat[1][0] +=      y * y *      x * x;
		mat[1][1] +=      y * y *      y * y;
		mat[1][2] +=      y * y * 2. * x * y;
		mat[2][0] += 2. * x * y *      x * x;
		mat[2][1] += 2. * x * y *      y * y;
		mat[2][2] += 2. * x * y * 2. * x * y;
	}

	complex float inv[3][3];
	complex float out[3];

	mat_inverse(3, inv, mat);
	mat_vecmul(3, 3, out, inv, lhs);

	qf[0] = out[0];
	qf[1] = out[1];
	qf[2] = out[2];
}


float harmonic(const float qf[3], float phi)
{
	return qf[0] * cosf(phi) + qf[1] * sinf(phi) + qf[2];
}


void fit_harmonic(float qf[3], int N, const float phi[N], const float v[N])
{
	complex float lhs[3] = { 0., 0., 0. };
	complex float mat[3][3] = { { 0. } };

	for (int i = 0; i < N; i++) {

		float x = cosf(phi[i]);
		float y = sinf(phi[i]);

		lhs[0] += x * v[i];
		lhs[1] += y * v[i];
		lhs[2] += v[i];

		mat[0][0] += x * x;
		mat[0][1] += x * y;
		mat[0][2] += x;
		mat[1][0] += y * x;
		mat[1][1] += y * y;
		mat[1][2] += y;
		mat[2][0] += x;
		mat[2][1] += y;
		mat[2][2] += 1.;
	}

	complex float inv[3][3];
	complex float out[3];

	mat_inverse(3, inv, mat);
	mat_vecmul(3, 3, out, inv, lhs);

	qf[0] = out[0];
	qf[1] = out[1];
	qf[2] = out[2];
}

