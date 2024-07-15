/* Copyright 2023. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 **/

#include <math.h>
#include <complex.h>
#include <assert.h>

#include "num/rand.h"
#include "num/linalg.h"
#include "num/multind.h"

#include "gaussians.h"


float gaussian_pdf(int N, const complex float m[N], const
		complex float isqr_cov[N][N], const complex float x[N])
{
	complex float u[N];

	vec_copy(N, u, x);
	vec_saxpy(N, u, -1., m);

	complex float v[N];
	mat_vecmul(N, N, v, isqr_cov, u);

	float sum = crealf(vec_dot(N, v, v));

	float f = pow(M_PI, -N);
	float d = pow(cabsf(mat_det(N, isqr_cov)), 2. * N);
	return f * d * exp(-1. * sum);
}


float gaussian_mix_pdf(int M, int N, const float coeff[M], const complex float m[M][N],
		const complex float isqr_cov[M][N][N], const complex float x[N])
{
	float sum = 0.;

	for (int i = 0; i < M; i++)
		sum += coeff[i] * gaussian_pdf(N, m[i], isqr_cov[i], x);

	return sum;
}



void gaussian_sample(int N, const complex float m[N],
		const complex float sqr_cov[N][N], complex float x[N])
{
	complex float u[N];
	memset(u, 0, sizeof u); // maybe-uninitialized
       
	for (int i = 0; i < N; i++)
		u[i] = gaussian_rand() / sqrtf(2.);

	mat_vecmul(N, N, x, sqr_cov, u);

	for (int i = 0; i < N; i++)
		x[i] += m[i];
}


void gaussian_mix_sample(int M, int N, const float coeff[M], const complex float m[M][N],
		const complex float sqr_cov[M][N][N], complex float x[N])
{
	float r = uniform_rand();

	int ind;
	float icoeff = 0.;
	for (ind = 0; ind < M; ind++) {

		icoeff += coeff[ind];

		if (r <= icoeff)
			break;
	}

	assert(1. >= icoeff);
	assert(ind < M);

	gaussian_sample(N, m[ind], sqr_cov[ind], x);
}



void gaussian_score(int N, const complex float m[N], const complex float isqr_cov[N][N],
		const complex float x[N], complex float sc[N])
{
	complex float icov[N][N];
	mat_mul(N, N, N, icov, isqr_cov, isqr_cov); // FIXME conj

	complex float u[N];
	vec_zero(N, u);
	vec_saxpy(N, u, -2., x);
	vec_saxpy(N, u, +2., m);
	mat_vecmul(N, N, sc, icov, u);
}


void gaussian_mix_score(int M, int N, const float coeff[M], const complex float m[M][N],
		const complex float isqr_cov[M][N][N],
		const complex float x[N], complex float sc[N])
{
	vec_zero(N, sc);
	float c[M];
	float no = 0.;

	for (int i = 0; i < M; i++) {

		c[i] = coeff[i] * gaussian_pdf(N, m[i], isqr_cov[i], x);
		no += c[i];
	}

	for (int i = 0; i < M; i++) {

		complex float u[N];
		gaussian_score(N, m[i], isqr_cov[i], x, u);

		vec_saxpy(N, sc, c[i] / no, u);
	}
}


void gaussian_convolve(int N, complex float m[N],
		complex float sqr_cov[N][N],
		const complex float m1[N],
		const complex float sqr_cov1[N][N],
		const complex float m2[N],
		const complex float sqr_cov2[N][N])
{
	assert(1 == N);
	assert(0. <= crealf(sqr_cov1[0][0]));
	assert(0. <= crealf(sqr_cov2[0][0]));
	assert(0. == cimagf(sqr_cov1[0][0]));
	assert(0. == cimagf(sqr_cov2[0][0]));

	m[0] = m1[0] + m2[0];

	sqr_cov[0][0] = sqrtf(powf(crealf(sqr_cov1[0][0]), 2.) + powf(crealf(sqr_cov2[0][0]), 2.));
}


void gaussian_multiply(int N, complex float m[N],
		complex float isqr_cov[N][N],
		const complex float m1[N],
		const complex float isqr_cov1[N][N],
		const complex float m2[N],
		const complex float isqr_cov2[N][N])
{
	assert(1 == N);
	assert(0. <= crealf(isqr_cov1[0][0]));
	assert(0. <= crealf(isqr_cov2[0][0]));
	assert(0. == cimagf(isqr_cov1[0][0]));
	assert(0. == cimagf(isqr_cov2[0][0]));

	float v1 = powf(crealf(isqr_cov1[0][0]), -2.);
	float v2 = powf(crealf(isqr_cov2[0][0]), -2.);

	m[0] = (v1 * m1[0] + v2 * m2[0]) / (v1 + v2);

	isqr_cov[0][0] = sqrtf(powf(crealf(isqr_cov1[0][0]), 2.) + powf(crealf(isqr_cov2[0][0]), 2.));
}


