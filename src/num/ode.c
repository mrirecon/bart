/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <math.h>
//sqrt

// #include "iter/vec_iter.h"

#include "ode.h"


static void vec_saxpy(unsigned int N, float dst[N], const float a[N], float alpha, const float b[N])
{
	for (unsigned int i = 0; i < N; i++)
		dst[i] = a[i] + alpha * b[i];
}

static void vec_copy(unsigned N, float dst[N], const float src[N])
{
	vec_saxpy(N, dst, src, 0., src);
}

static float vec_sdot(unsigned int N, const float a[N], const float b[N])
{
	float ret = 0.;

	for (unsigned int i = 0; i < N; i++)
		ret += a[i] * b[i];

	return ret;
}

static float vec_norm(unsigned int N, const float x[N])
{
	return sqrtf(vec_sdot(N, x, x));
}



#define tridiag(s) (s * (s + 1) / 2)

static void runga_kutta_step(float h, unsigned int s, const float a[tridiag(s)], const float b[s], const float c[s - 1], unsigned int N, unsigned int K, float k[K][N], float ynp[N], float tmp[N], float tn, const float yn[N], void* data, void (*f)(void* data, float* out, float t, const float* yn))
{
	vec_saxpy(N, ynp, yn, h * b[0], k[0]);

	for (unsigned int l = 0, t = 1; t < s; t++) {

		vec_copy(N, tmp, yn);

		for (unsigned int r = 0; r < t; r++, l++)
			vec_saxpy(N, tmp, tmp, h * a[l], k[r % K]);

		f(data, k[t % K], tn + h * c[t - 1], tmp);

		vec_saxpy(N, ynp, ynp, h * b[t], k[t % K]);
	}
}

// Runga-Kutta 4

void rk4_step(float h, unsigned int N, float ynp[N], float tn, const float yn[N], void* data, void (*f)(void* data, float* out, float t, const float* yn))
{
	const float c[3] = { 0.5, 0.5, 1. };

	const float a[6] = {
		0.5,
		0.0, 0.5,
		0.0, 0.0, 1.0,
	};
	const float b[4] = { 1. / 6., 1. / 3., 1. / 3., 1. / 6. };

	float k[1][N];	// K = 1 because only diagonal elements are used
	f(data, k[0], tn, yn);

	float tmp[N];
	runga_kutta_step(h, 4, a, b, c, N, 1, k, ynp, tmp, tn, yn, data, f);
}



/*
 * Dormand JR, Prince PJ. A family of embedded Runge-Kutta formulae,
 * Journal of Computational and Applied Mathematics 6:19-26 (1980).
 */
void dormand_prince_step(float h, unsigned int N, float ynp[N], float tn, const float yn[N], void* data, void (*f)(void* data, float* out, float t, const float* yn))
{
	const float c[6] = { 1. / 5., 3. / 10., 4. / 5., 8. / 9., 1., 1. };

	const float a[tridiag(7)] = {
		1. / 5.,
		3. / 40.,	9. / 40.,
		44. / 45.,	-56. / 15.,	32. / 9.,
		19372. / 6561.,	-25360. / 2187., 64448. / 6561., -212. / 729.,
		9017. / 3168.,  -355. / 33.,	46732. / 5247.,	49. / 176.,	-5103. / 18656.,
		35. / 384.,	0.,		500. / 1113.,	125. / 192.,	-2187. / 6784.,	11. / 84.,
	};

	const float b[7] = { 5179. / 57600., 0.,  7571. / 16695., 393. / 640., -92097. / 339200., 187. / 2100., 1. / 40. };

	float k[6][N];
	f(data, k[0], tn, yn);

	float tmp[N];
	runga_kutta_step(h, 7, a, b, c, N, 6, k, ynp, tmp, tn, yn, data, f);
}



float dormand_prince_scale(float tol, float err)
{
#if 0
	float sc = 0.75 * powf(tol / err, 1. / 5.);

	return (sc < 2.) ? sc : 2.;
#else
	float sc = 1.25 * powf(err / tol, 1. / 5.);

	return 1. / ((sc > 1. / 2.) ? sc : (1. / 2.));
#endif
}



float dormand_prince_step2(float h, unsigned int N, float ynp[N], float tn, const float yn[N], float k[6][N], void* data, void (*f)(void* data, float* out, float t, const float* yn))
{
	const float c[6] = { 1. / 5., 3. / 10., 4. / 5., 8. / 9., 1., 1. };

	const float a[tridiag(7)] = {
		1. / 5.,
		3. / 40.,	9. / 40.,
		44. / 45.,	-56. / 15.,	32. / 9.,
		19372. / 6561.,	-25360. / 2187., 64448. / 6561., -212. / 729.,
		9017. / 3168.,  -355. / 33.,	46732. / 5247.,	49. / 176.,	-5103. / 18656.,
		35. / 384.,	0.,		500. / 1113.,	125. / 192.,	-2187. / 6784.,	11. / 84.,
	};

	const float b[7] = { 5179. / 57600., 0.,  7571. / 16695., 393. / 640., -92097. / 339200., 187. / 2100., 1. / 40. };

	float tmp[N];
	runga_kutta_step(h, 7, a, b, c, N, 6, k, ynp, tmp, tn, yn, data, f);

	vec_saxpy(N, tmp, tmp, -1., ynp);
	return vec_norm(N, tmp);
}


void ode_interval(float h, float tol, unsigned int N, float x[N], float st, float end, void* data, void (*f)(void* data, float* out, float t, const float* yn))
{
	float k[6][N];
	f(data, k[0], st, x);

	if (h > end - st)
		h = end - st;

	for (float t = st; t < end; ) {

		float ynp[N];
	repeat:
		;
		float err = dormand_prince_step2(h, N, ynp, t, x, k, data, f);

		float h_new = h * dormand_prince_scale(tol, err);

		if (err > tol) {

			h = h_new;
			f(data, k[0], t, x);	// recreate correct k[0] which has been overwritten
			goto repeat;
		}

		t += h;
		h = h_new;

		if (t + h > end)
			h = end - t;

		for (unsigned int i = 0; i < N; i++)
			x[i] = ynp[i];
	}
}


struct ode_matrix_s {

	unsigned int N;
	const float* matrix;
};

static void ode_matrix_fun(void* _data, float* x, float t, const float* in)
{
	struct ode_matrix_s* data = _data;
	(void)t;

	unsigned int N = data->N;

	for (unsigned int i = 0; i < N; i++) {

		x[i] = 0.;

		for (unsigned int j = 0; j < N; j++)
			x[i] += (*(const float(*)[N][N])data->matrix)[i][j] * in[j];
	}
}

void ode_matrix_interval(float h, float tol, unsigned int N, float x[N], float st, float end, const float matrix[N][N])
{
	struct ode_matrix_s data = { N, &matrix[0][0] };
	ode_interval(h, tol, N, x, st, end, &data, ode_matrix_fun);
}



// the direct method for sensitivity analysis
// ode: d/dt y_i = f_i(y, t, p_j), y_i(0) = a_i
// d/dp_j y_i(0) = ...
// d/dt d/dp_j y_i(t) = d/dp_j f_i(y, t, p) = \sum_k d/dy_k f_i(y, t, p) * dy_k/dp_j + df_i / dp_j

struct seq_data {

	unsigned int N;
	unsigned int P;

	void* data;
	void (*f)(void* data, float* out, float t, const float* yn);
	void (*pdy)(void* data, float* out, float t, const float* yn);
	void (*pdp)(void* data, float* out, float t, const float* yn);
};

static void seq(void* _data, float* out, float t, const float* yn)
{
	struct seq_data* data = _data;
	int N = data->N;
	int P = data->P;

	data->f(data->data, out, t, yn);

	float dy[N][N];
	data->pdy(data->data, &dy[0][0], t, yn);

	float dp[P][N];
	data->pdp(data->data, &dp[0][0], t, yn);

	for (int i = 0; i < P; i++) {
		for (int j = 0; j < N; j++) {

			out[(1 + i) * N + j] = 0.;

			for (int k = 0; k < N; k++)
				out[(1 + i) * N + j] += dy[k][j] * yn[(1 + i) * N + k];

			out[(1 + i) * N + j] += dp[i][j];
		}
	}
}

void ode_direct_sa(float h, float tol, unsigned int N, unsigned int P, float x[P + 1][N],
	float st, float end, void* data,
	void (*f)(void* data, float* out, float t, const float* yn),
	void (*pdy)(void* data, float* out, float t, const float* yn),
	void (*pdp)(void* data, float* out, float t, const float* yn))
{
	struct seq_data data2 = { N, P, data, f, pdy, pdp };

	ode_interval(h, tol, N * (1 + P), &x[0][0], st, end, &data2, seq);
}







