/* Copyright 2018-2023. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>
#include <stdlib.h>

#include "misc/nested.h"

#include "num/quadrature.h"

// #include "iter/vec_iter.h"

#include "ode.h"


static void vec_saxpy(int N, float dst[N], const float a[N], float alpha, const float b[N])
{
	for (int i = 0; i < N; i++)
		dst[i] = a[i] + alpha * b[i];
}

static void vec_copy(int N, float dst[N], const float src[N])
{
	vec_saxpy(N, dst, src, 0., src);
}

static float vec_sdot(int N, const float a[N], const float b[N])
{
	float ret = 0.;

	for (int i = 0; i < N; i++)
		ret += a[i] * b[i];

	return ret;
}

static float vec_norm(int N, const float x[N])
{
	return sqrtf(vec_sdot(N, x, x));
}



#define tridiag(s) (s * (s + 1) / 2)

static void runga_kutta_step(float h, int s, const float a[tridiag(s)], const float b[s], const float c[s - 1], int N, int K, float k[K][N], float ynp[N], float tmp[N], float tn, const float yn[N], void CLOSURE_TYPE(f)(float* out, float t, const float* yn))
{
	vec_saxpy(N, ynp, yn, h * b[0], k[0]);

	for (int l = 0, t = 1; t < s; t++) {

		vec_copy(N, tmp, yn);

		for (int r = 0; r < t; r++, l++)
			vec_saxpy(N, tmp, tmp, h * a[l], k[r % K]);

		NESTED_CALL(f, (k[t % K], tn + h * c[t - 1], tmp));

		vec_saxpy(N, ynp, ynp, h * b[t], k[t % K]);
	}
}

// Runga-Kutta 4

void rk4_step(float h, int N, float ynp[N], float tn, const float yn[N],
		void CLOSURE_TYPE(f)(float* out, float t, const float* yn))
{
	const float c[3] = { 0.5, 0.5, 1. };

	const float a[6] = {
		0.5,
		0.0, 0.5,
		0.0, 0.0, 1.0,
	};
	const float b[4] = { 1. / 6., 1. / 3., 1. / 3., 1. / 6. };

	float k[1][N];	// K = 1 because only diagonal elements are used
	NESTED_CALL(f, (k[0], tn, yn));

	float tmp[N];
	runga_kutta_step(h, 4, a, b, c, N, 1, k, ynp, tmp, tn, yn, f);
}



/*
 * Dormand JR, Prince PJ. A family of embedded Runge-Kutta formulae,
 * Journal of Computational and Applied Mathematics 6:19-26 (1980).
 */
void dormand_prince_step(float h, int N, float ynp[N], float tn, const float yn[N],
		void CLOSURE_TYPE(f)(float* out, float t, const float* yn))
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
	NESTED_CALL(f, (k[0], tn, yn));

	float tmp[N];
	runga_kutta_step(h, 7, a, b, c, N, 6, k, ynp, tmp, tn, yn, f);
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



float dormand_prince_step2(float h, int N, float ynp[N], float tn, const float yn[N], float k[6][N],
		void CLOSURE_TYPE(f)(float* out, float t, const float* yn))
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
	runga_kutta_step(h, 7, a, b, c, N, 6, k, ynp, tmp, tn, yn, f);

	vec_saxpy(N, tmp, tmp, -1., ynp);
	return vec_norm(N, tmp);
}


void ode_interval(float h, float tol, int N, float x[N], float st, float end,
		void CLOSURE_TYPE(f)(float* out, float t, const float* yn))
{
	float k[6][N];
	f(k[0], st, x);

	if (h > end - st)
		h = end - st;

	for (float t = st; t < end; ) {

		float ynp[N];
	repeat:
		;
		float err = dormand_prince_step2(h, N, ynp, t, x, k, f);

		float h_new = h * dormand_prince_scale(tol, err);

		if (err > tol) {

			h = h_new;
			NESTED_CALL(f, (k[0], t, x));	// recreate correct k[0] which has been overwritten
			goto repeat;
		}

		t += h;
		h = h_new;

		if (t + h > end)
			h = end - t;

		for (int i = 0; i < N; i++)
			x[i] = ynp[i];
	}
}



void ode_matrix_interval(float h, float tol, int N, float x[N], float st, float end, const float matrix[N][N])
{
#ifdef __clang__
	const void* matrix2 = matrix;	// clang workaround
#endif
	NESTED(void, ode_matrix_fun, (float* x, float t, const float* in))
	{
		(void)t;
#ifdef __clang__
		const float (*matrix)[N] = matrix2;
#endif
		for (int i = 0; i < N; i++) {

			x[i] = 0.;

			for (int j = 0; j < N; j++)
				x[i] += matrix[i][j] * in[j];
		}
	};

	ode_interval(h, tol, N, x, st, end, ode_matrix_fun);
}



// the direct method for sensitivity analysis
// ode: d/dt y_i = f_i(y, t, p_j), y_i(0) = a_i
// d/dp_j y_i(0) = ...
// d/dt d/dp_j y_i(t) = d/dp_j f_i(y, t, p) = \sum_k d/dy_k f_i(y, t, p) * dy_k/dp_j + df_i / dp_j

struct seq_data {

	int N;
	int P;

	void CLOSURE_TYPE(f)(float* out, float t, const float* yn);
	void CLOSURE_TYPE(pdy)(float* out, float t, const float* yn);
	void CLOSURE_TYPE(pdp)(float* out, float t, const float* yn);
};

static void seq(const struct seq_data* data, float* out, float t, const float* yn)
{
	int N = data->N;
	int P = data->P;

	data->f(out, t, yn);

	float dy[N][N];
	data->pdy(&dy[0][0], t, yn);

	float dp[P][N];
	data->pdp(&dp[0][0], t, yn);

	for (int i = 0; i < P; i++) {
		for (int j = 0; j < N; j++) {

			out[(1 + i) * N + j] = 0.;

			for (int k = 0; k < N; k++)
				out[(1 + i) * N + j] += dy[k][j] * yn[(1 + i) * N + k];

			out[(1 + i) * N + j] += dp[i][j];
		}
	}
}

void ode_direct_sa(float h, float tol, int N, int P, float x[P + 1][N],
	float st, float end,
	void CLOSURE_TYPE(f)(float* out, float t, const float* yn),
	void CLOSURE_TYPE(pdy)(float* out, float t, const float* yn),
	void CLOSURE_TYPE(pdp)(float* out, float t, const float* yn))
{
	struct seq_data data2 = { N, P, f, pdy, pdp };

	NESTED(void, seq2, (float* out, float t, const float* yn))
	{
		seq(&data2, out, t, yn);
	};

	ode_interval(h, tol, N * (1 + P), &x[0][0], st, end, seq2);
}


// the adjoint method for sensitivity analysis
// void (*s)(void* data, float* out, float t)
// void M

void ode_adjoint_sa(float h, float tol,
	int N, const float t[N + 1],
	int M, float x[N + 1][M], float z[N + 1][M],
	const float x0[M],
	void CLOSURE_TYPE(sys)(float dst[M], float t, const float in[M]),
	void CLOSURE_TYPE(sysT)(float dst[M], float t, const float in[M]),
	void CLOSURE_TYPE(cost)(float dst[M], float t))
{
	// forward solution

	for (int m = 0; m < M; m++)
		x[0][m] = x0[m];

	for (int i = 0; i < N; i++) {

		for (int m = 0; m < M; m++)
			x[i + 1][m] = x[i][m];

		ode_interval(h, tol, M, x[i + 1], t[i], t[i + 1], sys);
	}

	// adjoint state

	for (int m = 0; m < M; m++)
		z[N][m] = 0.;

	for (int i = N; 0 < i; i--) {

		for (int m = 0; m < M; m++)
			z[i - 1][m] = z[i][m];

		// invert time -> ned. sign on RHS

		NESTED(void, asa_eval, (float out[M], float t, const float yn[M]))
		{
			NESTED_CALL(sysT, (out, -t, yn));

			float off[M];
			NESTED_CALL(cost, (off, -t));

			for (int m = 0; m < M; m++)
				out[m] += off[m];
		};

		ode_interval(h, tol, M, z[i - 1], -t[i], -t[i - 1], asa_eval);
	}
}


void ode_matrix_adjoint_sa(float h, float tol,
	int N, const float t[N + 1],
	int M, float x[N + 1][M], float z[N + 1][M],
	const float x0[M], const float sys[N][M][M],
	const float cost[N][M])
{
	// forward solution

	for (int m = 0; m < M; m++)
		x[0][m] = x0[m];

	for (int i = 0; i < N; i++) {

		for (int m = 0; m < M; m++)
			x[i + 1][m] = x[i][m];

		ode_matrix_interval(h, tol, M, x[i + 1], t[i], t[i + 1], sys[i]);
	}

	// adjoint state

	for (int m = 0; m < M; m++)
		z[N][m] = 0.;

	for (int i = N; 0 < i; i--) {

		for (int m = 0; m < M; m++)
			z[i - 1][m] = z[i][m];
#ifdef __clang__
		const void* cost2 = cost;
		const void* sys2 = sys;
#endif
		// invert time -> ned. sign on RHS

		NESTED(void, matrix_fun, (float x[M], float t, const float in[M]))
		{
			(void)t;
#ifdef __clang__
			const float (*cost)[M] = cost2;
			const float (*sys)[M][M] = sys2;
#endif
			for (int l = 0; l < M; l++) {

				x[l] = cost[i - 1][l];

				for (int k = 0; k < M; k++)
					x[l] += sys[i - 1][k][l] * in[k];
			}
		};

		ode_interval(h, tol, M, z[i - 1], -t[i], -t[i - 1], matrix_fun);
	}
}

static float adj_eval(int M, const float x[M], const float z[M], const float Adp[M][M])
{
	float ret = 0.;

	for (int l = 0; l < M; l++)
		for (int k = 0; k < M; k++)
			ret += z[l] * Adp[l][k] * x[k];

	return ret;
}

void ode_adjoint_sa_eval(int N, const float t[N + 1], int M,
		int P, float dj[P],
		const float x[N + 1][M], const float z[N + 1][M],
		const float Adp[P][M][M])
{
#ifdef __clang__
	const void* x2 = x;
	const void* z2 = z;
	const void* Adp2 = Adp;
#endif

	NESTED(void, eval, (float out[P], int i))
	{
#ifdef __clang__
		const float (*x)[M] = x2;
		const float (*z)[M] = z2;
		const float (*Adp)[M][M] = Adp2;
#endif
		for (int p = 0; p < P; p++)
			out[p] = adj_eval(M, x[i], z[i], Adp[p]);
	};

	quadrature_trapezoidal(N, t, P, dj, eval);
}


void ode_adjoint_sa_eq_eval(int N, int M, int P, float dj[P],
		const float x[N + 1][M], const float z[N + 1][M],
		const float Adp[P][M][M])
{
#ifdef __clang__
	const void* x2 = x;
	const void* z2 = z;
	const void* Adp2 = Adp;
#endif

	NESTED(void, eval, (float out[P], int i))
	{
#ifdef __clang__
		const float (*x)[M] = x2;
		const float (*z)[M] = z2;
		const float (*Adp)[M][M] = Adp2;
#endif
		for (int p = 0; p < P; p++)
			out[p] = adj_eval(M, x[i], z[i], Adp[p]);
	};

	quadrature_simpson_ext(N, 1., P, dj, eval);
}

