/* Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <math.h>
#include <assert.h>

#include "splines.h"


static long binomial(unsigned int n, unsigned int k)
{
	long result = 1;

	for (unsigned int i = 1; i <= k; i++)
		result *= (n + 1 - i);

	for (unsigned int i = 1; i <= k; i++)
		result /= i;

	return result;
}


/* basis */
double bernstein(unsigned int n, unsigned int v, double x)
{
	assert(v <= n);

	return binomial(n, v) * pow(x, v) * pow(1. - x, n - v);
}


static double lerp(double t, double a, double b)
{
	return (1. - t) * a + t * b;
}




static void de_casteljau_step(unsigned int N, double out[static N], double t, const double coeff[static N + 1])
{
	for (unsigned int i = 0; i < N; i++)
		out[i] = lerp(t, coeff[i], coeff[i + 1]);
}


static double de_casteljau(double t, unsigned int N, const double coeff[static N + 1])
{
	if (0 == N)
		return coeff[0];

	double coeff2[N];
	de_casteljau_step(N, coeff2, t, coeff);

	return de_casteljau(t, N - 1, coeff2);
}


static void de_casteljau_split(double t, unsigned int N, double coeffA[static N + 1], double coeffB[static N + 1], const double coeff[static N + 1])
{
	coeffA[0] = coeff[0];
	coeffB[N] = coeff[N];

	if (0 == N)
		return;

	double coeff2[N];
	de_casteljau_step(N, coeff2, t, coeff);

	de_casteljau_split(t, N - 1, coeffA + 1, coeffB, coeff2);	
}


void bezier_split(double t, unsigned int N, double coeffA[static N + 1], double coeffB[static N + 1], const double coeff[static N + 1])
{
	de_casteljau_split(t, N, coeffA, coeffB, coeff);
}


double bezier_curve(double u, unsigned int N, const double k[static N + 1])
{
	return de_casteljau(u, N, k);
}


void bezier_increase_degree(unsigned int N, double coeff2[static N + 2], const double coeff[static N + 1])
{
	coeff2[0] = coeff[0];

	for (unsigned int i = 1; i <= N; i++)
		coeff2[i] = lerp(i / (1. + N), coeff[i], coeff[i - 1]);

	coeff2[N + 1] = coeff[N];
}

	
double bezier_surface(double u, double v, unsigned int N, unsigned int M, const double k[static N + 1][M + 1])
{
	double coeff[N + 1];

	for (unsigned int i = 0; i <= N; i++)
		coeff[i] = bezier_curve(u, M, k[i]);

	return bezier_curve(v, N, coeff);
}


double bezier_patch(double u, double v, const double k[4][4])
{
	return bezier_surface(u, v, 3, 3, k);
}




static void cspline2bezier(double out[4], const double in[4])
{
	const double m[4][4] = {
		{	1.,	1.,	0.,	0.,	},
		{	0.,	1./ 3.,	0.,	0.,	},
		{	0.,	0.,	1.,	1.,	},
		{	0.,	0.,	-1./3.,	0.,	},
	};

	for (int i = 0; i < 4; i++) {

		out[i] = 0.;

		for (int j = 0; j < 4; j++)
			out[i] += m[j][i] * in[j];
	}
}


// cubic hermite spline
double cspline(double t, const double coeff[4])
{
	double coeff2[4];
	cspline2bezier(coeff2, coeff);

	return bezier_curve(t, 3, coeff2);
}



static double frac(double u, double v)
{
	if (0. == v) {

		// assert(0. == u);
		return 0.;
	}

	return u / v;
}



/*
 * bspline blending function of order p with n + 1 knots. i enumerates the basis.
 */
double bspline(unsigned int n, unsigned int i, unsigned int p, const double tau[static n + 1], double u)
{
	assert(i + p < n);
	assert(tau[i] <= tau[i + 1]);
	assert((tau[0] <= u) && (u <= tau[n]));

	if (0 == p)
		return ((tau[i] <= u) && (u < tau[i + 1])) ? 1. : 0.;

	assert(tau[i] <= tau[i + p + 1]);

	double a = frac(u - tau[i], tau[i + p] - tau[i]);
	double b = frac(tau[i + p + 1] - u, tau[i + p + 1] - tau[i + 1]);

	return a * bspline(n, i, p - 1, tau, u) + b * bspline(n, i + 1, p - 1, tau, u);
}



double bspline_derivative(unsigned int n, unsigned int i, unsigned int p, const double tau[static n + 1], double x)
{
	assert(p > 0);

	double a = frac(p, tau[i + p] - tau[i]);
	double b = frac(p, tau[i + p + 1] - tau[i + 1]);

	return a * bspline(n, i, p - 1, tau, x) - b * bspline(n, i + 1, p - 1, tau, x);
}


double nurbs(unsigned int n, unsigned int p, const double tau[static n + 1], const double coord[static n - p],
	const double w[static n - p], double x)
{
#if 0
	double sum = 0.;
	double nrm = 0.;

	for (unsigned int i = 0; i < n + 0 - p; i++) {

		double b = bspline(n, i, p, tau, x);

		sum += w[i] * coord[i] * b;
		nrm += w[i] * b;
	}
#else
	double coordw[n - p];

	for (unsigned int i = 0; i < n + 0 - p; i++)
		coordw[i] = w[i] * coord[i];

	double sum = bspline_curve(n, p, tau, coordw, x);
	double nrm = bspline_curve(n, p, tau, w, x);
#endif
	return sum / nrm;
}



static void cox_deboor_step(unsigned int N, double out[static N], double x, unsigned int p, const double tau[static N + p + 1], const double coeff[static N + 1])
{
	unsigned int k = p - N + 1;

	for (unsigned int s = 0; s < N; s++) {

		double t = frac(x - tau[s + k], tau[s + p + 1] - tau[s + k]);
                out[s] = lerp(t, coeff[s], coeff[s + 1]);
	}
}

static double cox_deboor_i(double x, unsigned int N, unsigned int p, const double tau[static N + 1],  const double coeff[static N + 1])
{
        if (0 == N)
                return coeff[0];

        double coeff2[N];
        cox_deboor_step(N, coeff2, x, p, tau, coeff);

        return cox_deboor_i(x, N - 1, p, tau, coeff2);
}

#if 0
static double cox_deboor_r(unsigned int n, unsigned int p, unsigned int k, unsigned int s, const double t2[static n + 1], const double v2[static n + 1 - p], double x)
{
	if (0 == k)
		return v2[s];

	double t = (x - t2[s]) / (t2[s + p - k + 1] - t2[s]);

	double a = cox_deboor_r(n, p, k - 1, s - 1, t2, v2, x);
	double b = cox_deboor_r(n, p, k - 1, s - 0, t2, v2, x);

	return lerp(t, a, b);
}
#endif

static unsigned int find_span(unsigned int n, const double t[static n + 1], double x)
{
	assert(x >= t[0]);

	unsigned int i = 0;

	while (x >= t[i])
		i++;

	i--;

	return i;
}


static double cox_deboor(unsigned int n, unsigned int p, const double t[static n + 1], const double v[static n + 1 - p], double x)
{
	int i = find_span(n, t, x);

//	return cox_deboor_r(n, p, p, p, t + i - p, v + i - p, x);
	return cox_deboor_i(x, p, p, t + i - p, v + i - p);
}


double bspline_curve(unsigned int n, unsigned int p, const double t[static n + 1], const double v[static n - p], double x)
{
	return cox_deboor(n, p, t, v, x);
}


static void bspline_coeff_derivative(unsigned int n, unsigned int p, double t2[static n - 1], double v2[static n - p - 1], const double t[static n + 1], const double v[static n - p])
{
	for (unsigned int i = 1; i < n; i++)
		t2[i - 1] = t[i];

	for (unsigned int i = 0; i < n - p - 1; i++)
		v2[i] = (float)p / (t[i + p + 1] - t[i + 1]) * (v[i + 1] - v[i]);
}


void bspline_coeff_derivative_n(unsigned int k, unsigned int n, unsigned int p, double t2[static n + 1 - 2 * k], double v2[static n - p - k], const double t[static n + 1], const double v[static n - p])
{
	if (0 == k) {

		for (unsigned int i = 0; i < n + 1; i++)
			t2[i] = t[i];

		for (unsigned int i = 0; i < n - p; i++)
			v2[i] = v[i];

	} else {

		double t1[n - 1];
		double v1[n - p - 1];

		bspline_coeff_derivative(n, p, t1, v1, t, v);
		bspline_coeff_derivative_n(k - 1, n - 1, p - 1, t2, v2, t1, v1);
	}
}


double bspline_curve_derivative(unsigned int k, unsigned int n, unsigned int p, const double t[static n + 1], const double v[static n - p], double x)
{
	double t2[n + 1 - 2 * k];
	double v2[n - p - k];
	bspline_coeff_derivative_n(k, n, p, t2, v2, t, v);

	return cox_deboor(n - 2 * k, p - k, t2, v2, x);
}


static double newton_raphson(int iter, double x0, void* data, double (*fun)(void* data, double x), double (*der)(void* data, double x))
{
	return (0 == iter) ? x0 : newton_raphson(iter - 1, x0 - fun(data, x0) / der(data, x0), data, fun, der);
}

struct bspline_s {

	unsigned int n;
	unsigned int p;
	const double* t;
	const double* v;
};

static double n_fun(void* _data, double x)
{
	struct bspline_s* data = _data;
	return bspline_curve(data->n, data->p, data->t, data->v, x);
}

static double n_der(void* _data, double x)
{
	struct bspline_s* data = _data;
	return bspline_curve_derivative(1, data->n, data->p, data->t, data->v, x);
}

double bspline_curve_zero(unsigned int n, unsigned int p, const double tau[static n + 1], const double v[static n - p])
{
	return newton_raphson(20, (tau[n] + tau[0]) / 2., &(struct bspline_s){ n, p, tau, v }, n_fun, n_der);
}




void bspline_knot_insert(double x, unsigned int n, unsigned int p, double t2[static n + 2], double v2[n - p + 1], const double tau[static n + 1], const double v[static n - p])
{
	unsigned int k = find_span(n, tau, x);

	// knots

	for (unsigned int i = 0; i <= k; i++)
		t2[i] = tau[i];

	t2[k + 1] = x;

	for (unsigned int i = k + 1; i < n; i++)
		t2[i + 1] = tau[i];

	unsigned int r = k - p + 1;
	unsigned int s = k;

	for (unsigned int i = 0; i < r; i++)
		v2[i] = v[i];

	for (unsigned int i = r; i <= s; i++) {

		double a = (x - tau[i]) / (tau[i + p] - tau[i]);

		v2[i] = (1. - a) * v[i - 1] + a * v[i];
	}

	for (unsigned int i = s; i < n - p; i++)
		v2[i + 1] = v[i];
}






