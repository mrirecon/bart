/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <math.h>
#include <assert.h>

#include "polynom.h"

complex double polynom_eval(complex double x, int N, const complex double coeff[N + 1])
{
	// Horner's method: a_0 + x * (a_1 + x * (a_2 + ...)
	return coeff[0] + ((0 == N) ? 0. : (x * polynom_eval(x, N - 1, coeff + 1)));
}

void (polynom_derivative)(int N, complex double out[N], const complex double in[N + 1])
{
	for (int i = 0; i < N; i++)
		out[i] = (i + 1) * in[i + 1];
}

void polynom_integral(int N, complex double out[N + 2], const complex double in[N + 1])
{
	out[0] = 0.;

	for (int i = 0; i <= N; i++)
		out[i + 1] = in[i] / (i + 1);
}

complex double polynom_integrate(complex double st, complex double end, int N, const complex double coeff[N + 1])
{
	complex double int_coeff[N + 2];
	polynom_integral(N, int_coeff, coeff);

	return polynom_eval(end, N + 1, int_coeff) - polynom_eval(st, N + 1, int_coeff);
}

void polynom_monomial(int N, complex double coeff[N + 1], int O)
{
	for (int i = 0; i <= N; i++)
		coeff[i] = (O == i) ? 1. : 0.;
}

void polynom_from_roots(int N, complex double coeff[N + 1], const complex double root[N])
{
	// Vieta's formulas

	for (int i = 0; i <= N; i++)
		coeff[i] = 0.;

	// assert N < 
	for (unsigned long b = 0; b < (1u << N); b++) {

		complex double prod = 1.;
		int count = 0;

		for (int i = 0; i < N; i++) {

			if (b & (1 << i)) {

				prod *= -root[i];
				count++;
			}
		}

		coeff[N - count] += prod;
	}
}


void polynom_scale(int N, complex double out[N + 1], complex double scale, const complex double in[N + 1])
{
	complex double prod = 1.;

	for (int i = 0; i <= N; i++) {

		out[i] = prod * in[i];
		prod *= scale;
	}
}


void polynom_shift(int N, complex double out[N + 1], complex double shift, const complex double in[N + 1])
{
	// Taylor shift (there are faster FFT-based methods)

	for (int i = 0; i <= N; i++)
		out[i] = 0.;

	complex double tmp[N + 1];

	for (int i = 0; i <= N; i++)
		tmp[i] = in[i];

	complex double prod = 1.;

	for (int i = 0; i <= N; i++) {

		for (int j = 0; j <= (N - i); j++)
			out[j] += prod * tmp[j];

		polynom_derivative(N - i, tmp, tmp);

		prod *= shift;
		prod /= (i + 1);
	}
}


void quadratic_formula(complex double x[2], complex double coeff[3])
{
	complex double c = coeff[0];
	complex double b = coeff[1];
	complex double a = coeff[2];

	assert(0. != a);

	complex double t = csqrt(cpow(b, 2.) - 4. * a * c);

	x[0] = (-b + t) / (2. * a);
	x[1] = (-b - t) / (2. * a);

	// FIXME: precision
	// Citardauq Formula
	//	x[1] = 2. * c / (-b + s * t);
}


void cubic_formula(complex double x[3], complex double coeff[4])
{
	complex double a = coeff[3];
	complex double b = coeff[2];
	complex double c = coeff[1];
	complex double d = coeff[0];

	assert(0. != a);

	// depressed form t^3 + p t + q with t = -b / (3 a)
	complex double p = (3. * a * c - cpow(b, 2.)) / (3. * cpow(a, 2.));
	complex double q = (2. * cpow(b, 3.) - 9. * a * b * c + 27. * cpow(a, 2.) * d) / (27. * cpow(a, 3.)); 

	// Vieta's substitution: quadratic in w^3 with t = w - p / (3 w)
	complex double qp[3] = { -cpow(p, 3.) / 27., q, 1. };
	complex double qw[2];

	quadratic_formula(qw, qp);

	if (0. == qw[0])
		qw[0] = qw[1];

	complex double w1 = cpow(qw[0], 1. / 3.);

	complex double ksi = 0.5 * (-1. + sqrt(3.) * 1.i);

	for (int i = 0; i < 3; i++) {

		complex double wi = cpow(ksi, i) * w1;
		complex double ti = (0. == wi) ? 0. : (wi - p / (3. * wi));

		x[i] = ti - b / (3. * a);
	}
}


