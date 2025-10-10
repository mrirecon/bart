/* Copyright 2023-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>
#include <complex.h>

#include "num/gaussians.h"

#include "utest.h"


static bool test_gaussian_pdf0(float s)
{
	complex float m[1] = { 0.1 + 0.2i };
	complex float v[1][1] = { { powf(s, -1.) } };
	complex float x[1] = { 0.5 - 0.1i };

	float val1 = gaussian_pdf(1, m, v, x);

	double dist = pow(creal(x[0] - m[0]), 2.) + pow(cimag(x[0] - m[0]), 2.);

	float val2 = exp(-dist / s) / (s * M_PI);

	if (fabsf(val1 - val2) > 1.E-6)
		return false;

	return true;
}

static bool test_gaussian_pdf1(void)
{
	for (float s = 0.1; s < 1.3; s += 0.1)
		if (!test_gaussian_pdf0(s))
			return false;

	return true;
}

UT_REGISTER_TEST(test_gaussian_pdf1);


static bool test_gaussian_pdf2(float s)
{
	complex float m[2] = { 0., 1. };
	complex float v[2][2] = {
		{ 0.5 * s, 0.1 },
		{ 0.1, 1. * s },
	};

	complex float x1[2] = { 0.3, 0.5 };
	float val1 = gaussian_pdf(2, m, v, x1);

	complex float x2[2] = { -0.3, 1.5 };
	float val2 = gaussian_pdf(2, m, v, x2);
	
	if (val1 != val2)	// symmetry
		return false;

	// normalization

	float sum = 0.;

	for (int i = 0; i < 20; i++) {
	for (int j = 0; j < 20; j++) {
	for (int k = 0; k < 20; k++) {
	for (int l = 0; l < 20; l++) {

		complex float x[2] = {
			(i / 2. - 5.) + 1.i * (j / 2. - 5.),
			(k / 2. - 5.) + 1.i * (l / 2. - 5.)
		};

		complex float m1[2] = { 0., 0. };
		complex float v1[2][2] = { { 0.5, 0. }, { 0., 0.5 } };
		sum += gaussian_pdf(2, m1, v1, x);
	}}}}

	// the 16.f is for the oversampling x2
	if (fabsf(1.f - sum / 16.f) > 1.E-3)
		return false;

	return true;
}



static bool test_gaussian_pdf(void)
{
	if (!test_gaussian_pdf2(1.))
		return false;

	if (!test_gaussian_pdf2(2.))
		return false;

	return true;
}

UT_REGISTER_TEST(test_gaussian_pdf);



static bool test_gaussian_mix_pdf(void)
{
	complex float m[2][2] = { { 0., 1. }, { 0., 1. } };
	complex float v[2][2][2] = {
		{ { 0.5, 0.1 }, { 0.1, 1. }, },
		{ { 0.5, 0.1 }, { 0.1, 1. }, },
	};

	complex float x1[2] = { 0.3, 0.5 };
	float val1 = gaussian_pdf(2, m[0], v[0], x1);

	float coeff[2] = { 0.2, 0.8 };
	float val2 = gaussian_mix_pdf(2, 2, coeff, m, v, x1);

	return fabsf(val1 - val2) < UT_TOL;
}

UT_REGISTER_TEST(test_gaussian_mix_pdf);


static bool test_gaussian_score0(float s)
{
	complex float m[1] = { 0. };
	complex float v[1][1] = {
		{ powf(s, -1.) },
	};

	complex float x[1] = { 0.3 };
	complex float sc[1];

	gaussian_score(1, m, v, x, sc);

	// \nabla_x log (exp(-x^2 / s)) = - \nabla_x x^2 / s = - 2 x / s

	if (cabsf(sc[0] - (-2.f * x[0] / s)) > 1.E-6)
		return false;

	return true;
}

static bool test_gaussian_score2(float s)
{
	complex float m[2] = { 0.3, 1.2 };
	complex float v[2][2] = {
		{ powf(s, -1.), 0. },
		{ 0., powf(s, -1.) },
	};

	complex float x[2] = { 0.5 - 0.1i, 0.5 + 0.1i };
	complex float sc[2];

	gaussian_score(2, m, v, x, sc);

	if (cabsf(x[0] - m[0] + 0.5f * s * sc[0]) > 1.E-7)
		return false;

	if (cabsf(x[1] - m[1] + 0.5f * s * sc[1]) > 1.E-7)
		return false;

	return true;
}

static bool test_gaussian_score()
{
	for (float s = 0.1; s < 1.3; s += 0.1)
		if (!test_gaussian_score0(s))
			return false;

	for (float s = 0.1; s < 1.3; s += 0.1)
		if (!test_gaussian_score2(s))
			return false;

	return true;
}

UT_REGISTER_TEST(test_gaussian_score);




static bool test_gaussian_mult(void)
{
	complex float m1[1] = { 0. };
	complex float m2[1] = { 1. };
	complex float v1[1][1] = { { 1. } };
	complex float v2[1][1] = { { 1. } };

	complex float m[1];
	complex float v[1][1];

	gaussian_multiply(1, m, v, m1, v1, m2, v2);

	if ((0.5 != m[0]) || (2. != v[0][0]))
		return false;

	complex float xa[1] = { 0.3 };
	float vala1 = gaussian_pdf(1, m1, v1, xa);
	float vala2 = gaussian_pdf(1, m2, v2, xa);
	float vala  = gaussian_pdf(1, m, v, xa);

	complex float xb[1] = { 0.1 + 0.2i };
	float valb1 = gaussian_pdf(1, m1, v1, xb);
	float valb2 = gaussian_pdf(1, m2, v2, xb);
	float valb  = gaussian_pdf(1, m, v, xb);

	float z = gaussian_multiply_factor(1, m1, v1, m2, v2);

	if (1.E-8 < fabsf(vala1 * vala2 - z * vala))
		return false;

	if (1.E-8 < fabsf(valb1 * valb2 - z * valb))
		return false;

	return true;
}

UT_REGISTER_TEST(test_gaussian_mult);


static bool test_gaussian_conv(void)
{
	complex float m1[1] = { 0.5 };
	complex float m2[1] = { 1. };
	complex float v1[1][1] = { { 0.25 } };
	complex float v2[1][1] = { { 0.5 } };

	complex float m[1];
	complex float v[1][1];

	gaussian_convolve(1, m, v, m1, v1, m2, v2);

	if ((1.5 != m[0]) || (1.E-7 < cabsf(1. - 6. * v[0][0])))
		return false;

	return true;
}

UT_REGISTER_TEST(test_gaussian_conv);


