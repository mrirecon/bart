/* Copyright 2022. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 *	Nick Scholand
 */

#include <complex.h>
#include <math.h>

#include "misc/debug.h"

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/specfun.h"

#include "utest.h"


static bool test_sine_integral(void)
{
	double tests[][2] = {	{ 0.,	0. },
				{ 1.,	0.946083 },
				{ M_PI,	1.851937 },	/* Wilbraham-Gibbs constant*/
				{ 10,	1.658348 },
				{ 100,	1.562226 }
	};

	for (unsigned int i = 0; i < ARRAY_SIZE(tests); i++) {

		if ((Si(tests[i][0]) - tests[i][1]) > 10E-5)
			return 0;

		// Test for Si(-z) = -Si(z)
	}

	return 1;
}

UT_REGISTER_TEST(test_sine_integral);



static bool test_complex_log(void)
{

	enum { N = 1 };
	long dims[N] = { 1 };

	complex float in[N] = { 2.5 };

	complex float out[N] = { 0. };

	float ref = 0.916291;	// Wolfram alpha -> log(2.5)

	md_zlog(N, dims, out, in);

	// debug_printf(DP_INFO, "Ref: %f\t, Estimate: %f\n", ref, cabsf(out[0]));

	if ( cabsf(out[0] - ref) > 10E-6)
			return 0;

	return 1;
}

UT_REGISTER_TEST(test_complex_log);

static bool test_gamma_func(void)
{
	double tests[][2] = {	{ 0.5, sqrt(M_PI)},
				{ sqrt(3.), 0.91510229697308},
				{ 3.5, 3.32335097044784}
		};

	for (unsigned int i = 0; i < ARRAY_SIZE(tests); i++)
		if (fabs(gamma_func(tests[i][0]) - tests[i][1]) > 10E-12)
			return 0;

	return 1;
}

UT_REGISTER_TEST(test_gamma_func);


static bool test_hyp2f1(void)
{
	double tests[][5] = {
		{ 1., 	-2., 3., 0., 1.},
		{ 1.,	-2., 3., 1., 0.5},
		{ 1.,	-1., 3., 1., 0.666666666666667},
		{ 1.5,	2.5, 4.5, 1., 10.30835089459151},
		{ 0., 2., 2., -1.0, 1.},
		{ -1., 2., 3., -0.4, 1.26666666666667},
		{ 0.5, 0.2, 1.5, 1., 1.1496439092239847},
		{ 12.3, 8., 20.31, 1, 69280986.75273195},
		{ -1., 2., 3., 0.2, 0.866666666666667},
		{ 1., -2., 3., 0.2, 0.873333333333333},
		{ -1., 2., 3., 0.4, 0.733333333333333} };

	for (unsigned int i = 0; i < ARRAY_SIZE(tests); i++) {

		double val = hyp2f1(tests[i][0], tests[i][1], tests[i][2], tests[i][3]);
		// debug_printf(DP_INFO, "%1.15e,\t%1.15e\n", val, tests[i][4]);

		if (fabs(val - tests[i][4]) > 10E-5)
			return 0;
	}

	return 1;
}

UT_REGISTER_TEST(test_hyp2f1);



static bool test_legendre(void)
{
	double tests[][3] = {
		{ 0., 	-1., 1.},
		{ 0.,	0., 1.},
		{ 0.,	1., 1.},
		{ 1., 	-1., -1.},
		{ 1.,	0., 0.},
		{ 1.,	1., 1.},
		{ 2., 	-1., 1.},
		{ 2.,	0., -0.5},
		{ 2.,	1., 1.},
		{ 3., 	-1., -1.},
		{ 3.,	0., 0.},
		{ 3.,	1., 1.},
		{ 4., 	-1., 1.},
		{ 4.,	0., 0.375},
		{ 4.,	1., 1.} };

	for (unsigned int i = 0; i < ARRAY_SIZE(tests); i++) {

		double val = legendre(tests[i][0], tests[i][1]);
		// debug_printf(DP_INFO, "%1.15e,\t%1.15e\n", val, tests[i][2]);

		if (fabs(val - tests[i][2]) > 10E-12)
			return 0;
	}

	return 1;
}

UT_REGISTER_TEST(test_legendre);
