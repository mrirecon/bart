/* Copyright 2022. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 *	Nick Scholand
 */

#include <complex.h>

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

