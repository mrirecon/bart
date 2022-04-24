/* Copyright 2022. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 	Martin Uecker
 */


#include <math.h>
#include <stdio.h>

#include "num/qform.h"

#include "utest.h"


static bool test_quadratic_fit(void)
{
	float qf[3] = { 3., -1., 0.4  };

	float angles[10];
	float val[10];

	for (int i = 0; i < 10; i++) {

		angles[i] = 2. * M_PI * i / 10.;
		val[i] = quadratic_form(qf, angles[i]);
	}

	float qf2[3];
	fit_quadratic_form(qf2, 10, angles, val);

	float d = powf(qf2[0] - qf[0], 2.)
		+ powf(qf2[1] - qf[1], 2.)
		+ powf(qf2[2] - qf[2], 2.);

	return (d < 1.E-12);
}

UT_REGISTER_TEST(test_quadratic_fit);


static bool test_harmonic_fit(void)
{
	float qf[3] = { 3., -1., 0.4  };

	float angles[10];
	float val[10];

	for (int i = 0; i < 10; i++) {

		angles[i] = 2. * M_PI * i / 10.;
		val[i] = harmonic(qf, angles[i]);
	}

	float qf2[3];
	fit_harmonic(qf2, 10, angles, val);

	float d = powf(qf2[0] - qf[0], 2.)
		+ powf(qf2[1] - qf[1], 2.)
		+ powf(qf2[2] - qf[2], 2.);

	return (d < 1.E-12);
}

UT_REGISTER_TEST(test_harmonic_fit);


