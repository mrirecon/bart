/* Copyright 2023. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>
#include <complex.h>

#include "num/gaussians.h"

#include "utest.h"


static bool test_gaussian_pdf(void)
{
	complex float m[2] = { 0., 1. };
	complex float v[2][2] = { 
		{ 0.5, 0.1 },
		{ 0.1, 1. },
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

			complex float x[1] = { 
				(i / 2. - 5.) + 1.i * (j / 2. - 5.)
			};

			sum += gaussian_pdf(1, m, v, x);
		}
	}

	if (fabsf(1. - sum / 4.) > 1.E-3)
		return false;

	return true;	
}

UT_REGISTER_TEST(test_gaussian_pdf);


static bool test_gaussian_mult(void)
{
	complex float m1[1] = { 0. };
	complex float m2[1] = { 1. };
	complex float v1[1][1] = { { 1. } };
	complex float v2[1][1] = { { 1. } };

	complex float m[1];
	complex float v[1][1];

	gaussian_multiply(1, m, v, m1, v1, m2, v2);

	if ((0.5 != m[0]) || (sqrtf(2.) != v[0][0]))
		return false;

	complex float xa[1] = { 0.3 };
	float vala1 = gaussian_pdf(1, m1, v1, xa);
	float vala2 = gaussian_pdf(1, m2, v2, xa);
	float vala  = gaussian_pdf(1, m, v, xa);

	complex float xb[1] = { 0.1 + 0.2i };
	float valb1 = gaussian_pdf(1, m1, v1, xb);
	float valb2 = gaussian_pdf(1, m2, v2, xb);
	float valb  = gaussian_pdf(1, m, v, xb);

	if (1.E-8 < fabsf(vala1 * vala2 * valb - vala * valb1 * valb2))
		return false;

	return true;
}

UT_REGISTER_TEST(test_gaussian_mult);



