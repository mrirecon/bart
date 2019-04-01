/* Copyright 2016-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <math.h>

#include "num/linalg.h"

#include "utest.h"



static bool test_mat_pinv(void)
{
	const complex float A[3][2] = {
		{ 1.i, 4. },
		{ 2.i, 5. },
		{ 3.i, 6. },
	};

	const complex float C[2][3] = {
		{ -0.00000 + 0.94444i, +0.00000 + 0.11111i, +0.00000 - 0.72222i },
		{ +0.44444 + 0.00000i, +0.11111 + 0.00000i, +-0.22222 - 0.00000i },
	};

	complex float B[2][3];

	mat_pinv(3, 2, B, A);

	float err = 0.;

	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 3; j++)
			err += powf(cabsf(C[i][j] - B[i][j]), 2.);

	return (err < 1.E-10);
}


UT_REGISTER_TEST(test_mat_pinv);


static bool test_thomas_algorithm(void)
{
	const complex float A[7][3] = {

		{ 0. , 1., 0.5 },
		{ 0.5, 1., 0.5 },
		{ 0.5, 1., 0.5 },
		{ 0.5, 1., 0.5 },
		{ 0.5, 1., 0.5 },
		{ 0.5, 1., 0.5 },
		{ 0.5, 1., 0.  },
	};

	const complex float d[7] = { 1., 0., 0., 0., 0., 0., 0. };
	complex float x[7];

	thomas_algorithm(7, x, A, d);

	bool ok = true;

	for (int i = 0; i < 7; i++)
		ok &= (cabsf(x[i] - 0.25f * (7 - i) * powf(-1., i)) < 1e-6);

	return ok;
}

UT_REGISTER_TEST(test_thomas_algorithm);

