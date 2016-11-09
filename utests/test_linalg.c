/* Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <math.h>

#include "num/linalg.h"

#include "utest.h"



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

