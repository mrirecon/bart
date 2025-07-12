/* Copyright 2025. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>

#include "num/lapack.h"

#include "utest.h"



static bool test_solve(void)
{
	float m[2][2] = {
		{ 0., 1. },
		{ -1., 0. },
	};

	float x[2] = { 0.5, 0.3 };
	lapack_solve_real(2, m, x);

	float g[2] = { 0.3, -0.5 };

	float err = 0;

	for (int i = 0; i < 2; i++)
		err += powf(x[i] - g[i], 2.);

	return (err < 1.E-10);
}

UT_REGISTER_TEST(test_solve);


