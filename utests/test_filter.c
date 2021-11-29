/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <math.h>

#include "num/filter.h"

#include "utest.h"



static bool test_weiszfeld(void)
{
	const float vec[][2] = {
		{ 1.5, 0.5 }, { -0.5, 0.5 }, { 0.5, 1.5 }, { 0.5, -0.5 },
	};

	const float g[2] = { 0.5, 0.5 };
	float m[2];

	weiszfeld(10, 4, 2, m, vec);

	return     (fabsf(m[0] - g[0]) < 1.E-3)
		&& (fabsf(m[1] - g[1]) < 1.E-3);
}


UT_REGISTER_TEST(test_weiszfeld);



static bool test_geometric_median(void)
{
	const complex float vec[] = {
		1.5 + 0.5i, -0.5 + 0.5i, 0.5 + 1.5i, 0.5 + -0.5i,
	};

	const complex float g = 0.5 + 0.5i;
	complex float m = median_complex_float(4, vec);

	return cabsf(m - g);
}


UT_REGISTER_TEST(test_geometric_median);

