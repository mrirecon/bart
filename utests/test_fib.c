/* Copyright 2023. Institute of Biomedical Imaging, TU Graz
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>
#include <stdbool.h>

#include "noncart/traj.h"

#include "utest.h"


static bool test_gen_fib(void)
{
	int ref1[10] = { 1, 1, 2, 3, 5, 8, 13, 21, 34, 55 };
	int ref7[10] = { 1, 7, 8, 15, 23, 38, 61, 99, 160, 259 };

	for (int i = 0; i < 10; i++) {

		if (0 != abs(ref1[i] - gen_fibonacci(1, i)))
			return false;

		if (0 != abs(ref7[i] - gen_fibonacci(7, i)))
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_gen_fib);
