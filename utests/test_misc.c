/* Copyright 2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include "misc/misc.h"

#include "utest.h"


static int cmp(const void* _data, int a, int b)
{
	(void)_data;
	return a - b;
}

static bool test_quicksort(void)
{
	int x[5] = { 0, 3, 2, 1, 4 };

	quicksort(5, x, NULL, cmp);

	for (int i = 0; i < 5; i++)
		if (i != x[i])
			return false;

	return true;
}

UT_REGISTER_TEST(test_quicksort);


static int cmp2(const void* _data, int a, int b)
{
	const int* data = _data;
	return data[a] - data[b];
}

static bool test_quicksort2(void)
{
	int x[6] = { 0, 1, 2, 3, 4, 5 };
	int d[6] = { 0, 4, 2, 2, 4, 2 };
	int g[6] = { 0, 2, 2, 2, 4, 4 };

	quicksort(6, x, d, cmp2);

	for (int i = 0; i < 6; i++)
		if (g[i] != d[x[i]])
			return false;

	return true;
}


UT_REGISTER_TEST(test_quicksort2);



