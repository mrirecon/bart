/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2025 Daniel Mackner
 */

#include "misc/debug.h"
#include "misc/misc.h"

#include "seq/pulseq.c"

#include "utest.h"

static bool test_shape(void)
{
	const double in[7] = { 0., 0.1, 0.2, 0.8, 0.3, 0.2, 1. };

	struct shape out = make_compressed_shape(0, (int)ARRAY_SIZE(in), in);

	for (int i = 0; i < (int)ARRAY_SIZE(in); i++)
		if (out.values->data[i] != in[i])
			return false;

	xfree(out.values);

	return true;
}

UT_REGISTER_TEST(test_shape);

static bool test_shape_compression1(void)
{
	const double in[10] = { 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };

	const double good[4] = { 0, 0.1, 0.1, 7 };

	struct shape out = make_compressed_shape(0, 10, in);

	for (int i = 0; i < (int)ARRAY_SIZE(good) ; i++)
		if (out.values->data[i] != good[i])
			return false;

	xfree(out.values);

	return true;
}

UT_REGISTER_TEST(test_shape_compression1);


static bool test_shape_compression2(void)
{
	const double in[10] = { 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5. };

	const double good[3] = { 0.5, 0.5, 8 };

	struct shape out = make_compressed_shape(0, 10, in);

	for (int i = 0; i < (int)ARRAY_SIZE(good); i++)
		if (out.values->data[i] != good[i])
			return false;

	xfree(out.values);

	return true;
}

UT_REGISTER_TEST(test_shape_compression2);

static bool test_shape_compression3(void)
{
	const double in[11] = { 0.5, 1., 1.5, 2., 2.5 , 15., 15., 15., 15., 15., 15. };

	const double good[7] = { 0.5, 0.5, 3, 12.5, 0., 0., 3. };

	struct shape out = make_compressed_shape(0, (int)ARRAY_SIZE(in), in);

	for (int i = 0; i < (int)ARRAY_SIZE(good); i++)
		if (out.values->data[i] != good[i])
			return false;

	xfree(out.values);

	return true;
}

UT_REGISTER_TEST(test_shape_compression3);
