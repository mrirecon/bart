/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2025 Daniel Mackner
 */

#include "misc/debug.h"
#include "misc/misc.h"

#include "seq/config.h"
#include "seq/event.h"

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


static bool test_rf_shape1(void)
{
	const struct seq_config seq = seq_config_defaults;
	struct rf_shape pulse[1];
	seq_sample_rf_shapes(1, pulse, &seq);

	struct pulseq ps;
	pulseq_init(&ps);
	
	pulse_shapes_to_pulseq(&ps, 1, pulse);

	if (3 != ps.shapes->len)
		return false;
	
	if (seq.phys.rf_duration != ps.shapes->data[0].values->len)
		return false;

	for (int i = 0; i < ps.shapes->data[0].values->len; i++) {

		if (1. < fabs(ps.shapes->data[0].values->data[i])) // assume uncompressed magnitude
			return false;
	}

	const double good[12] = {0.5, 0.0, 0.0, 144., -0.5, 0.0, 0.0, 324., 0.5, 0.0, 0.0, 143.};
	
	if ((int)ARRAY_SIZE(good) != ps.shapes->data[1].values->len)
		return false;

	for (int i = 0; i < ps.shapes->data[1].values->len; i++) {

		if (UT_TOL < fabs(good[i] - ps.shapes->data[1].values->data[i]))
			return false;
	}

	pulseq_free(&ps);

	return true;
}

UT_REGISTER_TEST(test_rf_shape1);


static bool test_rf_shape2(void)
{
	struct seq_config seq = seq_config_defaults;
	seq.magn.mag_prep = PREP_IR_NON;

	struct rf_shape pulse[2];
	seq_sample_rf_shapes(2, pulse, &seq);

	struct pulseq ps;
	pulseq_init(&ps);
	
	pulse_shapes_to_pulseq(&ps, 2, pulse);

	if (6 != ps.shapes->len)
		return false;
	
	if (5000 != ps.shapes->data[3].values->len)
		return false;

	if (5000 != ps.shapes->data[4].values->len)
		return false;

	for (int i = 0; i < ps.shapes->data[3].values->len; i++) {

		if (1. < fabs(ps.shapes->data[3].values->data[i])) // assume uncompressed magnitude/pha for hypsec
			return false;
		if (1. < fabs(ps.shapes->data[4].values->data[i])) // assume uncompressed magnitude/pha for hypsec
			return false;
	}

	const double good[4] = { 0.0, 2.0, 2., 4997. };
	
	if ((int)ARRAY_SIZE(good) != ps.shapes->data[5].values->len)
		return false;

	for (int i = 0; i < ps.shapes->data[5].values->len; i++)
		if (UT_TOL < fabs(good[i] - ps.shapes->data[5].values->data[i]))
			return false;

	pulseq_free(&ps);

	return true;
}

UT_REGISTER_TEST(test_rf_shape2);
