/* Copyright 2025-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <math.h>
#include <stdint.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mri.h"

#include "num/multind.h"

#include "seq/config.h"
#include "seq/flash.h"
#include "seq/helpers.h"
#include "seq/seq.h"

#include "seq/adc_rf.h"

#include "utest.h"


static bool rf_spoiling_spoiled(void)
{
	const double ref[20] = {

		50., 150., 300., 140., 30.,
		330.,  320.,  0.,  90., 230.,
		60.,  300.,  230.,  210.,  240.,
		320.,  90.,  270.,  140.,  60.
	};

	struct seq_config seq = seq_config_defaults;
	seq.phys.contrast = SEQ_CONTRAST_RF_SPOILED;

	struct seq_state seq_state = { };

	seq.loop_dims[SLICE_DIM] = 2;
	seq.loop_dims[PHS1_DIM] = 2;
	seq.loop_dims[TIME_DIM] = 10;

	seq_ui_interface_loop_dims(0, &seq, DIMS, seq.loop_dims);

	seq.loop_dims[COEFF_DIM] = 1;
	seq.loop_dims[COEFF2_DIM] = 1;

	int i = 0;

	do {
		if (UT_TOL < fabs(ref[i++] - rf_spoiling(DIMS, seq_state.pos, &seq)))
			return false;

	} while (seq_continue(&seq_state, &seq));

	return true;
}

UT_REGISTER_TEST(rf_spoiling_spoiled);



static bool test_sinc(void)
{
	struct seq_config seq = seq_config_defaults;

	struct rf_shape rf_shape[10];
	int rfs = seq_sample_rf_shapes(10, rf_shape, &seq);

	float shape_mag[SEQ_MAX_RF_SAMPLES];
	float shape_pha[SEQ_MAX_RF_SAMPLES];

	for (int i =0; i < rf_shape[0].samples; i++)
		seq_cfl_to_sample(&rf_shape[0], i, &shape_mag[i], &shape_pha[i]);

	if (rfs != 1)
		return false;

	if (rf_shape[0].samples != 1E6 * seq.phys.rf_duration)
		return false;

	// expected in reference implementation
	const double good_norm = 330.154932 / 2.;

	double s = seq_pulse_scaling(&rf_shape[0]);
	double n = seq_pulse_norm_sum(&rf_shape[0]);

	if (fabs(s - seq.phys.flip_angle) > 1E-6)
		return false;

	if (fabs(n - good_norm) > 1e-6)
		return false;

	// expected in reference implementation
	double good[2] = { 0.000445, 0.046339 };

	if (   ((shape_mag[310] - 1.0) > 1e-5)
	    || (fabs(good[0] - shape_mag[473]) > 1e-5)
	    || (fabs(good[0] - shape_mag[147]) > 1e-5)
	    || (fabs(good[1] - shape_mag[518]) > 1e-5)
	    || (fabs(good[1] - shape_mag[102]) > 1e-5))
			return false;

	for (int i = 0; i < rf_shape[0].samples; i++) {

		if (((i < 294/2) || (i > 946/2)) && (fabs(shape_pha[i] - M_PI) > 1e-4))
			return false;

		if ((i > 294/2) && (i < 946/2) && (fabs(shape_pha[i]) > 1e-5))
			return  false;
	}

	return true;
}

UT_REGISTER_TEST(test_sinc);

static bool test_sms(void)
{
	struct seq_config seq = seq_config_defaults;
	seq.loop_dims[SLICE_DIM] = 3;
	seq.geom.mb_factor = 3;

	struct rf_shape rf_shape[10];

	int rfs = seq_sample_rf_shapes(10, rf_shape, &seq);

	if (rfs != 3)
		return false;

	if ((rf_shape[0].samples != 1E6 * seq.phys.rf_duration)
	    || (rf_shape[1].samples != 1E6 * seq.phys.rf_duration)
	    || (rf_shape[2].samples != 1E6 * seq.phys.rf_duration))
		return false;

	// expected in reference implementation
	const double good_norm = 110.051648 / 2.;

	double s = seq_pulse_scaling(&rf_shape[0]);

	double n0 = seq_pulse_norm_sum(&rf_shape[0]);
	double n1 = seq_pulse_norm_sum(&rf_shape[1]);
	double n2 = seq_pulse_norm_sum(&rf_shape[2]);

	if (fabs(s - seq.phys.flip_angle) > 1e-6)
		return false;

	if ((fabs(n0 - good_norm) > 1e-6)
	    || (fabs(n1 - good_norm) > 1e-6)
	    || (fabs(n2 - good_norm) > 1e-6))
		return false;

	// values exported from reference implementation
	int idx[7] = { 310, 298, 322, 346, 274, 382, 238 };

	float mag[3][7] = {
		{ 1., 0.349166, 0.349166, 0.243029, 0.243031, 0.199427, 0.199427 },
		{ 0., 0.250700, 0.889000, 0.188331, 0.836786, 0.473933, 0.344311 },
		{ 0., 0.889000, 0.250700, 0.836787, 0.188333, 0.344310, 0.473932 },
	};

	float pha[3][7] = {
		{ 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 3.141593, 3.141593 },
		{ 5.176036, 5.235988, 2.094395, 5.235988, 2.094395, 2.094395, 2.094395 },
		{ 4.390638, 4.188790, 1.047197, 4.188790, 1.047198, 4.188790, 4.188790 }
	};

	for (int m = 0; m < seq.geom.mb_factor; m++) {

		float shape_mag[SEQ_MAX_RF_SAMPLES];
		float shape_pha[SEQ_MAX_RF_SAMPLES];

		for (int i =0; i < rf_shape[0].samples; i++)
			seq_cfl_to_sample(&rf_shape[m], i, &shape_mag[i], &shape_pha[i]);

		for (int i = 0; i < 7; i++) {

			if (fabsf(shape_mag[idx[i]] - mag[m][i]) > 1e-5)
				return false;

			if (fabsf(shape_pha[idx[i]] - pha[m][i]) > 1e-5)
				return false;
		}
	}

	return true;
}

UT_REGISTER_TEST(test_sms);

