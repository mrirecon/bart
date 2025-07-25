
#include <complex.h>
#include <math.h>
#include <stdint.h>

#include "misc/misc.h"
#include "misc/mri.h"

#include "seq/config.h"
#include "seq/flash.h"
#include "seq/seq.h"

#include "seq/adc_rf.h"

#include "utest.h"


static bool rf_spoiling_spoiled(void)
{
	const double ref[20] = { 

		50., 150., 300., 140., 30.,  
		330.,  320.,  0.,  90., 230.,
		60.,  300.,  230.,  150.,  120.,
		40.,  270.,  90.,  220.,  300.,
	};

	struct seq_config seq = seq_config_defaults;
	seq.phys.contrast = CONTRAST_RF_SPOILED;

	struct seq_state seq_state = { };;

	seq.enc.order = SEQ_ORDER_AVG_OUTER;
	set_loop_dims_and_sms(&seq, 1, 2, 2, 5, 1, 1, 1, 1, 0, 1);

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

	float shape_mag[MAX_RF_SAMPLES];
	float shape_pha[MAX_RF_SAMPLES];

	for (int i =0; i < rf_shape[0].samples; i++)
		idea_cfl_to_sample(&rf_shape[0], i, &shape_mag[i], &shape_pha[i]);

	if (rfs != 1)
		return false;

	if (rf_shape[0].samples != 2 * seq.phys.rf_duration)
		return 0;

	// expected in reference implementation
	const double good_norm = 330.154932;

	double s = idea_pulse_scaling(&rf_shape[0]);
	double n = idea_pulse_norm_sum(&rf_shape[0]);

	if (fabs(s - seq.phys.flip_angle) > 1E-6)
		return false;

	if (fabs(n - good_norm) > 1e-6)
		return false;

	// expected in reference implementation
	double good[2] = { 0.000445, 0.046478 };

	if (   ((shape_mag[620] - 1.0) > 1e-5)
	    || (fabs(good[0] - shape_mag[946]) > 1e-5)
	    || (fabs(good[0] - shape_mag[294]) > 1e-5)
	    || (fabs(good[1] - shape_mag[1035]) > 1e-5)
	    || (fabs(good[1] - shape_mag[205])  > 1e-5))
			return false;

	for (int i = 0; i < rf_shape[0].samples; i++) {

		if (((i < 294) || (i > 946)) && (fabs(shape_pha[i] - M_PI) > 1e-4))
			return false;

		if ((i > 294) && (i < 946) && (fabs(shape_pha[i]) > 1e-5))
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

	if ((rf_shape[0].samples != 2 * seq.phys.rf_duration)
	    || (rf_shape[1].samples != 2 * seq.phys.rf_duration)
	    || (rf_shape[2].samples != 2 * seq.phys.rf_duration))
		return 0;

	// expected in reference implementation
	const double good_norm = 110.051648;

	double s = idea_pulse_scaling(&rf_shape[0]);

	double n0 = idea_pulse_norm_sum(&rf_shape[0]);
	double n1 = idea_pulse_norm_sum(&rf_shape[1]);
	double n2 = idea_pulse_norm_sum(&rf_shape[2]);

	if (fabs(s - seq.phys.flip_angle) > 1e-6)
		return false;

	if ((fabs(n0 - good_norm) > 1e-6)
	    || (fabs(n1 - good_norm) > 1e-6)
	    || (fabs(n2 - good_norm) > 1e-6))
		return false;

	// values exported from reference implementation
	int idx[7] = { 620, 596, 644, 692, 548, 764, 476 };

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

		float shape_mag[MAX_RF_SAMPLES];
		float shape_pha[MAX_RF_SAMPLES];

		for (int i =0; i < rf_shape[0].samples; i++)
			idea_cfl_to_sample(&rf_shape[m], i, &shape_mag[i], &shape_pha[i]);

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

