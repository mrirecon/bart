#include <math.h>

#include "seq/config.h"
#include "seq/event.h"
#include "seq/kernel.h"
#include "seq/flash.h"
#include "seq/seq.h"
#include "seq/helpers.h"

#include "utest.h"

#define FLASH_EVENTS 14



static bool test_block_minv_init_delay(void)
{
	const enum block blocks[16] = { BLOCK_PRE, BLOCK_KERNEL_NOISE, BLOCK_PRE,
		BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE,
		BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE,
		BLOCK_PRE,
		BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE,
		BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE
	};

	struct seq_state seq_state = { 0 };
	struct seq_config seq = seq_config_defaults;
	seq.enc.order = SEQ_ORDER_AVG_OUTER;
	seq.magn.init_delay_sec = 1;
	seq.magn.ti = 100;
	seq.magn.mag_prep = PREP_IR_NON;

	int i = 0;

	const int max_E = 200;
	struct seq_event ev[max_E];

	seq.loop_dims[BATCH_DIM] = 2;
	set_loop_dims_and_sms(&seq, 1, 2, 3, 3, 1, 1, 1);
	int pre_blocks = 0;

	do {

		int E = seq_block(max_E, ev, &seq_state, &seq);

		if (0 > E)
			return false;

		if (0 == E)
			continue;

		if (blocks[i] != seq_state.mode)
			return false;

		if ((BLOCK_KERNEL_IMAGE == seq_state.mode) && (FLASH_EVENTS != E))
			return false;

		if (BLOCK_PRE == seq_state.mode)
			pre_blocks++;

		// correct delay_meas_time
		if ((BLOCK_PRE == seq_state.mode) && (1 == E) && (1.E6 * seq.magn.init_delay_sec != seq_block_end(E, ev, seq_state.mode, seq.phys.tr)))
			return false;
		else if ((BLOCK_PRE == seq_state.mode) && (1 < E) && (1. * seq.magn.ti != (ev[E - 1].end - ev[E - 1].start)))
			return false;

		i++;

	} while (seq_continue(&seq_state, &seq));

	if (3 != pre_blocks)
		return false;

	return true;
}

UT_REGISTER_TEST(test_block_minv_init_delay);


static bool test_block_minv_multislice(void)
{
	const enum block blocks[21] = {
		BLOCK_KERNEL_NOISE,
		BLOCK_PRE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_POST,
		BLOCK_PRE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_POST,
		BLOCK_PRE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_POST,
		BLOCK_PRE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_POST,
	};

	struct seq_state seq_state = { 0 };
	struct seq_config seq = seq_config_defaults;
	seq.enc.order = SEQ_ORDER_SEQ_MS;
	seq.magn.ti = 100;
	seq.magn.mag_prep = PREP_IR_NON;
	seq.magn.inv_delay_time_sec = 100.;

	int i = 0;
	int inversions = 0;

	const int max_E = 200;
	struct seq_event ev[max_E];

	seq.loop_dims[BATCH_DIM] = 2;
	set_loop_dims_and_sms(&seq, 1, 2, 3, 3, 1, 1, 1);

	do {

		int E = seq_block(max_E, ev, &seq_state, &seq);

		if (0 > E)
			return false;

		if (0 == E)
			continue;

		if (blocks[i] != seq_state.mode)
			return false;

		if ((BLOCK_KERNEL_IMAGE == seq_state.mode) && (FLASH_EVENTS != E))
			return false;

		if (BLOCK_PRE == seq_state.mode)
			inversions++;

		// correct ti in mag_prep block
		if ((BLOCK_PRE == seq_state.mode) && (2 == E) && (1. * seq.magn.ti != seq_block_end(E, ev, seq_state.mode, seq.phys.tr)))
			return false;

		i++;

	} while (seq_continue(&seq_state, &seq));

	if (4 != inversions)
		return false;

	return true;
}

UT_REGISTER_TEST(test_block_minv_multislice);


static bool test_fov_shift(void)
{
	const int slices = 3;

	float in[3] = {-27, 0, 27};
	float good[3] = {0, 0, 0};


	float gui_shift[slices][4];
	for (int i = 0; i < slices; i++) {

		gui_shift[i][0] = 0;
		gui_shift[i][1] = 0;
		gui_shift[i][2] = in[i]; // slice shift
	}

	struct seq_config seq = seq_config_defaults;

	seq.geom.mb_factor = 3;
	set_loop_dims_and_sms(&seq, 1, slices, 1, 1, 1, 1, 1);
	set_fov_pos(slices, 4, &gui_shift[0][0], &seq);
	
	if (0 < fabs(seq.geom.sms_distance - 27.))
		return false;

	for (int i = 0; i < slices; i++)
		if (0 < fabs(seq.geom.shift[i][2] - good[i]))
			return false;

	return true;
}

UT_REGISTER_TEST(test_fov_shift);

static bool test_fov_shift3x3(void)
{
	const int slices = 9;
	float in[9] = {-36, -27, -18, -9, 0, 9, 18, 27, 36};
	float good[9] = {-9, 0, 9, -9, 0, 9, -9, 0, 9};

	float gui_shift[slices][4];
	for (int i = 0; i < slices; i++) {

		gui_shift[i][0] = 0;
		gui_shift[i][1] = 0;
		gui_shift[i][2] = in[i]; // slice shift
	}

	struct seq_config seq = seq_config_defaults;

	seq.geom.mb_factor = 3;
	set_loop_dims_and_sms(&seq, 1, slices, 1, 1, 1, 1, 1);
	set_fov_pos(slices, 4, &gui_shift[0][0], &seq);
	
	if (0 < fabs(seq.geom.sms_distance - 27.))
		return false;
	for (int i = 0; i < slices; i++)
		if (0 < fabs(seq.geom.shift[i][2] - good[i]))
			return false;

	return true;
}

UT_REGISTER_TEST(test_fov_shift3x3);
