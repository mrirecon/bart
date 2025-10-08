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
		BLOCK_PRE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE,
		BLOCK_PRE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE,
		BLOCK_PRE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE,
		BLOCK_PRE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE,
	};

	struct seq_state seq_state = { 0 };;
	struct seq_config seq = seq_config_defaults;
	seq.enc.order = SEQ_ORDER_SEQ_MS;
	seq.magn.ti = 100;
	seq.magn.mag_prep = PREP_IR_NON;
	// seq.magn.inv_delay_time = 100;

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
