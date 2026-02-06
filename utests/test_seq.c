/* Copyright 2025-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>

#include "num/multind.h"

#include "seq/config.h"
#include "seq/event.h"
#include "seq/kernel.h"
#include "seq/flash.h"
#include "seq/seq.h"
#include "seq/helpers.h"

#include "utest.h"

#define FLASH_EVENTS 14



static int trigger_event_count(const struct seq_config* seq, const struct seq_state* seq_state)
{
	return (seq->trigger.trigger_out && (0 == seq_state->pos[PHS1_DIM])) ? 1 : 0;
}

static bool test_block_minv_init_delay(void)
{
	const enum seq_block blocks[16] = {
		SEQ_BLOCK_PRE, SEQ_BLOCK_KERNEL_NOISE, SEQ_BLOCK_PRE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_PRE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE
	};

	struct bart_seq* seq = bart_seq_alloc("");
	bart_seq_defaults(seq);

	seq->conf->magn.init_delay = 1.;
	seq->conf->magn.ti = 100.E-3;
	seq->conf->magn.mag_prep = SEQ_PREP_IR_NONSELECTIVE;

	seq->conf->loop_dims[BATCH_DIM] = 2;
	seq->conf->loop_dims[SLICE_DIM] = 2;
	seq->conf->loop_dims[PHS1_DIM] = 3;
	seq->conf->loop_dims[TIME_DIM] = 3;
	seq_ui_interface_loop_dims(0, seq->conf, DIMS, seq->conf->loop_dims);

	int i = 0;
	int pre_blocks = 0;

	do {
		int E = seq_block(seq->N, seq->event, seq->state, seq->conf);

		if (0 > E)
			return false;

		if (0 == E)
			continue;

		if (blocks[i] != seq->state->mode)
			return false;

		if ((SEQ_BLOCK_KERNEL_IMAGE == seq->state->mode) && (FLASH_EVENTS + trigger_event_count(seq->conf, seq->state) != E))
			return false;

		if (SEQ_BLOCK_PRE == seq->state->mode)
			pre_blocks++;

		// correct delay_meas_time
		if (   (SEQ_BLOCK_PRE == seq->state->mode)
		    && (1 == E)
		    && (seq->conf->magn.init_delay != seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad)))
			return false;

		if (   (SEQ_BLOCK_PRE == seq->state->mode)
		    && (1 < E)
		    && (1.E-4 * UT_TOL < fabs(seq->conf->magn.ti - (seq->event[E - 1].end - seq->event[E - 1].start))))
			return false;

		i++;

	} while (seq_continue(seq->state, seq->conf));

	if (3 != pre_blocks)
		return false;

	bart_seq_free(seq);

	return true;
}

UT_REGISTER_TEST(test_block_minv_init_delay);


static bool test_block_minv_multislice(void)
{
	const enum seq_block blocks[21] = {
		SEQ_BLOCK_KERNEL_NOISE,
		SEQ_BLOCK_PRE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_POST,
		SEQ_BLOCK_PRE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_POST,
		SEQ_BLOCK_PRE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_POST,
		SEQ_BLOCK_PRE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_POST,
	};

	struct bart_seq* seq = bart_seq_alloc("");
	bart_seq_defaults(seq);

	seq->conf->enc.order = SEQ_ORDER_SEQ_MS;
	seq->conf->magn.ti = 100.E-3;
	seq->conf->magn.mag_prep = SEQ_PREP_IR_NONSELECTIVE;
	seq->conf->magn.inv_delay_time = 100.;

	seq->conf->loop_dims[BATCH_DIM] = 2;
	seq->conf->loop_dims[SLICE_DIM] = 2;
	seq->conf->loop_dims[PHS1_DIM] = 3;
	seq->conf->loop_dims[TIME_DIM] = 3;
	seq_ui_interface_loop_dims(0, seq->conf, DIMS, seq->conf->loop_dims);

	int i = 0;
	int inversions = 0;

	do {
		int E = seq_block(seq->N, seq->event, seq->state, seq->conf);

		if (0 > E)
			return false;

		if (0 == E)
			continue;

		if (blocks[i] != seq->state->mode)
			return false;

		if ((SEQ_BLOCK_KERNEL_IMAGE == seq->state->mode) && (FLASH_EVENTS + trigger_event_count(seq->conf, seq->state) != E))
			return false;

		if (SEQ_BLOCK_PRE == seq->state->mode)
			inversions++;

		// correct ti in mag_prep block
		if (   (SEQ_BLOCK_PRE == seq->state->mode)
		    && (2 == E)
		    && (seq->conf->magn.ti != seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad)))
			return false;

		i++;

	} while (seq_continue(seq->state, seq->conf));

	if (4 != inversions)
		return false;

	bart_seq_free(seq);

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
		gui_shift[i][2] = 1.E-3 * in[i]; // slice shift
	}

	struct bart_seq* seq = bart_seq_alloc("");
	bart_seq_defaults(seq);

	seq->conf->geom.mb_factor = 3;
	seq->conf->loop_dims[SLICE_DIM] = slices;
	seq_ui_interface_loop_dims(0, seq->conf, DIMS, seq->conf->loop_dims);

	seq_set_fov_pos(slices, 4, &gui_shift[0][0], seq->conf);

	if (1.E-2 *  UT_TOL < fabs(seq->conf->geom.sms_distance - 27.E-3))
		return false;

	for (int i = 0; i < slices; i++)
		if (0 < fabs(seq->conf->geom.shift[i][2] - good[i]))
			return false;

	bart_seq_free(seq);

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
		gui_shift[i][2] = 1.E-3 * in[i]; // slice shift
	}

	struct bart_seq* seq = bart_seq_alloc("");
	bart_seq_defaults(seq);

	seq->conf->geom.mb_factor = 3;
	seq->conf->loop_dims[SLICE_DIM] = slices;
	seq_ui_interface_loop_dims(0, seq->conf, DIMS, seq->conf->loop_dims);

	seq_set_fov_pos(slices, 4, &gui_shift[0][0], seq->conf);

	if (1.E-2 *  UT_TOL < fabs(seq->conf->geom.sms_distance - 27.E-3))
		return false;

	for (int i = 0; i < slices; i++)
		if (1.E-2 *  UT_TOL < fabs(seq->conf->geom.shift[i][2] - 1.E-3 * good[i]))
			return false;

	bart_seq_free(seq);

	return true;
}

UT_REGISTER_TEST(test_fov_shift3x3);
