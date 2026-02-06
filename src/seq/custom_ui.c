/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <string.h>

#include "config.h"
#include "misc/misc.h"

#include "seq/config.h"
#include "seq/ui_enums.h"
#include "ui_enums.h"

#include "custom_ui.h"


#ifndef ARRAY_SIZE
#define ARRAY_SIZE(x) (sizeof(x)/sizeof(x[0]))
#endif


static const struct selection_opt pemode_opts[] = {

	{ SEQ_PEMODE_TURN, "1. TURN", },
	{ SEQ_PEMODE_RAGA, "2. RAGA", },
	{ SEQ_PEMODE_MEMS_HYB, "3. MEMS", },
};


static const struct selection_opt contrast_opts[] = {

	{ SEQ_CONTRAST_RF_RANDOM, "1. RF Random" },
	{ SEQ_CONTRAST_RF_SPOILED, "2. RF Spoiled" },
};


static const struct seq_ui_selection custom_selection_defaults[] = {

	{ "seq_wip5", cil_pe_mode, "PE Mode", ARRAY_SIZE(pemode_opts), &pemode_opts[0], SEQ_PEMODE_TURN, "" },
	{ "seq_wip2", cil_contrast, "Contrast", ARRAY_SIZE(contrast_opts), &contrast_opts[0], SEQ_CONTRAST_RF_RANDOM, "" },
};

static const struct seq_ui_long custom_bool_defaults[] = {

	{ "seq_wip12", cil_sms, "Simultaneous Multi-Slice", { 0, 0, 0, 1 }, "", "" },
};

static const struct seq_ui_long custom_long_defaults[] = {

	{ "seq_wip13", cil_sms_distance, "SMS Distance.", { 0, 100, 1, 0 }, "Slice distance within SMS group. [adjusted by slice thickness/gap; < 0: SMS off]", "mm" },
};

static const struct seq_ui_long custom_longarr_defaults[] = {

	{ "", cil_tiny, "Turns / Tiny Golden", { 1, 20, 1, 1 }, "Number of turns (repititions) of radial spoke pattern.", ""},
	{ "", cil_rf_duration, "RF pulse duration", { 20, 2560, 20, 400 }, "RF pulse duration.", "us"},
	{ "", cil_init_delay, "Delay Measurements", { 0, 300, 1, 0 }, "Delay measurements.", "s"},
	{ "", cil_inversions, "Inversions", { 0, 1000, 1, 1 }, "Number of IR experiments.", ""},
	{ "", cil_inv_delay, "Inversion Delay", { 0, 2000, 1, 0 }, "Delay between inversions.", "s"},
	{ "", cil_mb_factor, "Multiband factor (SMS)", { 1, 5, 1, 1 }, "SMS Multiband factor", ""},
	{ "", cil_RAGA_aligned_flags, "RAGA aligned flags", { 0, 65535, 1, 0 }, "Bitmask from dimension to align in RAGA sampling", ""}
};


static const struct seq_ui_double custom_double_defaults[] = {

	{ "seq_wip9", cid_cmd, "BART cmd", { -1000., 1000., 0.1, 0. }, "BART UI interface. Get/set config from file.", "" },
};

static const struct seq_ui_double custom_doublearr_defaults[] = {

	{ "", cid_BWTP, "BWTP", { 0., 200., 0.1, 1.6 }, "RF bandwidth-time-product.", "" },

};



struct custom_ui* seq_custom_ui_init(void)
{
	struct custom_ui* ui = xmalloc(sizeof (struct custom_ui));

	ui->sizes[SELECTION] = ARRAY_SIZE(custom_selection_defaults);
	ui->sizes[BOOL] = ARRAY_SIZE(custom_bool_defaults);
	ui->sizes[LONG] = ARRAY_SIZE(custom_long_defaults);
	ui->sizes[longarr] = ARRAY_SIZE(custom_longarr_defaults);
	ui->sizes[DOUBLE] = ARRAY_SIZE(custom_double_defaults);
	ui->sizes[doublearr] = ARRAY_SIZE(custom_doublearr_defaults);

	ui->selections = xmalloc((size_t)ui->sizes[SELECTION] * (sizeof (struct seq_ui_selection)));
	ui->checkboxes = xmalloc((size_t)ui->sizes[BOOL] * (sizeof (struct seq_ui_long)));
	ui->longs = xmalloc((size_t)ui->sizes[LONG] * (sizeof (struct seq_ui_long)));
	ui->longarr = xmalloc((size_t)ui->sizes[longarr] * (sizeof (struct seq_ui_long)));
	ui->doubles = xmalloc((size_t)ui->sizes[DOUBLE] * (sizeof (struct seq_ui_double)));
	ui->doublearr = xmalloc((size_t)ui->sizes[doublearr] * (sizeof (struct seq_ui_double)));

	memcpy(ui->selections, &custom_selection_defaults, (size_t)ui->sizes[SELECTION] * (sizeof (struct seq_ui_selection)));
	memcpy(ui->checkboxes, &custom_bool_defaults, (size_t)ui->sizes[BOOL] * (sizeof (struct seq_ui_long)));
	memcpy(ui->longs, &custom_long_defaults, (size_t)ui->sizes[LONG] * (sizeof (struct seq_ui_long)));
	memcpy(ui->longarr, &custom_longarr_defaults, (size_t)ui->sizes[longarr] * (sizeof (struct seq_ui_long)));
	memcpy(ui->doubles, &custom_double_defaults, (size_t)ui->sizes[DOUBLE] * (sizeof (struct seq_ui_double)));
	memcpy(ui->doublearr, &custom_doublearr_defaults, (size_t)ui->sizes[doublearr] * (sizeof (struct seq_ui_double)));

	return ui;
}

void seq_custom_ui_free(struct custom_ui* ui)
{
	xfree(ui->selections);
	xfree(ui->checkboxes);
	xfree(ui->longs);
	xfree(ui->longarr);
	xfree(ui->doubles);
	xfree(ui->doublearr);
	xfree(ui);
}
