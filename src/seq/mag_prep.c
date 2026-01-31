/* Copyright 2025-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "seq/config.h"
#include "seq/event.h"
#include "seq/adc_rf.h"
#include "seq/gradient.h"

#include "mag_prep.h"

int mag_prep(struct seq_event ev[6], const struct seq_config* seq)
{
	if (PREP_IR_NON != seq->magn.mag_prep)
		return 0;

	int i = 0;
	double proj_slice[3] = { 0., 0., 1. };

	i += prep_rf_inversion(ev + i, 800E-6, seq);

	struct grad_trapezoid spoil = {

		.ampl = .008,
		.rampup = 800E-6,
		.flat = 8200E-6,
		.rampdown = 600E-6,
	};

	i += seq_grad_to_event(ev + i, ev[i - 1].end, &spoil, proj_slice);
	i += wait_time_to_event(ev + i, ev[i - 1].end, seq->magn.ti);

	return i;
}

