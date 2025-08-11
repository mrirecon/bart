/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
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
	int i = 0;

	if (PREP_IR_NON == seq->magn.mag_prep) {

		double projSLICE[3] = { 0. , 0. , 1. };

		i += prep_rf_inv(ev + i, 800, seq);

		struct grad_trapezoid spoil = {

			.ampl = 8,
			.rampup = 800,
			.flat = 8200,
			.rampdown = 600,
		};

		i += seq_grad_to_event(ev + i, ev[i - 1].end, &spoil, projSLICE);
		
		i += wait_time_to_event(ev + i, ev[i - 1].end, seq->magn.ti);
	}

	return i;
}