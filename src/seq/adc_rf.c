/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <assert.h>
#include <complex.h>
#include <math.h>

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/list.h"

#include "num/multind.h"
#include "num/rand.h"

#include "noncart/traj.h"

#include "seq/event.h"
#include "seq/config.h"
#include "seq/anglecalc.h"
#include "seq/pulse.h"
#include "seq/misc.h"

#include "adc_rf.h"


#define RFSPOIL_INCREMENT_DEG	50.0


double phase_clamp(double phase)
{
	double ret = fmod(phase, 360.);
	return ret + ((ret > 180.) ? -360. : ((ret < -180.) ? 360. : 0));
}


double rf_spoiling(int D, const long pos[D], const struct seq_config* seq)
{
	double idx;

	switch (seq->phys.contrast) {

	case SEQ_CONTRAST_NO_SPOILING:

		return 0.;

	case SEQ_CONTRAST_RF_SPOILED:

		idx = 1. + md_ravel_index_permuted(DIMS, pos, SEQ_FLAGS & ~(COEFF_FLAG|COEFF2_FLAG), seq->loop_dims, seq->order);

		return fmod(0.5 * RFSPOIL_INCREMENT_DEG * (idx + pow(idx, 2.)), 360.);

	case SEQ_CONTRAST_RF_RANDOM:

		return uniform_rand() * 360. - 180.;

	default:
		assert(0);
	}
}


int prep_rf_excitation(struct seq_event* rf_ev, double start, double rf_spoil_phase,
		const struct seq_state* seq_state, const struct seq_config* seq)
{
	if (SEQ_BLOCK_KERNEL_NOISE == seq_state->mode)
		return 0;

	rf_ev->type = SEQ_EVENT_PULSE;	

	rf_ev->start = start;
	rf_ev->end = rf_ev->start + seq->phys.rf_duration;

	const double asym_pulse = 0.5;

	rf_ev->mid = rf_ev->start + (rf_ev->end - rf_ev->start) * asym_pulse;

	rf_ev->pulse.shape_id = 0;

	if (1 < seq->geom.mb_factor)
		rf_ev->pulse.shape_id = seq_state->pos[SLICE_DIM];

	if (seq->geom.mb_factor <= rf_ev->pulse.shape_id) // SLICE_DIM is only used for SMS
		return 0;	

	rf_ev->pulse.type = SEQ_RF_EXCITATION;
	rf_ev->pulse.fa = seq->phys.flip_angle;
	rf_ev->pulse.freq = seq->sys.gamma * seq->geom.shift[seq_state->chrono_slice][2] * slice_amplitude(seq);
	rf_ev->pulse.phase = phase_clamp(rf_spoil_phase);

	return 1;
}


int prep_rf_inversion(struct seq_event* rf_ev, double start, const struct seq_config* seq)
{
	if (seq->magn.mag_prep != SEQ_PREP_IR_NONSELECTIVE)
		return 0;

	rf_ev->type = SEQ_EVENT_PULSE;	

	rf_ev->start = start;

	struct pulse_hypsec hs = pulse_hypsec_defaults;
	struct pulse* pp = CAST_UP(&hs);

	rf_ev->end = rf_ev->start + pp->duration;

	const double asym_pulse = 0.5;

	rf_ev->mid = rf_ev->start + (rf_ev->end - rf_ev->start) * asym_pulse;

	rf_ev->pulse.shape_id = seq->geom.mb_factor;

	rf_ev->pulse.type = SEQ_RF_REFOCUSSING;
	rf_ev->pulse.fa = 180.;
	rf_ev->pulse.freq = 0.;
	rf_ev->pulse.phase = 0.;

	return 1;
}


long flash_ex_calls(const struct seq_config* seq)
{
	long dims[DIMS];
	md_select_dims(DIMS, PHS1_FLAG|TIME_FLAG|TIME2_FLAG|AVG_FLAG|BATCH_FLAG, dims, seq->loop_dims);

	if (SEQ_PEMODE_RAGA == seq->enc.pe_mode)
		dims[PHS1_DIM] = 1;

	if (1 < seq->geom.mb_factor)
		dims[PHS2_DIM] = seq->loop_dims[PHS2_DIM];
	else
		dims[SLICE_DIM] = seq->loop_dims[SLICE_DIM];

	return md_calc_size(DIMS, dims);
}


double adc_time_to_echo(const struct seq_config* seq)
{
	return 0.5 * seq->phys.dwell * seq->geom.baseres;
}


double adc_duration(const struct seq_config* seq)
{
	return round_up_raster(seq->phys.dwell * seq->geom.baseres, seq->sys.raster_rf);
}

static double adc_nco_freq(double proj_angle, long chrono_slice, const struct seq_config* seq)
{
	return seq->sys.gamma * 
		(seq->geom.shift[chrono_slice][0] * ro_amplitude(seq) * sin(proj_angle)
		+ seq->geom.shift[chrono_slice][1] * ro_amplitude(seq) * cos(proj_angle));
}

int prep_adc(struct seq_event* adc_ev, double start, double rf_spoil_phase,
		const struct seq_state* seq_state, const struct seq_config* seq)
{
	if (SEQ_BLOCK_KERNEL_DUMMY == seq_state->mode)
		return 0;

	adc_ev->type = SEQ_EVENT_ADC;

	adc_ev->start = start;
	adc_ev->end = adc_ev->start + adc_duration(seq);

	adc_ev->adc.dwell_ns = (long)(seq->phys.dwell * 1.E9 + 0.5);
	adc_ev->adc.columns = seq->geom.baseres;

	adc_ev->mid = adc_ev->start + adc_time_to_echo(seq);

	md_copy_dims(DIMS, adc_ev->adc.pos, seq_state->pos);

	if (SEQ_PEMODE_RAGA == seq->enc.pe_mode) {

		struct traj_conf conf;
		traj_conf_from_seq(&conf, seq);

		adc_ev->adc.pos[PHS1_DIM] = raga_increment_from_pos(seq->order, seq_state->pos,
							SEQ_FLAGS & ~(COEFF_FLAG | COEFF2_FLAG),
							seq->loop_dims, &conf);
	}

	adc_ev->adc.os = seq->phys.os;

	double proj_angle = get_rot_angle(seq_state->pos, seq);

	adc_ev->adc.freq = adc_nco_freq(proj_angle, seq_state->chrono_slice, seq);
	adc_ev->adc.phase = phase_clamp(rf_spoil_phase);

	return 1;
}

