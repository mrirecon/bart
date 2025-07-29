/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>

#include "num/multind.h"

#include "misc/mri.h"
#include "misc/misc.h"

#include "seq/config.h"
#include "seq/event.h"
#include "seq/misc.h"

#include "seq/adc_rf.h"
#include "seq/pulse.h"
#include "seq/flash.h"

#include "seq.h"


int seq_sample_rf_shapes(int N, struct rf_shape pulse[N], const struct seq_config* seq)
{
	int idx = 0;

	for (; idx < seq->geom.mb_factor; idx++) {

		if (idx >= N)
			return -1;

		pulse[idx].sar_calls = flash_ex_calls(seq);
		pulse[idx].sar_dur = seq->phys.rf_duration;
		pulse[idx].fa_prep = seq->phys.flip_angle;

		const float alpha = 0.5;

		pulse[idx].samples = 2 * seq->phys.rf_duration;

		if (MAX_RF_SAMPLES < pulse[idx].samples)
			return -1;

		double dwell = 1.E-6 * seq->phys.rf_duration / pulse[idx].samples;

		struct pulse_sms ps = pulse_sms_defaults;

		pulse_sms_init(&ps, 1.E-6 * seq->phys.rf_duration, seq->phys.flip_angle, 0., seq->phys.bwtp, alpha,
			seq->geom.mb_factor, idx, 1.E-3 * seq->geom.sms_distance, 1.E-3 * seq->geom.slice_thickness);

		pulse[idx].max = ps.A; // this is scaled by fa / fa_prep
		pulse[idx].integral = pulse_sms_integral(&ps);

		struct pulse* pp = CAST_UP(&ps);

		for (int j = 0; j < pulse[idx].samples; j++)
			pulse[idx].shape[j] = pulse_eval(pp, j * dwell);
	}

	if (PREP_IR_NON == seq->magn.mag_prep) {

		struct pulse_hypsec hs = pulse_hypsec_defaults;

		pulse_hypsec_init(seq->sys.gamma, &hs);

		pulse[idx].max = hs.A;
		pulse[idx].integral = pulse_hypsec_integral(&hs);

		struct pulse* pp = CAST_UP(&hs);

		pulse[idx].sar_calls = seq->loop_dims[BATCH_DIM];
		pulse[idx].sar_dur = 1.E6 * pp->duration;

		pulse[idx].samples = 0.5 * pulse[idx].sar_dur;

		if (MAX_RF_SAMPLES < pulse[idx].samples)
			return -1;

		double dwell = pp->duration / pulse[idx].samples;

		for (int j = 0; j < pulse[idx].samples; j++)
			pulse[idx].shape[j] = pulse_eval(pp, j * dwell);

		idx++;
	}

	return idx;
}



/*
 * Compute gradients on a raster. This also works
 * (i.e. yields correct 0 moment) if the abstract
 * gradients do not start and end on the raster.
 * We integrate over each interval to obtain the
 * average gradient.
 */
void seq_compute_gradients(int M, double gradients[M][3], double dt, int N, const struct seq_event ev[N])
{
	for (int i = 0; i < M; i++) 
		for (int a = 0; a < 3; a++)
			gradients[i][a] = 0.;

	for (int i = 0; i < N; i++) {

		if (SEQ_EVENT_GRADIENT != ev[i].type)
			continue;

		double s = ev[i].start;
		double e = ev[i].end;


		/*            |    /
                 *            |   /|
                 *            .../..
                 *            | /  |
                 *            |/   |
                 *            /    |
                 *       ..../|    |
                 *  |____|__/_|____|____|
                 *    0    1    2    3  
                 */

		assert(0. <= s);

		double om[3];

		for (int a = 0; a < 3; a++)
			om[a] = 0.;

		for (int p = trunc(s / dt); p <= ceil(e / dt); p++) {

			assert(0 <= p);

			double m0[3];
			moment(m0, (p + 1.) * dt, &ev[i]);

			for (int a = 0; a < 3; a++) {

				if (p < M)
					gradients[p][a] += (m0[a] - om[a]) / dt;

				om[a] = m0[a];
			}
		}
	}
}



double idea_phase_nco(int set, const struct seq_event* ev)
{
	double freq = (SEQ_EVENT_PULSE == ev->type) ? ev->pulse.freq : ev->adc.freq;
	double phase_mid = (SEQ_EVENT_PULSE == ev->type) ? ev->pulse.phase : ev->adc.phase;

	if (0 == set)
		phase_mid = -1. * phase_mid;

	double time = (set) ? (ev->mid - ev->start) : (ev->end - ev->mid);

	return phase_clamp(-freq * 0.000360 * time + phase_mid);
}

double idea_pulse_scaling(const struct rf_shape* pulse)
{
	return 180. / M_PI * pulse->integral;
}

double idea_pulse_norm_sum(const struct rf_shape* pulse)
{
	double dwell = 1.e-6 * pulse->sar_dur / pulse->samples;
	return ((pulse->integral / dwell) / pulse->max);
}

void idea_cfl_to_sample(const struct rf_shape* pulse, int idx, float* mag, float* pha)
{
	assert(idx < pulse->samples);

	complex float val = pulse->shape[idx];

	*mag = cabs(val) / pulse->max;
	*pha = fmod(carg(val) + 2. * M_PI, 2. * M_PI);
}




long seq_block_end(int N, const struct seq_event ev[N], enum block mode, long tr)
{
	if ((BLOCK_PRE == mode) || (BLOCK_POST == mode))
		return round_up_GRT(events_end_time(N, ev, 0, 0));
	else
		return tr;
}

long seq_block_end_flat(int N, const struct seq_event ev[N])
{
	return round_up_GRT(events_end_time(N, ev, 1, 1));
}


long seq_block_rdt(int N, const struct seq_event ev[N])
{
	return round_up_GRT(events_end_time(N, ev, 1, 0) - seq_block_end_flat(N, ev));
}


int seq_block(int N, struct seq_event ev[N], struct seq_state* seq_state, const struct seq_config* seq)
{
	seq_state->chrono_slice = seq_state->pos[SLICE_DIM];

	if (BLOCK_KERNEL_PREPARE == seq_state->mode) {

		seq_state->pos[SLICE_DIM] = (long)ceil(0.5 * seq->geom.mb_factor); // for SMS bSSFP

		return flash(N, ev, seq_state, seq);

	} else if (BLOCK_KERNEL_CHECK == seq_state->mode) {

		return flash(N, ev, seq_state, seq);
	}

	long zeros[DIMS] = { 0 };
	long last_idx[DIMS];

	for (int i = 0; i < DIMS; i++)
		last_idx[i] = seq->loop_dims[i] - 1;

	// changed beahvior for sequential multislice
	unsigned long msm_flag = 0UL;

	if (md_check_equal_order(DIMS, seq->order, seq_loop_order_multislice, SEQ_FLAGS))
	       msm_flag = SLICE_FLAG ;

	if (0 == seq_state->pos[COEFF_DIM]) {

		if (md_check_equal_dims(DIMS, (zeros[COEFF2_DIM] = 1, zeros), seq_state->pos, ~0UL)) {

			seq_state->mode = BLOCK_KERNEL_NOISE;

			return flash(N, ev, seq_state, seq);
		}

		if (seq_state->pos[COEFF2_DIM] > 1) {

			if (md_check_equal_dims(DIMS, zeros, seq_state->pos, ~(BATCH_FLAG | msm_flag | COEFF2_FLAG))) {

				if (PREP_OFF != seq->magn.mag_prep) {

					int i = 0;
					seq_state->mode = BLOCK_PRE;

					i += prep_rf_inv(ev + i, 0., seq);

					struct grad_trapezoid spoil = {
						.ampl = 8,
						.rampup = 800,
						.flat = 8200,
						.rampdown = 600,
					};

					double projSLICE[3] = { 0. , 0. , 1. };

					i+= seq_grad_to_event(ev + i, ev[i - 1].end, &spoil, projSLICE);
					
					ev[i].start = ev[i - 1].end;
					ev[i].end = ev[i].start + seq->magn.ti;
					ev[i].type = SEQ_EVENT_WAIT;

					i++;

					return i;
				}
			}

		} else if (seq_state->pos[PHS1_DIM] > 0) {

			md_max_dims(DIMS, (COEFF2_FLAG | PHS2_FLAG) &  ~msm_flag, seq_state->pos, seq_state->pos, last_idx);
		}

	} else if (1 == seq_state->pos[COEFF_DIM]) {

		seq_state->mode = BLOCK_KERNEL_IMAGE;
		md_max_dims(DIMS, (COEFF2_FLAG), seq_state->pos, seq_state->pos, last_idx);

		return flash(N, ev, seq_state, seq);

	} else if (2 == seq_state->pos[COEFF_DIM]) {

		md_max_dims(DIMS, (COEFF2_FLAG | PHS2_FLAG) & ~msm_flag, seq_state->pos, seq_state->pos, last_idx);
		return 0;
	}

	return 0;
}

int seq_continue(struct seq_state* seq_state, const struct seq_config* seq)
{
	return md_next_permuted(DIMS, seq->order, seq->loop_dims, SEQ_FLAGS, seq_state->pos);
}

