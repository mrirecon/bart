/* Copyright 2025-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>

#include "num/multind.h"
#include "num/rand.h"

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/version.h"

#include "seq/config.h"
#include "seq/event.h"
#include "seq/helpers.h"
#include "seq/misc.h"

#include "seq/adc_rf.h"
#include "seq/anglecalc.h"
#include "seq/pulse.h"
#include "seq/flash.h"
#include "seq/mag_prep.h"

#include "seq.h"

#define MAX_EVENTS 2048
#define MAX_RF_PULSES 32

struct bart_seq* bart_seq_alloc(const char* driver_version)
{
	struct bart_seq* seq = NULL;
	seq = xmalloc(sizeof *seq);

	seq->bart_version = bart_version;
	seq->driver_version = driver_version;

	seq->conf = xmalloc(sizeof *(seq->conf));
	seq->state = xmalloc(sizeof *(seq->state));

	seq->N = MAX_EVENTS;
	seq->event = xmalloc((size_t)seq->N * (sizeof *(seq->event)));

	seq->P = MAX_RF_PULSES;
	seq->rf_shape = xmalloc((size_t)seq->P * (sizeof *(seq->rf_shape)));

	return seq;
}

void bart_seq_defaults(struct bart_seq* seq)
{
	memcpy(seq->conf, &seq_config_defaults, sizeof *(seq->conf));
	memset(seq->state, 0, sizeof *(seq->state));
	memset(seq->event, 0, (size_t)seq->N * (sizeof *(seq->event)));
	memset(seq->rf_shape, 0, (size_t)seq->P * (sizeof *(seq->rf_shape)));
}

int bart_seq_prepare(struct bart_seq* seq)
{
	num_rand_init(0ULL); // initialize here since once called before actual sequence start

	seq->state->mode = BLOCK_KERNEL_PREPARE;
	
	int N = seq_block(seq->N, seq->event, seq->state, seq->conf);

	if (0 < N)
		N = seq_sample_rf_shapes(MAX_RF_PULSES, seq->rf_shape, seq->conf);
	
	for (int i = 0; i < DIMS; i++)
		seq->state->pos[i] = 0;

	seq->state->mode = BLOCK_UNDEFINED;

	return N;
}


void bart_seq_free(struct bart_seq* seq)
{
	xfree(seq->conf);
	xfree(seq->state);
	xfree(seq->event);
	xfree(seq->rf_shape);
	xfree(seq);
}

int bart_seq_version_check(const char* driver_version, const unsigned int min_bart_version[5])
{
	// FIXME check for stable event.h, seq.h, helpers.h, custom_ui.h

	unsigned int vd[5];
	if (!version_parse(vd, driver_version))
		return -1;

	const unsigned int min_driver[5] = { };
	if (version_compare(vd, min_driver) < 0)
		return -1;

	unsigned int vb[5];
	if (!version_parse(vb, bart_version))
		return -1;

	if (version_compare(vb, min_bart_version) < 0)
		return -1;

	return 1;
}


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

		pulse[idx].samples = lround(1.E6 * seq->phys.rf_duration);

		if (MAX_RF_SAMPLES < pulse[idx].samples)
			return -1;

		double dwell = seq->phys.rf_duration / pulse[idx].samples;

		struct pulse_sms ps = pulse_sms_defaults;

		pulse_sms_init(&ps, seq->phys.rf_duration, seq->phys.flip_angle, 0., seq->phys.bwtp, alpha,
			seq->geom.mb_factor, idx, seq->geom.sms_distance, seq->geom.slice_thickness);

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
		pulse[idx].fa_prep = 180.;

		struct pulse* pp = CAST_UP(&hs);

		pulse[idx].sar_calls = seq->loop_dims[BATCH_DIM];
		pulse[idx].sar_dur = pp->duration;

		pulse[idx].samples = lround(0.5 * 1E6 * pulse[idx].sar_dur);

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


double seq_nco_freq(const struct seq_event* ev)
{
	return (SEQ_EVENT_PULSE == ev->type) ? ev->pulse.freq : ev->adc.freq;
}

double seq_nco_phase(int set, const struct seq_event* ev)
{
	double phase_mid = (SEQ_EVENT_PULSE == ev->type) ? ev->pulse.phase : ev->adc.phase;

	if (0 == set)
		phase_mid = -1. * phase_mid;

	double time = (set) ? (ev->mid - ev->start) : (ev->end - ev->mid);

	return phase_clamp(- seq_nco_freq(ev) * 360. * time + phase_mid);
}

double seq_pulse_scaling(const struct rf_shape* pulse)
{
	return 180. / M_PI * pulse->integral;
}

double seq_pulse_norm_sum(const struct rf_shape* pulse)
{
	double dwell = pulse->sar_dur / pulse->samples;
	return ((pulse->integral / dwell) / pulse->max);
}

void seq_cfl_to_sample(const struct rf_shape* pulse, int idx, float* mag, float* pha)
{
	assert(idx < pulse->samples);

	complex float val = pulse->shape[idx];

	*mag = cabs(val) / pulse->max;
	*pha = fmod(carg(val) + 2. * M_PI, 2. * M_PI);
}




double seq_block_end(int N, const struct seq_event ev[N], enum block mode, double tr, double raster)
{
	if ((BLOCK_PRE == mode) || (BLOCK_POST == mode))
		return round_up_raster(events_end_time(N, ev, 0, 0), raster);
	else
		return tr;
}

double seq_block_end_flat(int N, const struct seq_event ev[N], double raster)
{
	return round_up_raster(events_end_time(N, ev, 1, 1), raster);
}


double seq_block_rdt(int N, const struct seq_event ev[N], double raster)
{
	return round_up_raster(events_end_time(N, ev, 1, 0) - seq_block_end_flat(N, ev, raster), raster);
}

static long get_chrono_slice(const struct seq_state* seq_state, const struct seq_config* seq)
{
	return (1 < seq->geom.mb_factor) ? seq_state->pos[PHS2_DIM] + seq_state->pos[SLICE_DIM] * seq->loop_dims[PHS2_DIM] : seq_state->pos[SLICE_DIM];
}

static int check_settings(const struct seq_state* seq_state, const struct seq_config* seq)
{
	if (0 > seq->loop_dims[PHS2_DIM])
		return ERROR_SETTING_DIM;

	if (MAX_SLICES < seq_get_slices(seq))
		return ERROR_SETTING_DIM;

	if (PEMODE_RAGA_MEMS == seq->enc.pe_mode)
		return ERROR_ROT_ANGLE;


	if (CONTEXT_BINARY != seq_state->context) {

		if ((PEMODE_RAGA == seq->enc.pe_mode) || (PEMODE_RAGA_ALIGNED == seq->enc.pe_mode)) {

			if (!check_gen_fib(seq->loop_dims[PHS1_DIM], seq->enc.tiny))
				return ERROR_SETTING_SPOKES_RAGA;
		}


		if (0 == (seq->loop_dims[PHS1_DIM] % 2))
			return ERROR_SETTING_SPOKES_EVEN;
	}

	return 1;
}

int seq_block(int N, struct seq_event ev[N], struct seq_state* seq_state, const struct seq_config* seq)
{
	int err = check_settings(seq_state, seq);
	if (1 > err)
		return err;

	seq_state->chrono_slice = get_chrono_slice(seq_state, seq);

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

		if (md_check_equal_dims(DIMS, zeros, seq_state->pos, ~0UL)) {

			seq_state->mode = BLOCK_PRE;
			return wait_time_to_event(ev, 0., seq->magn.init_delay);
		}
		else if (md_check_equal_dims(DIMS, (zeros[COEFF2_DIM] = 1, zeros), seq_state->pos, ~0UL)) {

			seq_state->mode = BLOCK_KERNEL_NOISE;

			return flash(N, ev, seq_state, seq);
		}

		if (seq_state->pos[COEFF2_DIM] > 1) {

			int i = 0;
			if (md_check_equal_dims(DIMS, zeros, seq_state->pos, ~(BATCH_FLAG | msm_flag | COEFF2_FLAG))) {

				seq_state->mode = BLOCK_PRE;
				i += mag_prep(ev + i, seq);

				return i;
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

		if (md_check_equal_dims(DIMS, last_idx, seq_state->pos, (SEQ_FLAGS & ~(BATCH_FLAG | msm_flag)))
			&& (0. < seq->magn.inv_delay_time)) {

				seq_state->mode = BLOCK_POST;
				ev[0].type = SEQ_EVENT_WAIT;
				ev[0].end = seq->magn.inv_delay_time;
				return 1;
		}

		return 0;
	}

	return 0;
}

int seq_continue(struct seq_state* seq_state, const struct seq_config* seq)
{
	return md_next_permuted(DIMS, seq->order, seq->loop_dims, SEQ_FLAGS, seq_state->pos);
}

