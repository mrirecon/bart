/* Copyright 2025-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>
#include <stdio.h>

#include "num/multind.h"

#include "misc/mri.h"
#include "misc/misc.h"

#include "noncart/traj.h"

#include "seq/config.h"
#include "seq/anglecalc.h"
#include "seq/adc_rf.h"
#include "seq/flash.h"
#include "seq/misc.h"
#include "seq/mag_prep.h"
#include "seq/opts.h"
#include "seq/ui_enums.h"

#include "helpers.h"

int seq_raga_spokes(const struct seq_config* seq)
{
	if (SEQ_PEMODE_RAGA == seq->enc.pe_mode)
		return raga_spokes(seq->geom.baseres, seq->enc.tiny);

	return seq->loop_dims[PHS1_DIM];
}

int seq_check_equal_dims(int D, const long dims1[D], const long dims2[D], unsigned long flags)
{
	return md_check_equal_dims(D, dims1, dims2, flags);
}


double seq_minimum_tr(const struct seq_config* seq)
{
	return min_tr_flash(seq);
}


void seq_minimum_te(const struct seq_config* seq, double* min_te, double* fil_te)
{
	min_te_flash(seq, min_te, fil_te);
}




static long kernels_per_measurement(const long loop_dims[DIMS])
{
	long dims[DIMS];
	md_select_dims(DIMS, PHS1_FLAG|TIME2_FLAG|AVG_FLAG|SLICE_FLAG|PHS2_FLAG, dims, loop_dims);

	return md_calc_size(DIMS, dims);
}

long seq_relevant_readouts_meas_time(const struct seq_config* seq)
{
	return kernels_per_measurement(seq->loop_dims) / seq->loop_dims[PHS1_DIM];
}

double seq_total_measure_time(const struct seq_config* seq)
{
	double pre_duration = seq->magn.init_delay;

	struct seq_event ev[6];
	int e = mag_prep(ev, seq);

	double prep_pulse_duration = seq_block_end(e, ev, SEQ_BLOCK_PRE, seq->phys.tr, seq->sys.raster_grad);
	prep_pulse_duration += seq->magn.inv_delay_time;
	// prep_pulse_duration *= inv_calls(seq);

	long dims[DIMS] = { };
	md_select_dims(DIMS, SEQ_FLAGS & ~(COEFF_FLAG|COEFF2_FLAG), dims, seq->loop_dims);

	long img_calls = md_calc_size(DIMS, dims);
	double imaging_duration = seq->phys.tr * img_calls;

	if ((SEQ_TRIGGER_OFF != seq->trigger.type) && (1 < seq->trigger.pulses)) {

		imaging_duration = 1. * (seq->trigger.delay_time + seq->phys.tr) * img_calls * (seq->trigger.pulses - 1);
	}

	return pre_duration + prep_pulse_duration + imaging_duration;
}



static void custom_params_to_config(struct seq_config* seq, int nl, const long custom_long[__VLA(nl)], int nd, const double custom_double[__VLA(nd)])
{
	seq->enc.pe_mode = (enum pe_mode)custom_long[cil_pe_mode];
	seq->phys.contrast = (enum flash_contrast)custom_long[cil_contrast];

	seq->geom.mb_factor = 1;

	if (CHECKBOX_ON == custom_long[cil_sms])
		seq->geom.mb_factor = custom_long[cil_mb_factor];

	seq->phys.os = 2.;

	seq->enc.tiny = custom_long[cil_tiny];
	seq->phys.rf_duration = 1E-6 * custom_long[cil_rf_duration];
	seq->magn.init_delay = custom_long[cil_init_delay];
	seq->loop_dims[BATCH_DIM] = custom_long[cil_inversions];
	seq->magn.inv_delay_time = custom_long[cil_inv_delay];
	seq->enc.aligned_flags = (unsigned long)custom_long[cil_RAGA_aligned_flags];

	seq->phys.bwtp = custom_double[cid_BWTP];
}


static void config_to_custom_params(int nl, long custom_long[__VLA(nl)], int nd, double custom_double[__VLA(nd)], const struct seq_config* seq)
{
	custom_long[cil_pe_mode] = seq->enc.pe_mode;;
	custom_long[cil_contrast] = seq->phys.contrast;

	custom_long[cil_mb_factor] = 1;

	if (CHECKBOX_ON == custom_long[cil_sms])
		custom_long[cil_mb_factor] = seq->geom.mb_factor;

	custom_long[cil_tiny] = seq->enc.tiny;
	custom_long[cil_rf_duration] = lround(1.E6 * seq->phys.rf_duration);
	custom_long[cil_init_delay] = seq->magn.init_delay;
	custom_long[cil_inversions] = seq->loop_dims[BATCH_DIM];
	custom_long[cil_inv_delay] = seq->magn.inv_delay_time;
	custom_double[cid_BWTP] = seq->phys.bwtp;

	custom_long[cil_sms] = CHECKBOX_OFF;
}


void seq_ui_interface_custom_params(int reverse, struct seq_config* seq, int nl, long params_long[__VLA(nl)], int nd, double params_double[__VLA(nd)])
{
	if (reverse)
		config_to_custom_params(nl, params_long, nd, params_double, seq);
	else
		custom_params_to_config(seq, nl, params_long, nd, params_double);
}

static void seq_bart_to_standard_conf(struct seq_standard_conf* std, struct seq_config* seq)
{
	std->tr = seq->phys.tr;

	for (int i = 0; i < SEQ_MAX_NO_ECHOES; i++)
		std->te[i] = seq->phys.te[i];

	std->dwell = seq->phys.dwell;
	std->flip_angle = seq->phys.flip_angle;

	std->fov = seq->geom.fov;
	std->baseres = seq->geom.baseres;
	std->slice_thickness = seq->geom.slice_thickness;
	// std->slice_os = 1. + seq->geom.slice_os;

	// std->is3D = seq->dim.is3D;

	std->gamma = seq->sys.gamma;
	std->b0 = seq->sys.b0;
	std->grad_min_rise_time = seq->sys.grad.inv_slew_rate;
	std->grad_max_ampl = seq->sys.grad.max_amplitude;
	std->coil_control_lead = seq->sys.coil_control_lead;
	std->min_duration_ro_rf = seq->sys.min_duration_ro_rf;

	std->mag_prep = seq->magn.mag_prep;
	std->ti = 0.;

	if (SEQ_PREP_OFF != seq->magn.mag_prep)
	       std->ti = seq->magn.ti;

	std->trigger_type = seq->trigger.type;
	std->trigger_delay_time = seq->trigger.delay_time;
	std->trigger_pulses = seq->trigger.pulses;
	std->trigger_out = 1;

	std->enc_order = seq->enc.order;

	for (int i = 0; i < SEQ_ACOUSTIC_RESONANCE_ENTRIES; i++) {

		std->acoustic_res_freq[i] = std->acoustic_res_freq[i];
		std->acoustic_res_bw[i] = std->acoustic_res_bw[i];
	}
}

static void seq_standard_conf_to_bart(struct seq_config* seq, struct seq_standard_conf* std)
{
	seq->phys.tr = std->tr;

	for (int i = 0; i < SEQ_MAX_NO_ECHOES; i++)
		seq->phys.te[i] = std->te[i];

	seq->phys.dwell = std->dwell;
	seq->phys.os = 2.;
	seq->phys.flip_angle = std->flip_angle;

	seq->geom.fov = std->fov;
	seq->geom.baseres = std->baseres;
	seq->geom.slice_thickness = std->slice_thickness;
	// seq->geom.slice_os = 1. + std->slice_os;

	// seq->dim.is3D = std->is3D;

	seq->sys.gamma = std->gamma;

	seq->sys.b0 = std->b0;

	seq->sys.grad.inv_slew_rate = std->grad_min_rise_time;
	seq->sys.grad.max_amplitude = std->grad_max_ampl;

	seq->sys.coil_control_lead = std->coil_control_lead;
	seq->sys.min_duration_ro_rf = std->min_duration_ro_rf;

	seq->magn.mag_prep = std->mag_prep;

	seq->magn.ti = 0;

	if (SEQ_PREP_OFF != seq->magn.mag_prep)
		seq->magn.ti = std->ti;

	seq->trigger.type = std->trigger_type;
	seq->trigger.delay_time = std->trigger_delay_time;
	seq->trigger.pulses = std->trigger_pulses;
	seq->trigger.trigger_out = 1;

	seq->enc.order = std->enc_order;
}


void seq_ui_interface_standard_conf(int reverse, struct seq_config* conf, struct seq_standard_conf* std_conf)
{
	if (reverse)
		seq_bart_to_standard_conf(std_conf, conf);
	else
		seq_standard_conf_to_bart(conf, std_conf);
}


static void loop_dims_to_conf(struct seq_config* seq, const int D, const long in_dims[D])
{
	switch (seq->enc.order) {

	case SEQ_ORDER_AVG_OUTER:
		md_copy_order(DIMS, seq->order, seq_loop_order_avg_outer);
		break;

	case SEQ_ORDER_SEQ_MS:
		md_copy_order(DIMS, seq->order, seq_loop_order_multislice);
		break;

	case SEQ_ORDER_AVG_INNER:
		md_copy_order(DIMS, seq->order, seq_loop_order_avg_inner);
		break;
	}

	long total_slices = in_dims[SLICE_DIM];

	if (1 < seq->geom.mb_factor) {

		seq->loop_dims[SLICE_DIM] = seq->geom.mb_factor;
		seq->loop_dims[PHS2_DIM] = total_slices / seq->geom.mb_factor;

	} else {

		seq->loop_dims[SLICE_DIM] = total_slices;
		seq->loop_dims[PHS2_DIM] = 1;
	}

	if ((seq->loop_dims[PHS2_DIM] * seq->loop_dims[SLICE_DIM]) != total_slices)
		seq->loop_dims[PHS2_DIM] = -1; //mb groups

	long frames = in_dims[TIME_DIM];
	seq->loop_dims[TIME_DIM] = frames;

	long radial_views = in_dims[PHS1_DIM];

	if (SEQ_PEMODE_RAGA == seq->enc.pe_mode) {

		seq->loop_dims[TIME_DIM] = (long)ceil(1. * frames / radial_views);
		seq->loop_dims[ITER_DIM] = frames % radial_views;

		if (0 == seq->loop_dims[ITER_DIM])
			seq->loop_dims[ITER_DIM] = radial_views;
	}

	seq->loop_dims[TIME2_DIM] = 1;

	if (SEQ_TRIGGER_OFF != seq->trigger.type)
		seq->loop_dims[TIME2_DIM] = in_dims[TIME2_DIM];

	seq->loop_dims[AVG_DIM] = in_dims[AVG_DIM];
	seq->loop_dims[PHS1_DIM] = radial_views;
	seq->loop_dims[TE_DIM] = in_dims[TE_DIM];

	seq->loop_dims[COEFF2_DIM] = 3; // 2 additional calls for delay_meas + noise_scan
	seq->loop_dims[COEFF_DIM] = 3; // pre-/post- and actual kernel calls
}

static void conf_to_loop_dims(const int D, long dims[D], struct seq_config* seq)
{
	dims[SLICE_DIM] = seq->loop_dims[SLICE_DIM];
	dims[PHS2_DIM] = (seq->geom.mb_factor > 1) ? seq->loop_dims[SLICE_DIM] / seq->geom.mb_factor : 1;
	if ((dims[PHS2_DIM] * seq->geom.mb_factor != seq->loop_dims[SLICE_DIM]))
		seq->loop_dims[PHS2_DIM] = -1; //mb groups

	dims[PHS1_DIM] = seq->loop_dims[PHS1_DIM];

	dims[TIME_DIM] = seq->loop_dims[TIME_DIM];
	dims[TE_DIM] = seq->loop_dims[TE_DIM];
	dims[AVG_DIM] = seq->loop_dims[AVG_DIM];
}

void seq_ui_interface_loop_dims(int reverse, struct seq_config* seq, const int D, long dims[__VLA(D)])
{
	if (reverse)
		conf_to_loop_dims(D, dims, seq);
	else
		loop_dims_to_conf(seq, D, dims);
}


struct seq_interface_conf seq_get_interface_conf(struct seq_config* conf)
{
	struct seq_interface_conf ret = { };

	ret.tr = conf->phys.tr;
	ret.radial_views = conf->loop_dims[PHS1_DIM];
	ret.slices = get_slices(conf);
	ret.echoes = conf->loop_dims[TE_DIM];
	ret.trigger_type = conf->trigger.type;
	ret.trigger_delay_time = conf->trigger.delay_time;
	ret.trigger_pulses = conf->trigger.pulses;
	ret.slice_thickness = conf->geom.slice_thickness;
	ret.sms_distance = conf->geom.sms_distance;
	ret.is3D = 0; // conf->dim.is3D;
	ret.isBSSFP = 0; // (CONTRAST_BALANCED == conf->phys.contrast) ? 1 : 0;
	ret.raster_grad = conf->sys.raster_grad;
	ret.raster_rf = conf->sys.raster_rf;
	ret.grad_max_ampl = conf->sys.grad.max_amplitude;

	return ret;
}


void seq_set_fov_pos(int N, int M, const float* shifts, struct seq_config* seq)
{
	long total_slices = get_slices(seq);
	assert(total_slices <= N);

	seq->geom.sms_distance = 0;

	if (1 < seq->geom.mb_factor)
		seq->geom.sms_distance = fabsf(shifts[2] - shifts[seq->loop_dims[PHS2_DIM] * M + 2]);

	for (int i = 0; i < total_slices; i++) {

		seq->geom.shift[i][0] = shifts[i * M + 0]; // RO shift
		seq->geom.shift[i][1] = shifts[i * M + 1]; // PE shift

		if (1 < seq->geom.mb_factor) {

			seq->geom.shift[i][2] = shifts[(total_slices / 2) * M + 2]
						+ (seq->geom.sms_distance / seq->loop_dims[PHS2_DIM]) 
						* (i % seq->loop_dims[PHS2_DIM] - floor(seq->loop_dims[PHS2_DIM] / 2.));

		} else {

			seq->geom.shift[i][2] = shifts[i * M + 2];
		}		
	}

	if (1 == seq->geom.mb_factor)
		seq->geom.sms_distance = -0.999; // UI information
}


int seq_config_from_string(struct seq_config* seq, int N, char* buffer)
{
	return read_config_from_str(seq, N, buffer);
}


int seq_print_info_config(int N, char* info, const struct seq_config* seq)
{
	int ctr = 0;

	ctr += snprintf(info + ctr, (size_t)(N - ctr), "\n\nseq_config\nTR\t\t\t\t\t%f", seq->phys.tr);

	for (int i = 0; i < seq->loop_dims[TE_DIM]; i++)
		ctr += snprintf(info + ctr, (size_t)(N - ctr), "\nTE[%d]\t\t\t\t\t%f", i, seq->phys.te[i]);

	ctr += snprintf(info + ctr, (size_t)(N - ctr), 
			"\ndwell/os\t\t\t\t%.8f/%.2f\ncontrast/rf duration/FA/BWTP\t\t%d/%.6f/%.2f/%.2f",
			seq->phys.dwell, seq->phys.os,
			seq->phys.contrast, seq->phys.rf_duration, seq->phys.flip_angle, seq->phys.bwtp);

	ctr += snprintf(info + ctr, (size_t)(N - ctr), 
			"\nFOV/slice-th\t\t\t%.3f/%.3f\nBR/mb_factor/SMS dist\t\t\t%d/%d/%.3f",
			seq->geom.fov, seq->geom.slice_thickness,
			seq->geom.baseres, seq->geom.mb_factor, seq->geom.sms_distance);
	
	ctr += snprintf(info + ctr, (size_t)(N - ctr),
			"\nPE_Mode/Turns-GA/aligned flags/order\t%d/%d/%ld/%d",
			seq->enc.pe_mode, seq->enc.tiny, seq->enc.aligned_flags, seq->enc.order);

	ctr += snprintf(info + ctr, (size_t)(N - ctr),
			"\ngamma/b0/max grad/inv slew\t\t%.0f/%.3f/%.3f/%.6f\n",
			seq->sys.gamma, seq->sys.b0, seq->sys.grad.max_amplitude, seq->sys.grad.inv_slew_rate);

	ctr += snprintf(info + ctr, (size_t)(N - ctr),
			"\nmag prep/TI/init delay/inv delay\t\t%d/%.6f/%.2f/%.2f",
			seq->magn.mag_prep, seq->magn.ti, seq->magn.init_delay, seq->magn.inv_delay_time);

	ctr += snprintf(info + ctr, (size_t)(N - ctr),
			"\nloop_dims\t: %ld|%ld|%ld|%ld\t\t%ld|%ld|%ld|%ld\t\t%ld|%ld|%ld|%ld\t\t%ld|%ld|%ld|%ld\t\t",
			seq->loop_dims[READ_DIM], seq->loop_dims[PHS1_DIM], seq->loop_dims[PHS2_DIM], seq->loop_dims[COIL_DIM],
			seq->loop_dims[MAPS_DIM], seq->loop_dims[TE_DIM], seq->loop_dims[COEFF_DIM], seq->loop_dims[COEFF2_DIM],
			seq->loop_dims[ITER_DIM], seq->loop_dims[CSHIFT_DIM], seq->loop_dims[TIME_DIM], seq->loop_dims[TIME2_DIM],
			seq->loop_dims[LEVEL_DIM], seq->loop_dims[SLICE_DIM], seq->loop_dims[AVG_DIM], seq->loop_dims[BATCH_DIM]);

	ctr += snprintf(info + ctr, (size_t)(N - ctr), "\n\nCrowthers no. of radial Spokes =\t%.2f", M_PI * seq->geom.baseres);

	if (ctr > N)
		return -1;

	return ctr;
}



void seq_print_info_radial_views(int N, char* info, const struct seq_config* seq)
{
	(void) N;

	if (SEQ_PEMODE_RAGA != seq->enc.pe_mode)
		return;

	int ctr = 0;

	ctr += snprintf(info + ctr, (size_t)(N - ctr), "Rational Approximation of Golden Angle Sampling. Tiny-GA: %d\nallowed spokes: ", seq->enc.tiny);

	int i = 1;

	// max number of spokes
	while (2050 > gen_fibonacci(seq->enc.tiny, i)) {

		if (check_gen_fib(gen_fibonacci(seq->enc.tiny, i), seq->enc.tiny)) {

			ctr += snprintf(info + ctr, (size_t)(N - ctr), "%d\t", gen_fibonacci(seq->enc.tiny, i));
		}

		i++;
	}
}
