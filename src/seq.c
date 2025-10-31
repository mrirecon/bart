/* Copyright 2025-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Daniel Mackner
 */

#include <complex.h>
#include <math.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/rand.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/opts.h"

#include "seq/config.h"
#include "seq/event.h"
#include "seq/helpers.h"
#include "seq/seq.h"

#include "seq/misc.h"
#include "seq/flash.h"
#include "seq/kernel.h"
#include "seq/pulseq.h"


#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char help_str[] = "Computes a GRE sequence.";

enum gradient_mode { GRAD_FAST, GRAD_NORMAL, GRAD_WHISPER };

int main_seq(int argc, char* argv[argc])
{
	const char* grad_file = NULL;
	const char* mom_file = NULL;
	const char* adc_file = NULL;
	const char* raga_file = NULL;
	const char* seq_file = NULL;

	struct arg_s args[] = {

		ARG_OUTFILE(false, &adc_file, "0th moment (x,y,z) at sample points, sample_points, phase of adc"),
		ARG_OUTFILE(false, &grad_file, "gradients (x,y,z) per imaging block"),
		ARG_OUTFILE(false, &mom_file, "0th moment (x,y,z) per imaging block"),
		ARG_OUTFILE(false, &seq_file,  "pulseq file"),
	};

	float dt = -1.;
	long samples = -1;
	double rel_shift[3] = { };
	long raga_full_frames = 0;
	float dist = 1.;

	struct bart_seq* seq = bart_seq_alloc();
	bart_seq_defaults(seq);

	enum gradient_mode gradient_mode = GRAD_FAST;

	bool chrono = false;
	bool support = false;

	long custom_params_long[MAX_PARAMS_LONG] = { 0 };
	double custom_params_double[MAX_PARAMS_DOUBLE] = { 0. };

	const struct opt_s opts[] = {

		OPT_FLOAT('d', &dt, "dt", "time-increment per sample (default: seq->conf->phys.tr / 1000)"),
		OPT_LONG('N', &samples, "samples", "Number of samples (default: 1000)"),

		OPT_DOVEC3('s', &seq->conf->geom.shift[0], "RO:PE:SL", "FOV shift of first slice"),
		OPT_DOVEC3('S', &rel_shift, "RO:PE:SL", "relative FOV shift of first slice"),
		OPTL_FLOAT(0, "dist", &dist, "dist", "slice distance factor [1 / slice_thickness] (default: 1.)"),

		// contrast mode
		OPTL_SELECT(0, "no-spoiling", enum flash_contrast, &seq->conf->phys.contrast, CONTRAST_NO_SPOILING, "spoiling off (default: rf random)"),
		OPTL_SELECT(0, "spoiled", enum flash_contrast, &seq->conf->phys.contrast, CONTRAST_RF_SPOILED, "RF_SPOILED (inc: 50 deg, gradient on) (default: rf random)"),

		// FOV and resolution
		OPTL_DOUBLE(0, "FOV", &seq->conf->geom.fov, "FOV", "Field Of View"),
		OPTL_PINT(0, "BR", &seq->conf->geom.baseres, "BR", "Base Resolution"),
		OPTL_DOUBLE(0, "slice_thickness", &seq->conf->geom.slice_thickness, "slice_thickness", "Slice thickness"),

		// basic sequence parameters
		OPTL_DOUBLE(0, "FA", &seq->conf->phys.flip_angle, "flip angle", "Flip angle [deg]"),
		OPTL_DOUBLE(0, "TR", &seq->conf->phys.tr, "TR", "TR"),
		OPTL_DOUBLE(0, "TE", &seq->conf->phys.te, "TE", "TE"),
		OPTL_DOUBLE(0, "BWTP", &seq->conf->phys.bwtp, "BWTP", "Bandwidth Time Product"),

		// others sequence parameters
		OPTL_DOUBLE(0, "rf_duration", &seq->conf->phys.rf_duration, "rf_duration", "RF pulse duration"),
		OPTL_DOUBLE(0, "dwell", &seq->conf->phys.dwell, "dwell", "Dwell time"),
		OPTL_DOUBLE(0, "os", &seq->conf->phys.os, "os", "Oversampling factor"),

		// encoding
		OPTL_UINT(0, "pe_mode", &seq->conf->enc.pe_mode, "pe_mode", "Phase-encoding mode"),
		OPTL_SELECT(0, "turn", enum pe_mode, &seq->conf->enc.pe_mode, PEMODE_TURN, "turn-based PE (default: RAGA)"),
		OPTL_SELECT(0, "raga", enum pe_mode, &seq->conf->enc.pe_mode, PEMODE_RAGA, "RAGA PE"),
		OPTL_SELECT(0, "raga_al", enum pe_mode, &seq->conf->enc.pe_mode, PEMODE_RAGA_ALIGNED, "RAGA-aligned PE (default: RAGA)"),
		OPTL_ULONG(0, "raga_flags", &seq->conf->enc.aligned_flags, "raga_aligned_flags", "RAGA aligned flags (by bitmask)"),

		OPTL_SET(0, "chrono", &chrono, "save gradients/moments/sampling in chronological order (RAGA)"),
		OPT_OUTFILE('R', &raga_file, "file", "raga indices"),

		OPTL_PINT(0, "tiny", &seq->conf->enc.tiny, "tiny", "Tiny golden-ratio index"),

		OPTL_LONG('r', "lines", &seq->conf->loop_dims[PHS1_DIM], "lines", "Number of phase encoding lines"),
		OPTL_LONG('z', "partitions", &seq->conf->loop_dims[PHS2_DIM], "partitions", "Number of partitions (3D) or SMS groups (2D)"),
		OPTL_LONG('t', "measurements", &seq->conf->loop_dims[TIME_DIM], "measurements", "Number of measurements / frames (RAGA: total number of spokes)"),
		OPTL_LONG('f', "raga_full_frames", &raga_full_frames, "raga_full_frames", "Number of full frames (only RAGA)"),
		OPTL_LONG('m', "slices", &seq->conf->loop_dims[SLICE_DIM], "slices", "Number of slices of multiband factor (SMS)"),
		OPTL_LONG('i', "inversions", &seq->conf->loop_dims[BATCH_DIM], "inversions", "Number of inversions"),

		// order
		OPTL_SELECT(0, "sequential-multislice", enum seq_order, &seq->conf->enc.order, SEQ_ORDER_SEQ_MS, "seq_order: sequential multislice (default: avg outer)"),
		OPTL_SELECT(0, "avg-inner", enum seq_order, &seq->conf->enc.order, SEQ_ORDER_AVG_INNER, "seq_order: average inner (default: avg outer)"),

		// sms
		OPTL_PINT(0, "mb_factor", &seq->conf->geom.mb_factor, "mb_factor", "Multi-band factor"),
		OPTL_DOUBLE(0, "sms_distance", &seq->conf->geom.sms_distance, "sms_distance", "SMS slice distance"),

		// magnetization preparation
		OPTL_SELECT(0, "IR_NON", enum mag_prep, &seq->conf->magn.mag_prep, PREP_IR_NON, "Magn. preparation: Nonselective Inversion (default: off)"),
		OPTL_DOUBLE(0, "TI", &seq->conf->magn.ti, "TI", "Inversion time"),
		OPTL_DOUBLE(0, "init_delay", &seq->conf->magn.init_delay, "init_delay", "Initial delay of measurement"),
		OPTL_DOUBLE(0, "inv_delay", &seq->conf->magn.inv_delay_time, "inv_delay_time", "Inversion delay time"),

		// gradient mode
		OPTL_SELECT(0, "gradient-normal", enum gradient_mode, &gradient_mode, GRAD_NORMAL, "Gradient normal mode (default: fast)"),
		OPTL_SELECT(0, "gradient-whisper", enum gradient_mode, &gradient_mode, GRAD_WHISPER, "Gradient whispher mode (default: fast)"),


		OPTL_VECN(0, "CUSTOM_LONG", custom_params_long, "custom long parameters"),
		OPTL_DOVECN(0, "CUSTOM_DOUBLE", custom_params_double, "custom double parameters"),
		OPTL_VECN(0, "LOOP", seq->conf->loop_dims, "sequence loop dimensions"),
	};

	num_rand_init(0ULL);

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_rand_init(0ULL);

	if (custom_params_long[0] > 0)
		seq_ui_interface_custom_params(0, seq->conf, MAX_PARAMS_LONG, custom_params_long, MAX_PARAMS_DOUBLE, custom_params_double);

	if ((1 == seq->conf->loop_dims[TIME_DIM]) &&
		(seq->conf->loop_dims[TIME_DIM] < seq->conf->loop_dims[PHS1_DIM]) &&
		((PEMODE_RAGA == seq->conf->enc.pe_mode) || (PEMODE_RAGA_ALIGNED == seq->conf->enc.pe_mode))) {

		if (0 < raga_full_frames)
			seq->conf->loop_dims[TIME_DIM] = raga_full_frames * seq->conf->loop_dims[PHS1_DIM];

		if (1 == seq->conf->loop_dims[TIME_DIM]) {

			debug_printf(DP_INFO, "Set total number of spokes to %ld (full frame for RAGA encoding)\n", seq->conf->loop_dims[PHS1_DIM]);
			seq->conf->loop_dims[TIME_DIM] = seq->conf->loop_dims[PHS1_DIM];
		}
	}


	seq_ui_interface_loop_dims(0, seq->conf, DIMS, seq->conf->loop_dims);

	const long total_slices = seq_get_slices(seq->conf);

	if ((0. < fabs(rel_shift[0])) || (0. < fabs(rel_shift[1])) || (0. < fabs(rel_shift[2]))) {

		if ((0. < fabs(seq->conf->geom.shift[0][0])) || (0. < fabs(seq->conf->geom.shift[0][1])) || (0. < fabs(seq->conf->geom.shift[0][2])))
			error("Choose either relative or absolute FOV shift");

		for (int i = 0; i < total_slices; i++) {

			seq->conf->geom.shift[i][0] = rel_shift[0] * seq->conf->geom.fov;
			seq->conf->geom.shift[i][1] = rel_shift[1] * seq->conf->geom.fov;
			seq->conf->geom.shift[i][2] = rel_shift[2] * seq->conf->geom.slice_thickness;
		}
	}

	if ((1 < total_slices) && (0. < dist)) {

		float shift[total_slices][3];
		memset(shift, 0, sizeof shift);

		for (int i = 0; i < total_slices; i++) {

			shift[i][0] = seq->conf->geom.shift[0][0];
			shift[i][1] = seq->conf->geom.shift[0][1];
			shift[i][2] = (i - 0.5 * (total_slices - 1)) * dist * seq->conf->geom.slice_thickness;
		}

		seq_set_fov_pos(total_slices, 3, &shift[0][0], seq->conf);

		debug_printf(DP_INFO, "slice shifts:\n\t%d %f \t\n", 0, seq->conf->geom.shift[0][2]);
		for (int i = 1; i < total_slices; i++)
			debug_printf(DP_INFO, "\t%d: %f \n", i, seq->conf->geom.shift[i][2]);
		debug_printf(DP_INFO, "\n");
	}

	if (0 > samples)
		samples = (0. > dt) ? 1000 : (seq->conf->phys.tr / dt);

	double ddt = (0 > dt) ? seq->conf->phys.tr / samples : ceil(dt * 1.e6) / 1.e6; //FIXME breaks with float

	if ((PEMODE_RAGA != seq->conf->enc.pe_mode) && (PEMODE_RAGA_ALIGNED != seq->conf->enc.pe_mode))
		chrono = true;

	if ((NULL != raga_file) && (((PEMODE_RAGA != seq->conf->enc.pe_mode) && (PEMODE_RAGA_ALIGNED != seq->conf->enc.pe_mode)) || chrono))
		error("RAGA indices only for raga pe mode and non chronologic mode\n");

	// FIXME, this should be moved in system configurations
	switch (gradient_mode) {

	case GRAD_NORMAL:
		seq->conf->sys.grad.max_amplitude = 22.E-3;
		seq->conf->sys.grad.inv_slew_rate = 10.E-3 * sqrt(2.);
		break;

	case GRAD_WHISPER:
		seq->conf->sys.grad.max_amplitude = 22.E-3;
		seq->conf->sys.grad.inv_slew_rate = 20.E-3 * sqrt(2.);
		break;

	case GRAD_FAST:
		break;
	}


	debug_printf(DP_INFO, "loops: %ld \t dims: ", md_calc_size(DIMS, seq->conf->loop_dims));
	debug_print_dims(DP_INFO, DIMS, seq->conf->loop_dims);

	long kernel_dims[DIMS];
	md_select_dims(DIMS, ~(COEFF_FLAG | COEFF2_FLAG | ITER_FLAG), kernel_dims, seq->conf->loop_dims);

	debug_printf(DP_INFO, "kernels: %ld \t dims: ", md_calc_size(DIMS, kernel_dims));
	debug_print_dims(DP_INFO, DIMS, kernel_dims);

	long mdims[DIMS];
	md_select_dims(DIMS, ~TE_FLAG, mdims, kernel_dims);

	int E = 0;

	if (support) {

		seq->state->mode = BLOCK_KERNEL_PREPARE;

		E = seq_block(seq->N, seq->event, seq->state, seq->conf);

		seq->state->mode = BLOCK_UNDEFINED;

		for (int i = 0; i < DIMS; i++)
			seq->state->pos[i] = 0;

		assert(NULL == mom_file);
	}

	mdims[PHS2_DIM] *= mdims[PHS1_DIM];
	mdims[PHS1_DIM] = support ? events_counter(SEQ_EVENT_GRADIENT, E, seq->event) : samples;
	mdims[READ_DIM] = support ? 6 : 3;

	double g2[samples][mdims[READ_DIM]];
	float m0[samples][3];

	long mstrs[DIMS];
	md_calc_strides(DIMS, mstrs, mdims, CFL_SIZE);

	long adims[DIMS];
	md_copy_dims(DIMS, adims, kernel_dims);

	adims[PHS2_DIM] *= adims[PHS1_DIM]; // consistency with traj tool
	adims[PHS1_DIM] = seq->conf->geom.baseres * seq->conf->phys.os;
	adims[READ_DIM] = 5;

	long adc_dims[DIMS];
	md_select_dims(DIMS, (READ_FLAG | PHS1_FLAG | TE_FLAG), adc_dims, adims);

	long adc_strs[DIMS];
	md_calc_strides(DIMS, adc_strs, adc_dims, CFL_SIZE);

	long astrs[DIMS];
	md_calc_strides(DIMS, astrs, adims, CFL_SIZE);

	long ind_dims[DIMS];
	md_select_dims(DIMS, ~(READ_FLAG | PHS1_FLAG), ind_dims, adims);

	long ind_strs[DIMS];
	md_calc_strides(DIMS, ind_strs, ind_dims, CFL_SIZE);

	complex float* out_grad = NULL;
	complex float* out_mom = NULL;
	complex float* out_adc = NULL;
	complex float* out_raga = NULL;

	if (NULL != grad_file)
		out_grad = create_cfl(grad_file, DIMS, mdims);

	if (NULL != mom_file)
		out_mom = create_cfl(mom_file, DIMS, mdims);

	if (NULL != adc_file)
		out_adc = create_cfl(adc_file, DIMS, adims);

	if (NULL != raga_file)
		out_raga = create_cfl(raga_file, DIMS, ind_dims);

	struct pulseq ps;

	int prepped_rfs = bart_seq_prepare(seq);
	
	if (0 > prepped_rfs)
		error("Sequence preparation failed! - check seq_config, %d] \n", prepped_rfs);

	debug_printf(DP_INFO, "Nr. of RF shapes: %d\n", prepped_rfs);

	for (int i = 0; i < prepped_rfs; i++) {

		double s = seq_pulse_scaling(&seq->rf_shape[i]);
		double n = seq_pulse_norm_sum(&seq->rf_shape[i]);

		debug_printf(DP_DEBUG3, "RF pulse %d: scale = %f, sum = %f\n", i, s, n);
	}

	if (NULL != seq_file) {

		pulseq_init(&ps, seq->conf);
		pulse_shapes_to_pulseq(&ps, prepped_rfs, seq->rf_shape);
	}

	do {
		debug_print_dims(DP_DEBUG2, DIMS, seq->state->pos);

		E = seq_block(seq->N, seq->event, seq->state, seq->conf);

		if (0 < E)
			debug_printf(DP_DEBUG2, "block mode: %d ; E: %d \n", seq->state->mode, E);


		if (0 > E)
			error("Sequence not possible! - check seq_config, %d] \n", E);

		if ((BLOCK_KERNEL_NOISE == seq->state->mode) || (0 == E)) // no noise_scan with pulseq
			goto debug_print_events;

		if (NULL != seq_file)
			events_to_pulseq(&ps, seq->state->mode, seq->conf->phys.tr, seq->conf->sys, prepped_rfs, seq->rf_shape, E, seq->event);

		if (BLOCK_KERNEL_IMAGE != seq->state->mode)
			goto debug_print_events;

		debug_printf(DP_DEBUG1, "end of last event: %.8f \t end of calc: %.8f\n",
				events_end_time(E, seq->event, 1, 0), samples * ddt);

		if (support)
			gradients_support(samples, g2, E, seq->event);
		else
			seq_compute_gradients(samples, g2, ddt, E, seq->event);

		compute_moment0(samples, m0, ddt, E, seq->event);


		long pos_save[DIMS];
		md_copy_dims(DIMS, pos_save, seq->state->pos);

		if (!chrono) {

			int adc_idx = events_idx(0, SEQ_EVENT_ADC, E, seq->event);

			if (0 > adc_idx)
				error("No ADC found - try chronologic ordering");

			if (NULL != out_raga) {

				pos_save[PHS2_DIM] = pos_save[PHS2_DIM] * seq->conf->loop_dims[PHS1_DIM] + pos_save[PHS1_DIM];
				pos_save[PHS1_DIM] = 0;
				MD_ACCESS(DIMS, ind_strs, pos_save, out_raga) = seq->event[adc_idx].adc.pos[PHS1_DIM];
			}

			md_copy_dims(DIMS, pos_save, seq->event[adc_idx].adc.pos);
		}

		pos_save[PHS2_DIM] = pos_save[PHS2_DIM] * seq->conf->loop_dims[PHS1_DIM] + pos_save[PHS1_DIM];
		pos_save[PHS1_DIM] = 0;

		do {
			if (NULL != out_grad)
				MD_ACCESS(DIMS, mstrs, pos_save, out_grad) = g2[pos_save[PHS1_DIM]][pos_save[READ_DIM]];

			if (NULL != out_mom)
				MD_ACCESS(DIMS, mstrs, pos_save, out_mom) = m0[pos_save[PHS1_DIM]][pos_save[READ_DIM]];

		} while (md_next(DIMS, mdims, (READ_FLAG | PHS1_FLAG), pos_save));

		if (NULL != out_adc) {

			complex float* adc = md_alloc(DIMS, adc_dims, CFL_SIZE);

			compute_adc_samples(DIMS, adc_dims, adc, E, seq->event);

			double m0_adc[3];

			float scale = seq->conf->phys.dwell * ro_amplitude(seq->conf);

			do {
				assert(0 == pos_save[READ_DIM]);

				moment_sum(m0_adc, MD_ACCESS(DIMS, adc_strs, pos_save, adc), E, seq->event);

				for (int i = 0; i < 3; i++)
					m0_adc[i] = m0_adc[i] / scale;

				MD_ACCESS(DIMS, astrs, (pos_save[READ_DIM] = 0, pos_save), out_adc) = m0_adc[0];
				MD_ACCESS(DIMS, astrs, (pos_save[READ_DIM] = 1, pos_save), out_adc) = m0_adc[1];
				MD_ACCESS(DIMS, astrs, (pos_save[READ_DIM] = 2, pos_save), out_adc) = m0_adc[2];

				MD_ACCESS(DIMS, astrs, (pos_save[READ_DIM] = 3, pos_save), out_adc) = MD_ACCESS(DIMS, adc_strs, (pos_save[READ_DIM] = 0, pos_save), adc);
				MD_ACCESS(DIMS, astrs, (pos_save[READ_DIM] = 4, pos_save), out_adc) = MD_ACCESS(DIMS, adc_strs, (pos_save[READ_DIM] = 1, pos_save), adc);

				pos_save[READ_DIM] = 0;

			} while (md_next(DIMS, adims, PHS1_FLAG | TE_FLAG, pos_save));

			md_free(adc);
		}

debug_print_events:
		linearize_events(E, seq->event, &seq->state->start_block, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad);

		for (int i = 0; i < E; i++) {

			debug_printf(DP_DEBUG3, "event[%d]:\t%.8f\t\t%.8f\t\t%.8f\t\t", i,
					seq->event[i].start, seq->event[i].mid, seq->event[i].end);

			switch (seq->event[i].type) {

			case SEQ_EVENT_GRADIENT:

				debug_printf(DP_DEBUG3, "||\t%.5f\t\t%.5f\t\t%.5f", seq->event[i].grad.ampl[0], seq->event[i].grad.ampl[1],seq->event[i].grad.ampl[2]);
				break;

			case SEQ_EVENT_PULSE:

				debug_printf(DP_DEBUG3, "|| SEQ_EVENT_PULSE \t freq: %.2f\t\t phase: %.2f", seq->event[i].pulse.freq, seq->event[i].pulse.phase);
				break;

			case SEQ_EVENT_ADC:

				debug_printf(DP_DEBUG3, "|| SEQ_EVENT_ADC \t freq: %.2f\t\t phase: %.2f", seq->event[i].adc.freq, seq->event[i].adc.phase);
				break;

			default:

			}

			debug_printf(DP_DEBUG3, "\n");
		}

		debug_printf(DP_DEBUG3, "seq_block_end_flat: %f\n", seq_block_end_flat(E, seq->event, seq->conf->sys.raster_grad));


	} while (seq_continue(seq->state, seq->conf));

	if (NULL != grad_file)
		unmap_cfl(DIMS, mdims, out_grad);

	if (NULL != mom_file)
		unmap_cfl(DIMS, mdims, out_mom);

	if (NULL != adc_file)
		unmap_cfl(DIMS, adims, out_adc);

	if (NULL != raga_file)
		unmap_cfl(DIMS, ind_dims, out_raga);

	if (NULL != seq_file) {

		FILE *fp = fopen(seq_file, "w+");
		if (NULL == fp)
			error("Opening file for .seq");
		pulseq_writef(fp, &ps);
		fclose(fp);
	}

	bart_seq_free(seq);

	return 0;
}

