/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
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

		ARG_OUTFILE(false, &grad_file, "gradients (x,y,z)"),
		ARG_OUTFILE(false, &mom_file, "0th moment (x,y,z)"),
		ARG_OUTFILE(false, &adc_file, "0th moment (x,y,z) at sample points, sample_points, phase of adc"),
		ARG_OUTFILE(false, &seq_file,  "pulseq file"),
	};

	float dt = -1.;
	long samples = -1;
	float dist = 1.;

	struct seq_state seq_state = { };
	struct seq_config seq = seq_config_defaults;

	enum gradient_mode gradient_mode = GRAD_FAST;

	seq.enc.order = SEQ_ORDER_AVG_OUTER;

	bool chrono = false;
	bool support = false;

	const struct opt_s opts[] = {

		OPT_FLOAT('d', &dt, "dt", "time-increment per sample (default: seq.phys.tr / 1000)"),
		OPT_LONG('N', &samples, "samples", "Number of samples (default: 1000)"),

		OPT_DOVEC3('s', &seq.geom.shift[0], "RO:PE:SL", "FOV shift (mm) of first slice"),
		OPTL_FLOAT(0, "dist", &dist, "dist", "slice distance factor [1 / slice_thickness] (default: 1.)"),

		// contrast mode
		OPTL_SELECT(0, "no-spoiling", enum flash_contrast, &seq.phys.contrast, CONTRAST_NO_SPOILING, "spoiling off (default: rf random)"),
		OPTL_SELECT(0, "spoiled", enum flash_contrast, &seq.phys.contrast, CONTRAST_RF_SPOILED, "RF_SPOILED (inc: 50 deg, gradient on) (default: rf random)"),

		// FOV and resolution
		OPTL_DOUBLE(0, "FOV", &seq.geom.fov, "FOV", "Field Of View [mm]"),
		OPTL_PINT(0, "BR", &seq.geom.baseres, "BR", "Base Resolution"),
		OPTL_DOUBLE(0, "slice_thickness", &seq.geom.slice_thickness, "slice_thickness", "Slice thickness [mm]"),

		// basic sequence parameters
		OPTL_DOUBLE(0, "FA", &seq.phys.flip_angle, "flip angle", "Flip angle [deg]"),
		OPTL_DOUBLE(0, "TR", &seq.phys.tr, "TR", "TR [us]"),
		OPTL_DOUBLE(0, "TE", &seq.phys.te, "TE", "TE [us]"),
		OPTL_DOUBLE(0, "BWTP", &seq.phys.bwtp, "BWTP", "Bandwidth Time Product"),

		// others sequence parameters
		OPTL_DOUBLE(0, "rf_duration", &seq.phys.rf_duration, "rf_duration", "RF pulse duration [us]"),
		OPTL_DOUBLE(0, "dwell", &seq.phys.dwell, "dwell", "Dwell time [us]"),
		OPTL_DOUBLE(0, "os", &seq.phys.os, "os", "Oversampling factor"),

		// encoding
		OPTL_UINT(0, "pe_mode", &seq.enc.pe_mode, "pe_mode", "Phase-encoding mode"),
		OPTL_SELECT(0, "turn", enum pe_mode, &seq.enc.pe_mode, PEMODE_TURN, "turn-based PE (default: RAGA)"),
		OPTL_SELECT(0, "raga", enum pe_mode, &seq.enc.pe_mode, PEMODE_RAGA, "RAGA PE"),
		OPTL_SELECT(0, "raga_al", enum pe_mode, &seq.enc.pe_mode, PEMODE_RAGA_ALIGNED, "RAGA-aligned PE (default: RAGA)"),

		OPTL_SET(0, "chrono", &chrono, "save gradients/moments/sampling in chronological order (RAGA)"),
		OPT_OUTFILE('R', &raga_file, "file", "raga indices"),

		OPTL_PINT(0, "tiny", &seq.enc.tiny, "tiny", "Tiny golden-ratio index"),

		OPTL_LONG('r', "lines", &seq.loop_dims[PHS1_DIM], "lines", "Number of phase encoding lines"),
		OPTL_LONG('z', "partitions", &seq.loop_dims[PHS2_DIM], "partitions", "Number of partitions (3D) or SMS groups (2D)"),
		OPTL_LONG('t', "measurements", &seq.loop_dims[TIME_DIM], "measurements", "Number of measurements / frames (RAGA: total number of spokes)"),
		OPTL_LONG('m', "slices", &seq.loop_dims[SLICE_DIM], "slices", "Number of slices of multiband factor (SMS)"),
		OPTL_LONG('i', "inversions", &seq.loop_dims[BATCH_DIM], "inversions", "Number of inversions"),

		// order
		OPTL_SELECT(0, "sequential-multislice", enum seq_order, &seq.enc.order, SEQ_ORDER_SEQ_MS, "seq_order: sequential multislice (default: avg outer)"),
		OPTL_SELECT(0, "avg-inner", enum seq_order, &seq.enc.order, SEQ_ORDER_AVG_INNER, "seq_order: average inner (default: avg outer)"),

		// sms
		OPTL_PINT(0, "mb_factor", &seq.geom.mb_factor, "mb_factor", "Multi-band factor"),
		OPTL_DOUBLE(0, "sms_distance", &seq.geom.sms_distance, "sms_distance", "SMS slice distance [mm]"),

		// magnetization preparation
		OPTL_SELECT(0, "IR_NON", enum mag_prep, &seq.magn.mag_prep, PREP_IR_NON, "Magn. preparation: Nonselective Inversion (default: off)"),
		OPTL_DOUBLE(0, "TI", &seq.magn.ti, "TI", "Inversion time [us]"),
		OPTL_DOUBLE(0, "init_delay", &seq.magn.init_delay_sec, "init_delay_sec", "Initial delay of measurement (seconds)"),
		OPTL_DOUBLE(0, "inv_delay", &seq.magn.inv_delay_time_sec, "inv_delay_time_sec", "Inversion delay time (seconds)"),

		// gradient mode
		OPTL_SELECT(0, "gradient-normal", enum gradient_mode, &gradient_mode, GRAD_NORMAL, "Gradient normal mode (default: fast)"),
		OPTL_SELECT(0, "gradient-whisper", enum gradient_mode, &gradient_mode, GRAD_WHISPER, "Gradient whispher mode (default: fast)"),

		OPTL_SET(0, "support", &support, "save support points of gradient"),
	};

	num_rand_init(0ULL);

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_rand_init(0ULL);

	if ((1 == seq.loop_dims[TIME_DIM]) &&
		(seq.loop_dims[TIME_DIM] < seq.loop_dims[PHS1_DIM]) &&
		((PEMODE_RAGA == seq.enc.pe_mode) || (PEMODE_RAGA_ALIGNED == seq.enc.pe_mode))) {

		debug_printf(DP_INFO, "Set total number of spokes to %ld (full frame for RAGA encoding)\n", seq.loop_dims[PHS1_DIM]);
		seq.loop_dims[TIME_DIM] = seq.loop_dims[PHS1_DIM];
	}


	set_loop_dims_and_sms(&seq, seq.loop_dims[PHS2_DIM], seq.loop_dims[SLICE_DIM], seq.loop_dims[PHS1_DIM],
				seq.loop_dims[TIME_DIM], seq.loop_dims[TE_DIM], 1, 1);


	const long total_slices = get_slices(&seq);

	if ((1 < total_slices) && (0. < dist)) {

		float shift[total_slices][3];
		memset(shift, 0, sizeof shift);

		for (int i = 0; i < total_slices; i++) {

			shift[i][0] = seq.geom.shift[i][0];
			shift[i][1] = seq.geom.shift[i][1];
			shift[i][2] = (i - 0.5 * (total_slices - 1)) * dist * seq.geom.slice_thickness;
		}

		set_fov_pos(total_slices, 3, &shift[0][0], &seq);

		debug_printf(DP_INFO, "slice shifts [mm]:\n\t%d %f \t\n", 0, seq.geom.shift[0][2]);
		for (int i = 1; i < total_slices; i++)
			debug_printf(DP_INFO, "\t%d: %f \n", i, seq.geom.shift[i][2]);
		debug_printf(DP_INFO, "\n");
	}

	if (0 > samples)
		samples = (0. > dt) ? 1000 : (seq.phys.tr / dt);

	double ddt = (0 > dt) ? seq.phys.tr / samples : ceil(dt * 1.e6) / 1.e6; //FIXME breaks with float

	if ((PEMODE_RAGA != seq.enc.pe_mode) && (PEMODE_RAGA_ALIGNED != seq.enc.pe_mode))
		chrono = true;

	if ((NULL != raga_file) && (((PEMODE_RAGA != seq.enc.pe_mode) && (PEMODE_RAGA_ALIGNED != seq.enc.pe_mode)) || chrono))
		error("RAGA indices only for raga pe mode and non chronologic mode\n");

	// FIXME, this should be moved in system configurations
	switch (gradient_mode) {

	case GRAD_NORMAL:
		seq.sys.grad.max_amplitude = 22.;
		seq.sys.grad.inv_slew_rate = 10 * sqrt(2.);
		break;

	case GRAD_WHISPER:
		seq.sys.grad.max_amplitude = 22.;
		seq.sys.grad.inv_slew_rate = 20 * sqrt(2.);
		break;

	case GRAD_FAST:
		break;
	}


	struct seq_event ev[MAX_EVENTS];

	debug_printf(DP_INFO, "loops: %ld \t dims: ", md_calc_size(DIMS, seq.loop_dims));
	debug_print_dims(DP_INFO, DIMS, seq.loop_dims);

	long kernel_dims[DIMS];
	md_select_dims(DIMS, ~(COEFF_FLAG | COEFF2_FLAG | ITER_FLAG), kernel_dims, seq.loop_dims);

	debug_printf(DP_INFO, "kernels: %ld \t dims: ", md_calc_size(DIMS, kernel_dims));
	debug_print_dims(DP_INFO, DIMS, kernel_dims);

	long mdims[DIMS];
	md_select_dims(DIMS, ~TE_FLAG, mdims, kernel_dims);

	int E = 0;

	if (support) {

		seq_state.mode = BLOCK_KERNEL_PREPARE;

		E = seq_block(MAX_EVENTS, ev, &seq_state, &seq);

		seq_state.mode = BLOCK_UNDEFINED;

		for (int i = 0; i < DIMS; i++)
			seq_state.pos[i] = 0;

		assert(NULL == mom_file);
	}

	mdims[PHS2_DIM] *= mdims[PHS1_DIM];
	mdims[PHS1_DIM] = support ? events_counter(SEQ_EVENT_GRADIENT, E, ev) : samples;
	mdims[READ_DIM] = support ? 6 : 3;

	double g2[samples][mdims[READ_DIM]];
	float m0[samples][3];

	long mstrs[DIMS];
	md_calc_strides(DIMS, mstrs, mdims, CFL_SIZE);

	long adims[DIMS];
	md_copy_dims(DIMS, adims, kernel_dims);

	adims[PHS2_DIM] *= adims[PHS1_DIM]; // consistency with traj tool
	adims[PHS1_DIM] = seq.geom.baseres * seq.phys.os;
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
	struct rf_shape rf_shapes[MAX_RF_PULSES];
	int rfs = seq_sample_rf_shapes(MAX_RF_PULSES, rf_shapes, &seq);

	debug_printf(DP_INFO, "Nr. of RF shapes: %d\n", rfs);

	for (int i = 0; i < rfs; i++) {

		double s = idea_pulse_scaling(&rf_shapes[i]);
		double n = idea_pulse_norm_sum(&rf_shapes[i]);

		debug_printf(DP_DEBUG3, "RF pulse %d: scale = %f, sum = %f\n", i, s, n);
	}

	if (NULL != seq_file) {

		pulseq_init(&ps, &seq);
		pulse_shapes_to_pulseq(&ps, rfs, rf_shapes);
	}

	do {
		debug_print_dims(DP_DEBUG2, DIMS, seq_state.pos);

		E = seq_block(MAX_EVENTS, ev, &seq_state, &seq);

		if (0 < E)
			debug_printf(DP_DEBUG2, "block mode: %d ; E: %d \n", seq_state.mode, E);


		if (0 > E)
			error("Sequence not possible! - check seq_config, %d] \n", E);

		if ((BLOCK_KERNEL_NOISE == seq_state.mode) || (0 == E)) // no noise_scan with pulseq
			goto debug_print_events;

		if (NULL != seq_file)
			events_to_pulseq(&ps, seq_state.mode, seq.phys.tr, seq.sys, rfs, rf_shapes, E, ev);

		if (BLOCK_KERNEL_IMAGE != seq_state.mode)
			goto debug_print_events;

		debug_printf(DP_DEBUG1, "end of last event: %.2f \t end of calc: %.2f\n",
				events_end_time(E, ev, 1, 0), samples * ddt);

		if (support)
			gradients_support(samples, g2, E, ev);
		else
			seq_compute_gradients(samples, g2, ddt, E, ev);

		compute_moment0(samples, m0, ddt, E, ev);


		long pos_save[DIMS];
		md_copy_dims(DIMS, pos_save, seq_state.pos);


		if (!chrono) {

			int adc_idx = events_idx(0, SEQ_EVENT_ADC, E, ev);

			if (0 > adc_idx)
				error("No ADC found - try chronologic ordering");

			if (NULL != out_raga) {

				pos_save[PHS2_DIM] = pos_save[PHS2_DIM] * seq.loop_dims[PHS1_DIM] + pos_save[PHS1_DIM];
				pos_save[PHS1_DIM] = 0;
				MD_ACCESS(DIMS, ind_strs, pos_save, out_raga) = ev[adc_idx].adc.pos[PHS1_DIM];
			}

			md_copy_dims(DIMS, pos_save, ev[adc_idx].adc.pos);
		}

		pos_save[PHS2_DIM] = pos_save[PHS2_DIM] * seq.loop_dims[PHS1_DIM] + pos_save[PHS1_DIM];
		pos_save[PHS1_DIM] = 0;

		do {
			if (NULL != out_grad)
				MD_ACCESS(DIMS, mstrs, pos_save, out_grad) = g2[pos_save[PHS1_DIM]][pos_save[READ_DIM]];

			if (NULL != out_mom)
				MD_ACCESS(DIMS, mstrs, pos_save, out_mom) = m0[pos_save[PHS1_DIM]][pos_save[READ_DIM]];

		} while (md_next(DIMS, mdims, (READ_FLAG | PHS1_FLAG), pos_save));

		if (NULL != out_adc) {

			complex float* adc = md_alloc(DIMS, adc_dims, CFL_SIZE);

			compute_adc_samples(DIMS, adc_dims, adc, E, ev);

			double m0_adc[3];

			float scale = seq.phys.dwell * ro_amplitude(&seq);

			do {
				assert(0 == pos_save[READ_DIM]);

				moment_sum(m0_adc, MD_ACCESS(DIMS, adc_strs, pos_save, adc), E, ev);

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
		linearize_events(E, ev, &seq_state.start_block, seq_state.mode, seq.phys.tr);

		for (int i = 0; i < E; i++) {

			debug_printf(DP_DEBUG3, "event[%d]:\t%.2f\t\t%.2f\t\t%.2f\t\t", i,
					ev[i].start, ev[i].mid, ev[i].end);

			if (SEQ_EVENT_GRADIENT == ev[i].type)
				debug_printf(DP_DEBUG3, "||\t%.2f\t\t%.2f\t\t%.2f", ev[i].grad.ampl[0], ev[i].grad.ampl[1],ev[i].grad.ampl[2]);

			debug_printf(DP_DEBUG3, "\n");
		}

		debug_printf(DP_DEBUG3, "seq_block_end_flat: %ld\n", seq_block_end_flat(E, ev));


	} while (seq_continue(&seq_state, &seq));

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

	return 0;
}

