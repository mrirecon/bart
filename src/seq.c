/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Daniel Mackner
 */

#include <complex.h>
#include <math.h>

#include "num/multind.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/opts.h"
#include "misc/version.h"

#include "seq/config.h"
#include "seq/event.h"
#include "seq/seq.h"

#include "seq/adc_rf.h"
#include "seq/misc.h"
#include "seq/flash.h"
#include "seq/kernel.h"


#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char help_str[] = "Computes sequence.";


int main_seq(int argc, char* argv[argc])
{
	const char* grad_file = NULL;
	const char* mom_file = NULL;
	const char* adc_file = NULL;

	struct arg_s args[] = {

		ARG_OUTFILE(false, &grad_file, "gradients (x,y,z)"),
		ARG_OUTFILE(false, &mom_file, "0. moment (x,y,z)"),
		ARG_OUTFILE(false, &adc_file,  "phase of adc, and 0. moment (x,y,z) at sample points"),
	};

	float dt = -1.;
	long samples = -1;
	long single_idx = -1;
	bool unchecked_gr = false;
	const char* seq_hash = NULL;

	struct seq_state seq_state = { 0 };;
	struct seq_config seq = seq_config_defaults;
	bool norm_k = false;
	bool gradient_normal_mode = false;
	bool gradient_whisper_mode = false;

	bool support = false;

	const struct opt_s opts[] = {

		OPT_INT('l', &debug_level, "level", "Debug level"),
		OPT_FLOAT('d', &dt, "dt", "time-increment per sample (default: seq.phys.tr / 1000)"),
		OPT_LONG('N', &samples, "samples", "Number of samples (default: 1000)"),
		OPT_LONG('I', &single_idx, "samples", "Single kernel call at index"),
		OPTL_SET(0, "no_gradient_check", &unchecked_gr, "Disable check for complete gradient calculation - gradient can be longer than TR"),
		OPT_STRING('V', &seq_hash, "seq", "reproducibility check"),

		// OPT_FLVEC3('s', &seq.geom.shift[0], "RO:PE:SL", "FOV shift (mm)"),
		OPTL_SET(0, "norm-kspace", &norm_k, "Normalize kspace with 2*ro_amplitude"),

		OPTL_DOUBLE(0, "fov", &seq.geom.fov, "fov", "Field of view [mm]"),
		OPTL_PINT(0, "br", &seq.geom.baseres, "br", "Base resolution"),
		OPTL_DOUBLE(0, "dwell", &seq.phys.dwell, "dwell", "Dwell time [us]"),
		OPTL_DOUBLE(0, "os", &seq.phys.os, "os", "Oversampling factor"),

		OPTL_DOUBLE(0, "rf_duration", &seq.phys.rf_duration, "rf_duration", "RF pulse duration [us]"),
		OPTL_DOUBLE(0, "flip_angle", &seq.phys.flip_angle, "flip_angle", "Flip angle [deg]"),
		OPTL_PINT(0, "mb_factor", &seq.geom.mb_factor, "mb_factor", "Multi-band factor"),
		OPTL_DOUBLE(0, "bwtp", &seq.phys.bwtp, "bwtp", "Bandwidth Time Product"),
		OPTL_DOUBLE(0, "slice_thickness", &seq.geom.slice_thickness, "slice_thickness", "Slice thickness [mm]"),
		OPTL_DOUBLE(0, "sms_distance", &seq.geom.sms_distance, "sms_distance", "SMS slice distance [mm]"),

		OPTL_UINT(0, "pe_mode", &seq.enc.pe_mode, "pe_mode", "Phase-encoding mode"),
		OPTL_PINT(0, "tiny", &seq.enc.tiny, "turns", "Turns or GA index"),

		OPTL_UINT(0, "mag_prep", &seq.magn.mag_prep, "mag_prep", "Magn. preparation [OFF, IR_NON]"),
		OPTL_DOUBLE(0, "ti", &seq.magn.ti, "ti", "Inversion time [us]"),
		OPTL_UINT(0, "contrast", &seq.phys.contrast, "spoiling", "Spoiling [RF_RANDOM]"),

		OPTL_DOUBLE(0, "tr", &seq.phys.tr, "tr", "TR [us]"),
		OPTL_DOUBLE(0, "te", &seq.phys.te, "te0", "TE [us]"),

		OPTL_LONG('r', "lines", &seq.loop_dims[PHS1_DIM], "lines", "Number of phase encoding lines"),
		OPTL_LONG('z', "partitions", &seq.loop_dims[PHS2_DIM], "partitions", "Number of partitions (3D) or SMS groups (2D)"),
		OPTL_LONG('t', "measurements", &seq.loop_dims[TIME_DIM], "measurements", "Number of measurements / frames"),
		OPTL_LONG('m', "slices", &seq.loop_dims[SLICE_DIM], "slices", "Number of slices of multiband factor (SMS)"),
		OPTL_LONG('i', "inversions", &seq.loop_dims[BATCH_DIM], "inversions", "Number of inversions"),
		OPTL_SET(0, "gradient-normal", &gradient_normal_mode, "Gradient normal mode (default: fast)"),
		OPTL_SET(0, "gradient-whisper", &gradient_whisper_mode, "Gradient whisper mode (default: fast)"),

		OPTL_SET(0, "support", &support, "save support points of gradient"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if (unchecked_gr)
		debug_printf(DP_WARN, "End of gradient calculation may end before end of last event!\n");


	if (NULL != seq_hash) {

		unsigned int va[4];
		char bart_hash[8];
		int ret = sscanf(bart_version, "v%u.%u.%u-%u-g%7s", &va[0], &va[1], &va[2], &va[3], bart_hash);

		if ((5 != ret) || strcmp(bart_hash, seq_hash))
			error("failed compatibility check: %s \t %s\n", bart_hash, seq_hash);
	}	

	set_loop_dims_and_sms(&seq, 0, seq.loop_dims[SLICE_DIM], seq.loop_dims[PHS1_DIM],
				seq.loop_dims[TIME_DIM], seq.loop_dims[TE_DIM],
				seq.loop_dims[BATCH_DIM], 1, 1, 1, seq.geom.mb_factor);

	if ((0 > samples) && (0 > dt))
		samples = 1000;
	else if (0 < dt)
		samples = seq.phys.tr / dt;

	assert(!(gradient_normal_mode && gradient_whisper_mode));

	// FIXME, this should be moved in system configurations
	if (gradient_normal_mode)
		seq.sys.grad.inv_slew_rate = 10 * sqrt(2.);
	else if (gradient_whisper_mode)
		seq.sys.grad.inv_slew_rate = 20 * sqrt(2.);

	seq.sys.grad.max_amplitude = (gradient_normal_mode || gradient_whisper_mode) ? 22. : seq.sys.grad.max_amplitude;

	struct seq_event ev[MAX_EVENTS];
	int E;

	debug_printf(DP_INFO, "loops: %ld \t dims: ", md_calc_size(DIMS, seq.loop_dims));
	debug_print_dims(DP_INFO, DIMS, seq.loop_dims);
	long kernel_dims[DIMS];
	md_select_dims(DIMS, ~(COEFF_FLAG | COEFF2_FLAG | ITER_FLAG), kernel_dims, seq.loop_dims);
	debug_printf(DP_INFO, "kernels: %ld \t dims: ", md_calc_size(DIMS, kernel_dims));
	debug_print_dims(DP_INFO, DIMS, kernel_dims);

	if (-1 < single_idx) {

		debug_printf(DP_INFO, "evaluate sequence only at index: %ld\n", single_idx);

		if (single_idx >=  md_calc_size(DIMS, kernel_dims))
			error("Index out of bounds! %ld >= %ld\n", single_idx, md_calc_size(DIMS, kernel_dims));

		md_unravel_index_permuted(DIMS, seq_state.pos, SEQ_FLAGS, kernel_dims, single_idx, seq.order);
	}
		
	long mdims[DIMS];

	if (-1 < single_idx)
		md_singleton_dims(DIMS, mdims);
	else
		md_select_dims(DIMS, ~TE_FLAG, mdims, kernel_dims);

	if (support) {

		seq_state.mode = BLOCK_KERNEL_PREPARE;

		E = seq_block(MAX_EVENTS, ev, &seq_state, &seq);

		seq_state.mode = BLOCK_UNDEFINED;

		for (int i = 0; i < DIMS; i++)
			seq_state.pos[i] = 0;

		assert(NULL == mom_file);
	}

	mdims[READ_DIM] = support ? events_counter(SEQ_EVENT_GRADIENT, E, ev) : samples;
	mdims[MAPS_DIM] = support ? 6 : 3;

	long mstrs[DIMS];
	md_calc_strides(DIMS, mstrs, mdims, CFL_SIZE);

	long adims[DIMS];

	if (-1 < single_idx)
		md_singleton_dims(DIMS, adims);
	else
		md_copy_dims(DIMS, adims, kernel_dims);

	adims[READ_DIM] = seq.geom.baseres * seq.phys.os;
	adims[MAPS_DIM] = 5;

	long adc_dims[DIMS];
	md_select_dims(DIMS, (READ_FLAG | MAPS_FLAG | TE_FLAG), adc_dims, adims);

	long adc_strs[DIMS];
	md_calc_strides(DIMS, adc_strs, adc_dims, CFL_SIZE);

	long astrs[DIMS];
	md_calc_strides(DIMS, astrs, adims, CFL_SIZE);

	complex float* out_grad = NULL;
	complex float* out_mom = NULL;
	complex float* out_adc = NULL;

	if (NULL != grad_file)
		out_grad = create_cfl(grad_file, DIMS, mdims);

	if (NULL != mom_file)
		out_mom = create_cfl(mom_file, DIMS, mdims);
	
	if (NULL != adc_file)
		out_adc = create_cfl(adc_file, DIMS, adims);

	struct rf_shape rf_shapes[MAX_RF_PULSES];
	int rfs = seq_sample_rf_shapes(MAX_RF_PULSES, rf_shapes, &seq);

	debug_printf(DP_INFO, "sampled RF shapes: %d\n", rfs);

	for (int i = 0; i < rfs; i++) {

		double s = idea_pulse_scaling(&rf_shapes[i]);
		double n = idea_pulse_norm_sum(&rf_shapes[i]);

		debug_printf(DP_DEBUG3, "RF pulse %d: scale = %f, sum = %f\n", i, s, n);
	}

	do {
		debug_print_dims(DP_DEBUG2, DIMS, seq_state.pos);

		E = seq_block(MAX_EVENTS, ev, &seq_state, &seq);

		if (0 < E)
			debug_printf(DP_DEBUG2, "block mode: %d ; E: %d \n", seq_state.mode, E);

		if ((BLOCK_KERNEL_IMAGE != seq_state.mode) || (0 == E))
			continue;

		if (0 > E)
			error("Sequence not possible! - check seq_config, %d] \n", E);

		for (int i = 0; i < E; i++) {

			debug_printf(DP_DEBUG3, "event[%d]:\t%.2f\t\t%.2f\t\t%.2f\t\t", i, 
					ev[i].start, ev[i].mid, ev[i].end);

			if (SEQ_EVENT_GRADIENT == ev[i].type)
				debug_printf(DP_DEBUG3, "||\t%.2f\t\t%.2f\t\t%.2f", ev[i].grad.ampl[0], ev[i].grad.ampl[1],ev[i].grad.ampl[2]);

			debug_printf(DP_DEBUG3, "\n");
		}

		debug_printf(DP_DEBUG3, "seq_block_end_flat: %ld\n", seq_block_end_flat(E, ev));

		double ddt = (0 > dt) ? 1. * seq.phys.tr / samples : ceil(dt * 1.e6)/ 1.e6; //FIXME breaks with float
		double g2[samples][mdims[MAPS_DIM]];

		if (unchecked_gr) {

			debug_printf(DP_INFO, "end of last event: %.2f \t end of calc: %.2f\n",
				    events_end_time(E, ev, 1, 0), samples * ddt);
		}

		if (support)
			seq_gradients_support(samples, g2, E, ev);
		else
			seq_compute_gradients(samples, g2, ddt, E, ev);

		float m0[samples][3];
		seq_compute_moment0(samples, m0, ddt, E, ev);


		do {
			if (NULL != out_grad)
				MD_ACCESS(DIMS, mstrs, seq_state.pos, out_grad) = g2[seq_state.pos[READ_DIM]][seq_state.pos[MAPS_DIM]];

			if (NULL != out_mom)
				MD_ACCESS(DIMS, mstrs, seq_state.pos, out_mom) = m0[seq_state.pos[READ_DIM]][seq_state.pos[MAPS_DIM]];

		} while (md_next(DIMS, mdims, (READ_FLAG | MAPS_FLAG), seq_state.pos));

		if (NULL != out_adc) {

			complex float* adc = md_alloc(DIMS, adc_dims, CFL_SIZE);

			seq_compute_adc_samples(DIMS, adc_dims, adc, E, ev);

			double m0_adc[3];

			float scale = 1.;

			if (norm_k)
				scale = (seq.phys.dwell / seq.phys.os) * ro_amplitude(&seq);

			do {
				moment_sum(m0_adc, MD_ACCESS(DIMS, adc_strs, (seq_state.pos[MAPS_DIM] = 0, seq_state.pos), adc), E, ev);
				
				MD_ACCESS(DIMS, astrs, (seq_state.pos[MAPS_DIM] = 0, seq_state.pos), out_adc) = MD_ACCESS(DIMS, adc_strs, (seq_state.pos[MAPS_DIM] = 0, seq_state.pos), adc);
				MD_ACCESS(DIMS, astrs, (seq_state.pos[MAPS_DIM] = 1, seq_state.pos), out_adc) = MD_ACCESS(DIMS, adc_strs, (seq_state.pos[MAPS_DIM] = 1, seq_state.pos), adc);
				MD_ACCESS(DIMS, astrs, (seq_state.pos[MAPS_DIM] = 2, seq_state.pos), out_adc) = m0_adc[0] / scale;
				MD_ACCESS(DIMS, astrs, (seq_state.pos[MAPS_DIM] = 3, seq_state.pos), out_adc) = m0_adc[1] / scale;
				MD_ACCESS(DIMS, astrs, (seq_state.pos[MAPS_DIM] = 4, seq_state.pos), out_adc) = m0_adc[2] / scale;
				seq_state.pos[MAPS_DIM] = 0;

			} while (md_next(DIMS, adims, (READ_FLAG | TE_FLAG), seq_state.pos));

			md_free(adc);
		}

	} while (seq_continue(&seq_state, &seq) && (0 > single_idx));

	if (NULL != grad_file)
		unmap_cfl(DIMS, mdims, out_grad);

	if (NULL != mom_file)
		unmap_cfl(DIMS, mdims, out_mom);
		
	if (NULL != adc_file)
		unmap_cfl(DIMS, adims, out_adc);

	return 0;
}


