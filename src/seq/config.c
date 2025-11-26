/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/mri.h"

#include "config.h"

const int seq_loop_order_avg_inner[DIMS] = {
	AVG_DIM,
	PHS2_DIM,
	SLICE_DIM,
	COEFF2_DIM,
	COEFF_DIM,
	PHS1_DIM,
	TIME2_DIM,
	TIME_DIM,
	BATCH_DIM,

	READ_DIM,
	COIL_DIM,
	MAPS_DIM,
	TE_DIM,
	ITER_DIM,
	CSHIFT_DIM,
	LEVEL_DIM
};


const int seq_loop_order_avg_outer[DIMS] = {
	PHS2_DIM,
	SLICE_DIM,
	COEFF2_DIM,
	COEFF_DIM,
	PHS1_DIM,
	AVG_DIM,
	TIME2_DIM,
	TIME_DIM,
	BATCH_DIM,

	READ_DIM,
	COIL_DIM,
	MAPS_DIM,
	TE_DIM,
	ITER_DIM,
	CSHIFT_DIM,
	LEVEL_DIM
};

const int seq_loop_order_multislice[DIMS] = {
	PHS2_DIM,
	COEFF2_DIM,
	COEFF_DIM,
	PHS1_DIM,
	AVG_DIM,
	TIME2_DIM,
	TIME_DIM,
	BATCH_DIM,

	SLICE_DIM,
	READ_DIM,
	COIL_DIM,
	MAPS_DIM,
	TE_DIM,
	ITER_DIM,
	CSHIFT_DIM,
	LEVEL_DIM
};


const struct seq_config seq_config_defaults = {

	.phys = {
		.tr = 3110.,
		.te = 1900.,
		.dwell = 4.0,
		.os = 2.,
		.contrast = CONTRAST_RF_RANDOM,
		.rf_duration = 620.,
		.flip_angle = 6.,
		.bwtp = 3.8,
	},

	.geom = {
		.fov = 256,
		.slice_thickness = 6.,
		.shift = { [0 ... MAX_SLICES - 1] = { 0., 0., 0. } },
		.baseres = 256,
		.mb_factor = 1,
		.sms_distance = 20.,
	},

	.enc = {
		.pe_mode = PEMODE_RAGA,
		.tiny = 1,
		.aligned_flags = 0,
	},

	.magn = {
		.mag_prep = PREP_OFF,
		.ti = 0.,
		.init_delay_sec = 0.,
		.inv_delay_time_sec = 0.,
	},

	.sys = {
		.gamma = 42.575575,
		.b0 = 2.893620,
		.grad.inv_slew_rate = 7.848885540911,
		.grad.max_amplitude = 24.,
		.coil_control_lead = 100.,
		.min_duration_ro_rf = 213.,
		.raster_grad = 10.,
		.raster_rf = 1.,
		.raster_dwell = .1,
	},

	.order = { [0 ... DIMS - 1] = 1 },
	.loop_dims = { [0 ... DIMS - 1] = 1 },
};


