/* Copyright 2019-2020. Uecker Lab, University Medical Center Goettingen.
 * Copyright 2022. Institute of Biomedical Imaging. Graz University of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 **/
 
#include <stdbool.h>

#include "moba/meco.h"
#include "misc/debug.h"

#include "moba.h"

struct moba_conf moba_defaults = {

	.mode = MDB_T1,

	.iter = 8,
	.opt_reg = 1.,
	.alpha = 1.,
	.alpha_min = 0.,
	.alpha_min_exp_decay = true,
	.redu = 2.,
	.step = 0.9,
	.lower_bound = 0.,
	.tolerance = 0.01,
	.damping = 0.9,
	.inner_iter = 250,
	.sobolev_a = 880.f,
	.sobolev_b = 32.f,
	.noncartesian = false,
	.sms = false,
        .k_filter = false,
	.k_filter_type = EF1,
	.auto_norm = false,
	.algo = 3,
	.rho = 0.01,
	.stack_frames = false,

	// T1
	.scaling_M0 = 1.,
	.scaling_R1s = 1.,

	// MECO
	.mgre_model = MECO_WFR2S,
	.fat_spec = FAT_SPEC_1,
	.scale_fB0 = { 222., 1. },
	.out_origin_maps = false,

	.use_gpu = false,
};


struct moba_other_conf moba_other_defaults = {

        .fov_reduction_factor = 1.,
        .scale = { 1., 1., 1., 1. },
        .initval = { 1., 1., 1., 1. },
	.tvscales_N = 0,
	.tvscales = { 1., 1., 1., 1. },
};


int moba_get_nr_of_coeffs(const struct moba_conf* conf, int in)
{
	int coeffs = -1;

	switch (conf->mode) {

	case MDB_T1:
        case MDB_T1_PHY:
		coeffs = 3;
		break;

	case MDB_T2:
		coeffs = 2;
		break;

	case MDB_MGRE:
		coeffs = (MECO_PI != conf->mgre_model) ? get_num_of_coeff(conf->mgre_model) : in;
		break;

        case MDB_BLOCH:
		coeffs = 4;
		break;
	}

	return coeffs;
}


void debug_other(struct moba_other_conf* data)
{
        debug_printf(DP_WARN, "Other Debug-Output for MOBA\n\n");
        debug_printf(DP_INFO, "\tR1:%f\n", data->fov_reduction_factor);
        debug_printf(DP_INFO, "\tscale:%f:%f:%f:%f\n", data->scale[0], data->scale[1], data->scale[2], data->scale[3]);
}

