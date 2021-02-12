/* Copyright 2019-2020. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 **/
 
#include <stdbool.h>

#include "moba.h"

struct moba_conf moba_defaults = {

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
	.auto_norm_off = false,
	.algo = 3,
	.rho = 0.01,
	.stack_frames = false,
};

