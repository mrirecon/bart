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
	.redu = 2.,
	.step = 0.9,
	.lower_bound = 0.,
	.tolerance = 0.01,
	.inner_iter = 250,
	.noncartesian = false,
	.sms = false,
        .k_filter = false,
	.auto_norm_off = false,
};


