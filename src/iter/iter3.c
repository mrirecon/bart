/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2016-2021. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2023-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <assert.h>
#include <stdbool.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/types.h"
#include "misc/misc.h"

#include "iter/italgos.h"
#include "iter/vec.h"

#include "iter3.h"


DEF_TYPEID(iter3_irgnm_conf);
DEF_TYPEID(iter3_landweber_conf);
DEF_TYPEID(iter3_lbfgs_conf);
DEF_TYPEID(iter3_levenberg_marquardt_conf);

const struct iter3_irgnm_conf iter3_irgnm_defaults = {

	.super.TYPEID = &TYPEID2(iter3_irgnm_conf),

	.iter = 8,
	.alpha = 1.,
	.alpha_min = 0.,
	.alpha_min0 = 0.,
	.redu = 2.,

	.cgiter = 100,
	.cgtol = 0.1,

	.nlinv_legacy = false,
};

const struct iter3_landweber_conf iter3_landweber_defaults = {

	.super.TYPEID = &TYPEID2(iter3_landweber_conf),

	.iter = 8,
	.alpha = 1.,
	.epsilon = 0.1,
};

const struct iter3_lbfgs_conf iter3_lbfgs_defaults = {

	.super.TYPEID = &TYPEID2(iter3_lbfgs_conf),

	.iter = -1,
	.M = 6,
	.step = 1.,
	.c1 = 1.e-4,
	.c2 = 0.95,
	.ftol = 1.e-4,
	.gtol = 1.e-4,
};


const struct iter3_levenberg_marquardt_conf iter3_levenberg_marquardt_defaults = {

	.super.TYPEID = &TYPEID2(iter3_levenberg_marquardt_conf),

	.iter = 15,
	.cgiter = 50,
	.redu = 0.1,
	.Bi = 1,
	.Bo = 1,
	.l2lambda = 0.1,
};


