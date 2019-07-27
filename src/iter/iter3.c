/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2016-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
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

const struct iter3_irgnm_conf iter3_irgnm_defaults = {

	.INTERFACE.TYPEID = &TYPEID2(iter3_irgnm_conf),

	.iter = 8,
	.alpha = 1.,
	.alpha_min = 0.,
	.redu = 2.,

	.cgiter = 100,
	.cgtol = 0.1,

	.nlinv_legacy = false,
};

const struct iter3_landweber_conf iter3_landweber_defaults = {

	.INTERFACE.TYPEID = &TYPEID2(iter3_landweber_conf),

	.iter = 8,
	.alpha = 1.,
	.epsilon = 0.1,
};




