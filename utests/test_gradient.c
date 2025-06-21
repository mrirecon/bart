/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <math.h>
#include <stdint.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "seq/gradient.h"

#include "utest.h"

static struct grad_limits sys = {
	.inv_slew_rate = 1000.,
	.max_amplitude = 1.,
};


static bool test_softest_gradient1(void)
{
	long available_time = 3000;
	float moment = 2000;

	struct grad_trapezoid grad;

	if (!grad_soft(&grad, available_time, moment, sys))
		return false;

	if (1. != grad.ampl)
		return false;

	if (grad_total_time(&grad) != available_time)
		return false;

	return true;
}

static bool test_softest_gradient2(void)
{
	long available_time = 6000;
	float moment = 2000;

	struct grad_trapezoid grad;

	if (!grad_soft(&grad, available_time, moment, sys))
		return false;

	if (0.4 != grad.ampl)
		return false;

	if (grad_total_time(&grad) != available_time)
		return false;

	return true;
}

static bool test_hardest_gradient1(void)
{
	float moment = 2000;

	struct grad_trapezoid grad;

	if (!grad_hard(&grad, moment, sys))
		return false;

	if (1. != grad.ampl)
		return false;

	if (3000. != grad_total_time(&grad))
		return false;

	return true;
}

static bool test_hardest_gradient2(void)
{
	float moment = -2000.;

	struct grad_trapezoid grad;

	if (!grad_hard(&grad, moment, sys))
		return false;

	if (-1. != grad.ampl)
		return false;

	if (3000. != grad_total_time(&grad))
		return false;

	return true;
}

UT_REGISTER_TEST(test_softest_gradient1);
UT_REGISTER_TEST(test_softest_gradient2);
UT_REGISTER_TEST(test_hardest_gradient1);
UT_REGISTER_TEST(test_hardest_gradient2);

