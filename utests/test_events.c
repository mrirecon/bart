/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>

#include "misc/debug.h"

#include "seq/gradient.h"
#include "seq/seq_event.h"

#include "utest.h"


static struct grad_limits limits = {
	.inv_slew_rate = 10.,
	.max_amplitude = 20.,
};

static bool test_moment_sum(void)
{
	double available_time = 1000.;
	double momentum = 10000.;
	struct grad_trapezoid grad;

	if (!grad_soft(&grad, available_time, momentum, limits))
		return false;

	struct seq_event ev[2];
	double proj[3] = { 0. , 0. , 1. };

	int i = seq_grad_to_event(ev, 0, &grad, proj);

	if (2 != i)
		return false;

	double m0[3];
	moment_sum(m0, available_time, 2, ev);

	if (UT_TOL < fabs(m0[2] - momentum))
		return false; 

	return true;
}

UT_REGISTER_TEST(test_moment_sum);

