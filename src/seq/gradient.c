/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>
#include <assert.h>

#include "misc/mri.h"
#include "misc/misc.h"

#include "gradient.h"


double grad_total_time(const struct grad_trapezoid* grad)
{
	return (grad->rampup + grad->flat + grad->rampdown);
}



double grad_momentum(struct grad_trapezoid* grad)
{
	return grad->ampl * (0.5 * grad->rampup + grad->flat + 0.5 * grad->rampdown);
}


/* compute parameters for softest symmetric trapezoid
 * */
bool grad_soft(struct grad_trapezoid* grad, double dur, double moment, struct grad_limits sys)
{
	if (0 > dur)
		return false;

	double rise = sys.max_amplitude * sys.inv_slew_rate;
	double flat = dur - 2. * rise;
	double ampl = moment / (rise + flat);

	if (0. > flat)
		return false;

	if (rise < (fabs(ampl) * sys.inv_slew_rate))
		return false;

	if ((fabs(ampl) > sys.max_amplitude))
		return false;

	grad->start = 0.;
	grad->flat = flat;
	grad->ampl = ampl;
	grad->rampdown = rise;
	grad->rampup = rise;

	return true;
}


/* compute parameters for hardest symmetric trapezoid
 * */
bool grad_hard(struct grad_trapezoid* grad, double moment, struct grad_limits sys)
{
	double rise = sys.max_amplitude * sys.inv_slew_rate;
	double flat = 0.;

	if (rise < fabs(moment) / sys.max_amplitude) {

		flat = fabs(moment) / sys.max_amplitude - rise;

	} else {

		rise = sqrt(fabs(moment) * sys.inv_slew_rate);
	}

	grad->start = 0.;
	grad->flat = flat;
	grad->rampdown = rise;
	grad->rampup = rise;
	grad->ampl = moment / (flat + rise);

	return true;
}

