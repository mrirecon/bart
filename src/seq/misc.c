/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "math.h"

#include "seq/gradient.h"
#include "seq/config.h"

#include "misc.h"


long round_up_GRT(double val_usec)
{
	if (val_usec < 0)
		return 0;

	long lVal_usec = ceil(val_usec);
	long rem = lVal_usec % GRAD_RASTER_TIME;

	if (rem)
		lVal_usec += GRAD_RASTER_TIME - rem;

	return lVal_usec;
}

double ro_amplitude(const struct seq_config* seq)
{
	return 1.e6 / (seq->geom.fov * seq->phys.dwell * seq->sys.gamma);
}

double slice_amplitude(const struct seq_config* seq)
{
	return 1.e6 * seq->phys.bwtp / (seq->sys.gamma * seq->phys.rf_duration * seq->geom.slice_thickness);
}

int gradient_prepare_with_timing(struct grad_trapezoid* grad, double moment, struct grad_limits sys)
{
	if (2 * GRAD_RASTER_TIME > (grad->rampup + grad->flat + grad->rampdown))
		return 0;

	grad->ampl = moment / (0.5 * grad->rampup + grad->flat + 0.5 * grad->rampdown);

	if ((fabs(grad->ampl) > sys.max_amplitude)
	    || (fabs(grad->ampl) * sys.inv_slew_rate > grad->rampup)
		|| (fabs(grad->ampl) * sys.inv_slew_rate > grad->rampdown))
			return 0;

	return 1;
}
