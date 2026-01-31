/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "math.h"

#include "seq/gradient.h"
#include "seq/config.h"

#include "misc.h"


double round_up_raster(double time, double raster_time)
{
	if (0. > time)
		return 0.;

	return ceil(time / raster_time - 1E2 * raster_time) * raster_time;
}

double ro_amplitude(const struct seq_config* seq)
{
	return 1. / (seq->geom.fov * seq->phys.dwell * seq->sys.gamma);
}

double slice_amplitude(const struct seq_config* seq)
{
	return seq->phys.bwtp / (seq->sys.gamma * seq->phys.rf_duration * seq->geom.slice_thickness);
}

int gradient_prepare_with_timing(struct grad_trapezoid* grad, double moment, const struct seq_config* seq)
{
	if (2 * seq->sys.raster_grad > (grad->rampup + grad->flat + grad->rampdown))
		return 0;

	grad->ampl = moment / (0.5 * grad->rampup + grad->flat + 0.5 * grad->rampdown);

	if (   (fabs(grad->ampl) > seq->sys.grad.max_amplitude)
	    || (   (fabs(grad->ampl) * seq->sys.grad.inv_slew_rate > grad->rampup))
		|| (fabs(grad->ampl) * seq->sys.grad.inv_slew_rate > grad->rampdown))
			return 0;

	return 1;
}

