/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>

#include "num/multind.h"

#include "misc/mri.h"
#include "misc/misc.h"

#include "seq/config.h"
#include "seq/event.h"
#include "seq/anglecalc.h"
#include "seq/adc_rf.h"
#include "seq/gradient.h"
#include "seq/misc.h"
#include "seq/seq.h"

#include "flash.h"

// for time-optimized overlapping gradients, otherwise sqrt(2.)
#define SCALE_GRAD 0.82


static double start_rf(const struct seq_config* seq)
{
	double sli_ampl = slice_amplitude(seq);

	return round_up_raster(MAX(sli_ampl * seq->sys.grad.inv_slew_rate, seq->sys.coil_control_lead), seq->sys.raster_rf);
}

static double ro_shift(const struct seq_config* seq)
{
	double start_flat = start_rf(seq) + seq->phys.rf_duration / 2.
				+ 1. * seq->phys.te - adc_time_to_echo(seq);

	return (GRAD_RASTER_TIME - (round_up_GRT(start_flat) - (long)start_flat)) % GRAD_RASTER_TIME;
}


static double available_time_RF_SLI(int ro, const struct seq_config* seq)
{
	double ampl = ro ? ro_amplitude(seq) : slice_amplitude(seq);

	return seq->phys.te - seq->phys.rf_duration / 2.
		- ampl * seq->sys.grad.inv_slew_rate
		- adc_time_to_echo(seq)
		- ro_shift(seq);
}

static int prep_grad_ro_deph(struct grad_trapezoid* grad, const struct seq_config* seq)
{
	double amp = ro_amplitude(seq);
	double mom = amp *
		(0.5 * amp * seq->sys.grad.inv_slew_rate
		+ ro_shift(seq) + adc_time_to_echo(seq));

	struct grad_limits limits = seq->sys.grad;
	limits.max_amplitude *= SCALE_GRAD;

	if (!grad_soft(grad, available_time_RF_SLI(1, seq), - mom, limits))
		return 0;

	return 1;
}


static int prep_grad_ro(struct grad_trapezoid* grad, const struct seq_config* seq)
{
	*grad = (struct grad_trapezoid){ 0 };

	double ampl = ro_amplitude(seq);

	if (seq->sys.grad.max_amplitude < ampl)
		return 0;

	grad->rampup = ampl * seq->sys.grad.inv_slew_rate;
	grad->flat = round_up_GRT(adc_duration(seq) + ro_shift(seq));
	grad->rampdown = ampl * seq->sys.grad.inv_slew_rate;
	grad->ampl = ampl;

	return 1;
}



static int prep_grad_sli(struct grad_trapezoid* grad, const struct seq_config* seq)
{
	*grad = (struct grad_trapezoid){ 0 };

	double ampl = slice_amplitude(seq);

	if (seq->sys.grad.max_amplitude < ampl)
		return 0;

	grad->rampup = round_up_raster(MAX(ampl * seq->sys.grad.inv_slew_rate, seq->sys.coil_control_lead), seq->sys.raster_rf); //round_up for start of rf pulse
	grad->flat = seq->phys.rf_duration;
	grad->rampdown = ampl * seq->sys.grad.inv_slew_rate;
	grad->ampl = ampl;

	return 1;
}


static int prep_grad_sli_reph(struct grad_trapezoid* grad, const struct seq_config* seq)
{
	*grad = (struct grad_trapezoid){ 0 };

	double amp = slice_amplitude(seq);
	double mom = amp * (0.5 * seq->phys.rf_duration + 0.5 * amp * seq->sys.grad.inv_slew_rate);

	struct grad_limits limits = seq->sys.grad;
	limits.max_amplitude *= SCALE_GRAD;

	if (!grad_soft(grad, available_time_RF_SLI(0, seq), -mom, limits))
		return 0;

	return 1;
}


struct flash_timing {

	double RF;
	double slice;
	double slice_rephaser;
	double readout_dephaser;
	double readout;
	double adc;
};


static struct flash_timing flash_compute_timing(const struct seq_config *seq)
{
	struct flash_timing timing;

	timing.slice = 0.;
	timing.RF = start_rf(seq);
	timing.readout_dephaser = timing.RF + seq->phys.rf_duration;
	timing.slice_rephaser = timing.readout_dephaser + seq->sys.grad.inv_slew_rate * slice_amplitude(seq);
	timing.readout = timing.readout_dephaser + available_time_RF_SLI(1, seq);
	timing.adc = timing.RF + seq->phys.rf_duration / 2. + seq->phys.te - adc_time_to_echo(seq);

	return timing;
}


int flash(int N, struct seq_event ev[N], struct seq_state* seq_state, const struct seq_config* seq)
{
	struct flash_timing timing = flash_compute_timing(seq);

	int i = 0;

	double rf_spoil_phase = rf_spoiling(DIMS, seq_state->pos, seq);
	double proj_angle = get_rot_angle(seq_state->pos, seq);

	if (0. > proj_angle)
		return ERROR_ROT_ANGLE;

	double projX[3] = { 0., cos(proj_angle), 0. };
	double projY[3] = { sin(proj_angle), 0., 0. };
	double projSLICE[3] = { 0. , 0. , 1. };


	struct grad_trapezoid slice;

	if (!prep_grad_sli(&slice, seq))
		return ERROR_PREP_GRAD_SLI;

	i += seq_grad_to_event(ev + i, timing.slice, &slice, projSLICE);

	i += prep_rf_ex(ev + i, timing.RF, rf_spoil_phase, seq_state, seq);


	struct grad_trapezoid slice_rephaser;

	if (!prep_grad_sli_reph(&slice_rephaser, seq))
		return ERROR_PREP_GRAD_SLI_REPH;

	i += seq_grad_to_event(ev + i, timing.slice_rephaser, &slice_rephaser, projSLICE);


	struct grad_trapezoid readout_dephaser;

	if (!prep_grad_ro_deph(&readout_dephaser, seq))
		return ERROR_PREP_GRAD_RO_DEPH;

	//check for overlapping gradients!
	if (seq->sys.grad.max_amplitude < sqrtf(powf(fabs(slice_rephaser.ampl), 2.) + powf(readout_dephaser.ampl, 2.)))
		return ERROR_MAX_GRAD_RO_SLI;

	i += seq_grad_to_event(ev + i, timing.readout_dephaser, &readout_dephaser, projX);
	i += seq_grad_to_event(ev + i, timing.readout_dephaser, &readout_dephaser, projY);


	struct grad_trapezoid readout;

	if (!prep_grad_ro(&readout, seq))
		return ERROR_PREP_GRAD_RO_RO;

	i += seq_grad_to_event(ev + i, timing.readout, &readout, projX);
	i += seq_grad_to_event(ev + i, timing.readout, &readout, projY);



	i += prep_adc(ev + i, timing.adc, rf_spoil_phase, seq_state, seq);


	if (seq_block_end_flat(i, ev) > seq->phys.tr)
		return ERROR_END_FLAT_KERNEL;

	if (((PEMODE_RAGA == seq->enc.pe_mode) || (PEMODE_RAGA_ALIGNED == seq->enc.pe_mode))
		&& ((seq->loop_dims[TIME_DIM] - 1) == seq_state->pos[TIME_DIM])
		&& ((seq->loop_dims[ITER_DIM] - 1) == seq_state->pos[PHS1_DIM]))
			seq_state->pos[PHS1_DIM] = seq->loop_dims[PHS1_DIM] - 1;

	return i;
}


