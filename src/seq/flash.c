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

	return ceil(MAX(sli_ampl * seq->sys.grad.inv_slew_rate, seq->sys.coil_control_lead));
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

	grad->rampup = ceil(MAX(ampl * seq->sys.grad.inv_slew_rate, seq->sys.coil_control_lead)); //round_up for start of rf pulse
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

	if (!grad_soft(grad, available_time_RF_SLI(0, seq), - mom, limits))
		return 0;

	return 1;
}


int flash(int N, struct seq_event ev[N], struct seq_state* seq_state, const struct seq_config* seq)
{
	struct grad_trapezoid sli;
	struct grad_trapezoid sli_reph;
	struct grad_trapezoid ro_deph;
	struct grad_trapezoid ro;

	double sli_start = 0;
	double rf_start = start_rf(seq);
	double ro_deph_start = rf_start + seq->phys.rf_duration;
	double sli_reph_start = ro_deph_start + seq->sys.grad.inv_slew_rate * slice_amplitude(seq);
	double ro_grad_start = ro_deph_start + available_time_RF_SLI(1, seq);
	double adc_start = start_rf(seq) + seq->phys.rf_duration / 2. + 1. * seq->phys.te - adc_time_to_echo(seq);

	int i = 0;

	double rf_spoil_phase = rf_spoiling(DIMS, seq_state->pos, seq);

	double proj_angle = get_rot_angle(seq_state->pos, seq);

	if (0. > proj_angle)
		return ERROR_ROT_ANGLE;

	double projX[3] = { cos(proj_angle), 0.	, 0. };
	double projY[3] = { 0., sin(proj_angle)	, 0. };
	double projSLICE[3] = { 0. , 0. , 1. };

	if (!prep_grad_sli(&sli, seq))
		return ERROR_PREP_GRAD_SLI;

	i += seq_grad_to_event(ev + i, sli_start, &sli, projSLICE);

	i += prep_rf_ex(ev + i, rf_start, rf_spoil_phase, seq_state, seq);

	if (!prep_grad_sli_reph(&sli_reph, seq))
		return ERROR_PREP_GRAD_SLI_REPH;

	i += seq_grad_to_event(ev + i, sli_reph_start, &sli_reph, projSLICE);

	if (!prep_grad_ro_deph(&ro_deph, seq))
		return ERROR_PREP_GRAD_RO_DEPH;

	//check for overlapping gradients!
	if (seq->sys.grad.max_amplitude < sqrtf(powf(fabs(sli_reph.ampl), 2.) + powf(ro_deph.ampl, 2.)))
		return ERROR_MAX_GRAD_RO_SLI;

	i += seq_grad_to_event(ev + i, ro_deph_start, &ro_deph, projX);
	i += seq_grad_to_event(ev + i, ro_deph_start, &ro_deph, projY);

	if (!prep_grad_ro(&ro, seq))
		return ERROR_PREP_GRAD_RO_RO;

	i += seq_grad_to_event(ev + i, ro_grad_start, &ro, projX);
	i += seq_grad_to_event(ev + i, ro_grad_start, &ro, projY);

	i += prep_adc(ev + i, adc_start, rf_spoil_phase, seq_state, seq);

	if (seq_block_end_flat(i, ev) > seq->phys.tr)
		return ERROR_END_FLAT_KERNEL;

	if (((PEMODE_RATION_APPROX_GA == seq->enc.pe_mode) || (PEMODE_RATION_APPROX_GAAL == seq->enc.pe_mode)) 
		&& ((seq->loop_dims[TIME_DIM] - 1)  == seq_state->pos[TIME_DIM]) && ((seq->loop_dims[ITER_DIM] - 1) == seq_state->pos[PHS1_DIM]))
			seq_state->pos[PHS1_DIM] = seq->loop_dims[PHS1_DIM] - 1;

	return i;
}



void set_loop_dims_and_sms(struct seq_config* seq, long /* partitions */, long total_slices, long radial_views,
	long frames, long echoes, long inv_reps, long phy_phases, long averages, int checkbox_sms, long mb_factor)
{
	switch (seq->enc.order) {

	case ORDER_AVG_OUTER:
		md_copy_order(DIMS, seq->order, seq_loop_order_avg_outer);
		break;

	case ORDER_SEQ_MS:
		md_copy_order(DIMS, seq->order, seq_loop_order_multislice);
		break;

	case ORDER_AVG_INNER:
		md_copy_order(DIMS, seq->order, seq_loop_order_avg_inner);
		break;
	}

	seq->geom.mb_factor = (checkbox_sms) ? mb_factor : 1;
	seq->loop_dims[SLICE_DIM] = (checkbox_sms) ? seq->geom.mb_factor : total_slices;
	seq->loop_dims[PHS2_DIM] = (checkbox_sms) ? total_slices / seq->geom.mb_factor : 1;

	if ((seq->loop_dims[PHS2_DIM] * seq->loop_dims[SLICE_DIM]) != total_slices)
		seq->loop_dims[PHS2_DIM] = -1; //mb groups

	seq->loop_dims[BATCH_DIM] = inv_reps;
	seq->loop_dims[TIME_DIM] = frames;

	if ((PEMODE_RATION_APPROX_GA == seq->enc.pe_mode) || (PEMODE_RATION_APPROX_GAAL == seq->enc.pe_mode)) {

		assert(frames >= radial_views);

		seq->loop_dims[TIME_DIM] = ceil(1. * frames / radial_views);
		seq->loop_dims[ITER_DIM] = frames % radial_views;

		if (0 == seq->loop_dims[ITER_DIM])
			seq->loop_dims[ITER_DIM] = radial_views;
	}
	seq->loop_dims[TIME2_DIM] = phy_phases;
	seq->loop_dims[AVG_DIM] = averages;
	seq->loop_dims[PHS1_DIM] = radial_views;
	seq->loop_dims[TE_DIM] = echoes;

	// 2 additional calls for pre_sequence (delay_meas + noise_scan) 
	// now we assume one block for magn_prep
	// FIXME: max of prep_scans and loops demanded for (asl-) saturation etc.
	seq->loop_dims[COEFF2_DIM] = 3;
	seq->loop_dims[COEFF_DIM] = 3; // pre-/post- and actual kernel calls
}


