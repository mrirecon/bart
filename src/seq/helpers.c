/* Copyright 2025-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>

#include "num/multind.h"

#include "misc/mri.h"
#include "misc/misc.h"

#include "noncart/traj.h"

#include "seq/config.h"
#include "seq/anglecalc.h"
#include "seq/adc_rf.h"
#include "seq/flash.h"
#include "seq/misc.h"
#include "seq/mag_prep.h"

#include "helpers.h"


int seq_raga_spokes(const struct seq_config* seq)
{
	if ((PEMODE_RAGA == seq->enc.pe_mode) || (PEMODE_RAGA_ALIGNED == seq->enc.pe_mode))
		return raga_spokes(seq->geom.baseres, seq->enc.tiny);
	else
		return seq->loop_dims[PHS1_DIM];
}

int seq_check_equal_dims(int D, const long dims1[D], const long dims2[D], unsigned long flags)
{
	return md_check_equal_dims(D, dims1, dims2, flags);
}


long seq_minimum_tr(const struct seq_config* seq)
{
	return min_tr_flash(seq);
}


void seq_minimum_te(const struct seq_config* seq, long* min_te, long* fil_te)
{
	min_te_flash(seq, min_te, fil_te);
}




long seq_get_slices(const struct seq_config* seq)
{
	return (1 < seq->geom.mb_factor) ? seq->loop_dims[SLICE_DIM] * seq->loop_dims[PHS2_DIM] : seq->loop_dims[SLICE_DIM];
}


static long kernels_per_measurement(const long loop_dims[DIMS])
{
	long dims[DIMS];
	md_select_dims(DIMS, (PHS1_FLAG|TIME2_FLAG|AVG_FLAG|SLICE_FLAG|PHS2_FLAG), dims, loop_dims);

	return md_calc_size(DIMS, dims);
}

long seq_relevant_readouts_meas_time(const struct seq_config* seq)
{
	return kernels_per_measurement(seq->loop_dims) / seq->loop_dims[PHS1_DIM];
}

double seq_total_measure_time(const struct seq_config* seq)
{
	double pre_duration = seq->magn.init_delay;

	struct seq_event ev[6];
	int e = mag_prep(ev, seq);

	double prep_pulse_duration = seq_block_end(e, ev, BLOCK_PRE, seq->phys.tr, seq->sys.raster_grad);
	prep_pulse_duration += seq->magn.inv_delay_time;
	// prep_pulse_duration *= inv_calls(seq);

	long dims[DIMS] = { };
	md_select_dims(DIMS, SEQ_FLAGS & ~(COEFF_FLAG|COEFF2_FLAG), dims, seq->loop_dims);
	long img_calls = md_calc_size(DIMS, dims);
	double imaging_duration = seq->phys.tr * img_calls;

	if ((TRIGGER_OFF != seq->trigger.type) && (1 < seq->trigger.pulses)) {

		imaging_duration = 1. * (seq->trigger.delay_time + seq->phys.tr) * img_calls * (seq->trigger.pulses - 1);
	}

	return pre_duration + prep_pulse_duration + imaging_duration;
}



void set_loop_dims_and_sms(struct seq_config* seq, long /* partitions*/ , long total_slices, long radial_views,
	long frames, long echoes, long phy_phases, long averages)
{
	switch (seq->enc.order) {

	case SEQ_ORDER_AVG_OUTER:
		md_copy_order(DIMS, seq->order, seq_loop_order_avg_outer);
		break;

	case SEQ_ORDER_SEQ_MS:
		md_copy_order(DIMS, seq->order, seq_loop_order_multislice);
		break;

	case SEQ_ORDER_AVG_INNER:
		md_copy_order(DIMS, seq->order, seq_loop_order_avg_inner);
		break;
	}

	seq->loop_dims[SLICE_DIM] = (seq->geom.mb_factor > 1) ? seq->geom.mb_factor : total_slices;
	seq->loop_dims[PHS2_DIM] = (seq->geom.mb_factor > 1) ? total_slices / seq->geom.mb_factor : 1;
	if ((seq->loop_dims[PHS2_DIM] * seq->loop_dims[SLICE_DIM]) != total_slices)
		seq->loop_dims[PHS2_DIM] = -1; //mb groups

	seq->loop_dims[TIME_DIM] = frames;

	if ((PEMODE_RAGA == seq->enc.pe_mode)
	    || (PEMODE_RAGA_ALIGNED == seq->enc.pe_mode)) {

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

void seq_set_fov_pos(int N, int M, const float* shifts, struct seq_config* seq)
{
	long total_slices = seq_get_slices(seq);
	assert(total_slices <= N);

	if (1 < seq->geom.mb_factor)
		seq->geom.sms_distance = fabsf(shifts[2] - shifts[seq->loop_dims[PHS2_DIM] * M + 2]);
	else
		seq->geom.sms_distance = 0;

	for (int i = 0; i < total_slices; i++) {

		seq->geom.shift[i][0] = shifts[i * M + 0]; // RO shift
		seq->geom.shift[i][1] = shifts[i * M + 1]; // PE shift

		if (1 < seq->geom.mb_factor) {

			seq->geom.shift[i][2] = shifts[(total_slices / 2) * M + 2]
						+ (seq->geom.sms_distance / seq->loop_dims[PHS2_DIM]) 
						* (i % seq->loop_dims[PHS2_DIM] - floor(seq->loop_dims[PHS2_DIM] / 2.));
		}
		else {

			seq->geom.shift[i][2] = shifts[i * M + 2];
		}		
	}

	if (1 == seq->geom.mb_factor)
		seq->geom.sms_distance = -999.; // UI information
}
