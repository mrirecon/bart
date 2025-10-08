/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>

#include "num/multind.h"

#include "misc/mri.h"

#include "seq/config.h"

#include "helpers.h"



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
