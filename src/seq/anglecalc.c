/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */


#include <math.h>
#include <assert.h>

#include "anglecalc.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "noncart/traj.h"

#include "seq/config.h"

void traj_conf_from_seq(struct traj_conf *conf, const struct seq_config* seq)
{
	*conf = traj_defaults;

	conf->full_circle = true; // double_angle always on! FIXME: to delete everywhere

	conf->Y = seq->loop_dims[PHS1_DIM];
	conf->tiny_gold = seq->enc.tiny;

	conf->raga_inc = raga_increment(conf->Y, conf->tiny_gold);

	conf->golden = true;
	conf->rational = true;
	conf->double_base = true;

	conf->mb = seq->loop_dims[SLICE_DIM];

	switch (seq->enc.pe_mode) {

	case PEMODE_RATION_APPROX_GA:
		conf->aligned_flags = 0;
		break;
	case PEMODE_RATION_APPROX_GAAL:
		conf->aligned_flags = SLICE_FLAG;
		break;
	default:
		assert(0);
	}
}

double get_rot_angle(const long pos[DIMS], const struct seq_config* seq)
{
	struct traj_conf conf;

	traj_conf_from_seq(&conf, seq);

	double atom = calc_angle_atom(&conf);
	long inc = raga_increment_from_pos(seq->order, pos, (SEQ_FLAGS & ~(COEFF_FLAG|COEFF2_FLAG)), seq->loop_dims, &conf);

	return atom * inc;
}

