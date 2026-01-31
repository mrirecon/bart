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

	conf->full_circle = true;

	switch (seq->enc.pe_mode) {

	case SEQ_PEMODE_TURN:

		conf->golden = false;
		conf->aligned = true;
		conf->turns = seq->enc.tiny;
		break;

	case SEQ_PEMODE_MEMS_HYB:

		conf->mems_traj = true;
		conf->tiny_gold = seq->enc.tiny;
		break;

	case SEQ_PEMODE_RAGA:

		conf->rational = true;
		conf->aligned_flags = 0;
		break;

	case SEQ_PEMODE_RAGA_ALIGNED:

		conf->rational = true;
		conf->aligned_flags = PHS2_FLAG | SLICE_FLAG;
		break;

	case SEQ_PEMODE_RAGA_MEMS:

		assert(0);
	}

	if (conf->rational) {

		conf->golden = true;
		conf->double_base = true;

		conf->Y = seq->loop_dims[PHS1_DIM];
		conf->tiny_gold = seq->enc.tiny;

		conf->aligned_flags |= seq->enc.aligned_flags;

		conf->raga_inc = raga_increment(conf->Y, conf->tiny_gold);
	}
}

double get_rot_angle(const long pos[DIMS], const struct seq_config* seq)
{
	struct traj_conf conf;

	traj_conf_from_seq(&conf, seq);

	if (conf.rational) {

		double atom = calc_angle_atom(&conf);
		long inc = raga_increment_from_pos(seq->order, pos, (SEQ_FLAGS & ~(COEFF_FLAG|COEFF2_FLAG)), seq->loop_dims, &conf);

		return atom * inc;
	}

	double base_angle[DIMS] = { 0. };
	calc_base_angles(base_angle, seq->loop_dims[PHS1_DIM], seq->loop_dims[TE_DIM], conf);

	long pos2[DIMS] = { 0L };

	pos2[PHS2_DIM] = pos[PHS1_DIM];
	pos2[SLICE_DIM] = pos[SLICE_DIM];
	pos2[TE_DIM] = pos[TE_DIM];
	pos2[TIME_DIM] = pos[TIME_DIM];

	long ind[DIMS] = { 0L };
	indices_from_position(ind, pos2, conf);

	double angle = 0.;

	for (int d = 1; d < DIMS; d++)
		angle += ind[d] * base_angle[d];

	return angle;
}



int check_gen_fib(int spokes, int tiny_ga)
{
	if (0 == spokes % 2)
		return 0;

	int i = 0;

	while (spokes >= gen_fibonacci(tiny_ga, i)) {

		if (spokes == gen_fibonacci(tiny_ga, i))
			return 1;

		i++;
	}

	return 0;
}
