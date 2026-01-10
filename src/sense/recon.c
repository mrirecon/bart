/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2016-2020. Martin Uecker.
 * Copyright 2022-2024. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2016 Martin Uecker
 * 2013-2014 Jonathan Tamir
 * 2014      Frank Ong
 *
 *
 * Ra JB, Rim CY. Fast imaging using subencoding data sets from multiple
 * detectors. Magn Reson Med 1993; 30:142-145.
 *
 * Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P. SENSE: Sensitivity
 * encoding for fast MRI. Magn Reson Med 1999; 42:952-962.
 *
 * Pruessmann KP, Weiger M, Boernert P, Boesiger P. Advances in sensitivity
 * encoding with arbitrary k-space trajectories. 
 * Magn Reson Med 2001; 46:638-651.
 *
 * Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS,
 * Lustig M. ESPIRiT - An Eigenvalue Approach to Autocalibrating Parallel MRI:
 * Where SENSE meets GRAPPA. Magn Reson Med 2014; 71:990-1001.
 */

#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/realval.h"

#include "iter/iter.h"
#include "iter/lsqr.h"
#include "iter/lad.h"

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mri.h"

#include "sense/model.h"

#include "recon.h"

const struct sense_conf sense_defaults = {

	.rvc = false,
	.gpu = false,
	.rwiter = 1,
	.gamma = -1.,
	.cclambda = 0.,
	.bpsense = false,
	.precond = false,
};





void debug_print_sense_conf(int level, const struct sense_conf* conf)
{
	debug_printf(level, "sense conf:\n");
	debug_printf(level, "\trvc:          %s\n", conf->rvc ? "on" : "off");
	debug_printf(level, "\trwiter:       %d\n", conf->rwiter);
	debug_printf(level, "\tgamma:        %f\n", conf->gamma);
	debug_printf(level, "\tcclambda:     %f\n", conf->cclambda);
	debug_printf(level, "\n\n");
}





const struct operator_p_s* sense_recon_create(const struct sense_conf* conf,
		  const struct linop_s* sense_op,
		  const long pat_dims[DIMS],
		  italgo_fun2_t italgo, iter_conf* iconf,
		  const complex float* init,
		  int num_funs,
		  const struct operator_p_s* thresh_op[num_funs],
		  const struct linop_s* thresh_funs[num_funs],
		  const struct operator_s* precond_op,
		  struct iter_monitor_s* monitor)
{
	struct lsqr_conf lsqr_conf = lsqr_defaults;
	lsqr_conf.lambda = conf->cclambda;
	lsqr_conf.it_gpu = conf->gpu;

	const struct operator_p_s* op = NULL;

	assert(DIMS == linop_codomain(sense_op)->N);

	long ksp_dims[DIMS];
	md_copy_dims(DIMS, ksp_dims, linop_codomain(sense_op)->dims);

	if (1 < conf->rwiter) {

		assert(!conf->bpsense); // not compatible

		unsigned long flags = 0;
		for (int i = 0; i < DIMS; i++)
			if (pat_dims[i] > 1)
				flags = MD_SET(flags, i);

		const struct lad_conf lad_conf = { conf->rwiter, conf->gamma, flags, &lsqr_conf };

		op = lad2_create(&lad_conf, italgo, iconf, (const float*)init, sense_op, num_funs, thresh_op, thresh_funs);

		linop_free(sense_op);

	} else {

		if ((conf->bpsense) || (conf->precond))
			op = lsqr2_create(&lsqr_conf, italgo, iconf, (const float*)init, NULL, NULL,
					num_funs, thresh_op, thresh_funs, monitor);
		else
			op = lsqr2_create(&lsqr_conf, italgo, iconf, (const float*)init, sense_op, precond_op,
					num_funs, thresh_op, thresh_funs, monitor);

		linop_free(sense_op);
	}

	return op;
}

