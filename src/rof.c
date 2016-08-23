/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014 Martin Uecker <uecker@eecs.berkeley.edu>
 *
 *
 * Rudin LI, Osher S, Fatemi E. Nonlinear total variation based
 * noise removal algorithms, Physica D: Nonlinear Phenomena
 * 60:259-268 (1992)
 * 
 */

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <complex.h>
#include <assert.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/ops.h"
#include "num/init.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/grad.h"

#include "misc/mmio.h"
#include "misc/misc.h"

#include "iter/prox.h"
#include "iter/thresh.h"
#include "iter/iter2.h"
#include "iter/iter.h"


#ifndef DIMS
#define DIMS 16
#endif


static const char usage_str[] = "<lambda> <flags> <input> <output>";
static const char help_str[] = "Perform total variation denoising along dims <flags>.\n";

	
int main_rof(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 4, usage_str, help_str);

	num_init();

	long dims[DIMS];

	float lambda = atof(argv[1]);
	int flags = atoi(argv[2]);
	
	complex float* in_data = load_cfl(argv[3], DIMS, dims);
	complex float* out_data = create_cfl(argv[4], DIMS, dims);

	// TV operator

	const struct linop_s* tv_op = linop_grad_create(DIMS, dims, flags);
//	const struct linop_s* tv_op = linop_identity_create(DIMS, dims);

	struct iter_admm_conf conf = iter_admm_defaults;

	conf.maxiter = 50;
	conf.rho = .1;

	const struct operator_p_s* thresh_prox = prox_thresh_create(DIMS + 1, linop_codomain(tv_op)->dims, 
								lambda, MD_BIT(DIMS), false);

	iter2_admm(CAST_UP(&conf), linop_identity_create(DIMS, dims)->forward,
		   1, MAKE_ARRAY(thresh_prox), MAKE_ARRAY(tv_op), NULL,
		   NULL, 2 * md_calc_size(DIMS, dims), (float*)out_data, (const float*)in_data, NULL);

	linop_free(tv_op);
	operator_p_free(thresh_prox);
	
	unmap_cfl(DIMS, dims, in_data);
	unmap_cfl(DIMS, dims, out_data);
	exit(0);
}


