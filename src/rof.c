/* Copyright 2014. The Regents of the University of California.
 * Copyright 2022. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014, 2022 Martin Uecker
 *
 *
 * Rudin LI, Osher S, Fatemi E. Nonlinear total variation based
 * noise removal algorithms, Physica D: Nonlinear Phenomena
 * 60:259-268 (1992)
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
#include "num/ops_p.h"
#include "num/init.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/grad.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "iter/prox.h"
#include "iter/thresh.h"
#include "iter/iter2.h"
#include "iter/iter.h"


#ifndef DIMS
#define DIMS 16
#endif


static const char help_str[] = "Perform total variation denoising along dims <flags>.";

struct reg {

	const struct linop_s* linop;
	const struct operator_p_s* prox;
};

static struct reg tvreg(unsigned long flags, float lambda, int N, const long dims[N])
{
	struct reg reg;

	reg.linop = linop_grad_create(N, dims, N, flags);
	reg.prox = prox_thresh_create(N + 1, linop_codomain(reg.linop)->dims, lambda, MD_BIT(N));

	return reg;
}

	
int main_rof(int argc, char* argv[argc])
{
	float lambda = 0.;
	unsigned long flags = 0;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_FLOAT(true, &lambda, "lambda"),
		ARG_ULONG(true, &flags, "flags"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = { };

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long dims[DIMS];
	
	complex float* in_data = load_cfl(in_file, DIMS, dims);
	complex float* out_data = create_cfl(out_file, DIMS, dims);

	auto id_op  = linop_identity_create(DIMS, dims);

	struct reg reg = tvreg(flags, lambda, DIMS, dims);

	struct iter_admm_conf conf = iter_admm_defaults;

	conf.maxiter = 50;
	conf.rho = .1;

	iter2_admm(CAST_UP(&conf), id_op->forward,
		   1, MAKE_ARRAY(reg.prox), MAKE_ARRAY(reg.linop), NULL,
		   NULL, 2 * md_calc_size(DIMS, dims), (float*)out_data, (const float*)in_data, NULL);

	linop_free(id_op);
	linop_free(reg.linop);

	operator_p_free(reg.prox);
	
	unmap_cfl(DIMS, dims, in_data);
	unmap_cfl(DIMS, dims, out_data);

	return 0;
}


