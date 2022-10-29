/* Copyright 2014,2019. The Regents of the University of California.
 * Copyright 2022. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014,2019,2022 Martin Uecker.
 *
 *
 * Rudin LI, Osher S, Fatemi E. Nonlinear total variation based
 * noise removal algorithms, Physica D: Nonlinear Phenomena
 * 60:259-268 (1992)
 *
 * Bredies K, Kunisch K, Pock T. Total generalized variation.
 * SIAM Journal on Imaging Sciences
 * 3:492-526 (2010)
 */

#include <stdlib.h>
#include <assert.h>
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

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "iter/tgv.h"
#include "iter/iter2.h"
#include "iter/iter.h"


#ifndef DIMS
#define DIMS 16
#endif


static const char help_str[] = "Perform total generalized variation denoising along dims specified by flags.";


	
int main_tgv(int argc, char* argv[argc])
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
	
	long in_dims[DIMS];

	complex float* in_data = load_cfl(in_file, DIMS, in_dims);

	assert(1 == in_dims[DIMS - 1]);

	long out_dims[DIMS];
	md_copy_dims(DIMS, out_dims, in_dims);

	out_dims[DIMS - 1] = 1 + bitcount(flags);

	int ext_shift = 1;
	struct reg2 reg2 = tgv_reg(flags, /*MD_BIT(DIMS - 1) |*/ MD_BIT(DIMS), lambda, DIMS, out_dims, &ext_shift);


	complex float* out_data = create_cfl(out_file, DIMS, out_dims);

	auto id = linop_extract_create(DIMS, (long[DIMS]){ 0 }, in_dims, out_dims);

	struct iter_admm_conf conf = iter_admm_defaults;

	complex float* adj = md_alloc(DIMS, out_dims, CFL_SIZE);

	linop_adjoint(id, DIMS, out_dims, adj, DIMS, in_dims, in_data);

	conf.maxiter = 100;
	conf.rho = .1;

	iter2_admm(CAST_UP(&conf), id->normal,
		   2, MAKE_ARRAY(reg2.prox[0], reg2.prox[1]), MAKE_ARRAY(reg2.linop[0], reg2.linop[1]),
		   NULL, NULL,
		   2 * md_calc_size(DIMS, out_dims), (float*)out_data, (const float*)adj,
		   NULL);

	md_free(adj);

	linop_free(id);
	linop_free(reg2.linop[0]);
	linop_free(reg2.linop[1]);

	operator_p_free(reg2.prox[0]);
	operator_p_free(reg2.prox[1]);

	unmap_cfl(DIMS, in_dims, in_data);
	unmap_cfl(DIMS, out_dims, out_data);

	return 0;
}

