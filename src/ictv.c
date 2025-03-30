/* Copyright 2022-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
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


static const char help_str[] = "Infimal convolution of total variation along dims specified by flags.";


	
int main_ictv(int argc, char* argv[argc])
{
	float lambda = 0.;
	unsigned long flags = 0;
	const char* in_file = NULL;
	const char* out_file = NULL;

	int tvscales_N = 5;
	float tvscales[5] = { 0., 0., 0., 0., 0. };

	int tvscales2_N = 5;
	float tvscales2[5] = { 0., 0., 0., 0., 0. };
	
	float gamma[] = { 1., 1. };

	struct iter_admm_conf conf = iter_admm_defaults;

	struct arg_s args[] = {

		ARG_FLOAT(true, &lambda, "lambda"),
		ARG_ULONG(true, &flags, "flags"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = {
	
		OPT_PINT('i', &conf.maxiter, "i", "max. iterations"),
		OPT_FLOAT('u', &conf.rho, "rho", "rho in ADMM"),
		OPTL_FLVECN(0, "tvscales", tvscales, "Scaling of derivatives of the first gradient"),
		OPTL_FLVECN(0, "tvscales2", tvscales2, "Scaling of derivatives of the second gradient"),
		OPTL_FLVEC2(0, "gamma", &gamma, "gamma1:gamma2", "gamma1 * || grad (x - z) ||_1, gamma2 * || grad z ||_1")
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();
	
	long in_dims[DIMS];

	complex float* in_data = load_cfl(in_file, DIMS, in_dims);

	assert(1 == in_dims[DIMS - 1]);

	long out_dims[DIMS];
	md_copy_dims(DIMS, out_dims, in_dims);

	out_dims[DIMS - 1] = 2;

	const struct linop_s* lop_trafo = NULL;

	long ext_shift = md_calc_size(DIMS, in_dims);
	struct reg2 reg2 = ictv_reg(flags, /*MD_BIT(DIMS - 1) |*/ MD_BIT(DIMS), lambda, DIMS, in_dims, 2 * ext_shift, &ext_shift, gamma, tvscales_N, tvscales, tvscales2_N, tvscales2, lop_trafo);


	complex float* out_data = create_cfl(out_file, DIMS, out_dims);

	auto id = linop_extract_create(DIMS, (long[DIMS]){ }, in_dims, out_dims);
	id = linop_reshape_out_F(id, 1, MD_DIMS(2 * md_calc_size(DIMS, in_dims)));


	complex float* adj = md_alloc(DIMS, out_dims, CFL_SIZE);

	linop_adjoint(id, DIMS, out_dims, adj, DIMS, in_dims, in_data);

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

