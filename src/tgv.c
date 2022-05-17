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


static const char help_str[] = "Perform total generalized variation denoising along dims specified by flags.";

struct reg2 {

	const struct linop_s* linop[2];
	const struct operator_p_s* prox[2];
};

static struct reg2 tgvreg(unsigned long flags, float lambda, int N, const long in_dims[N])
{
	long out_dims[N];
	struct reg2 reg2;

	const struct linop_s* grad1 = linop_grad_create(N - 1, in_dims, N - 1, flags);
	const struct linop_s* grad2x = linop_grad_create(N + 0, linop_codomain(grad1)->dims, N + 0, flags);


	auto grad2a = linop_transpose_create(N + 1, N - 1, N + 0, linop_codomain(grad2x)->dims);
	auto grad2b = linop_identity_create(N + 1, linop_codomain(grad2x)->dims);
	auto grad2 = linop_chain_FF(grad2x, linop_plus_FF(grad2a, grad2b));


	long grd_dims[N];
	md_copy_dims(N, grd_dims, linop_codomain(grad1)->dims);

	md_copy_dims(N, out_dims, grd_dims);
	out_dims[N - 1]++;


	long pos1[N];

	for (int i = 0; i < N; i++)
		pos1[i] = 0;

	pos1[N - 1] = 0;

	auto grad1b = linop_extract_create(N, pos1, in_dims, out_dims);
	auto grad1c = linop_reshape_create(N - 1, linop_domain(grad1)->dims, N, in_dims);
	auto grad1d = linop_chain_FF(linop_chain_FF(grad1b, grad1c), grad1);


	long pos1b[N];

	for (int i = 0; i < N; i++)
		pos1b[i] = 0;

	pos1b[N - 1] = 1;

	auto grad1e = linop_extract_create(N, pos1b, grd_dims, out_dims);
	reg2.linop[0] = linop_plus_FF(grad1e, grad1d);


	long pos2[N];

	for (int i = 0; i < N; i++)
		pos2[i] = 0;

	pos2[N - 1] = 1;

	auto grad2e = linop_extract_create(N, pos2, grd_dims, out_dims);
	reg2.linop[1] = linop_chain_FF(grad2e, grad2);

	reg2.prox[0] = prox_thresh_create(N + 0, linop_codomain(reg2.linop[0])->dims, lambda, 0u);
	reg2.prox[1] = prox_thresh_create(N + 1, linop_codomain(reg2.linop[1])->dims, lambda, 0u);

	return reg2;
}

/* TGV
 * 
 * min x \|Ix - y\|_2^2 + min z \alpha \|grad x - z \|_1 + \beta \|eps z \|_1
 *
 * min x,z \| Ix - y \|_2^2 + \alpha \|grad x - z\|_1 + \beta \|eps z\|_1
 * */
	
int main_tgv(int argc, char* argv[argc])
{
	float lambda = 0.;
	int flags = -1;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_FLOAT(true, &lambda, "lambda"),
		ARG_INT(true, &flags, "flags"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = { };

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();
	
	long in_dims[DIMS];

	complex float* in_data = load_cfl(in_file, DIMS, in_dims);

	assert(1 == in_dims[DIMS - 1]);


	struct reg2 reg2 = tgvreg(flags, lambda, DIMS, in_dims);

	long out_dims[DIMS];
	md_copy_dims(DIMS, out_dims, linop_domain(reg2.linop[0])->dims);


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

