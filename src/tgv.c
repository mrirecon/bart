/* Copyright 2014,2019. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014,2019 Martin Uecker <uecker@eecs.berkeley.edu>
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

#include "iter/prox.h"
#include "iter/thresh.h"
#include "iter/iter2.h"
#include "iter/iter.h"


#ifndef DIMS
#define DIMS 16
#endif


static const char usage_str[] = "<lambda> <flags> <input> <output>";
static const char help_str[] = "Perform total generalized variation denoising along dims <flags>.\n";


/* TGV
 * 
 * min x \|Ix - y\|_2^2 + min z \alpha \|grad x - z \|_1 + \beta \|eps z \|_1
 *
 * min x,z \| Ix - y \|_2^2 + \alpha \|grad x - z\|_1 + \beta \|eps z\|_1
 * */
	
int main_tgv(int argc, char* argv[])
{
	mini_cmdline(&argc, argv, 4, usage_str, help_str);

	num_init();

	float lambda = atof(argv[1]);
	int flags = atoi(argv[2]);
	
	long in_dims[DIMS];

	complex float* in_data = load_cfl(argv[3], DIMS, in_dims);

	assert(1 == in_dims[DIMS - 1]);

	long out_dims[DIMS];
	
	const struct linop_s* grad1 = linop_grad_create(DIMS - 1, in_dims, DIMS - 1, flags);
	const struct linop_s* grad2x = linop_grad_create(DIMS + 0, linop_codomain(grad1)->dims, DIMS + 0, flags);


	auto grad2a = linop_transpose_create(DIMS + 1, DIMS - 1, DIMS + 0, linop_codomain(grad2x)->dims);
	auto grad2b = linop_identity_create(DIMS + 1, linop_codomain(grad2x)->dims);
	auto grad2c = linop_plus(grad2a, grad2b);

	linop_free(grad2a);
	linop_free(grad2b);

	auto grad2 = linop_chain(grad2x, grad2c);

	linop_free(grad2x);
	linop_free(grad2c);


	long grd_dims[DIMS];
	md_copy_dims(DIMS, grd_dims, linop_codomain(grad1)->dims);

	md_copy_dims(DIMS, out_dims, grd_dims);
	out_dims[DIMS - 1]++;

	complex float* out_data = create_cfl(argv[4], DIMS, out_dims);


	long pos1[DIMS] = { [DIMS - 1] = 0 };
	auto grad1b = linop_extract_create(DIMS, pos1, in_dims, out_dims);
	auto grad1c = linop_reshape_create(DIMS - 1, linop_domain(grad1)->dims, DIMS, in_dims);

	auto grad1d1 = linop_chain(grad1b, grad1c);
	auto grad1d = linop_chain(grad1d1, grad1);

	linop_free(grad1);
	linop_free(grad1b);
	linop_free(grad1c);
	linop_free(grad1d1);

	long pos1b[DIMS] = { [DIMS - 1] = 1 };
	auto grad1e = linop_extract_create(DIMS, pos1b, grd_dims, out_dims);
	const struct linop_s* grad1f = linop_plus(grad1e, grad1d);

	linop_free(grad1e);
	linop_free(grad1d);

	long pos2[DIMS] = { [DIMS - 1] = 1 };
	auto grad2e = linop_extract_create(DIMS, pos2, grd_dims, out_dims);
	const struct linop_s* grad2f = linop_chain(grad2e, grad2);

	linop_free(grad2);
	linop_free(grad2e);

	auto p1 = prox_thresh_create(DIMS + 0, linop_codomain(grad1f)->dims, lambda, 0u);
	auto p2 = prox_thresh_create(DIMS + 1, linop_codomain(grad2f)->dims, lambda, 0u);

	auto id = linop_extract_create(DIMS, (long[DIMS]){ 0 }, in_dims, out_dims);

	struct iter_admm_conf conf = iter_admm_defaults;

	complex float* adj = md_alloc(DIMS, out_dims, CFL_SIZE);

	linop_adjoint(id, DIMS, out_dims, adj, DIMS, in_dims, in_data);

	conf.maxiter = 100;
	conf.rho = .1;

	iter2_admm(CAST_UP(&conf), id->normal,
		   2, MAKE_ARRAY(p1, p2), MAKE_ARRAY(grad1f, grad2f),
		   NULL, NULL,
		   2 * md_calc_size(DIMS, out_dims), (float*)out_data, (const float*)adj,
		   NULL);

	md_free(adj);

	linop_free(id);
	linop_free(grad1f);
	linop_free(grad2c);

	operator_p_free(p1);
	operator_p_free(p2);
	
	unmap_cfl(DIMS, in_dims, in_data);
	unmap_cfl(DIMS, out_dims, out_data);

	return 0;
}

