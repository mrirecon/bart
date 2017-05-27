/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>

#include "misc/debug.h"

#include "num/rand.h"
#include "num/iovec.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "linops/linop.h"

#include "lintest.h"


float linop_test_adjoint(const struct linop_s* op)
{
	int N_dom = linop_domain(op)->N;
	int N_cod = linop_codomain(op)->N;

	long dims_dom[N_dom];
	md_copy_dims(N_dom, dims_dom, linop_domain(op)->dims);

	long dims_cod[N_cod];
	md_copy_dims(N_cod, dims_cod, linop_codomain(op)->dims);

	complex float* tmp1 = md_alloc(N_dom, dims_dom, CFL_SIZE);
	complex float* tmp2 = md_alloc(N_cod, dims_cod, CFL_SIZE);
	complex float* tmp3 = md_alloc(N_cod, dims_cod, CFL_SIZE);
	complex float* tmp4 = md_alloc(N_dom, dims_dom, CFL_SIZE);

	md_gaussian_rand(N_dom, dims_dom, tmp1);
	linop_forward_unchecked(op, tmp3, tmp1);

	md_gaussian_rand(N_cod, dims_cod, tmp2);
	linop_adjoint_unchecked(op, tmp4, tmp2);

	complex float sc1 = md_zscalar(N_dom, dims_dom, tmp1, tmp4);
	complex float sc2 = md_zscalar(N_cod, dims_cod, tmp3, tmp2);

	md_free(tmp1);
	md_free(tmp2);
	md_free(tmp3);
	md_free(tmp4);

	debug_printf(DP_DEBUG4, "- %f+%fi - %f+%fi -\n", crealf(sc1), cimagf(sc1), crealf(sc2), cimagf(sc2));

	return cabsf(sc1 - sc2);
}



float linop_test_normal(const struct linop_s* op)
{
	int N_dom = linop_domain(op)->N;
	int N_cod = linop_codomain(op)->N;

	long dims_dom[N_dom];
	md_copy_dims(N_dom, dims_dom, linop_domain(op)->dims);

	long dims_cod[N_cod];
	md_copy_dims(N_cod, dims_cod, linop_codomain(op)->dims);

	complex float* tmp1 = md_alloc(N_dom, dims_dom, CFL_SIZE);
	complex float* tmp2 = md_alloc(N_dom, dims_dom, CFL_SIZE);
	complex float* tmp3 = md_alloc(N_cod, dims_cod, CFL_SIZE);
	complex float* tmp4 = md_alloc(N_dom, dims_dom, CFL_SIZE);

	md_gaussian_rand(N_dom, dims_dom, tmp1);

	linop_forward_unchecked(op, tmp3, tmp1);
	linop_adjoint_unchecked(op, tmp4, tmp3);

	linop_normal_unchecked(op, tmp2, tmp1);

	float nrmse = md_znrmse(N_dom, dims_dom, tmp2, tmp4);

	md_free(tmp1);
	md_free(tmp2);
	md_free(tmp3);
	md_free(tmp4);

	return nrmse;
}


float linop_test_inverse(const struct linop_s* op)
{
	int N_dom = linop_domain(op)->N;

	long dims_dom[N_dom];
	md_copy_dims(N_dom, dims_dom, linop_domain(op)->dims);

	complex float* tmp1 = md_alloc(N_dom, dims_dom, CFL_SIZE);
	complex float* tmp2 = md_alloc(N_dom, dims_dom, CFL_SIZE);
	complex float* tmp3 = md_alloc(N_dom, dims_dom, CFL_SIZE);

	md_gaussian_rand(N_dom, dims_dom, tmp1);

	linop_normal_unchecked(op, tmp3, tmp1);
	linop_norm_inv_unchecked(op, 0., tmp2, tmp3);

	float nrmse = md_znrmse(N_dom, dims_dom, tmp2, tmp1);

	md_free(tmp1);
	md_free(tmp2);
	md_free(tmp3);

	return nrmse;
}


