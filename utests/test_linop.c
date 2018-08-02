/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/ops.h"

#include "linops/someops.h"
#include "linops/linop.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "utest.h"




static bool test_linop_plus(void)
{
	enum { N = 3 };
	long dims[N] = { 8, 4, 2 };


	complex float val1a = 2.;
	complex float val1b = 3.;
	struct linop_s* diaga = linop_cdiag_create(N, dims, 0, &val1a);
	struct linop_s* diagb = linop_cdiag_create(N, dims, 0, &val1b);
	struct linop_s* plus = linop_plus(diaga, diagb);

	complex float val2 = 5.;
	struct linop_s* diag2 = linop_cdiag_create(N, dims, 0, &val2);

	complex float* in = md_alloc(N, dims, CFL_SIZE);
	complex float* dst1 = md_alloc(N, dims, CFL_SIZE);
	complex float* dst2 = md_alloc(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, in);

	linop_forward(plus, N, dims, dst1, N, dims, in);
	linop_forward(diag2, N, dims, dst2, N, dims, in);

	double err = md_znrmse(N, dims, dst1, dst2);

	linop_free(diaga);
	linop_free(diagb);
	linop_free(plus);
	linop_free(diag2);

	md_free(in);
	md_free(dst1);
	md_free(dst2);

	return (err < UT_TOL);
}



UT_REGISTER_TEST(test_linop_plus);


static bool test_linop_null(void)
{
	long dims[1] = { 5 };
	const struct linop_s* l = linop_null_create(1, dims, dims);

	bool ok = true;

	ok &= operator_zero_or_null_p(l->forward);
	ok &= operator_zero_or_null_p(l->adjoint);
	ok &= operator_zero_or_null_p(l->normal);

	linop_free(l);

	return ok;
}


UT_REGISTER_TEST(test_linop_null);

