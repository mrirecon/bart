/* Copyright 2018-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <math.h>

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




static bool test_linop_stack(void)
{
	enum { N = 3 };
	long dims[N] = { 8, 4, 1 };
	long dims2[N] = { 8, 4, 2 };

	complex float val1a = 2.;
	complex float val1b = 3.;

	struct linop_s* diaga = linop_cdiag_create(N, dims, 0, &val1a);
	struct linop_s* diagb = linop_cdiag_create(N, dims, 0, &val1b);
	struct linop_s* stack = linop_stack(2, 2, diaga, diagb);

	complex float* in = md_alloc(N, dims2, CFL_SIZE);
	complex float* out = md_alloc(N, dims2, CFL_SIZE);

	md_zfill(N, dims2, in, 1.);

	bool ok = true;
	double n, n2, err;


	linop_forward(stack, N, dims2, out, N, dims2, in);

	n = powf(md_znorm(N, dims2, out), 2);
	n2 = (powf(val1a, 2.) + powf(val1b, 2.)) * md_calc_size(N, dims);
	err = fabs(n - n2);

#ifdef  __clang__
	ok &= (err < 100. * UT_TOL);
#else
	ok &= (err < UT_TOL);
#endif

	linop_adjoint(stack, N, dims2, out, N, dims2, in);

	n = powf(md_znorm(N, dims2, out), 2);
	n2 = (powf(val1a, 2.) + powf(val1b, 2.)) * md_calc_size(N, dims);
	err = fabs(n - n2);

#ifdef  __clang__
	ok &= (err < 100. * UT_TOL);
#else
	ok &= (err < UT_TOL);
#endif


	linop_normal(stack, N, dims2, out, in);

	n = powf(md_znorm(N, dims2, out), 2);
	n2 = (powf(val1a, 4.) + powf(val1b, 4.)) * md_calc_size(N, dims);
	err = fabs(n - n2);

	ok &= (err < 1.E-3);


	linop_free(diaga);
	linop_free(diagb);
	linop_free(stack);

	md_free(in);
	md_free(out);

	return ok;
}


UT_REGISTER_TEST(test_linop_stack);



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

