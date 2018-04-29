/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/iovec.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"

#include "linops/linop.h"
#include "linops/lintest.h"
#include "linops/someops.h"

#include "nlops/zexp.h"
#include "nlops/tenmul.h"
#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/nltest.h"

#include "utest.h"







static bool test_nlop_cast_pos(void)
{
	bool ok = true;
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	struct linop_s* l = linop_identity_create(N, dims);
	struct nlop_s* d = nlop_from_linop(l);

	if (l == linop_from_nlop(d)) // maybe just require != NULL ?
		ok = false;

	linop_free(l);
	nlop_free(d);

	return ok;
}



UT_REGISTER_TEST(test_nlop_cast_pos);



static bool test_nlop_cast_neg(void)
{
	bool ok = true;
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	struct nlop_s* d = nlop_zexp_create(N, dims);

	if (NULL != linop_from_nlop(d))
		ok = false;

	nlop_free(d);

	return ok;
}



UT_REGISTER_TEST(test_nlop_cast_neg);








static bool test_nlop_chain(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	complex float val = 2.;

	struct linop_s* cdiag = linop_cdiag_create(N, dims, 0, &val);
	struct nlop_s* diag = nlop_from_linop(cdiag);
	struct nlop_s* zexp = nlop_zexp_create(N, dims);
	struct nlop_s* zexp2 = nlop_chain(zexp, diag);

	double err = nlop_test_derivative(zexp2);

	nlop_free(zexp2);
	nlop_free(zexp);
	nlop_free(diag);
	linop_free(cdiag);

	UT_ASSERT(err < 1.E-2);
}



UT_REGISTER_TEST(test_nlop_chain);




static bool test_nlop_tenmul(void)
{
	enum { N = 3 };
	long odims[N] = { 10, 1, 3 };
	long idims1[N] = { 1, 7, 3 };
	long idims2[N] = { 10, 7, 1 };

	complex float* dst1 = md_alloc(N, odims, CFL_SIZE);
	complex float* dst2 = md_alloc(N, odims, CFL_SIZE);
	complex float* src1 = md_alloc(N, idims1, CFL_SIZE);
	complex float* src2 = md_alloc(N, idims2, CFL_SIZE);

	md_gaussian_rand(N, idims1, src1);
	md_gaussian_rand(N, idims2, src2);

	struct nlop_s* tenmul = nlop_tenmul_create(N, odims, idims1, idims2);

	md_ztenmul(N, odims, dst1, idims1, src1, idims2, src2);

	nlop_generic_apply_unchecked(tenmul, 3, (void*[]){ dst2, src1, src2 });

	double err = md_znrmse(N, odims, dst2, dst1);

	nlop_free(tenmul);

	md_free(src1);
	md_free(src2);
	md_free(dst1);
	md_free(dst2);

	UT_ASSERT(err < UT_TOL);
}



UT_REGISTER_TEST(test_nlop_tenmul);





static bool test_nlop_tenmul_der(void)
{
	enum { N = 3 };
	long odims[N] = { 10, 1, 3 };
	long idims1[N] = { 1, 7, 3 };
	long idims2[N] = { 10, 7, 1 };

	complex float* dst1 = md_alloc(N, odims, CFL_SIZE);
	complex float* dst2 = md_alloc(N, odims, CFL_SIZE);
	complex float* dst3 = md_alloc(N, odims, CFL_SIZE);
	complex float* src1 = md_alloc(N, idims1, CFL_SIZE);
	complex float* src2 = md_alloc(N, idims2, CFL_SIZE);

	md_gaussian_rand(N, idims1, src1);
	md_gaussian_rand(N, idims2, src2);

	struct nlop_s* tenmul = nlop_tenmul_create(N, odims, idims1, idims2);

	nlop_generic_apply_unchecked(tenmul, 3, (void*[]){ dst1, src1, src2 });

	const struct linop_s* der1 = nlop_get_derivative(tenmul, 0, 0);
	const struct linop_s* der2 = nlop_get_derivative(tenmul, 0, 1);

	linop_forward(der1, N, odims, dst2, N, idims1, src1);
	linop_forward(der2, N, odims, dst3, N, idims2, src2);
	

	double err = md_znrmse(N, odims, dst2, dst1)
		   + md_znrmse(N, odims, dst3, dst1);

	nlop_free(tenmul);

	md_free(src1);
	md_free(src2);
	md_free(dst1);
	md_free(dst2);
	md_free(dst3);

	UT_ASSERT(err < UT_TOL);
}

UT_REGISTER_TEST(test_nlop_tenmul_der);




static bool test_nlop_zexp(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	complex float* dst1 = md_alloc(N, dims, CFL_SIZE);
	complex float* dst2 = md_alloc(N, dims, CFL_SIZE);
	complex float* src = md_alloc(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, src);

	struct nlop_s* zexp = nlop_zexp_create(N, dims);

	md_zexp(N, dims, dst1, src);

	nlop_apply(zexp, N, dims, dst2, N, dims, src);

	double err = md_znrmse(N, dims, dst2, dst1);

	nlop_free(zexp);

	md_free(src);
	md_free(dst1);
	md_free(dst2);

	UT_ASSERT(err < UT_TOL);
}



UT_REGISTER_TEST(test_nlop_zexp);



static bool test_nlop_tenmul_der2(void)
{
	enum { N = 3 };
	long odims[N] = { 10, 1, 3 };
	long idims1[N] = { 1, 7, 3 };
	long idims2[N] = { 10, 7, 1 };

	struct nlop_s* tenmul = nlop_tenmul_create(N, odims, idims1, idims2);
	struct nlop_s* flat = nlop_flatten(tenmul);

	double err = nlop_test_derivative(flat);

	nlop_free(flat);
	nlop_free(tenmul);

	UT_ASSERT((!safe_isnanf(err)) && (err < 1.5E-2));
}

UT_REGISTER_TEST(test_nlop_tenmul_der2);


static void random_application(const struct nlop_s* nlop)
{
	auto dom = nlop_domain(nlop);
	auto cod = nlop_codomain(nlop);

	complex float* in = md_alloc(dom->N, dom->dims, dom->size);
	complex float* dst = md_alloc(cod->N, cod->dims, cod->size);

	md_gaussian_rand(dom->N, dom->dims, in);

	// define position for derivatives
	nlop_apply(nlop, cod->N, cod->dims, dst, dom->N, dom->dims, in);

	md_free(in);
	md_free(dst);
}


static bool test_nlop_tenmul_der_adj(void)
{
	enum { N = 3 };
	long odims[N] = { 10, 1, 3 };
	long idims1[N] = { 1, 7, 3 };
	long idims2[N] = { 10, 7, 1 };

	struct nlop_s* tenmul = nlop_tenmul_create(N, odims, idims1, idims2);

	struct nlop_s* flat = nlop_flatten(tenmul);

	random_application(flat);

	double err = linop_test_adjoint(nlop_get_derivative(flat, 0, 0));

	nlop_free(flat);
	nlop_free(tenmul);

	UT_ASSERT((!safe_isnanf(err)) && (err < 6.E-2));
}

UT_REGISTER_TEST(test_nlop_tenmul_der_adj);






static bool test_nlop_zexp_derivative(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	struct nlop_s* zexp = nlop_zexp_create(N, dims);

	double err = nlop_test_derivative(zexp);

	nlop_free(zexp);

	UT_ASSERT(err < 1.E-2);
}



UT_REGISTER_TEST(test_nlop_zexp_derivative);




static bool test_nlop_zexp_der_adj(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	struct nlop_s* zexp = nlop_zexp_create(N, dims);

	complex float* dst = md_alloc(N, dims, CFL_SIZE);
	complex float* src = md_alloc(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, src);

	nlop_apply(zexp, N, dims, dst, N, dims, src);

	md_free(src);
	md_free(dst);

	float err = linop_test_adjoint(nlop_get_derivative(zexp, 0, 0));

	nlop_free(zexp);

	UT_ASSERT(err < 1.E-2);
}



UT_REGISTER_TEST(test_nlop_zexp_der_adj);




static bool test_nlop_combine(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	struct nlop_s* zexp = nlop_zexp_create(N, dims);
	struct linop_s* lid = linop_identity_create(N, dims);
	struct nlop_s* id = nlop_from_linop(lid);
	linop_free(lid);
	struct nlop_s* comb = nlop_combine(zexp, id);

	complex float* in1 = md_alloc(N, dims, CFL_SIZE);
	complex float* in2 = md_alloc(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, in1);
	md_gaussian_rand(N, dims, in2);


	complex float* out1 = md_alloc(N, dims, CFL_SIZE);
	complex float* out2 = md_alloc(N, dims, CFL_SIZE);
	complex float* out3 = md_alloc(N, dims, CFL_SIZE);
	complex float* out4 = md_alloc(N, dims, CFL_SIZE);

	nlop_apply(zexp, N, dims, out1, N, dims, in1);
	nlop_apply(id, N, dims, out2, N, dims, in2);

	nlop_generic_apply_unchecked(comb, 4, (void*[]){ out3, out4, in1, in2 });

	double err = md_znrmse(N, dims, out4, out2)
		   + md_znrmse(N, dims, out3, out1);

	md_free(in1);
	md_free(in2);
	md_free(out1);
	md_free(out2);
	md_free(out3);
	md_free(out4);

	nlop_free(comb);
	nlop_free(id);
	nlop_free(zexp);

	UT_ASSERT((!safe_isnanf(err)) && (err < 1.E-2));
}



UT_REGISTER_TEST(test_nlop_combine);



static bool test_nlop_combine_der1(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };	// FIXME: this test is broken

	struct nlop_s* zexp = nlop_zexp_create(N, dims);
	struct linop_s* lid = linop_identity_create(N, dims);
	struct nlop_s* id = nlop_from_linop(lid);
	linop_free(lid);
	struct nlop_s* comb = nlop_combine(zexp, id);

	random_application(zexp);

	complex float* in1 = md_alloc(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, in1);

	complex float* out1 = md_alloc(N, dims, CFL_SIZE);
	complex float* out2 = md_alloc(N, dims, CFL_SIZE);
	complex float* out3 = md_alloc(N, dims, CFL_SIZE);
	complex float* out4 = md_alloc(N, dims, CFL_SIZE);

	linop_forward(nlop_get_derivative(comb, 0, 1), N, dims, out1, N, dims, in1);

	if (0. != md_znorm(N, dims, out1))
		return false;

	linop_forward(nlop_get_derivative(comb, 1, 0), N, dims, out1, N, dims, in1);

	if (0. != md_znorm(N, dims, out1))
		return false;

	nlop_derivative(zexp, N, dims, out1, N, dims, in1);

	linop_forward(nlop_get_derivative(comb, 0, 0), N, dims, out2, N, dims, in1);

	nlop_derivative(id, N, dims, out3, N, dims, in1);

	linop_forward(nlop_get_derivative(comb, 1, 1), N, dims, out4, N, dims, in1);

	double err = md_znrmse(N, dims, out1, out2);

	md_free(in1);
	md_free(out1);
	md_free(out2);
	md_free(out3);
	md_free(out4);


	nlop_free(comb);
	nlop_free(id);
	nlop_free(zexp);

	return (0. == err);
}



UT_REGISTER_TEST(test_nlop_combine_der1);



static bool test_nlop_comb_flat_der(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	struct nlop_s* zexp1 = nlop_zexp_create(N, dims);
	struct nlop_s* zexp2 = nlop_zexp_create(N, dims);

	random_application(zexp1);
	random_application(zexp2);

	struct nlop_s* comb = nlop_combine(zexp1, zexp2);
	struct nlop_s* flat = nlop_flatten(comb);

	auto iov = nlop_domain(flat);

	complex float* in = md_alloc(iov->N, iov->dims, iov->size);
	complex float* dst2 = md_alloc(iov->N, iov->dims, iov->size);
	complex float* dst = md_alloc(N, dims, CFL_SIZE);


	md_gaussian_rand(N, dims, in);

	nlop_derivative(zexp1, N, dims, dst, N, dims, in);

	nlop_derivative(flat, iov->N, iov->dims, dst2, iov->N, iov->dims, in);

	double err = md_znrmse(N, dims, dst2, dst);

	md_free(in);
	md_free(dst);
	md_free(dst2);

	nlop_free(flat);
	nlop_free(comb);
	nlop_free(zexp1);
	nlop_free(zexp2);

	UT_ASSERT((!safe_isnanf(err)) && (err < 1.E-2));
}



UT_REGISTER_TEST(test_nlop_comb_flat_der);



static bool test_nlop_combine_derivative(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };	// FIXME: this test is broken

	struct nlop_s* zexp1 = nlop_zexp_create(N, dims);
	struct nlop_s* zexp2 = nlop_zexp_create(N, dims);
	struct nlop_s* comb = nlop_combine(zexp1, zexp2);
	struct nlop_s* flat = nlop_flatten(comb);

	double err = nlop_test_derivative(flat);

	nlop_free(flat);
	nlop_free(comb);
	nlop_free(zexp1);
	nlop_free(zexp2);

	UT_ASSERT((!safe_isnanf(err)) && (err < 2.E-2));
}



UT_REGISTER_TEST(test_nlop_combine_derivative);




static bool test_nlop_link(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	complex float val = 2.;

	struct linop_s* cdiag = linop_cdiag_create(N, dims, 0, &val);
	struct nlop_s* diag = nlop_from_linop(cdiag);
	linop_free(cdiag);

	struct nlop_s* zexp = nlop_zexp_create(N, dims);
	struct nlop_s* zexp2 = nlop_chain(zexp, diag);
	struct nlop_s* zexp3 = nlop_combine(diag, zexp);
	struct nlop_s* zexp4 = nlop_link(zexp3, 1, 0);

	complex float* in = md_alloc(N, dims, CFL_SIZE);
	complex float* dst1 = md_alloc(N, dims, CFL_SIZE);
	complex float* dst2 = md_alloc(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, in);

	nlop_apply(zexp2, N, dims, dst1, N, dims, in);
	nlop_apply(zexp4, N, dims, dst2, N, dims, in);

	double err = md_znrmse(N, dims, dst2, dst1);

	md_free(in);
	md_free(dst1);
	md_free(dst2);

	nlop_free(zexp4);
	nlop_free(zexp3);
	nlop_free(zexp2);
	nlop_free(zexp);
	nlop_free(diag);

	UT_ASSERT(err < 1.E-6);
}



UT_REGISTER_TEST(test_nlop_link);



