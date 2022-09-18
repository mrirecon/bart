/* Copyright 2018-2021. Uecker Lab. University Medical Center Göttingen.
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
#include "num/ops.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"

#include "linops/linop.h"
#include "linops/lintest.h"
#include "linops/someops.h"

#include "nlops/zexp.h"
#include "nlops/someops.h"
#include "nlops/tenmul.h"
#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/nltest.h"
#include "nlops/stack.h"
#include "nlops/const.h"
#include "nlops/checkpointing.h"
#include "nlops/mri_ops.h"

#include "utest.h"







static bool test_nlop_cast_pos(void)
{
	bool ok = true;
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	struct linop_s* l = linop_identity_create(N, dims);
	struct nlop_s* d = nlop_from_linop(l);
	const struct linop_s* l2;

	if (l == (l2 = linop_from_nlop(d))) // maybe just require != NULL ?
		ok = false;

	linop_free(l2);
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

	UT_ASSERT(ok);
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




static bool test_nlop_tenmul2(bool permute)
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

	struct nlop_s* tenmul = NULL;

	if (!permute) {

		tenmul = nlop_tenmul_create(N, odims, idims1, idims2);

	} else {

		struct nlop_s* tmp = nlop_tenmul_create(N, odims, idims2, idims1);

		tenmul = nlop_permute_inputs(tmp, 2, (int[]){ 1, 0 });

		nlop_free(tmp);
	}

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


static bool test_nlop_tenmul(void)
{
	UT_ASSERT(test_nlop_tenmul2(false));
}

UT_REGISTER_TEST(test_nlop_tenmul);



static bool test_nlop_permute(void)
{
	UT_ASSERT(test_nlop_tenmul2(true));
}

UT_REGISTER_TEST(test_nlop_permute);




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
		UT_ASSERT(false);

	linop_forward(nlop_get_derivative(comb, 1, 0), N, dims, out1, N, dims, in1);

	if (0. != md_znorm(N, dims, out1))
		UT_ASSERT(false);

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

	 UT_ASSERT(0. == err);
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


	md_gaussian_rand(iov->N, iov->dims, in);

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


static bool test_nlop_reshape(void)
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

	auto op = nlop_tenmul_create(N, odims, idims1, idims2);

	long nodims[1] = { md_calc_size(N, odims) };
	long nidims1[3] = { 1, 1, md_calc_size(N, idims1) };
	long nidims2[2] = { md_calc_size(N, idims2), 1 };

	auto op_reshape = nlop_reshape_out(op, 0, 1, nodims);
	op_reshape = nlop_reshape_in_F(op_reshape, 0, 3, nidims1);
	op_reshape = nlop_reshape_in_F(op_reshape, 1, 2, nidims2);

	nlop_generic_apply_unchecked(op, 3, (void*[]){ dst1, src1, src2 });
	nlop_generic_apply_unchecked(op_reshape, 3, (void*[]){ dst2, src1, src2 });

	double err = md_znrmse(N, odims, dst2, dst1);

	auto der1 = nlop_get_derivative(op, 0, 0);
	auto der1_resh = nlop_get_derivative(op_reshape, 0, 0);

	linop_forward_unchecked(der1, dst1, src1);
	linop_forward_unchecked(der1_resh, dst2, src1);

	err += md_znrmse(N, odims, dst2, dst1);

	auto der2 = nlop_get_derivative(op, 0, 1);
	auto der2_resh = nlop_get_derivative(op_reshape, 0, 1);

	linop_forward_unchecked(der2, dst1, src2);
	linop_forward_unchecked(der2_resh, dst2, src2);

	err += md_znrmse(N, odims, dst2, dst1);

	nlop_free(op_reshape);
	nlop_free(op);

	md_free(src1);
	md_free(src2);
	md_free(dst1);
	md_free(dst2);

	UT_ASSERT(err < UT_TOL);
}

UT_REGISTER_TEST(test_nlop_reshape);



struct count_op_s {

	INTERFACE(linop_data_t);
	int* counter;
};

static DEF_TYPEID(count_op_s);

static void count_forward(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct count_op_s* data = CAST_DOWN(count_op_s, _data);

	(*data->counter)++;

	long dim[2] = { 1 };
	md_copy(1, dim, dst, src, CFL_SIZE);
}

static void count_free(const linop_data_t* _data)
{
	const struct count_op_s* data = CAST_DOWN(count_op_s, _data);
	xfree(data);
}

static struct linop_s* linop_counter_create(int* counter)
{
	PTR_ALLOC(struct count_op_s, data);
	SET_TYPEID(count_op_s, data);

	data->counter = counter;

	long dim[1] = { 1 };

	return linop_create(1, dim, 1, dim, CAST_UP(PTR_PASS(data)), count_forward, count_forward, count_forward, NULL, count_free);
}

static bool test_nlop_parallel_derivatives(void)
{
	long dim[1] = { 1 };
	int counter = 0;

	complex float in[1] = { 1. };
	complex float out1[1] = { 1. };
	complex float out2[1] = { 1. };
	complex float* outs[2] = { out1, out2 };

	auto countop1 = linop_counter_create(&counter);
	auto chain = linop_chain(countop1, countop1);

	operator_apply_joined_unchecked(2, MAKE_ARRAY(chain->forward, chain->forward), outs, in);

	bool result = (2 == counter);

	counter = 0;

	auto plus_op = linop_plus(chain, chain);

	linop_forward_unchecked(plus_op, out1, in);

	linop_free(plus_op);

	result = result && (2 == counter);

	auto tenmul_op = nlop_tenmul_create(1, dim, dim ,dim);

	const struct nlop_s* tenmul_chain = nlop_chain2_FF(tenmul_op, 0, nlop_from_linop(chain), 0);

	tenmul_chain = nlop_chain2_FF(nlop_from_linop(chain), 0, tenmul_chain, 0);
	tenmul_chain = nlop_chain2_FF(nlop_from_linop(chain), 0, tenmul_chain, 0);
	tenmul_chain = nlop_reshape_in_F(tenmul_chain, 0, 1, dim);

	nlop_generic_apply_unchecked(tenmul_chain, 3, (void*[]){ out1, out2, in });

	counter = 0;

	const struct operator_s* op1 = nlop_get_derivative(tenmul_chain, 0, 0)->adjoint;
	const struct operator_s* op2 = nlop_get_derivative(tenmul_chain, 0, 1)->adjoint;

	operator_apply_joined_unchecked(2, MAKE_ARRAY(op1, op2), outs, in);

	result = result && (6 == counter);

	auto tmp = nlop_reshape_in(tenmul_chain, 0, 1, dim);
	const struct nlop_s* bridge = nlop_chain2(tmp, 0, tenmul_chain, 0);
	bridge = nlop_dup_F(bridge, 0, 1);
	bridge = nlop_dup_F(bridge, 0, 1);

	counter = 0;

	linop_adjoint_unchecked(nlop_get_derivative(bridge, 0, 0), out1, in);

	result = result && ((12 == counter) || (8 == counter)); // 8 if linop sums are optimized

	nlop_free(bridge);
	nlop_free(tmp);
	nlop_free(tenmul_chain);

	linop_free(countop1);
	linop_free(chain);

	UT_ASSERT(result);
}

UT_REGISTER_TEST(test_nlop_parallel_derivatives);



static bool test_stack(void)
{
	enum { N = 3 };
	long dims1[N] = { 3, 2, 7 };
	long dims2[N] = { 3, 5, 7 };
	long dims[N] = { 3, 7, 7 };

	complex float* in = md_alloc(N, dims, CFL_SIZE);
	complex float* out = md_alloc(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, in);

	float err = 0;
	const struct nlop_s* nlop_test;

	nlop_test = nlop_stack_create(N, dims, dims1, dims2, 1);
	nlop_test = nlop_stack_inputs_F(nlop_test, 0, 1, 1);

	nlop_apply(nlop_test, N, dims, out, N, dims, in);

	err += md_zrmse(N, dims, in, out);

	nlop_free(nlop_test);

	nlop_test = nlop_stack_create(N, dims, dims1, dims2, 1);
	nlop_test = nlop_permute_inputs_F(nlop_test, 2, MAKE_ARRAY(1, 0));
	nlop_test = nlop_stack_inputs_F(nlop_test, 1, 0, 1);

	nlop_apply(nlop_test, N, dims, out, N, dims, in);

	err += md_zrmse(N, dims, in, out);

	nlop_free(nlop_test);

	md_free(in);
	md_free(out);

	UT_ASSERT(1.e-7 > err);
}

UT_REGISTER_TEST(test_stack);

static bool test_stack_multiple(void)
{
	enum { N = 3 };
	long idims1_1[N] = { 3, 5, 3 };
	long idims2_1[N] = { 1, 5, 1 };
	long odims_1[N] = { 3, 5, 3 };

	long idims1_2[N] = { 3, 5, 1 };
	long idims2_2[N] = { 1, 5, 1 };
	long odims_2[N] = { 3, 5, 1 };

	const struct nlop_s* nlops[3];

	for (int i = 0; i < 3; i++)
		nlops[i] = nlop_tenmul_create(N, odims_2, idims1_2, idims2_2);
	
	const struct nlop_s* nlop1 = nlop_stack_multiple_F(3, nlops, 2, (int[2]){ 2, -1}, 1, (int[1]){ 2 });
	const struct nlop_s* nlop2 = nlop_tenmul_create(N, odims_1, idims1_1, idims2_1);

	bool ok = compare_nlops(nlop1, nlop2, true, true, true, UT_TOL);

	nlop_free(nlop1);
	nlop_free(nlop2);

	UT_ASSERT(ok);
}

UT_REGISTER_TEST(test_stack_multiple);

static bool test_stack_multiple2(void)
{
	enum { N = 3 };
	long idims1_1[N] = { 3, 5, 3 };
	long idims2_1[N] = { 1, 1, 3 };
	long odims_1[N] = { 3, 5, 3 };

	long idims1_2[N] = { 3, 1, 3 };
	long idims2_2[N] = { 1, 1, 3 };
	long odims_2[N] = { 3, 1, 3 };

	const struct nlop_s* nlops[5];

	for (int i = 0; i < 5; i++)
		nlops[i] = nlop_tenmul_create(N, odims_2, idims1_2, idims2_2);
	
	const struct nlop_s* nlop1 = nlop_stack_multiple_F(5, nlops, 2, (int[2]){ 1, -1}, 1, (int[1]){ 1 });
	const struct nlop_s* nlop2 = nlop_tenmul_create(N, odims_1, idims1_1, idims2_1);

	bool ok = compare_nlops(nlop1, nlop2, true, true, true, UT_TOL);

	nlop_free(nlop1);
	nlop_free(nlop2);

	UT_ASSERT(ok);
}

UT_REGISTER_TEST(test_stack_multiple2);

static bool test_stack_multiple_container(void)
{
	enum { N = 3 };
	long idims1_1[N] = { 3, 5, 3 };
	long idims2_1[N] = { 1, 5, 1 };
	long odims_1[N] = { 3, 5, 3 };

	long idims1_2[N] = { 3, 5, 1 };
	long idims2_2[N] = { 1, 5, 1 };
	long odims_2[N] = { 3, 5, 1 };

	const struct nlop_s* nlops[3];

	for (int i = 0; i < 3; i++)
		nlops[i] = nlop_tenmul_create(N, odims_2, idims1_2, idims2_2);
	
	const struct nlop_s* nlop1 = nlop_stack_container_create_F(3, nlops, 2, (int[2]){ 2, -1}, 1, (int[1]){ 2 });
	const struct nlop_s* nlop2 = nlop_tenmul_create(N, odims_1, idims1_1, idims2_1);

	bool ok = compare_nlops(nlop1, nlop2, true, true, true, UT_TOL);

	nlop_free(nlop1);
	nlop_free(nlop2);

	UT_ASSERT(ok);
}

UT_REGISTER_TEST(test_stack_multiple_container);

static bool test_stack_multiple_container2(void)
{
	enum { N = 3 };
	long idims1_1[N] = { 3, 5, 3 };
	long idims2_1[N] = { 1, 1, 3 };
	long odims_1[N] = { 3, 5, 3 };

	long idims1_2[N] = { 3, 1, 3 };
	long idims2_2[N] = { 1, 1, 3 };
	long odims_2[N] = { 3, 1, 3 };

	const struct nlop_s* nlops[5];

	for (int i = 0; i < 5; i++)
		nlops[i] = nlop_tenmul_create(N, odims_2, idims1_2, idims2_2);
	
	const struct nlop_s* nlop1 = nlop_stack_container_create_F(5, nlops, 2, (int[2]){ 1, -1}, 1, (int[1]){ 1 });
	const struct nlop_s* nlop2 = nlop_tenmul_create(N, odims_1, idims1_1, idims2_1);

	bool ok = compare_nlops(nlop1, nlop2, true, true, true, UT_TOL);

	nlop_free(nlop1);
	nlop_free(nlop2);

	UT_ASSERT(ok);
}

UT_REGISTER_TEST(test_stack_multiple_container2);


static bool test_stack_multiple_container_flatten(void)
{
	enum { N = 3 };
	long idims1_1[N] = { 3, 5, 3 };
	long idims2_1[N] = { 1, 5, 1 };
	long odims_1[N] = { 3, 5, 3 };

	long idims1_2[N] = { 3, 5, 1 };
	long idims2_2[N] = { 1, 5, 1 };
	long odims_2[N] = { 3, 5, 1 };

	const struct nlop_s* nlops[3];

	for (int i = 0; i < 3; i++)
		nlops[i] = nlop_tenmul_create(N, odims_2, idims1_2, idims2_2);
	
	const struct nlop_s* nlop1 = nlop_stack_container_create_F(3, nlops, 2, (int[2]){ 2, -1}, 1, (int[1]){ 2 });
	const struct nlop_s* nlop2 = nlop_tenmul_create(N, odims_1, idims1_1, idims2_1);

	nlop1 = nlop_flatten_F(nlop1);
	nlop2 = nlop_flatten_F(nlop2);

	bool ok = compare_nlops(nlop1, nlop2, true, true, true, UT_TOL);

	nlop_free(nlop1);
	nlop_free(nlop2);

	UT_ASSERT(ok);
}

UT_REGISTER_TEST(test_stack_multiple_container_flatten);

static bool test_stack_multiple_container_flatten2(void)
{
	enum { N = 3 };
	long idims1_1[N] = { 3, 5, 3 };
	long idims2_1[N] = { 1, 1, 3 };
	long odims_1[N] = { 3, 5, 3 };

	long idims1_2[N] = { 3, 1, 3 };
	long idims2_2[N] = { 1, 1, 3 };
	long odims_2[N] = { 3, 1, 3 };

	const struct nlop_s* nlops[5];

	for (int i = 0; i < 5; i++)
		nlops[i] = nlop_tenmul_create(N, odims_2, idims1_2, idims2_2);
	
	const struct nlop_s* nlop1 = nlop_stack_container_create_F(5, nlops, 2, (int[2]){ 1, -1}, 1, (int[1]){ 1 });
	const struct nlop_s* nlop2 = nlop_tenmul_create(N, odims_1, idims1_1, idims2_1);

	nlop1 = nlop_flatten_F(nlop1);
	nlop2 = nlop_flatten_F(nlop2);

	bool ok = compare_nlops(nlop1, nlop2, true, true, true, UT_TOL);

	nlop_free(nlop1);
	nlop_free(nlop2);

	UT_ASSERT(ok);
}

UT_REGISTER_TEST(test_stack_multiple_container_flatten2);



static bool test_nlop_select_derivatives(void)
{
	long dim[1] = { 1 };

	auto tenmul1 = nlop_tenmul_create(1, dim, dim, dim);

	complex float ptr1[1] = { 1. };
	complex float ptr2[1] = { 1. };
	complex float ptr3[1] = { 1. };

	void* args[3] = { ptr1, ptr2, ptr3 };

	UT_ASSERT2(!nlop_tenmul_der_available(tenmul1, 0));
	UT_ASSERT2(!nlop_tenmul_der_available(tenmul1, 1));

	nlop_generic_apply_unchecked(tenmul1, 3, args);

	UT_ASSERT2(nlop_tenmul_der_available(tenmul1, 0));
	UT_ASSERT2(nlop_tenmul_der_available(tenmul1, 1));

	nlop_generic_apply_select_derivative_unchecked(tenmul1, 3, args, 1l, 3l);

	UT_ASSERT2(nlop_tenmul_der_available(tenmul1, 0));
	UT_ASSERT2(nlop_tenmul_der_available(tenmul1, 1));

	nlop_generic_apply_select_derivative_unchecked(tenmul1, 3, args, 0l, 0l);

	UT_ASSERT2(!nlop_tenmul_der_available(tenmul1, 0));
	UT_ASSERT2(!nlop_tenmul_der_available(tenmul1, 1));

	nlop_free(tenmul1);

	return true;
}

UT_REGISTER_TEST(test_nlop_select_derivatives);


static bool test_nlop_select_derivatives_dup(void)
{
	long dim[1] = { 1 };

	auto tenmul1 = nlop_tenmul_create(1, dim, dim, dim);

	complex float ptr1[1] = { 1. };
	complex float ptr2[1] = { 1. };

	void* args[2] = { ptr1, ptr2 };

	auto op = nlop_dup(tenmul1, 0, 1);

	UT_ASSERT2(!nlop_tenmul_der_available(tenmul1, 0));
	UT_ASSERT2(!nlop_tenmul_der_available(tenmul1, 1));

	nlop_generic_apply_select_derivative_unchecked(op, 2, args, 1l, 1l);

	UT_ASSERT2(nlop_tenmul_der_available(tenmul1, 0));
	UT_ASSERT2(nlop_tenmul_der_available(tenmul1, 1));

	nlop_generic_apply_select_derivative_unchecked(op, 2, args, 0l, 0l);

	UT_ASSERT2(!nlop_tenmul_der_available(tenmul1, 0));
	UT_ASSERT2(!nlop_tenmul_der_available(tenmul1, 1));

	nlop_free(tenmul1);
	nlop_free(op);

	return true;
}

UT_REGISTER_TEST(test_nlop_select_derivatives_dup);


static bool test_nlop_select_derivatives_combine(void)
{
	long dim[1] = { 1 };

	auto tenmul1 = nlop_tenmul_create(1, dim, dim, dim);
	auto tenmul2 = nlop_tenmul_create(1, dim, dim, dim);

	complex float ptr1[1] = { 1. };
	complex float ptr2[1] = { 1. };
	complex float ptr3[1] = { 1. };
	complex float ptr4[1] = { 1. };
	complex float ptr5[1] = { 1. };
	complex float ptr6[1] = { 1. };

	void* args[6] = { ptr1, ptr2, ptr3, ptr4, ptr5, ptr6 };

	auto op = nlop_combine(tenmul1, tenmul2);

	UT_ASSERT2(!nlop_tenmul_der_available(tenmul1, 0));
	UT_ASSERT2(!nlop_tenmul_der_available(tenmul1, 1));
	UT_ASSERT2(!nlop_tenmul_der_available(tenmul2, 0));
	UT_ASSERT2(!nlop_tenmul_der_available(tenmul2, 1));

	nlop_generic_apply_select_derivative_unchecked(op, 6, args, 0l, 0l);

	UT_ASSERT2(!nlop_tenmul_der_available(tenmul1, 0));
	UT_ASSERT2(!nlop_tenmul_der_available(tenmul1, 1));
	UT_ASSERT2(!nlop_tenmul_der_available(tenmul2, 0));
	UT_ASSERT2(!nlop_tenmul_der_available(tenmul2, 1));

	nlop_generic_apply_select_derivative_unchecked(op, 6, args, 3l, 6l);

	UT_ASSERT2(!nlop_tenmul_der_available(tenmul1, 0));
	UT_ASSERT2(nlop_tenmul_der_available(tenmul1, 1));
	UT_ASSERT2(nlop_tenmul_der_available(tenmul2, 0));
	UT_ASSERT2(!nlop_tenmul_der_available(tenmul2, 1));

	nlop_free(tenmul1);
	nlop_free(tenmul2);
	nlop_free(op);

	return true;
}

UT_REGISTER_TEST(test_nlop_select_derivatives_combine);


static bool test_nlop_select_derivatives_link(void)
{
	long dim[1] = { 1 };

	auto tenmul1 = nlop_tenmul_create(1, dim, dim, dim);
	auto tenmul2 = nlop_tenmul_create(1, dim, dim, dim);

	complex float ptr1[1] = { 1. };
	complex float ptr2[1] = { 1. };
	complex float ptr3[1] = { 1. };
	complex float ptr4[1] = { 1. };

	void* args[4] = { ptr1, ptr2, ptr3, ptr4 };

	auto op = nlop_chain2(tenmul1, 0, tenmul2, 1);

	UT_ASSERT2(!nlop_tenmul_der_available(tenmul1, 0));
	UT_ASSERT2(!nlop_tenmul_der_available(tenmul1, 1));
	UT_ASSERT2(!nlop_tenmul_der_available(tenmul2, 0));
	UT_ASSERT2(!nlop_tenmul_der_available(tenmul2, 1));

	nlop_generic_apply_select_derivative_unchecked(op, 4, args, 0l, 0l);

	UT_ASSERT2(!nlop_tenmul_der_available(tenmul1, 0));
	UT_ASSERT2(!nlop_tenmul_der_available(tenmul1, 1));
	UT_ASSERT2(!nlop_tenmul_der_available(tenmul2, 0));
	UT_ASSERT2(!nlop_tenmul_der_available(tenmul2, 1));

	nlop_generic_apply_select_derivative_unchecked(op, 4, args, 1l, 4l);

	UT_ASSERT2(!nlop_tenmul_der_available(tenmul1, 0));
	UT_ASSERT2(nlop_tenmul_der_available(tenmul1, 1));
	UT_ASSERT2(!nlop_tenmul_der_available(tenmul2, 0));
	UT_ASSERT2(nlop_tenmul_der_available(tenmul2, 1));

	nlop_generic_apply_select_derivative_unchecked(op, 4, args, 1l, 1l);

	UT_ASSERT2(!nlop_tenmul_der_available(tenmul1, 0));
	UT_ASSERT2(!nlop_tenmul_der_available(tenmul1, 1));
	UT_ASSERT2(nlop_tenmul_der_available(tenmul2, 0));
	UT_ASSERT2(!nlop_tenmul_der_available(tenmul2, 1));

	nlop_free(tenmul1);
	nlop_free(tenmul2);
	nlop_free(op);

	return true;
}

UT_REGISTER_TEST(test_nlop_select_derivatives_link);


static bool test_nlop_zinv(void)
{
	enum { N = 3 };
	long dims[N] = { 5, 1, 3 };

	auto nlop = nlop_zinv_create(N, dims);

	float err_adj = nlop_test_adj_derivatives(nlop, false);
	float err_der = nlop_test_derivatives(nlop);

	nlop_free(nlop);

	debug_printf(DP_DEBUG1, "zinv errors:, adj: %.8f, %.8f\n", err_der, err_adj);

	UT_ASSERT((err_adj < UT_TOL) && (err_der < 2.E-2));
}

UT_REGISTER_TEST(test_nlop_zinv);


static bool test_zmax(void)
{
	unsigned int N = 3;

	long indims[] = { 2, 2, 1 };
	long outdims[] = { 2, 2, 4 };

	complex float stacked[] = {	1., 2., 3., 3.,
					2., 2., 4., 2.,
					2., 1., 4., 3.,
					1., 1., 1., 1.};
	complex float zmax[] = { 2., 2., 4., 3. };

	const struct nlop_s* zmax_op = nlop_zmax_create(N, outdims, 4);
	complex float* output_zmax = md_alloc(N, indims, CFL_SIZE);

	nlop_generic_apply_unchecked(zmax_op, 2, (void*[]){ output_zmax, stacked }); //output, in, mask
	nlop_free(zmax_op);

	float err =  md_zrmse(N, indims, output_zmax, zmax);

	md_free(output_zmax);

	UT_ASSERT(0.01 > err);
}

UT_REGISTER_TEST(test_zmax);


static const struct nlop_s* get_test_nlop(int N, const long dims[N])
{
	auto tenmul1 = nlop_tenmul_create(N, dims, dims, dims);
	auto tenmul2 = nlop_tenmul_create(N, dims, dims, dims);
	auto tenmul3 = nlop_tenmul_create(N, dims, dims, dims);

	const struct nlop_s* result = nlop_chain2_FF(tenmul1, 0, tenmul2, 0);

	result = nlop_combine_FF(result, tenmul3);
	result = nlop_permute_inputs_F(result, 5, (int[5]){ 4, 2, 1, 0, 3 });

	return result;
}

static bool test_nlop_checkpointing(void)
{
	enum { N = 3 };
	long dims[N] = { 3, 1, 3 };

	auto nlop = get_test_nlop(N, dims);
	auto nlop_cp = nlop_checkpoint_create_F(get_test_nlop(N, dims), true, true);

	int OO = 2;
	int II = 5;

	void* args[OO + II];
	void* args_cp[OO + II];

	for(int i = 0; i < OO; i++) {

		args[i] = md_alloc(N, dims, CFL_SIZE);
		args_cp[i] = md_alloc(N, dims, CFL_SIZE);
	}

	for(int i = OO; i < OO + II; i++) {

		args[i] = md_alloc(N, dims, CFL_SIZE);

		md_gaussian_rand(N, dims, args[i]);

		args_cp[i] = args[i];
	}

	unsigned long out_der_flag = MD_BIT(1) | MD_BIT(2);
	unsigned long in_der_flag = MD_BIT(2) | MD_BIT(3) | MD_BIT(4);

	nlop_generic_apply_select_derivative_unchecked(nlop, OO + II, args, out_der_flag, in_der_flag);
	nlop_generic_apply_select_derivative_unchecked(nlop_cp, OO + II, args_cp, out_der_flag, in_der_flag);

	float err = 0.;

	for (int o = 0; o < OO; o++)
		err += md_zrmse(N, dims, args[o], args_cp[o]);

	nlop_generic_apply_select_derivative_unchecked(nlop, OO + II, args, out_der_flag, in_der_flag);
	nlop_generic_apply_select_derivative_unchecked(nlop_cp, OO + II, args_cp, out_der_flag, in_der_flag);

	for (int o = 0; o < OO; o++)
		err += md_zrmse(N, dims, args[o], args_cp[o]);

	for (int i = 0; i < II; i++) {

		if (!MD_IS_SET(in_der_flag, i))
				continue;

		for (int o = 0; o < OO; o++) {

			if (!MD_IS_SET(out_der_flag, o))
				continue;

			complex float* src = md_alloc(N, dims, CFL_SIZE);

			md_gaussian_rand(N, dims, src);

			complex float* dst = md_alloc(N, dims, CFL_SIZE);
			complex float* dst_cp = md_alloc(N, dims, CFL_SIZE);

			linop_forward(nlop_get_derivative(nlop, o, i), N, dims, dst, N, dims, src);
			linop_forward(nlop_get_derivative(nlop_cp, o, i), N, dims, dst_cp, N, dims, src);

			err += md_zrmse(N, dims, dst, dst_cp);

			linop_adjoint(nlop_get_derivative(nlop, o, i), N, dims, dst, N, dims, src);
			linop_adjoint(nlop_get_derivative(nlop_cp, o, i), N, dims, dst_cp, N, dims, src);

			err += md_zrmse(N, dims, dst, dst_cp);

			md_free(src);
			md_free(dst);
			md_free(dst_cp);
		}
	}

	for (int i = 0; i < II; i++) {

		if (!MD_IS_SET(in_der_flag, i))
				continue;

		for (int o = 0; o < OO; o++) {

			if (!MD_IS_SET(out_der_flag, o))
				continue;

			complex float* src = md_alloc(N, dims, CFL_SIZE);

			md_gaussian_rand(N, dims, src);

			complex float* dst = md_alloc(N, dims, CFL_SIZE);
			complex float* dst_cp = md_alloc(N, dims, CFL_SIZE);

			linop_forward(nlop_get_derivative(nlop, o, i), N, dims, dst, N, dims, src);
			linop_forward(nlop_get_derivative(nlop_cp, o, i), N, dims, dst_cp, N, dims, src);

			err += md_zrmse(N, dims, dst, dst_cp);

			linop_adjoint(nlop_get_derivative(nlop, o, i), N, dims, dst, N, dims, src);
			linop_adjoint(nlop_get_derivative(nlop_cp, o, i), N, dims, dst_cp, N, dims, src);

			err += md_zrmse(N, dims, dst, dst_cp);

			md_free(src);
			md_free(dst);
			md_free(dst_cp);
		}
	}

	for(int i = 0; i < OO; i++) {

		md_free(args[i]);
		md_free(args_cp[i]);
	}

	for(int i = OO; i < OO + II; i++)
		md_free(args[i]);

	nlop_free(nlop);
	nlop_free(nlop_cp);

	UT_ASSERT(err < UT_TOL);
}

UT_REGISTER_TEST(test_nlop_checkpointing);


static bool test_mriop_normalinv_config(bool batch_independent, bool share_pattern)
{
	// Here we test the basic case of a fully sampled k-space
	// => The normal operator is the identity
	// => out = in / (1+lambda)
	enum { N = 16 };
	long dims[N] = { 8, 8, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3};

	struct config_nlop_mri_s mri_conf = conf_nlop_mri_simple;

	if (!share_pattern)
		mri_conf.pattern_flags = MD_CLEAR(mri_conf.pattern_flags, 15);

	if (!batch_independent)
		mri_conf.batch_flags = 0;

	long pdims[N];
	long idims[N];
	long ldims[N];

	md_select_dims(N, mri_conf.pattern_flags, pdims, dims);
	md_select_dims(N, mri_conf.image_flags, idims, dims);
	md_select_dims(N, MD_BIT(15), ldims, dims);


	auto nlop_inv = nlop_mri_normal_inv_create(N, dims, ldims, N, pdims, &mri_conf, NULL); // in: x0, coil, pattern, lambda; out:

	complex float* pattern = md_alloc(N, pdims, CFL_SIZE);

	md_zfill(N, pdims, pattern, 1.);


	complex float* coils = md_alloc(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, coils);


	complex float* coils_scale = md_alloc(N, idims, CFL_SIZE);

	md_zrss(N, dims, MD_BIT(3), coils_scale, coils);
	md_zdiv2(N, dims, MD_STRIDES(N, dims, CFL_SIZE), coils, MD_STRIDES(N, dims, CFL_SIZE), coils, MD_STRIDES(N, idims, CFL_SIZE), coils_scale);

	md_free(coils_scale);


	nlop_inv = nlop_set_input_const_F(nlop_inv, 1, N, dims, true, coils);
	nlop_inv = nlop_set_input_const_F(nlop_inv, 1, N, pdims, true, pattern);

	md_free(coils);
	md_free(pattern);


	complex float* lambda = md_alloc(N, ldims, CFL_SIZE);

	md_zfill(N, ldims, lambda, 3.);

	complex float* in = md_alloc(N, idims, CFL_SIZE);
	complex float* out = md_alloc(N, idims, CFL_SIZE);

	md_gaussian_rand(N, idims, in);

	nlop_generic_apply_unchecked(nlop_inv, 3, MAKE_ARRAY((void*)out, (void*)in, (void*)lambda));

	md_zdiv(N, idims, out, out, in);
	md_zsmul(N, idims, out, out, 4);

	complex float* ones = md_alloc(N, idims, CFL_SIZE);

	md_zfill(N, idims, ones, 1.);

	float err = md_znrmse(N, idims, ones, out);

	linop_forward_unchecked(nlop_get_derivative(nlop_inv, 0, 0), out, in);

	md_zdiv(N, idims, out, out, in);
	md_zsmul(N, idims, out, out, 4);

	err += md_znrmse(N, idims, ones, out);

	linop_forward_unchecked(nlop_get_derivative(nlop_inv, 0, 1), out, ones);

	md_zdiv(N, idims, out, out, in);
	md_zsmul(N, idims, out, out, -16);

	err += md_znrmse(N, idims, ones, out);

	md_free(ones);
	md_free(in);
	md_free(out);
	nlop_free(nlop_inv);
	md_free(lambda);

	UT_ASSERT(1.e-5 > err);
}

static bool test_mriop_normalinv(void)
{
	return    test_mriop_normalinv_config(true, true)
	       && test_mriop_normalinv_config(true, false)
	       && test_mriop_normalinv_config(false, true)
	       && test_mriop_normalinv_config(false, false);
}

UT_REGISTER_TEST(test_mriop_normalinv);

