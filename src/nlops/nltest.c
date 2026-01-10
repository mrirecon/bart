/* Copyright 2017-2022. Uecker Lab. University Center Göttingen.
 * Copyright 2022-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <stdbool.h>
#include <math.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "num/init.h"
#include "num/rand.h"
#include "num/iovec.h"
#include "num/multind.h"
#include "num/flpmath.h"

#if 1
#include "nlops/nlop.h"
#include "nlops/const.h"

#include "nlops/chain.h"
#include "nlops/tenmul.h"

#include "linops/lintest.h"

#include "nltest.h"


static bool linear_derivative(const struct nlop_s* op)
{
	int N_dom = nlop_domain(op)->N;
	int N_cod = nlop_codomain(op)->N;

	long dims_dom[N_dom];
	md_copy_dims(N_dom, dims_dom, nlop_domain(op)->dims);

	long dims_cod[N_cod];
	md_copy_dims(N_cod, dims_cod, nlop_codomain(op)->dims);

	complex float* x0 = md_calloc(N_dom, dims_dom, CFL_SIZE);

	complex float* x = md_alloc(N_dom, dims_dom, CFL_SIZE);
	complex float* y1 = md_alloc(N_cod, dims_cod, CFL_SIZE);
	complex float* y2 = md_alloc(N_cod, dims_cod, CFL_SIZE);

	bool result = true;

	for (int i = 0; i < 5; i++) {

		md_gaussian_rand(N_dom, dims_dom, x);

		nlop_apply(op, N_cod, dims_cod, y1, N_dom, dims_dom, x);
		nlop_apply(op, N_cod, dims_cod, y2, N_dom, dims_dom, x0);

		md_zsub(N_cod, dims_cod, y1, y1, y2);

		nlop_derivative(op, N_cod, dims_cod, y2, N_dom, dims_dom, x);

		float scale = (md_znorm(N_cod, dims_cod, y1) + md_znorm(N_cod, dims_cod, y2)) / 2.;

		md_zsub(N_cod, dims_cod, y1, y1, y2);

		debug_printf(DP_DEBUG2, "%.8f, %.8f\n", scale, md_znorm(N_cod, dims_cod, y1) / (1. < scale ? scale : 1.));

		if (1.e-6 < md_znorm(N_cod, dims_cod, y1) / (1. < scale ? scale : 1.))
			result = false;
	}

	md_free(y1);
	md_free(y2);
	md_free(x);
	md_free(x0);

	return result;
}

static float nlop_test_derivative_priv(const struct nlop_s* op, const complex float* in, bool lin)
{
	// This test does not make sense for operators with linear derivative:
	if (lin && linear_derivative(op))
		return 0.;

	int N_dom = nlop_domain(op)->N;
	int N_cod = nlop_codomain(op)->N;

	long dims_dom[N_dom];
	md_copy_dims(N_dom, dims_dom, nlop_domain(op)->dims);

	long dims_cod[N_cod];
	md_copy_dims(N_cod, dims_cod, nlop_codomain(op)->dims);

	complex float* x1 = md_alloc(N_dom, dims_dom, CFL_SIZE);

	if (NULL != in)
		md_copy(N_dom, dims_dom, x1, in, CFL_SIZE);

	complex float* h = md_alloc(N_dom, dims_dom, CFL_SIZE);
	complex float* x2 = md_alloc(N_dom, dims_dom, CFL_SIZE);
	complex float* d1 = md_alloc(N_cod, dims_cod, CFL_SIZE);
	complex float* d2 = md_alloc(N_cod, dims_cod, CFL_SIZE);
	complex float* d3 = md_alloc(N_cod, dims_cod, CFL_SIZE);
	complex float* y1 = md_alloc(N_cod, dims_cod, CFL_SIZE);
	complex float* y2 = md_alloc(N_cod, dims_cod, CFL_SIZE);


	float val = 0.;
	float val0 = 1.;
	float max_ratio = 0.;

	int failed_rounds = 0;
	const int rounds = 20; // Repeat this test, so that it is less likely to only pass for specific random values

	for (int r = 0; r < rounds; r++) {

		if (NULL == in)
			md_gaussian_rand(N_dom, dims_dom, x1);

		md_gaussian_rand(N_dom, dims_dom, h);

		nlop_apply(op, N_cod, dims_cod, y1, N_dom, dims_dom, x1);
		nlop_derivative(op, N_cod, dims_cod, d1, N_dom, dims_dom, h);

		float scale = 1.;
		float vall = 0.;
		val0 = 1.; // do not divide by zero if val0 is never changed
		val = 0.;

		for (int i = 0; i < 10; i++) {

			// d2 = F(x + s * h) - F(x)
			md_copy(N_dom, dims_dom, x2, x1, CFL_SIZE);
			md_zaxpy(N_dom, dims_dom, x2, scale, h);
			nlop_apply(op, N_cod, dims_cod, y2, N_dom, dims_dom, x2);
			md_zsub(N_cod, dims_cod, d2, y2, y1);

			// d3 = DF(s * h) = s * DF(h)
			md_zsmul(N_cod, dims_cod, d3, d1, scale);

			// d2 = (F(x + s * h) - F(x)) - DF(s * h)
			md_zsub(N_cod, dims_cod, d2, d2, d3);

			val = md_znorm(N_cod, dims_cod, d2);

			if (!safe_isfinite(val)) {

				debug_printf(DP_ERROR, "nlop_test_derivative_priv: %3d, %3d: norm is infinite! Aborting test...\n", r, i);
				max_ratio = NAN;

				goto failout;
			}

			debug_printf(DP_DEBUG2, "\t%3d, %3d: %f/%f=%f\n", r, i, val, scale, val / scale);

			val /= scale;

			if ((0 == i) || (val > vall))
				val0 = val;

			vall = val;
			scale /= 2.;

			if (1.e-6 * md_znorm(N_cod, dims_cod, y1) > md_znorm(N_cod, dims_cod, d2))
				break;
		}

		float ratio = val / val0;
		debug_printf(DP_DEBUG2, "%3d: %f/%f=%f\n", r, val, val0, ratio);

		if (ratio > 0.99) {

			debug_printf(DP_ERROR, "nlop_test_derivative_priv: %3d: ratio too large! ratio: %e\n", r, ratio);
			failed_rounds++;
		}

		max_ratio = MAX(max_ratio, ratio);
	}

	if (0 < failed_rounds)
		debug_printf(DP_ERROR, "nlop_test_derivative_priv: %3d of %3d rounds failed!\n", failed_rounds, rounds);

failout:
	md_free(h);
	md_free(x1);
	md_free(x2);
	md_free(y1);
	md_free(y2);
	md_free(d1);
	md_free(d2);
	md_free(d3);

	return max_ratio;
}


static bool der_success(int reduce_target, float val_target, int reduce, float val)
{
	bool res1 = (reduce_target <= reduce);
	bool res2 = (val_target >= val) || (0 >= val_target);

	return res1 && res2;
}

static bool nlop_test_derivative_priv_reduce(const struct nlop_s* op, bool lin, int iter_max, int reduce_target, float val_target)
{
	int N_dom = nlop_domain(op)->N;
	int N_cod = nlop_codomain(op)->N;

	long dims_dom[N_dom];
	md_copy_dims(N_dom, dims_dom, nlop_domain(op)->dims);

	long dims_cod[N_cod];
	md_copy_dims(N_cod, dims_cod, nlop_codomain(op)->dims);

	complex float* h = md_alloc(N_dom, dims_dom, CFL_SIZE);
	complex float* x1 = md_alloc(N_dom, dims_dom, CFL_SIZE);
	complex float* x2 = md_alloc(N_dom, dims_dom, CFL_SIZE);
	complex float* d1 = md_alloc(N_cod, dims_cod, CFL_SIZE);
	complex float* d2 = md_alloc(N_cod, dims_cod, CFL_SIZE);
	complex float* d3 = md_alloc(N_cod, dims_cod, CFL_SIZE);
	complex float* y1 = md_alloc(N_cod, dims_cod, CFL_SIZE);
	complex float* y2 = md_alloc(N_cod, dims_cod, CFL_SIZE);

	md_gaussian_rand(N_dom, dims_dom, x1);
	md_gaussian_rand(N_dom, dims_dom, h);


	nlop_apply(op, N_cod, dims_cod, y1, N_dom, dims_dom, x1);
	nlop_derivative(op, N_cod, dims_cod, d1, N_dom, dims_dom, h);

	float scale = 1.;
	float val0 = 0.;
	float val = 0.;
	float vall = 0.;

	int iter_reduce = 0;

	for (int i = 0; i < iter_max; i++) {

		// d = F(x + s * h) - F(x)
		md_copy(N_dom, dims_dom, x2, x1, CFL_SIZE);
		md_zaxpy(N_dom, dims_dom, x2, scale, h);

		nlop_apply(op, N_cod, dims_cod, y2, N_dom, dims_dom, x2);

		md_zsub(N_cod, dims_cod, d2, y2, y1);


		// DF(s * h)
		md_zsmul(N_cod, dims_cod, d3, d1, scale);
		md_zsub(N_cod, dims_cod, d2, d2, d3);

		val = md_znorm(N_cod, dims_cod, d2);

		debug_printf(DP_DEBUG2, "%f/%f=%f\n", val, scale, val / scale);

		val /= scale;

		if ((0 == i) || (val > vall)) {

			val0 = val;
			iter_reduce = 0;

		} else {

			iter_reduce++;
		}

		vall = val;
		scale /= 2.;

		if (der_success(reduce_target, val_target, iter_reduce, val / val0))
			break;
	}

	md_free(h);
	md_free(x1);
	md_free(x2);
	md_free(y1);
	md_free(y2);
	md_free(d1);
	md_free(d2);
	md_free(d3);

	if (der_success(reduce_target, val_target, iter_reduce, val / val0))
		return true;

	if (!lin)
		return false;


	auto nlop_square = nlop_tenmul_create(N_cod, dims_cod, dims_cod, dims_cod);

	nlop_square = nlop_dup_F(nlop_square, 0, 1);
	auto nlop_tmp = nlop_chain(op, nlop_square);

	nlop_free(nlop_square);


	bool result = nlop_test_derivative_priv_reduce(nlop_tmp, false, iter_max, reduce_target, val_target);

	nlop_free(nlop_tmp);

	return result;
}

float nlop_test_derivative(const struct nlop_s* op)
{
	return nlop_test_derivative_priv(op, NULL, false);
}

float nlop_test_derivative_at(const struct nlop_s* op, const complex float* in)
{
	return nlop_test_derivative_priv(op, in, false);
}

float nlop_test_derivatives(const struct nlop_s* op)
{
	int nr_in_args = nlop_get_nr_in_args(op);
	int nr_out_args = nlop_get_nr_out_args(op);

	complex float* src[nr_in_args];

	float err = 0.;

	for (int in = 0; in < nr_in_args; in++) {

		auto iov = nlop_generic_domain(op, in);

		src[in] = md_alloc(iov->N, iov->dims, CFL_SIZE);

		md_gaussian_rand(iov->N, iov->dims, src[in]);
	}

	for (int in = 0; in < nr_in_args; in++) {

		for (int out = 0; out < nr_out_args; out++) {

			const struct nlop_s* test_op = nlop_clone(op);

			for (int in_del = 0; in_del < nr_in_args; in_del++) {

				auto iov = nlop_generic_domain(op, in_del);

				if (in_del < in)
					test_op = nlop_set_input_const_F(test_op, 0, iov->N, iov->dims, true, src[in_del]);

				if (in_del > in)
					test_op = nlop_set_input_const_F(test_op, 1, iov->N, iov->dims, true, src[in_del]);
			}

			for (int out_del = 0; out_del < nr_out_args; out_del++) {

				if (out_del < out)
					test_op = nlop_del_out_F(test_op, 0);

				if (out_del > out)
					test_op = nlop_del_out_F(test_op, 1);
			}

			float tmp = nlop_test_derivative_priv(test_op, NULL, true);

			debug_printf(DP_DEBUG2, "der error (in=%d, out=%d): %.8f\n", in, out, tmp);

			err = (err > tmp ? err : tmp);

			nlop_free(test_op);
		}
	}

	for (int in = 0; in < nr_in_args; in++)
		md_free(src[in]);

	return err;
}

bool nlop_test_derivatives_reduce(const struct nlop_s* op, int iter_max, int reduce_target, float val_target)
{
	int nr_in_args = nlop_get_nr_in_args(op);
	int nr_out_args = nlop_get_nr_out_args(op);

	complex float* src[nr_in_args];

	bool result = true;

	for (int in = 0; in < nr_in_args; in++) {

		auto iov = nlop_generic_domain(op, in);

		src[in] = md_alloc(iov->N, iov->dims, CFL_SIZE);

		md_gaussian_rand(iov->N, iov->dims, src[in]);
	}

	for (int in = 0; in < nr_in_args; in++) {

		for (int out = 0; out < nr_out_args; out++) {

			const struct nlop_s* test_op = nlop_clone(op);

			for (int in_del = 0; in_del < nr_in_args; in_del++) {

				auto iov = nlop_generic_domain(op, in_del);

				if (in_del < in)
					test_op = nlop_set_input_const_F(test_op, 0, iov->N, iov->dims, true, src[in_del]);

				if (in_del > in)
					test_op = nlop_set_input_const_F(test_op, 1, iov->N, iov->dims, true, src[in_del]);
			}

			for (int out_del = 0; out_del < nr_out_args; out_del++) {

				if (out_del < out)
					test_op = nlop_del_out_F(test_op, 0);

				if (out_del > out)
					test_op = nlop_del_out_F(test_op, 1);
			}

			bool tmp = nlop_test_derivative_priv_reduce(test_op, true, iter_max, reduce_target, val_target);

			if (!tmp)
				debug_printf(DP_WARN, "der test (in=%d, out=%d) failed\n", in, out);

			result = result && tmp;

			nlop_free(test_op);
		}
	}

	for (int in = 0; in < nr_in_args; in++)
		md_free(src[in]);

	return result;
}

float nlop_test_adj_derivatives(const struct nlop_s* op, bool real)
{
	int nr_in_args = nlop_get_nr_in_args(op);
	int nr_out_args = nlop_get_nr_out_args(op);

	complex float* src[nr_in_args];
	complex float* dst[nr_out_args];

	float err = 0.;

	for (int in = 0; in < nr_in_args; in++) {

		auto iov = nlop_generic_domain(op, in);

		src[in] = md_alloc(iov->N, iov->dims, CFL_SIZE);

		md_gaussian_rand(iov->N, iov->dims, src[in]);
	}


	for (int out = 0; out < nr_out_args; out++) {

		auto iov = nlop_generic_codomain(op, out);

		dst[out] = md_alloc(iov->N, iov->dims, CFL_SIZE);
	}


	for (int in = 0; in < nr_in_args; in++) {

		for (int out = 0; out < nr_out_args; out++) {

			const struct nlop_s* test_op = nlop_clone(op);

			for (int in_del = 0; in_del < nr_in_args; in_del++) {

				auto iov = nlop_generic_domain(op, in_del);

				if (in_del < in)
					test_op = nlop_set_input_const_F(test_op, 0, iov->N, iov->dims, true, src[in_del]);

				if (in_del > in)
					test_op = nlop_set_input_const_F(test_op, 1, iov->N, iov->dims, true, src[in_del]);
			}

			for (int out_del = 0; out_del < nr_out_args; out_del++) {

				if (out_del < out)
					test_op = nlop_del_out_F(test_op, 0);

				if (out_del > out)
					test_op = nlop_del_out_F(test_op, 1);
			}

			auto iovdo = nlop_generic_domain(op, in);
			auto iovco = nlop_generic_codomain(op, out);

			nlop_apply(test_op,
				   iovco->N, iovco->dims, dst[out],
				   iovdo->N, iovdo->dims, src[in]);

			float tmp = (real ? linop_test_adjoint_real : linop_test_adjoint)(nlop_get_derivative(test_op, 0, 0));

			debug_printf(DP_DEBUG2, "adj der error (in=%d, out=%d): %.8f\n", in, out, tmp);

			err = (err > tmp ? err : tmp);

			nlop_free(test_op);
		}
	}

	for (int in = 0; in < nr_in_args; in ++)
		md_free(src[in]);

	for (int out = 0; out < nr_out_args; out ++)
		md_free(dst[out]);

	return err;
}

#endif



static bool compare_linops(const struct linop_s* lop1, const struct linop_s* lop2, bool frw, bool adj, float tol)
{
	bool result = true;

	auto dom = linop_domain(lop1);
	auto cod = linop_codomain(lop2);

	if (frw) {

		complex float* src = md_alloc(dom->N, dom->dims, dom->size);
		complex float* dst1 = md_alloc(cod->N, cod->dims, cod->size);
		complex float* dst2 = md_alloc(cod->N, cod->dims, cod->size);

		md_gaussian_rand(dom->N, dom->dims, src);

		linop_forward_unchecked(lop1, dst1, src);
		linop_forward_unchecked(lop2, dst2, src);

		result = result && (tol >= md_znrmse(cod->N, cod->dims, dst1, dst2));

		if (!result)
			debug_printf(DP_INFO, "linop compare frw failed!\n");

		md_free(src);
		md_free(dst1);
		md_free(dst2);
	}

	if (result && adj) {

		complex float* src = md_alloc(cod->N, cod->dims, cod->size);
		complex float* dst1 = md_alloc(dom->N, dom->dims, dom->size);
		complex float* dst2 = md_alloc(dom->N, dom->dims, dom->size);

		md_gaussian_rand(cod->N, cod->dims, src);

		linop_adjoint_unchecked(lop1, dst1, src);
		linop_adjoint_unchecked(lop2, dst2, src);

		result = result && (tol >= md_znrmse(dom->N, dom->dims, dst1, dst2));

		if (!result)
			debug_printf(DP_INFO, "linop compare adj failed!\n");

		md_free(src);
		md_free(dst1);
		md_free(dst2);
	}

	return result;
}



bool compare_nlops(const struct nlop_s* nlop1, const struct nlop_s* nlop2, bool shape, bool der, bool adj, float tol)
{
	int II = nlop_get_nr_in_args(nlop1);
	int OO = nlop_get_nr_out_args(nlop1);

	assert(II == nlop_get_nr_in_args(nlop2));
	assert(OO == nlop_get_nr_out_args(nlop2));

	complex float* args1[OO + II];
	complex float* args2[OO + II];

	bool result = true;

	for (int i = 0; i < II; i++) {

		auto iov1 = nlop_generic_domain(nlop1, i);
		auto iov2 = nlop_generic_domain(nlop2, i);

		if (shape)
			result = result && iovec_check(iov2, iov1->N, iov1->dims, iov1->strs);
		else
			result = result && (md_calc_size(iov1->N, iov1->dims) == md_calc_size(iov2->N, iov2->dims));

		args1[OO + i] = md_alloc(iov1->N, iov1->dims, iov1->size);
		args2[OO + i] = args1[OO + i];

		md_gaussian_rand(iov1->N, iov1->dims, args1[OO + i]);
	}

	for (int i = 0; i < OO; i++) {

		auto iov1 = nlop_generic_codomain(nlop1, i);
		auto iov2 = nlop_generic_codomain(nlop2, i);

		if (shape)
			result = result && iovec_check(iov2, iov1->N, iov1->dims, iov1->strs);
		else
			result = result && md_calc_size(iov1->N, iov1->dims) == md_calc_size(iov2->N, iov2->dims);

		args1[i] = md_alloc(iov1->N, iov1->dims, iov1->size);
		args2[i] = md_alloc(iov1->N, iov1->dims, iov1->size);
	}

	if (!result) {

		debug_printf(DP_INFO, "nlop compare shape failed!\n");
		goto cleanup;
	}

	nlop_generic_apply_unchecked(nlop1, II + OO, (void**)args1);
	nlop_generic_apply_unchecked(nlop2, II + OO, (void**)args2);


	for (int i = 0; i < OO; i++) {

		auto iovc1 = nlop_generic_codomain(nlop1, i);

		result = result && (tol >= md_znrmse(iovc1->N, iovc1->dims, args1[i], args2[i]));
	}

	if (!result) {

		debug_printf(DP_INFO, "nlop compare forward failed!\n");
		goto cleanup;
	}


	for (int i = 0; i < II; i++) {

		for (int o = 0; o < OO; o++) {

			auto der1 = nlop_get_derivative(nlop1, o, i);
			auto der2 = nlop_get_derivative(nlop2, o, i);

			result = result && compare_linops(der1, der2, der, adj, tol);
		}
	}

cleanup:
	for (int i = 0; i < II; i++)
		md_free(args1[OO + i]);

	for (int i = 0; i < OO; i++){

		md_free(args1[i]);
		md_free(args2[i]);
	}

	return result;
}

