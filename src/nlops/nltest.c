/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <stdbool.h>

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

	complex float* x = md_alloc(N_dom, dims_dom, CFL_SIZE);
	complex float* x0 = md_alloc(N_dom, dims_dom, CFL_SIZE);
	md_clear(N_dom, dims_dom, x0, CFL_SIZE);
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
		debug_printf(DP_DEBUG1, "%.8f, %.8f\n", scale, md_znorm(N_cod, dims_cod, y1) / (1. < scale ? scale : 1.));
		if (1.e-6 < md_znorm(N_cod, dims_cod, y1) / (1. < scale ? scale : 1.))
			result = false;
	}

	md_free(y1);
	md_free(y2);
	md_free(x);
	md_free(x0);

	return result;
}

static float nlop_test_derivative_priv(const struct nlop_s* op, bool lin)
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

	for (int i = 0; i < 10; i++) {

		// d = F(x + s * h) - F(x)
		md_copy(N_dom, dims_dom, x2, x1, CFL_SIZE);
		md_zaxpy(N_dom, dims_dom, x2, scale, h);
		nlop_apply(op, N_cod, dims_cod, y2, N_dom, dims_dom, x2);
		md_zsub(N_cod, dims_cod, d2, y2, y1);

		// DF(s * h)
		md_zsmul(N_cod, dims_cod, d3, d1, scale);
		md_zsub(N_cod, dims_cod, d2, d2, d3);

		val = md_znorm(N_cod, dims_cod, d2);

		debug_printf(DP_DEBUG1, "%f/%f=%f\n", val, scale, val / scale);

		val /= scale;

		if ((0 == i) || (val > vall))
			val0 = val;

		vall = val;
		scale /= 2.;
	}


	md_free(h);
	md_free(x1);
	md_free(x2);
	md_free(y1);
	md_free(y2);
	md_free(d1);
	md_free(d2);
	md_free(d3);

	return (lin && linear_derivative(op)) ? 0. : val / val0;
}

static bool der_succes(int reduce_target, float val_target, int reduce, float val)
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

		debug_printf(DP_DEBUG1, "%f/%f=%f\n", val, scale, val / scale);

		val /= scale;

		if ((0 == i) || (val > vall)) {

			val0 = val;
			iter_reduce = 0;
		} else {

			iter_reduce++;
		}

		vall = val;
		scale /= 2.;

		if (der_succes(reduce_target, val_target, iter_reduce, val/val0))
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

	if (der_succes(reduce_target, val_target, iter_reduce, val/val0))
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
	return nlop_test_derivative_priv(op, false);
}

float nlop_test_derivatives(const struct nlop_s* op)
{
	int nr_in_args = nlop_get_nr_in_args(op);
	int nr_out_args = nlop_get_nr_out_args(op);

	complex float* src[nr_in_args];

	float err = 0.0;

	for (int in = 0; in < nr_in_args; in ++){

		src[in] = md_alloc(nlop_generic_domain(op, in)->N, nlop_generic_domain(op, in)->dims, CFL_SIZE);
		md_gaussian_rand(nlop_generic_domain(op, in)->N, nlop_generic_domain(op, in)->dims, src[in]);
	}

	for (int in = 0; in < nr_in_args; in ++){
		for (int out = 0; out < nr_out_args; out ++){

			const struct nlop_s* test_op = nlop_clone(op);
			for (int in_del = 0; in_del < nr_in_args; in_del++){

				if(in_del < in)
					test_op = nlop_set_input_const_F(test_op, 0, nlop_generic_domain(op, in_del)->N, nlop_generic_domain(op, in_del)->dims, true, src[in_del]);
				if(in_del > in)
					test_op = nlop_set_input_const_F(test_op, 1, nlop_generic_domain(op, in_del)->N, nlop_generic_domain(op, in_del)->dims, true, src[in_del]);
			}

			for (int out_del = 0; out_del < nr_out_args; out_del ++){

				if(out_del < out)
					test_op = nlop_del_out_F(test_op, 0);
				if(out_del > out)
					test_op = nlop_del_out_F(test_op, 1);
			}

			float tmp = nlop_test_derivative_priv(test_op, true);
			debug_printf(DP_DEBUG2, "der error (in=%d, out=%d): %.8f\n", in, out, tmp);
			err = (err > tmp ? err : tmp);
			nlop_free(test_op);
		}
	}

	for (int in = 0; in < nr_in_args; in ++)
		md_free(src[in]);

	return err;
}

bool nlop_test_derivatives_reduce(const struct nlop_s* op, int iter_max, int reduce_target, float val_target)
{
	int nr_in_args = nlop_get_nr_in_args(op);
	int nr_out_args = nlop_get_nr_out_args(op);

	complex float* src[nr_in_args];

	bool result = true;

	for (int in = 0; in < nr_in_args; in ++){

		src[in] = md_alloc(nlop_generic_domain(op, in)->N, nlop_generic_domain(op, in)->dims, CFL_SIZE);
		md_gaussian_rand(nlop_generic_domain(op, in)->N, nlop_generic_domain(op, in)->dims, src[in]);
	}

	for (int in = 0; in < nr_in_args; in ++){
		for (int out = 0; out < nr_out_args; out ++){

			const struct nlop_s* test_op = nlop_clone(op);
			for (int in_del = 0; in_del < nr_in_args; in_del++){

				if(in_del < in)
					test_op = nlop_set_input_const_F(test_op, 0, nlop_generic_domain(op, in_del)->N, nlop_generic_domain(op, in_del)->dims, true, src[in_del]);
				if(in_del > in)
					test_op = nlop_set_input_const_F(test_op, 1, nlop_generic_domain(op, in_del)->N, nlop_generic_domain(op, in_del)->dims, true, src[in_del]);
			}

			for (int out_del = 0; out_del < nr_out_args; out_del ++){

				if(out_del < out)
					test_op = nlop_del_out_F(test_op, 0);
				if(out_del > out)
					test_op = nlop_del_out_F(test_op, 1);
			}

			bool tmp = nlop_test_derivative_priv_reduce(test_op, true, iter_max, reduce_target, val_target);
			if (!tmp)
				debug_printf(DP_WARN, "der test (in=%d, out=%d) failed\n", in, out);
			result = result && tmp;
			nlop_free(test_op);
		}
	}

	for (int in = 0; in < nr_in_args; in ++)
		md_free(src[in]);

	return result;
}

float nlop_test_adj_derivatives(const struct nlop_s* op, _Bool real)
{
	int nr_in_args = nlop_get_nr_in_args(op);
	int nr_out_args = nlop_get_nr_out_args(op);

	complex float* src[nr_in_args];
	complex float* dst[nr_out_args];

	float err = 0.0;

	for (int in = 0; in < nr_in_args; in ++){

		src[in] = md_alloc(nlop_generic_domain(op, in)->N, nlop_generic_domain(op, in)->dims, CFL_SIZE);
		md_gaussian_rand(nlop_generic_domain(op, in)->N, nlop_generic_domain(op, in)->dims, src[in]);
	}

	for (int out = 0; out < nr_out_args; out ++){
		dst[out] = md_alloc(nlop_generic_codomain(op, out)->N, nlop_generic_codomain(op, out)->dims, CFL_SIZE);
	}

	for (int in = 0; in < nr_in_args; in ++){
		for (int out = 0; out < nr_out_args; out ++){

			const struct nlop_s* test_op = nlop_clone(op);

			for (int in_del = 0; in_del < nr_in_args; in_del++){

				if(in_del < in)
					test_op = nlop_set_input_const_F(test_op, 0, nlop_generic_domain(op, in_del)->N, nlop_generic_domain(op, in_del)->dims, true, src[in_del]);
				if(in_del > in)
					test_op = nlop_set_input_const_F(test_op, 1, nlop_generic_domain(op, in_del)->N, nlop_generic_domain(op, in_del)->dims, true, src[in_del]);
			}

			for (int out_del = 0; out_del < nr_out_args; out_del ++){

				if(out_del < out)
					test_op = nlop_del_out_F(test_op, 0);

				if(out_del > out)
					test_op = nlop_del_out_F(test_op, 1);
			}


			nlop_apply(test_op,
				   nlop_generic_codomain(op, out)->N, nlop_generic_codomain(op, out)->dims, dst[out],
				   nlop_generic_domain(op, in)->N, nlop_generic_domain(op, in)->dims, src[in]);

			float tmp = (real ? linop_test_adjoint_real(nlop_get_derivative(test_op, 0, 0)) : linop_test_adjoint(nlop_get_derivative(test_op, 0, 0)));


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

float compare_gpu(const struct nlop_s* cpu_op, const struct nlop_s* gpu_op)
{
#ifdef  USE_CUDA
	num_init_gpu();
	int II = nlop_get_nr_in_args(cpu_op);
	int OO = nlop_get_nr_out_args(cpu_op);

	complex float* args[OO + II];
	complex float* args_tmp[OO + II];
	complex float* args_gpu[OO + II];

	for (int i = 0; i < II; i++){

		args[OO + i] = md_alloc(nlop_generic_domain(cpu_op, i)->N, nlop_generic_domain(cpu_op, i)->dims, nlop_generic_domain(cpu_op, i)->size);
		args_tmp[OO + i] = NULL;
		args_gpu[OO + i] = md_alloc_gpu(nlop_generic_domain(cpu_op, i)->N, nlop_generic_domain(cpu_op, i)->dims, nlop_generic_domain(cpu_op, i)->size);

		md_gaussian_rand(nlop_generic_domain(cpu_op, i)->N, nlop_generic_domain(cpu_op, i)->dims, args[OO + i]);
		md_copy(nlop_generic_domain(cpu_op, i)->N, nlop_generic_domain(cpu_op, i)->dims, args_gpu[OO + i], args[OO + i], nlop_generic_domain(cpu_op, i)->size);

	}

	for (int i = 0; i < OO; i++){

		args[i] = md_alloc(nlop_generic_codomain(cpu_op, i)->N, nlop_generic_codomain(cpu_op, i)->dims, nlop_generic_codomain(cpu_op, i)->size);
		args_tmp[i] = md_alloc(nlop_generic_codomain(cpu_op, i)->N, nlop_generic_codomain(cpu_op, i)->dims, nlop_generic_codomain(cpu_op, i)->size);
		args_gpu[i] = md_alloc_gpu(nlop_generic_codomain(cpu_op, i)->N, nlop_generic_codomain(cpu_op, i)->dims, nlop_generic_codomain(cpu_op, i)->size);
	}

	nlop_generic_apply_unchecked(cpu_op, II + OO, (void**)args);
	nlop_generic_apply_unchecked(gpu_op, II + OO, (void**)args_gpu);

	float result = 0;

	for (int i = 0; i < OO; i++){

		md_copy(nlop_generic_codomain(cpu_op, i)->N, nlop_generic_codomain(cpu_op, i)->dims, args_tmp[i], args_gpu[i], nlop_generic_codomain(cpu_op, i)->size);
		result += md_znrmse(nlop_generic_codomain(cpu_op, i)->N, nlop_generic_codomain(cpu_op, i)->dims, args[i], args_tmp[i]);
	}

	for(int i = 0; i < II + OO; i++){

		md_free(args[i]);
		md_free(args_tmp[i]);
		md_free(args_gpu[i]);
	}

	debug_printf(DP_DEBUG2, "operator error (cpu/gpu): %.8f\n", result);

	for (int i = 0; i < II; i++){

		for (int o = 0; o < OO; o++) {

			auto der_cpu = nlop_get_derivative(cpu_op, o, i);
			auto der_gpu = nlop_get_derivative(gpu_op, o, i);

			complex float* in_cpu = md_alloc(linop_domain(der_cpu)->N, linop_domain(der_cpu)->dims, CFL_SIZE);
			complex float* in_gpu = md_alloc_gpu(linop_domain(der_cpu)->N, linop_domain(der_cpu)->dims, CFL_SIZE);
			complex float* in_tmp = md_alloc(linop_domain(der_cpu)->N, linop_domain(der_cpu)->dims, CFL_SIZE);
			complex float* out_cpu = md_alloc(linop_codomain(der_cpu)->N, linop_codomain(der_cpu)->dims, CFL_SIZE);
			complex float* out_gpu = md_alloc_gpu(linop_codomain(der_cpu)->N, linop_codomain(der_cpu)->dims, CFL_SIZE);
			complex float* out_tmp = md_alloc(linop_codomain(der_cpu)->N, linop_codomain(der_cpu)->dims, CFL_SIZE);

			md_gaussian_rand(linop_domain(der_cpu)->N, linop_domain(der_cpu)->dims, in_cpu);
			md_copy(linop_domain(der_cpu)->N, linop_domain(der_cpu)->dims, in_gpu, in_cpu, CFL_SIZE);
			linop_forward_unchecked(der_cpu, out_cpu, in_cpu);
			linop_forward_unchecked(der_gpu, out_gpu, in_gpu);
			md_copy(linop_codomain(der_cpu)->N, linop_codomain(der_cpu)->dims, out_tmp, out_gpu, CFL_SIZE);
			result += md_znrmse(linop_codomain(der_cpu)->N, linop_codomain(der_cpu)->dims, out_tmp, out_cpu);

			debug_printf(DP_DEBUG2, "der[o=%d, i =%d] error (cpu/gpu): %.8f\n", o, i, md_znrmse(linop_codomain(der_cpu)->N, linop_codomain(der_cpu)->dims, out_tmp, out_cpu));

			md_gaussian_rand(linop_codomain(der_cpu)->N, linop_codomain(der_cpu)->dims, out_cpu);
			md_copy(linop_codomain(der_cpu)->N, linop_codomain(der_cpu)->dims, out_gpu, out_cpu, CFL_SIZE);
			linop_adjoint_unchecked(der_cpu, in_cpu, out_cpu);
			linop_adjoint_unchecked(der_gpu, in_gpu, out_gpu);
			md_copy(linop_domain(der_cpu)->N, linop_domain(der_cpu)->dims, in_tmp, in_gpu, CFL_SIZE);
			result += md_znrmse(linop_domain(der_cpu)->N, linop_domain(der_cpu)->dims, in_tmp, in_cpu);

			debug_printf(DP_DEBUG2, "adj[o=%d, i =%d] error (cpu/gpu): %.8f\n", o, i, md_znrmse(linop_domain(der_cpu)->N, linop_domain(der_cpu)->dims, in_tmp, in_cpu));

			md_free(in_cpu);
			md_free(in_gpu);
			md_free(in_tmp);
			md_free(out_cpu);
			md_free(out_gpu);
			md_free(out_tmp);
		}
	}

	return result;
#else
	UNUSED(cpu_op);
	UNUSED(gpu_op);
	assert(0);
#endif
}
