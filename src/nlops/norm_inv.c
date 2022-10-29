/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */


#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"

#include "num/iovec.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"

#include "iter/italgos.h"
#include "iter/iter.h"
#include "iter/iter2.h"
#include "iter/misc.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/someops.h"
#include "nlops/tenmul.h"

#include "norm_inv.h"

struct nlop_norm_inv_conf nlop_norm_inv_default = {

	.store_nlop = false,
	.iter_conf = NULL,
};


struct norm_inv_s {

	INTERFACE(nlop_data_t);

	const struct nlop_s* normal_op;

	complex float* dout;	//share same intermediate result
	complex float* AhAdout;	//share same intermediate result

	int II;
	const struct iovec_s** dom;
	complex float** in_args;	//output stored in in_args[0];

	bool store_nlop;		//store data necessary for inversion in nlop
	struct iter_conjgrad_conf iter_conf;

	complex float* noise;
	int maxeigen_iter;
};

DEF_TYPEID(norm_inv_s);


static void norm_inv_alloc(struct norm_inv_s* d, const void* ref)
{
	for (int i = 0; i < d->II; i++)
		if (NULL == d->in_args[i])
			d->in_args[i] = md_alloc_sameplace(d->dom[i]->N, d->dom[i]->dims, CFL_SIZE, ref);

	md_clear(d->dom[0]->N, d->dom[0]->dims, d->in_args[0], CFL_SIZE);
}


static void norm_inv_set_ops(const struct norm_inv_s* d, unsigned long der_flag)
{
	complex float* tmp_out = md_alloc_sameplace(d->dom[0]->N, d->dom[0]->dims, CFL_SIZE, d->in_args[0]);

	void* args[d->II + 1];
	args[0] = tmp_out;

	for (int i = 0; i < d->II; i++) {

		assert(NULL != d->in_args[i]);
		args[i + 1] = d->in_args[i];
	}

	nlop_generic_apply_select_derivative_unchecked(d->normal_op, d->II + 1, args, MD_BIT(0), der_flag);

	md_free(tmp_out);
}


static void norm_inv_free_ops(const struct norm_inv_s* d)
{
	nlop_clear_derivatives(d->normal_op);
}


static void norm_inv(const struct norm_inv_s* d, complex float* dst, const complex float* src)
{
	const struct operator_s* normal_op = nlop_get_derivative(d->normal_op, 0, 0)->forward;

	md_clear(d->dom[0]->N, d->dom[0]->dims, dst, CFL_SIZE);

	iter2_conjgrad(	CAST_UP(&(d->iter_conf)), normal_op,
			0, NULL, NULL, NULL, NULL,
			2 * md_calc_size(d->dom[0]->N, d->dom[0]->dims),
			(float*)dst,
			(const float*)src,
			NULL);
}


static void norm_inv_compute_adjoint(struct norm_inv_s* d, complex float* dst, const complex float* src)
{
	if (NULL == d->dout) {

		d->dout = md_alloc_sameplace(d->dom[0]->N, d->dom[0]->dims, CFL_SIZE, dst);

		md_clear(d->dom[0]->N, d->dom[0]->dims, d->dout, CFL_SIZE);
	}

	if (NULL == d->AhAdout) {

		d->AhAdout = md_alloc_sameplace(d->dom[0]->N, d->dom[0]->dims, CFL_SIZE, dst);

		md_clear(d->dom[0]->N, d->dom[0]->dims, d->AhAdout, CFL_SIZE);
	}

	if (0 != md_zrmse(d->dom[0]->N, d->dom[0]->dims, d->dout, src)) {

		md_copy(d->dom[0]->N, d->dom[0]->dims, d->dout, src, CFL_SIZE);

		if (d->store_nlop) {

			norm_inv(d, d->AhAdout, src);

		} else {

			norm_inv_set_ops(d, MD_BIT(0));
			norm_inv(d, d->AhAdout, src);
			norm_inv_free_ops(d);
		}
	}

	md_copy(d->dom[0]->N, d->dom[0]->dims, dst, d->AhAdout, CFL_SIZE);
}


static void norm_inv_fun(const nlop_data_t* _data, int Narg, complex float* args[Narg])
{
	const auto d = CAST_DOWN(norm_inv_s, _data);

	assert(1 + d->II == Narg);

	complex float* dst = args[0];
	const complex float* src = args[1];

	norm_inv_alloc(d, dst);

	for (int i = 1; i < d->II; i++)
		md_copy(d->dom[i]->N, d->dom[i]->dims, d->in_args[i], args[i + 1], CFL_SIZE);

	d->iter_conf.INTERFACE.alpha = 0;

	norm_inv_set_ops(d, MD_BIT(0));
	norm_inv(d, dst, src);

	md_copy(d->dom[0]->N, d->dom[0]->dims, d->in_args[0], dst, CFL_SIZE);

	unsigned long der_flags = 0;

	for (int i = 0; i < d->II; i++)
		if (nlop_der_requested(_data, i, 0))
			der_flags = MD_SET(der_flags, i);

	if (d->store_nlop)
		norm_inv_set_ops(d, (0 == der_flags) ? 0 : MD_BIT(0) | der_flags);
	else
		norm_inv_free_ops(d);


	if (d->store_nlop || (0 == der_flags)) {

		for (int i = 0; i < d->II; i++) {

			md_free(d->in_args[i]);

			d->in_args[i] = NULL;
		}
	}


	md_free(d->dout);
	md_free(d->AhAdout);

	d->dout = NULL;
	d->AhAdout = NULL;
}


static void norm_inv_der_src(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(norm_inv_s, _data);

	assert(0. == d->iter_conf.tol);

	if (d->store_nlop) {

		norm_inv(d, dst, src);

	} else {

		norm_inv_set_ops(d, MD_BIT(0));
		norm_inv(d, dst, src);
		norm_inv_free_ops(d);
	}
}


static void norm_inv_adj_src(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(norm_inv_s, _data);

	assert(0. == d->iter_conf.tol);

	norm_inv_compute_adjoint(d, dst, src);
}


static void norm_inv_der_par(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	assert(0 == o);
	assert(0 < i);

	const auto d = CAST_DOWN(norm_inv_s, _data);

	assert(0 == d->iter_conf.tol);

	complex float* tmp = md_alloc_sameplace(d->dom[0]->N, d->dom[0]->dims, CFL_SIZE, dst);

	if (d->store_nlop) {

		linop_forward(nlop_get_derivative(d->normal_op, 0, i), d->dom[0]->N, d->dom[0]->dims, tmp, d->dom[i]->N, d->dom[i]->dims, src);

		md_zsmul(d->dom[0]->N, d->dom[0]->dims, tmp, tmp, -1);

		norm_inv(d, dst, tmp);

	} else {

		norm_inv_set_ops(d, MD_BIT(0) | MD_BIT(i));

		linop_forward(nlop_get_derivative(d->normal_op, 0, i), d->dom[0]->N, d->dom[0]->dims, tmp, d->dom[i]->N, d->dom[i]->dims, src);

		md_zsmul(d->dom[0]->N, d->dom[0]->dims, tmp, tmp, -1);

		norm_inv(d, dst, tmp);

		norm_inv_free_ops(d);
	}

	md_free(tmp);
}


static void norm_inv_adj_par(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	assert(0 == o);
	assert(0 < i);

	const auto d = CAST_DOWN(norm_inv_s, _data);

	assert(0 == d->iter_conf.tol);

	complex float* tmp = md_alloc_sameplace(d->dom[0]->N, d->dom[0]->dims, CFL_SIZE, dst);

	norm_inv_compute_adjoint(d, tmp, src);

	md_zsmul(d->dom[0]->N, d->dom[0]->dims, tmp, tmp, -1);

	if (d->store_nlop) {

		linop_adjoint(nlop_get_derivative(d->normal_op, 0, i), d->dom[i]->N, d->dom[i]->dims, dst, d->dom[0]->N, d->dom[0]->dims, tmp);

	} else {

		norm_inv_set_ops(d, MD_BIT(i));

		linop_adjoint(nlop_get_derivative(d->normal_op, 0, i), d->dom[i]->N, d->dom[i]->dims, dst, d->dom[0]->N, d->dom[0]->dims, tmp);

		norm_inv_free_ops(d);
	}

	md_free(tmp);
}


static void norm_inv_free(const nlop_data_t* _data)
{
	const auto d = CAST_DOWN(norm_inv_s, _data);

	nlop_free(d->normal_op);

	md_free(d->dout);
	md_free(d->AhAdout);
	md_free(d->noise);

	for (int i = 0; i < d->II; i++) {

		iovec_free(d->dom[i]);

		md_free(d->in_args[i]);
	}

	xfree(d->dom);
	xfree(d->in_args);

	xfree(d);
}


static struct norm_inv_s* norm_inv_data_create(struct nlop_norm_inv_conf* conf, const struct nlop_s* normal_op)
{
	conf = conf ? conf : &nlop_norm_inv_default;

	PTR_ALLOC(struct norm_inv_s, data);
	SET_TYPEID(norm_inv_s, data);

	data->II = nlop_get_nr_in_args(normal_op);
	data->normal_op = nlop_clone(normal_op);

	assert(1 <= nlop_get_nr_in_args(normal_op));
	assert(1 == nlop_get_nr_out_args(normal_op));

	data->in_args = *TYPE_ALLOC(complex float*[data->II]);
	data->dom = *TYPE_ALLOC(const struct iovec_s*[data->II]);

	for (int i = 0; i < data->II; i ++){

		data->in_args[i] = NULL;
		data->dom[i] = iovec_create(nlop_generic_domain(normal_op, i)->N, nlop_generic_domain(normal_op, i)->dims, nlop_generic_domain(normal_op, i)->size);
	}

	assert(iovec_check(data->dom[0], nlop_generic_codomain(normal_op, 0)->N, nlop_generic_codomain(normal_op, 0)->dims, nlop_generic_codomain(normal_op, 0)->strs));

	data->dout = NULL;
	data->AhAdout = NULL;

	if (NULL == conf->iter_conf) {

		data->iter_conf = iter_conjgrad_defaults;
		data->iter_conf.l2lambda = 1.;
		data->iter_conf.maxiter = 30;

	} else {

		data->iter_conf = *conf->iter_conf;
	}

	data->store_nlop = conf->store_nlop;

	data->noise = NULL;
	data->maxeigen_iter = 0;

	return PTR_PASS(data);
}


const struct nlop_s* norm_inv_create(struct nlop_norm_inv_conf* conf, const struct nlop_s* normal_op)
{
	auto data = norm_inv_data_create(conf, normal_op);

	int II = nlop_get_nr_in_args(normal_op);
	int OO = 1;

	int NO = nlop_generic_codomain(normal_op, 0)->N;
	int NI = nlop_generic_domain(normal_op, 0)->N;

	for (int i = 0; i < II; i++)
		NI = MAX(NI, nlop_generic_domain(normal_op, i)->N);


	long nl_odims[OO][NO];
	long nl_idims[II][NI];

	nlop_der_fun_t der[II][OO];
	nlop_der_fun_t adj[II][OO];

	md_copy_dims(NO, nl_odims[0], nlop_generic_codomain(normal_op, 0)->dims);

	for (int i = 0; i <II; i++){

		md_singleton_dims(nlop_generic_domain(normal_op, i)->N, nl_idims[i]);
		md_copy_dims(nlop_generic_domain(normal_op, i)->N, nl_idims[i], nlop_generic_domain(normal_op, i)->dims);

		der[i][0] = norm_inv_der_par;
		adj[i][0] = norm_inv_adj_par;
	}

	der[0][0] = norm_inv_der_src;
	adj[0][0] = norm_inv_adj_src;

	const struct nlop_s* result = nlop_generic_create(	OO, NO, nl_odims, II, NI, nl_idims, CAST_UP(data),
								norm_inv_fun, der, adj, NULL, NULL, norm_inv_free);

	for (int i = 0; i < II; i++)
		result = nlop_reshape_in_F(result, i, nlop_generic_domain(normal_op, i)->N, nl_idims[i]);

	return result;
}


const struct nlop_s* norm_inv_lambda_create(struct nlop_norm_inv_conf* conf, const struct nlop_s* normal_op, unsigned long lflags)
{
	int II = nlop_get_nr_in_args(normal_op);
	auto iov = nlop_generic_domain(normal_op, 0);

	int N = iov->N;
	long dims[N];
	long ldims[N];

	md_copy_dims(N, dims, iov->dims);
	md_select_dims(N, lflags, ldims, dims);

	normal_op = nlop_combine_FF(nlop_clone(normal_op), nlop_tenmul_create(N, dims, dims, ldims));
	normal_op = nlop_dup_F(normal_op, 0, II);
	normal_op = nlop_combine_FF(nlop_zaxpbz_create(N, dims, 1, 1), normal_op);
	normal_op = nlop_link_F(normal_op, 1, 0);
	normal_op = nlop_link_F(normal_op, 1, 0);

	auto result = norm_inv_create(conf, normal_op);

	nlop_free(normal_op);

	return result;
}


const struct nlop_s* norm_inv_lop_lambda_create(struct nlop_norm_inv_conf* conf, const struct linop_s* lop, unsigned long lflags)
{
	struct nlop_norm_inv_conf tconf = conf ? *conf : nlop_norm_inv_default;
	tconf.store_nlop = true;

	auto normal_op = nlop_from_linop_F(linop_get_normal(lop));
	auto result = norm_inv_lambda_create(&tconf, normal_op, lflags);

	nlop_free(normal_op);

	return result;
}



static void normal_power_iter_fun(const nlop_data_t* _data, int Narg, complex float* args[Narg])
{
	const auto d = CAST_DOWN(norm_inv_s, _data);

	assert(d->II == Narg);

	complex float* dst = args[0];

	norm_inv_alloc(d, dst);

	for (int i = 1; i < d->II; i++)
		md_copy(d->dom[i]->N, d->dom[i]->dims, d->in_args[i], args[i], CFL_SIZE);

	norm_inv_set_ops(d, MD_BIT(0));

	if (NULL == d->noise) {

		d->noise = md_alloc_sameplace(d->dom[0]->N, d->dom[0]->dims, CFL_SIZE, dst);
		md_gaussian_rand(d->dom[0]->N, d->dom[0]->dims, d->noise);
	}

	complex float result = iter_power(d->maxeigen_iter, nlop_get_derivative(d->normal_op, 0, 0)->forward, 2 * md_calc_size(d->dom[0]->N, d->dom[0]->dims), (float*)d->noise);

	norm_inv_free_ops(d);

	for (int i = 0; i < d->II; i++) {

		md_free(d->in_args[i]);

		d->in_args[i] = NULL;
	}

	md_copy(1, MD_DIMS(1), dst, &result, CFL_SIZE);
}



const struct nlop_s* nlop_maxeigen_create(const struct nlop_s* normal_op)
{
	auto data = norm_inv_data_create(NULL, normal_op);

	data->maxeigen_iter = 30;

	int II = nlop_get_nr_in_args(normal_op) - 1;
	int OO = 1;

	int NO = nlop_generic_codomain(normal_op, 0)->N;
	int NI = nlop_generic_domain(normal_op, 0)->N;

	for (int i = 0; i < II; i++)
		NI = MAX(NI, nlop_generic_domain(normal_op, i + 1)->N);


	long nl_odims[OO][NO];
	long nl_idims[II ? II : 1][NI];


	md_singleton_dims(NO, nl_odims[0]);

	for (int i = 0; i < II; i++) {

		md_singleton_dims(nlop_generic_domain(normal_op, i + 1)->N, nl_idims[i]);
		md_copy_dims(nlop_generic_domain(normal_op, i + 1)->N, nl_idims[i], nlop_generic_domain(normal_op, i + 1)->dims);
	}

	const struct nlop_s* result = nlop_generic_create(
			1, NO, nl_odims, II, NI, nl_idims, CAST_UP(PTR_PASS(data)),
			normal_power_iter_fun,
			NULL, NULL,
			NULL, NULL, norm_inv_free);

	for (int i = 0; i < II; i++)
		result = nlop_reshape_in_F(result, i, nlop_generic_domain(normal_op, i + 1)->N, nl_idims[i]);

	return result;
}
