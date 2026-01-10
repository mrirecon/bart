/* Copyright 2023-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * Moritz Blumenthal
 */

#include <complex.h>

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/iovec.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/multiplace.h"
#include "num/ops.h"
#include "num/ops_graph.h"

#include "iter/italgos.h"
#include "iter/iter.h"
#include "iter/iter2.h"

#include "linops/someops.h"
#include "linops/sum.h"

#include "nlops/nlop.h"
#include "nlops/checkpointing.h"
#include "nlops/const.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/someops.h"
#include "nlops/tenmul.h"
#include "nlops/norm_inv.h"

#include "noncart/nufft.h"

#include "nn/nn_ops.h"

#include "noir/model2.h"
#include "noir/recon2.h"

#include "noir/model_net.h"

static bool multigpu = false;

void model_net_activate_multigpu(void)
{
	multigpu = true;
}

void model_net_deactivate_multigpu(void)
{
	multigpu = false;
}

struct noir2_net_config_s {

	int N;
	long* trj_dims;
	long* wgh_dims;
	long* bas_dims;
	long* msk_dims;
	long* ksp_dims;
	long* cim_dims;
	long* img_dims;
	long* col_dims;

	unsigned long batch_flag;
	struct noir2_model_conf_s mconf;

	bool noncart;

	const complex float* mask;
	const complex float* basis;
};

struct noir2_net_config_s* noir2_net_config_create(int N,
	const long trj_dims[N],
	const long wgh_dims[N],
	const long bas_dims[N], const complex float* basis,
	const long msk_dims[N], const complex float* mask,
	const long ksp_dims[N],
	const long cim_dims[N],
	const long img_dims[N],
	const long col_dims[N],
	unsigned long batch_flag,
	struct noir2_model_conf_s* model_conf)
{
	PTR_ALLOC(struct noir2_net_config_s, x);

	x->N = N;

	x->trj_dims = ARR_CLONE(long[N], trj_dims ?: MD_SINGLETON_DIMS(N));
	x->wgh_dims = ARR_CLONE(long[N], wgh_dims ?: MD_SINGLETON_DIMS(N));
	x->bas_dims = ARR_CLONE(long[N], bas_dims ?: MD_SINGLETON_DIMS(N));
	x->msk_dims = ARR_CLONE(long[N], msk_dims ?: MD_SINGLETON_DIMS(N));
	x->ksp_dims = ARR_CLONE(long[N], ksp_dims ?: MD_SINGLETON_DIMS(N));
	x->cim_dims = ARR_CLONE(long[N], cim_dims ?: MD_SINGLETON_DIMS(N));
	x->img_dims = ARR_CLONE(long[N], img_dims ?: MD_SINGLETON_DIMS(N));
	x->col_dims = ARR_CLONE(long[N], col_dims ?: MD_SINGLETON_DIMS(N));

	x->basis = basis;
	x->mask = mask;

	x->batch_flag = batch_flag;
	x->mconf = *model_conf;

	x->noncart = (NULL != trj_dims) && (1 != md_calc_size(N, trj_dims));

	return PTR_PASS(x);
}


void noir2_net_config_free(struct noir2_net_config_s* x)
{
	xfree(x->trj_dims);
	xfree(x->wgh_dims);
	xfree(x->bas_dims);
	xfree(x->msk_dims);
	xfree(x->ksp_dims);
	xfree(x->cim_dims);
	xfree(x->img_dims);
	xfree(x->col_dims);

	xfree(x);
}


struct noir2_net_s {

	struct noir2_net_config_s* config;

	int Nb;
	struct noir2_s* models;
};

struct noir2_net_s* noir2_net_create(struct noir2_net_config_s* config, int NB)
{
	int N = config->N;
	long trj_dims[N];
	long wgh_dims[N];
	long bas_dims[N];
	long msk_dims[N];
	long ksp_dims[N];
	long cim_dims[N];
	long img_dims[N];
	long col_dims[N];

	md_select_dims(N, config->batch_flag, trj_dims, config->trj_dims);
	md_select_dims(N, config->batch_flag, wgh_dims, config->wgh_dims);
	md_select_dims(N, config->batch_flag, ksp_dims, config->ksp_dims);
	md_select_dims(N, config->batch_flag, cim_dims, config->cim_dims);
	md_select_dims(N, config->batch_flag, img_dims, config->img_dims);
	md_select_dims(N, config->batch_flag, col_dims, config->col_dims);

	long NB_model = md_calc_size(N, ksp_dims);
	assert(NB_model == md_calc_size(N, wgh_dims));
	assert(NB_model == md_calc_size(N, cim_dims));
	assert(NB_model == md_calc_size(N, img_dims));
	assert(NB_model == md_calc_size(N, col_dims));
	assert(!config->noncart || (NB_model == md_calc_size(N, trj_dims)));


	md_select_dims(N, ~(config->batch_flag), trj_dims, config->trj_dims);
	md_select_dims(N, ~(config->batch_flag), wgh_dims, config->wgh_dims);
	md_select_dims(N, ~(config->batch_flag), bas_dims, config->bas_dims);
	md_select_dims(N, ~(config->batch_flag), msk_dims, config->msk_dims);
	md_select_dims(N, ~(config->batch_flag), ksp_dims, config->ksp_dims);
	md_select_dims(N, ~(config->batch_flag), cim_dims, config->cim_dims);
	md_select_dims(N, ~(config->batch_flag), img_dims, config->img_dims);
	md_select_dims(N, ~(config->batch_flag), col_dims, config->col_dims);

	struct noir2_s models[NB * NB_model];
	
	long img_col_dims[N];
	md_copy_dims(N, img_col_dims, col_dims);
	for (int i = 0; i < N; i++)
		if (1 != img_dims[i] && 1 != col_dims[i])
			img_col_dims[i] = img_dims[i];

	for (long i = 0; i < NB * NB_model; i++) {

		if (config->noncart)
			models[i] = noir2_noncart_create(N, trj_dims, NULL, wgh_dims, NULL, bas_dims, config->basis, msk_dims, config->mask, ksp_dims, cim_dims, img_dims, col_dims, img_col_dims, &(config->mconf));
		else
			models[i] = noir2_cart_create(N, wgh_dims, NULL, bas_dims, config->basis, msk_dims, config->mask, ksp_dims, cim_dims, img_dims, col_dims, img_col_dims, &(config->mconf));		

		if (NULL != config->basis)
			models[i].basis = multiplace_move(N, bas_dims, CFL_SIZE, config->basis);
	}

	PTR_ALLOC(struct noir2_net_s, x);

	x->Nb = NB * NB_model;
	x->models = ARR_CLONE(struct noir2_s[NB * NB_model], models);

	md_copy_dims(N, trj_dims, config->trj_dims);
	md_copy_dims(N, wgh_dims, config->wgh_dims);
	md_copy_dims(N, bas_dims, config->bas_dims);
	md_copy_dims(N, msk_dims, config->msk_dims);
	md_copy_dims(N, ksp_dims, config->ksp_dims);
	md_copy_dims(N, cim_dims, config->cim_dims);
	md_copy_dims(N, img_dims, config->img_dims);
	md_copy_dims(N, col_dims, config->col_dims);

	trj_dims[BATCH_DIM] = (config->noncart) ? NB : 1;
	wgh_dims[BATCH_DIM] = NB;
	ksp_dims[BATCH_DIM] = NB;
	cim_dims[BATCH_DIM] = NB;
	img_dims[BATCH_DIM] = NB;
	col_dims[BATCH_DIM] = NB;

	x->config = noir2_net_config_create(N, trj_dims, wgh_dims, bas_dims, config->basis, msk_dims, config->mask, ksp_dims, cim_dims, img_dims, col_dims, config->batch_flag | BATCH_FLAG, &(config->mconf));

	return PTR_PASS(x);
}

void noir2_net_free(struct noir2_net_s* x)
{
	for (int i = 0; i < x->Nb; i++)
		noir2_free(x->models + i);
	
	xfree(x->models);

	noir2_net_config_free(x->config);
	xfree(x);
}

int noir2_net_get_N(struct noir2_net_s* x)
{
	return x->config->N;
}

void noir2_net_get_img_dims(struct noir2_net_s* x, int N, long img_dims[N])
{
	assert(N == x->config->N);
	md_copy_dims(N, img_dims, x->config->img_dims);
}

void noir2_net_get_cim_dims(struct noir2_net_s* x, int N, long cim_dims[N])
{
	assert(N == x->config->N);
	md_copy_dims(N, cim_dims, x->config->cim_dims);
}


static const struct nlop_s* noir_get_forward(struct noir2_s* model)
{
	int N = model->N;

	const struct nlop_s* result = nlop_tenmul_create(N, model->cim_dims, model->img_dims, model->col_ten_dims);
	result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_get_normal(model->lop_fft)), 0);
	result = nlop_chain2_swap_FF(nlop_from_linop(model->lop_im), 0, result, 0);
	result = nlop_chain2_FF(nlop_from_linop(model->lop_coil), 0, result, 1);
	result = nlop_flatten_in_F(result, 0);
	result = nlop_flatten_in_F(result, 1);
	result = nlop_stack_inputs_F(result, 0, 1, 0);

	return result;
}

static const struct nlop_s* noir_get_adjoint(struct noir2_s* model)
{
	int N = model->N;
	const struct nlop_s* nlop_dim = nlop_tenmul_create(N, model->img_dims, model->col_ten_dims, model->cim_dims);	//in: c, z; dx_im = coils * dz
	nlop_dim = nlop_chain2_FF(nlop_from_linop_F(linop_zconj_create(N, model->col_ten_dims)), 0 , nlop_dim, 0);	//in: z, c; dx_im = \bar{coils} * dz
	nlop_dim = nlop_chain2_FF(nlop_from_linop(model->lop_coil), 0 , nlop_dim, 1); 					//in: z, c; dx_im = \bar{lop_coil(coils)} * dz
	nlop_dim = nlop_chain2_FF(nlop_dim, 0, nlop_from_linop_F(linop_get_adjoint(model->lop_im)), 0);			//dx_im = lop_im^H(\bar{lop_coil(coils)} * dz)

	nlop_dim = nlop_flatten_in_F(nlop_dim, 1);
	nlop_dim = nlop_flatten_out_F(nlop_dim, 0);

	const struct nlop_s* nlop_dcoil = nlop_tenmul_create(N, model->col_ten_dims, model->img_dims, model->cim_dims);	//dx_coil = im * dz
	nlop_dcoil = nlop_chain2_FF(nlop_from_linop_F(linop_zconj_create(N, model->img_dims)), 0 , nlop_dcoil, 0);	//dx_coil = \bar{im} * dz
	nlop_dcoil = nlop_chain2_FF(nlop_from_linop(model->lop_im), 0 , nlop_dcoil, 1); 				//dx_coil = \bar{lop_im(im)} * dz
	nlop_dcoil = nlop_chain2_FF(nlop_dcoil, 0, nlop_from_linop_F(linop_get_adjoint(model->lop_coil)), 0);		//dx_coil = lop_coil^H(\bar{lop_im(im)} * dz)
	
	nlop_dcoil = nlop_flatten_in_F(nlop_dcoil, 1);
	nlop_dcoil = nlop_flatten_out_F(nlop_dcoil, 0);

	const struct nlop_s* result = nlop_combine_FF(nlop_dim, nlop_dcoil);						// out: dx_im, dx_coil; in: dz, coils, dz, im
	result = nlop_permute_inputs_F(result, 4, (const int[4]){ 0, 2, 3, 1});						// out: dx_im, dx_coil; in: dz, dz, im, coils
	result = nlop_dup_F(result, 0, 1);										// out: dx_im, dx_coil; in: dz, im, coils

	result = nlop_stack_outputs_F(result, 0, 1, 0);
	result = nlop_stack_inputs_F(result, 1, 2, 0);									// out: dx; in: dz, xn

	return result;
}

static const struct nlop_s* noir_get_derivative(struct noir2_s* model)
{
	int N = model->N;

	const struct nlop_s* nlop1 = nlop_tenmul_create(N, model->cim_dims, model->img_dims, model->col_ten_dims);	//dz1 = im * dcoils
	nlop1 = nlop_chain2_FF(nlop_from_linop(model->lop_coil), 0 , nlop1, 1);	 					//dz1 = im * lop_coils(dcoils)
	nlop1 = nlop_chain2_swap_FF(nlop_from_linop(model->lop_im), 0 , nlop1, 0);	 				//dz1 = lop_im(im) * lop_coils(dcoils)
	nlop1 = nlop_flatten_in_F(nlop1, 0);
	nlop1 = nlop_flatten_in_F(nlop1, 1);

	const struct nlop_s* nlop2 = nlop_tenmul_create(N, model->cim_dims, model->img_dims, model->col_ten_dims);	//dz2 = dim * coils
	nlop2 = nlop_chain2_FF(nlop_from_linop(model->lop_coil), 0 , nlop2, 1);	 					//dz2 = dim * lop_coils(coils)
	nlop2 = nlop_chain2_swap_FF(nlop_from_linop(model->lop_im), 0 , nlop2, 0);	 				//dz2 = lop_im(dim) * lop_coils(coils)
	nlop2 = nlop_flatten_in_F(nlop2, 0);
	nlop2 = nlop_flatten_in_F(nlop2, 1);

	const struct nlop_s* result = nlop_combine_FF(nlop1, nlop2);							//out: dz1, dz2; in: im, dcoils, dim, coil;
	result = nlop_permute_inputs_F(result, 4, (const int[4]){2, 1, 0, 3});						//out: dz1, dz2; in: dim, dcoils, im, coil;
	result = nlop_stack_inputs_F(result, 0, 1, 0);
	result = nlop_stack_inputs_F(result, 1, 2, 0);									//out: z1, z2; in: dx, xn;
	result = nlop_chain2_FF(result, 0, nlop_zaxpbz_create(N, model->cim_dims, 1, 1), 0);
	result = nlop_link_F(result, 1, 0);										//out: dz; in: dx, xn;

	result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_get_normal(model->lop_fft)), 0);


	return result;
}

static const struct nlop_s* noir_get_normal(struct noir2_s* model)
{
	auto der = noir_get_derivative(model);			//out: dz; in: dx, xn
	auto adj = noir_get_adjoint(model);			//out: dx; in: dz, xn

	auto result = nlop_chain2_swap_FF(der, 0, adj, 0);	//out: dx; in: dx, xn, xn
	result = nlop_dup_F(result, 1, 2);			//out: dx; in: dx, xn
	return result;
}


static const struct nlop_s* noir_normal_inversion_create(struct noir2_s* model, const struct iter_conjgrad_conf* iter_conf)
{
	struct iter_conjgrad_conf cgconf = iter_conjgrad_defaults;

	if (NULL == iter_conf) {

		cgconf.l2lambda = 0.;
		cgconf.maxiter = 30;
		cgconf.tol = 0;

	} else {

		cgconf = *iter_conf;
	}

	struct nlop_norm_inv_conf conf = nlop_norm_inv_default;
	conf.iter_conf = &cgconf;

	auto normal_op = noir_get_normal(model);
	auto result = norm_inv_lambda_create(&conf, normal_op, ~0UL);

	nlop_free(normal_op);

	return result;
}

static const struct nlop_s* noir_gauss_newton_step_create_s(struct noir2_s* model, const struct iter_conjgrad_conf* iter_conf)
{
	auto result = noir_get_forward(model);	//out: F(xn); in: xn

	auto dom = nlop_domain(result);
	auto cod = nlop_codomain(result);

	assert(1 == dom->N);
	int N = cod->N;

	long dims[1];
	long kdims[N];

	md_copy_dims(1, dims, dom->dims);
	md_copy_dims(N, kdims, cod->dims);

	result = nlop_chain2_FF(result, 0, nlop_zaxpbz_create(N, kdims, 1, -1), 1);			//out: y - F(xn); in: y, xn
	result = nlop_chain2_swap_FF(result, 0, noir_get_adjoint(model), 0);				//out: DF(xn)^H(y - F(xn)); in: y, xn, xn
	result = nlop_dup_F(result, 1, 2);								//out: DF(xn)^H(y - F(xn)); in: y, xn
	result = nlop_chain2_swap_FF(result, 0, nlop_zaxpbz_create(1, dims, 1, -1), 0);			//out: DF(xn)^H(y - F(xn)) - lambda(xn - x0); in: y, xn, lambda(xn - x0)

	auto nlop_reg = nlop_zaxpbz_create(1, dims, 1, -1);						//out: xn - x0; in: xn, x0
	nlop_reg = nlop_chain2_swap_FF(nlop_reg, 0, nlop_tenmul_create(1, dims, dims, dims), 0);	//out: lambda(x_n - x_0); in: xn, x0, lambda

	result = nlop_chain2_FF(nlop_reg, 0, result, 2);						//out: DF(xn)^H(y - F(xn)) - lambda(xn - x0); in: y, xn, xn, x0, lambda
	result = nlop_dup_F(result, 1, 2);								//out: DF(xn)^H(y - F(xn)) - lambda(xn - x0); in: y, xn, x0, lambda

	result = nlop_checkpoint_create_F(result, true, true);

	auto nlop_inv = noir_normal_inversion_create(model, iter_conf);

	result = nlop_chain2_swap_FF(result, 0, nlop_inv, 0);						//out: (DF(xn)^H DF(xn) + lambda)^-1 DF(xn)^H(y - F(xn)) - lambda(xn - x0); in: y, xn, x0, lambda, xn, lambda
	result = nlop_dup_F(result, 1, 4);								//out: (DF(xn)^H DF(xn) + lambda)^-1 DF(xn)^H(y - F(xn)) - lambda(xn - x0); in: y, xn, x0, lambda, lambda
	result = nlop_dup_F(result, 3, 4);								//out: (DF(xn)^H DF(xn) + lambda)^-1 DF(xn)^H(y - F(xn)) - lambda(xn - x0); in: y, xn, x0, lambda

	result = nlop_chain2_swap_FF(result, 0, nlop_zaxpbz_create(1, dims, 1, 1), 0);
	result = nlop_dup_F(result, 1, 4);

	return result;
}

static const struct nlop_s* noir_gauss_newton_iter_create_s(struct noir2_s* model, const struct iter_conjgrad_conf* iter_conf, int iter, float redu, float alpha_min)
{
	auto result = nlop_checkpoint_create_F(noir_gauss_newton_step_create_s(model, iter_conf), false, true);
	iter--;

	while (0 < iter--) {

		auto iov = nlop_generic_domain(result, 3);

		auto nlop_scale = nlop_from_linop_F(linop_scale_create(iov->N, iov->dims, 1. / redu));
		nlop_scale = nlop_chain_FF(nlop_zsadd_create(iov->N, iov->dims, -alpha_min), nlop_scale);
		nlop_scale = nlop_chain_FF(nlop_scale, nlop_zsadd_create(iov->N, iov->dims, alpha_min));

		result = nlop_chain2_FF(nlop_scale, 0, result, 3);

		result = nlop_chain2_swap_FF(nlop_checkpoint_create_F(noir_gauss_newton_step_create_s(model, iter_conf), false, true), 0, result, 1);
		result = nlop_dup_F(result, 0, 4);
		result = nlop_dup_F(result, 2, 4);
		result = nlop_dup_F(result, 3, 4);
	}

	return nlop_checkpoint_create_F(result, true, true);
}


const struct nlop_s* noir_gauss_newton_step_create(struct noir2_net_s* model, const struct iter_conjgrad_conf* iter_conf)
{
	const struct nlop_s* nlops[model->Nb];

	for (int i = 0; i < model->Nb; i++){

		nlops[i] = noir_gauss_newton_step_create_s(&(model->models[i]), iter_conf);
		nlops[i] = nlop_append_singleton_dim_in_F(nlops[i], 1);
		nlops[i] = nlop_append_singleton_dim_in_F(nlops[i], 2);
		nlops[i] = nlop_append_singleton_dim_in_F(nlops[i], 3);
		nlops[i] = nlop_append_singleton_dim_out_F(nlops[i], 0);
		nlops[i] = nlop_checkpoint_create_F(nlops[i], true, false);
	}

	int istack_dims[] = { BATCH_DIM, 1, 1, 1 };
	int ostack_dims[] = { 1 };

	const struct nlop_s* ret = nlop_stack_multiple_F(model->Nb, nlops, 4, istack_dims, 1, ostack_dims, true, multigpu);

	return nlop_reshape2_in_F(ret, 0, model->config->N, model->config->batch_flag, model->config->cim_dims);
}

const struct nlop_s* noir_gauss_newton_iter_create_create(struct noir2_net_s* model, const struct iter_conjgrad_conf* iter_conf, int iter, float redu, float alpha_min)
{
	const struct nlop_s* nlops[model->Nb];

	for (int i = 0; i < model->Nb; i++){

		nlops[i] = noir_gauss_newton_iter_create_s(&(model->models[i]), iter_conf, iter, redu, alpha_min);
		nlops[i] = nlop_append_singleton_dim_in_F(nlops[i], 1);
		nlops[i] = nlop_append_singleton_dim_in_F(nlops[i], 2);
		nlops[i] = nlop_append_singleton_dim_in_F(nlops[i], 3);
		nlops[i] = nlop_append_singleton_dim_out_F(nlops[i], 0);
	}

	int istack_dims[] = { BATCH_DIM, 1, 1, 1 };
	int ostack_dims[] = { 1 };

	const struct nlop_s* ret = nlop_stack_multiple_F(model->Nb, nlops, 4, istack_dims, 1, ostack_dims, true, multigpu);

	return nlop_reshape2_in_F(ret, 0, model->config->N, model->config->batch_flag, model->config->cim_dims);
}

static const struct nlop_s* noir_rtnlinv_iter_s_create(int T, struct noir2_s* models[T], const struct iter_conjgrad_conf* iter_conf, int iter, int iter_skip, float redu, float alpha_min, float temp_damp)
{
	const struct nlop_s* ret = noir_gauss_newton_iter_create_s(models[0], iter_conf, iter, redu, alpha_min);

	ret = nlop_no_der_F(ret, 0, 0);
	ret = nlop_no_der_F(ret, 0, 1);
	ret = nlop_no_der_F(ret, 0, 2);
	ret = nlop_no_der_F(ret, 0, 3);

	for (int i = 1; i < T; i++) {

		assert(iter_skip < iter);

		auto tmp = noir_gauss_newton_iter_create_s(models[i], iter_conf, iter - iter_skip, redu, alpha_min);

		for (int i = 0; i < iter_skip; i++) {

			auto iov = nlop_generic_domain(tmp, 3);

			auto nlop_scale = nlop_from_linop_F(linop_scale_create(iov->N, iov->dims, 1. / redu));
			nlop_scale = nlop_chain_FF(nlop_zsadd_create(iov->N, iov->dims, -alpha_min), nlop_scale);
			nlop_scale = nlop_chain_FF(nlop_scale, nlop_zsadd_create(iov->N, iov->dims, alpha_min));

			tmp = nlop_chain2_FF(nlop_scale, 0, tmp, 3);
		}
		
		tmp = nlop_no_der_F(tmp, 0, 0);
		tmp = nlop_no_der_F(tmp, 0, 1);
		tmp = nlop_no_der_F(tmp, 0, 2);
		tmp = nlop_no_der_F(tmp, 0, 3);

		auto cod = nlop_generic_codomain(tmp, 0);
		tmp = nlop_prepend_FF(nlop_from_linop_F(linop_scale_create(cod->N, cod->dims, temp_damp)), tmp, 2);
		tmp = nlop_dup_F(tmp, 1, 2);			// y, x_i-1, lambda; out: x_i

		ret = nlop_chain2_keep_swap_FF(ret, 0, tmp, 1);	// y, xn, x0, lambda y, lambda; out: x_i, x_i-1, x_0:i-2
		ret = nlop_dup_F(ret, 3, 5);			// y, xn, x0, lambda y; out: x_i, x_i-1, x_0:i-2
		ret = nlop_stack_inputs_F(ret, 0, 4, TIME_DIM);	// y, xn, x0, lambda; out: x_i, x_i-1, x_0:i-2

		ret = nlop_append_singleton_dim_out_F(ret, 1);

		if (1 < i)
			ret = nlop_stack_outputs_F(ret, 2, 1, 1);

	}

	ret = nlop_append_singleton_dim_out_F(ret, 0);

	if (1 < T)
		ret = nlop_stack_outputs_F(ret, 1, 0, 1);

	ret = nlop_optimize_graph(ret);

	return ret;
}


const struct nlop_s* noir_rtnlinv_iter_create(struct noir2_net_s* model, const struct iter_conjgrad_conf* iter_conf, int iter, int iter_skip, float redu, float alpha_min, float temp_damp)
{
	int N = model->config->N;
	long bat_dims[N];

	md_select_dims(N,  model->config->batch_flag, bat_dims, model->config->ksp_dims);
	assert(1 < bat_dims[TIME_DIM]);

	long T = bat_dims[TIME_DIM];
	long B = md_calc_size(N, bat_dims) / T;

	const struct nlop_s* nlops[B];
	long b = 0;

	long pos[N];
	md_set_dims(N, pos, 0);

	struct noir2_s* models[T];

	do {
		assert(1 == MD_STRIDES(N, bat_dims, 1)[TIME_DIM]);
	
		for (pos[TIME_DIM] = 0; pos[TIME_DIM] < T; pos[TIME_DIM]++)
			models[pos[TIME_DIM]] = model->models + md_calc_offset(N, MD_STRIDES(N, bat_dims, 1), pos);
		
		nlops[b] = noir_rtnlinv_iter_s_create(T, models, iter_conf, iter, iter_skip, redu, alpha_min, temp_damp);
		nlops[b] = nlop_append_singleton_dim_in_F(nlops[b], 1);
		nlops[b] = nlop_append_singleton_dim_in_F(nlops[b], 2);
		nlops[b] = nlop_append_singleton_dim_in_F(nlops[b], 3);

		b++;

	} while (md_next(N, bat_dims, ~TIME_FLAG, pos));

	int istack_dims[] = { BATCH_DIM, 1, 1, 1 };
	int ostack_dims[] = { 1 };

	const struct nlop_s* ret = nlop_stack_multiple_F(B, nlops, 4, istack_dims, 1, ostack_dims, true, multigpu);

	return nlop_reshape2_in_F(ret, 0, model->config->N, model->config->batch_flag & ~TIME_FLAG, model->config->cim_dims);
}


const struct nlop_s* noir_decomp_create_s(struct noir2_s* model)
{
	const struct nlop_s* nlop_decomp = nlop_combine_FF(nlop_from_linop(model->lop_im), nlop_from_linop(model->lop_coil));
	nlop_decomp = nlop_flatten_in_F(nlop_decomp, 0);
	nlop_decomp = nlop_flatten_in_F(nlop_decomp, 1);
	nlop_decomp = nlop_stack_inputs_F(nlop_decomp, 0, 1, 0);

	return nlop_decomp;
}

const struct nlop_s* noir_decomp_create(struct noir2_net_s* model)
{
	const struct nlop_s* nlops[model->Nb];

	for (int i = 0; i < model->Nb; i++)
		nlops[i] = nlop_append_singleton_dim_in_F(noir_decomp_create_s(&(model->models[i])), 0);

	int istack_dims[] = { 1 };
	int ostack_dims[] = { BATCH_DIM, BATCH_DIM };

	int N = model->config->N;
	long col_dims[N];
	md_copy_dims(N, col_dims, model->models[0].col_ten_dims);

	for (int i = 0; i < N; i++)
		if (MD_IS_SET(model->config->batch_flag, i))
			col_dims[i] = model->config->col_dims[i];

	const struct nlop_s* ret = nlop_stack_multiple_F(model->Nb, nlops, 1, istack_dims, 2, ostack_dims, true, false);

	ret = nlop_reshape2_out_F(ret, 0, model->config->N, model->config->batch_flag, model->config->img_dims);
	ret = nlop_reshape2_out_F(ret, 1, model->config->N, model->config->batch_flag, col_dims);

	return ret;
}

const struct nlop_s* noir_split_create_s(struct noir2_s* model)
{
	auto id1 = linop_identity_create(model->N, model->img_dims);
	auto id2 = linop_identity_create(model->N, model->col_dims);
	
	const struct nlop_s* nlop_decomp = nlop_combine_FF(nlop_from_linop_F(id1), nlop_from_linop_F(id2));
	nlop_decomp = nlop_flatten_in_F(nlop_decomp, 0);
	nlop_decomp = nlop_flatten_in_F(nlop_decomp, 1);
	nlop_decomp = nlop_stack_inputs_F(nlop_decomp, 0, 1, 0);

	return nlop_decomp;
}

const struct nlop_s* noir_split_create(struct noir2_net_s* model)
{
	const struct nlop_s* nlops[model->Nb];

	for (int i = 0; i < model->Nb; i++)
		nlops[i] = nlop_append_singleton_dim_in_F(noir_split_create_s(&(model->models[i])), 0);

	int istack_dims[] = { 1 };
	int ostack_dims[] = { BATCH_DIM, BATCH_DIM };

	const struct nlop_s* ret = nlop_stack_multiple_F(model->Nb, nlops, 1, istack_dims, 2, ostack_dims, true, false);

	ret = nlop_reshape2_out_F(ret, 0, model->config->N, model->config->batch_flag, model->config->img_dims);
	ret = nlop_reshape2_out_F(ret, 1, model->config->N, model->config->batch_flag, model->config->col_dims);

	return ret;
}


const struct nlop_s* noir_cim_create(struct noir2_net_s* model)
{
	auto result = noir_decomp_create(model);

	int N = model->config->N;
	auto cod = nlop_generic_codomain(result, 1);

	result = nlop_combine_FF(nlop_tenmul_create(N, model->config->cim_dims, model->config->img_dims, cod->dims), result);
	result = nlop_link_F(result, 1, 0);
	result = nlop_link_F(result, 1, 0);

	return result;
}


const struct nlop_s* noir_join_create_s(struct noir2_s* model)
{
	auto id1 = linop_identity_create(model->N, model->img_dims);
	auto id2 = linop_identity_create(model->N, model->col_dims);

	const struct nlop_s* nlop_join = nlop_combine_FF(nlop_from_linop_F(id1), nlop_from_linop_F(id2));
	nlop_join = nlop_flatten_out_F(nlop_join, 0);
	nlop_join = nlop_flatten_out_F(nlop_join, 1);
	nlop_join = nlop_stack_outputs_F(nlop_join, 0, 1, 0);

	return nlop_join;
}

const struct nlop_s* noir_join_create(struct noir2_net_s* model)
{
	const struct nlop_s* nlops[model->Nb];

	for (int i = 0; i < model->Nb; i++)
		nlops[i] = nlop_append_singleton_dim_out_F(noir_join_create_s(&(model->models[i])), 0);

	int ostack_dims[] = { 1 };
	int istack_dims[] = { BATCH_DIM, BATCH_DIM };

	const struct nlop_s* ret = nlop_stack_multiple_F(model->Nb, nlops, 2, istack_dims, 1, ostack_dims, true, false);

	ret = nlop_reshape2_in_F(ret, 0, model->config->N, model->config->batch_flag, model->config->img_dims);
	ret = nlop_reshape2_in_F(ret, 1, model->config->N, model->config->batch_flag, model->config->col_dims);

	return ret;
}

const struct nlop_s* noir_extract_img_create(struct noir2_net_s* model)
{
	auto result = noir_split_create(model);
	return nlop_del_out_F(result, 1);
}

const struct nlop_s* noir_set_img_create(struct noir2_net_s* model)
{
	auto result = noir_join_create(model);
	auto dom = nlop_generic_domain(result, 1);

	complex float zero = 0;

	return nlop_set_input_const_F2(result, 1, dom->N, dom->dims, MD_SINGLETON_STRS(dom->N), true, &zero);
}

const struct nlop_s* noir_set_col_create(struct noir2_net_s* model)
{
	auto result = noir_join_create(model);
	auto dom = nlop_generic_domain(result, 0);

	complex float zero = 0;

	return nlop_set_input_const_F2(result, 0, dom->N, dom->dims, MD_SINGLETON_STRS(dom->N), true, &zero);
}


struct noir_adjoint_fft_s {

	nlop_data_t super;

	struct noir2_s* model;
};

DEF_TYPEID(noir_adjoint_fft_s);

static void noir_adjoint_fft_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(noir_adjoint_fft_s, _data);
	assert(3 == N);

	complex float* dst = args[0];
	const complex float* src = args[1];
	const complex float* pat = args[2];

	linop_gdiag_set_diag(data->model->lop_pattern, data->model->N, data->model->pat_dims, pat);
	linop_adjoint_unchecked(data->model->lop_fft, dst, src);
}

static void noir_adjoint_nufft_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(noir_adjoint_fft_s, _data);
	assert(4 == N);

	complex float* dst = args[0];
	const complex float* src = args[1];
	const complex float* wgh = args[2];
	const complex float* trj = args[3];

	auto model = data->model;

	nufft_update_traj(model->lop_nufft, model->N, model->trj_dims, trj, model->pat_dims, wgh, model->bas_dims, multiplace_read(model->basis, trj));

	linop_adjoint_unchecked(data->model->lop_fft, dst, src);
}

static void noir_adjoint_fft_der(const nlop_data_t* _data, int o, int i, complex float* dst, const complex float* src)
{
	assert(0 == o);
	assert(0 == i);
	const auto data = CAST_DOWN(noir_adjoint_fft_s, _data);
	linop_adjoint_unchecked(data->model->lop_fft, dst, src);
}

static void noir_adjoint_fft_adj(const nlop_data_t* _data, int o, int i, complex float* dst, const complex float* src)
{
	assert(0 == o);
	assert(0 == i);
	const auto data = CAST_DOWN(noir_adjoint_fft_s, _data);
	linop_forward_unchecked(data->model->lop_fft, dst, src);
}


static void noir_adjoint_fft_del(const nlop_data_t* _data)
{
	xfree(_data);
}

const struct nlop_s* noir_adjoint_fft_create_s(struct noir2_s* model)
{
	PTR_ALLOC(struct noir_adjoint_fft_s, data);
	SET_TYPEID(noir_adjoint_fft_s, data);

	data->model = model;

	auto cod = linop_codomain(model->lop_fft);
	auto dom = linop_domain(model->lop_fft);

	int N = model->N;
	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], dom->dims);


	long nl_idims[2][N];
	md_copy_dims(N, nl_idims[0], cod->dims);
	md_copy_dims(N, nl_idims[1], data->model->pat_dims);


	return nlop_generic_create(1, N, nl_odims, 2, N, nl_idims, CAST_UP(PTR_PASS(data)),
					noir_adjoint_fft_fun,
					(nlop_der_fun_t[2][1]){ { noir_adjoint_fft_der }, { NULL } },
					(nlop_der_fun_t[2][1]){ { noir_adjoint_fft_adj }, { NULL } },
					NULL, NULL, noir_adjoint_fft_del);
}

const struct nlop_s* noir_adjoint_fft_create(struct noir2_net_s* model)
{
	const struct nlop_s* nlops[model->Nb];

	for (int i = 0; i < model->Nb; i++)
		nlops[i] = noir_adjoint_fft_create_s(&(model->models[i]));

	int istack_dims[] = { BATCH_DIM, BATCH_DIM };
	int ostack_dims[] = { BATCH_DIM };

	const struct nlop_s* ret = nlop_stack_multiple_F(model->Nb, nlops, 2, istack_dims, 1, ostack_dims, true, multigpu);
	ret = nlop_reshape2_in_F(ret, 0, model->config->N, model->config->batch_flag, model->config->ksp_dims);
	ret = nlop_reshape2_in_F(ret, 1, model->config->N, model->config->batch_flag, model->config->wgh_dims);
	ret = nlop_reshape2_out_F(ret, 0, model->config->N, model->config->batch_flag, model->config->cim_dims);

	return ret;
}

const struct nlop_s* noir_adjoint_nufft_create_s(struct noir2_s* model)
{
	PTR_ALLOC(struct noir_adjoint_fft_s, data);
	SET_TYPEID(noir_adjoint_fft_s, data);

	data->model = model;

	auto cod = linop_codomain(model->lop_fft);
	auto dom = linop_domain(model->lop_fft);

	int N = model->N;
	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], dom->dims);


	long nl_idims[3][N];
	md_copy_dims(N, nl_idims[0], cod->dims);
	md_copy_dims(N, nl_idims[1], data->model->pat_dims);
	md_copy_dims(N, nl_idims[2], data->model->trj_dims);


	return nlop_generic_create(1, N, nl_odims, 3, N, nl_idims, CAST_UP(PTR_PASS(data)),
					noir_adjoint_nufft_fun,
					(nlop_der_fun_t[3][1]){ { noir_adjoint_fft_der }, { NULL }, { NULL } },
					(nlop_der_fun_t[3][1]){ { noir_adjoint_fft_adj }, { NULL }, { NULL } },
					NULL, NULL, noir_adjoint_fft_del);
}

const struct nlop_s* noir_adjoint_nufft_create(struct noir2_net_s* model)
{
	const struct nlop_s* nlops[model->Nb];

	for (int i = 0; i < model->Nb; i++)
		nlops[i] = noir_adjoint_nufft_create_s(&(model->models[i]));

	int istack_dims[] = { BATCH_DIM, BATCH_DIM, BATCH_DIM };
	int ostack_dims[] = { BATCH_DIM };

	const struct nlop_s* ret = nlop_stack_multiple_F(model->Nb, nlops, 3, istack_dims, 1, ostack_dims, true, multigpu);

	ret = nlop_reshape2_in_F(ret, 0, model->config->N, model->config->batch_flag, model->config->ksp_dims);
	ret = nlop_reshape2_in_F(ret, 1, model->config->N, model->config->batch_flag, model->config->wgh_dims);
	ret = nlop_reshape2_in_F(ret, 2, model->config->N, model->config->batch_flag, model->config->trj_dims);
	ret = nlop_reshape2_out_F(ret, 0, model->config->N, model->config->batch_flag, model->config->cim_dims);

	return ret;
}


const struct nlop_s* noir_fft_create_s(struct noir2_s* model)
{
	return nlop_from_linop_F(linop_fftc_create(model->N, model->cim_dims, model->model_conf.fft_flags));
}

const struct nlop_s* noir_fft_create(struct noir2_net_s* model)
{
	const struct nlop_s* nlops[model->Nb];

	for (int i = 0; i < model->Nb; i++)
		nlops[i] = noir_fft_create_s(&(model->models[i]));

	int istack_dims[] = { BATCH_DIM };
	int ostack_dims[] = { BATCH_DIM };

	const struct nlop_s* ret = nlop_stack_multiple_F(model->Nb, nlops, 1, istack_dims, 1, ostack_dims, true, multigpu);
	ret = nlop_reshape2_out_F(ret, 0, model->config->N, model->config->batch_flag, model->config->ksp_dims);
	ret = nlop_reshape2_in_F(ret, 0, model->config->N, model->config->batch_flag, model->config->cim_dims);

	return ret;
}

struct noir_nufft_s {

	nlop_data_t super;

	const struct linop_s* nufft;

	struct noir2_s* model;
};

DEF_TYPEID(noir_nufft_s);

static void noir_nufft_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(noir_nufft_s, _data);
	assert(3 == N);

	complex float* dst = args[0];
	const complex float* src = args[1];
	const complex float* trj = args[2];

	nufft_update_traj(data->nufft, data->model->N, data->model->trj_dims, trj, data->model->pat_dims, NULL, data->model->bas_dims, multiplace_read(data->model->basis, trj));
	linop_forward_unchecked(data->nufft, dst, src);
}

static void noir_nufft_der(const nlop_data_t* _data, int o, int i, complex float* dst, const complex float* src)
{
	assert(0 == o);
	assert(0 == i);
	const auto data = CAST_DOWN(noir_nufft_s, _data);
	linop_forward_unchecked(data->nufft, dst, src);
}

static void noir_nufft_adj(const nlop_data_t* _data, int o, int i, complex float* dst, const complex float* src)
{
	assert(0 == o);
	assert(0 == i);

	const auto data = CAST_DOWN(noir_nufft_s, _data);

	linop_adjoint_unchecked(data->nufft, dst, src);
}

static void noir_nufft_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(noir_nufft_s, _data);

	linop_free(data->nufft);

	xfree(_data);
}

const struct nlop_s* noir_nufft_create_s(struct noir2_s* model)
{
	PTR_ALLOC(struct noir_nufft_s, data);
	SET_TYPEID(noir_nufft_s, data);

	auto conf = *(model->model_conf.nufft_conf);
	conf.toeplitz = false;

	data->nufft = nufft_create2(model->N, model->ksp_dims, model->cim_dims, model->trj_dims, NULL, model->pat_dims, NULL, model->bas_dims, multiplace_read(model->basis, NULL), conf);
	data->model = model;

	int N = model->N;
	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], model->ksp_dims);

	long nl_idims[2][N];
	md_copy_dims(N, nl_idims[0], model->cim_dims);
	md_copy_dims(N, nl_idims[1], model->trj_dims);


	return nlop_generic_create(1, N, nl_odims, 2, N, nl_idims, CAST_UP(PTR_PASS(data)),
					noir_nufft_fun,
					(nlop_der_fun_t[2][1]){ { noir_nufft_der }, { NULL } },
					(nlop_der_fun_t[2][1]){ { noir_nufft_adj }, { NULL } },
					NULL, NULL, noir_nufft_del);
}

const struct nlop_s* noir_nufft_create(struct noir2_net_s* model)
{
	const struct nlop_s* nlops[model->Nb];

	for (int i = 0; i < model->Nb; i++)
		nlops[i] = noir_nufft_create_s(&(model->models[i]));

	int istack_dims[] = { BATCH_DIM, BATCH_DIM };
	int ostack_dims[] = { BATCH_DIM };

	const struct nlop_s* ret = nlop_stack_multiple_F(model->Nb, nlops, 2, istack_dims, 1, ostack_dims, true, multigpu);

	ret = nlop_reshape2_out_F(ret, 0, model->config->N, model->config->batch_flag, model->config->ksp_dims);
	ret = nlop_reshape2_in_F(ret, 0, model->config->N, model->config->batch_flag, model->config->cim_dims);
	ret = nlop_reshape2_in_F(ret, 1, model->config->N, model->config->batch_flag, model->config->trj_dims);

	return ret;
}



static const struct nlop_s* noir_sense_forward(struct noir2_s* model)
{
	int N = model->N;

	const struct nlop_s* result = nlop_tenmul_create(N, model->cim_dims, model->img_dims, model->col_ten_dims);
	result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_get_normal(model->lop_fft)), 0);
	result = nlop_chain2_swap_FF(nlop_from_linop(model->lop_im), 0, result, 0);
	result = nlop_chain2_FF(nlop_from_linop(model->lop_coil), 0, result, 1);
	return result;
}

static const struct nlop_s* noir_sense_adjoint(struct noir2_s* model)
{
	int N = model->N;
	const struct nlop_s* nlop_dim = nlop_tenmul_create(N, model->img_dims, model->col_ten_dims, model->cim_dims);	//in: c, z; dx_im = coils * dz
	nlop_dim = nlop_chain2_FF(nlop_from_linop_F(linop_zconj_create(N, model->col_ten_dims)), 0 , nlop_dim, 0);	//in: z, c; dx_im = \bar{coils} * dz
	nlop_dim = nlop_chain2_FF(nlop_from_linop(model->lop_coil), 0 , nlop_dim, 1); 					//in: z, c; dx_im = \bar{lop_coil(coils)} * dz
	nlop_dim = nlop_chain2_FF(nlop_dim, 0, nlop_from_linop_F(linop_get_adjoint(model->lop_im)), 0);			//dx_im = lop_im^H(\bar{lop_coil(coils)} * dz)

	return nlop_dim;
}

static const struct nlop_s* noir_sense_normal(struct noir2_s* model)
{
	auto der = noir_sense_forward(model);			//out: dz; in: dx, xn
	auto adj = noir_sense_adjoint(model);			//out: dx; in: dz, xn

	auto result = nlop_chain2_swap_FF(der, 0, adj, 0);	//out: dx; in: dx, xn, xn
	result = nlop_dup_F(result, 1, 2);			//out: dx; in: dx, xn, xn
	return result;
}

static const struct nlop_s* noir_sense_normal_inversion_create(struct noir2_s* model, const struct iter_conjgrad_conf* iter_conf)
{
	struct iter_conjgrad_conf cgconf = iter_conjgrad_defaults;

	if (NULL == iter_conf) {

		cgconf.l2lambda = 0.;
		cgconf.maxiter = 30;
		cgconf.tol = 0;

	} else {

		cgconf = *iter_conf;
	}

	struct nlop_norm_inv_conf conf = nlop_norm_inv_default;
	conf.iter_conf = &cgconf;

	auto normal_op = noir_sense_normal(model);
	auto result = norm_inv_lambda_create(&conf, normal_op, ~0UL);

	nlop_free(normal_op);

	return result;
}


static const struct nlop_s* noir_sense_recon_create_s(struct noir2_s* model, const struct iter_conjgrad_conf* iter_conf)
{
	int N = model->N;
	long img_dims[N];
	md_copy_dims(N, img_dims, model->img_dims);

	auto result = noir_sense_adjoint(model);								//out: A^Hy; in: y, coln
	result = nlop_chain2_swap_FF(result, 0, nlop_zaxpbz_create(N, img_dims, 1., 1.), 0);			//out: A^Hy + l*img0; in: y, coln, l*img0
	result = nlop_chain2_FF(nlop_tenmul_create(N, img_dims, img_dims, img_dims), 0, result, 2);		//out: A^Hy + l*img0; in: y, coln, img0, l
	result = nlop_chain2_swap_FF(result, 0, noir_sense_normal_inversion_create(model, iter_conf), 0);	//out: (A^HA+l)^-1(A^Hy + l*img0); in: y, coln, img0, l, coln, l
	result = nlop_dup_F(result, 1, 4);
	result = nlop_dup_F(result, 3, 4);									//out: (A^HA+l)^-1(A^Hy + l*img0); in: y, coln, img0, l
	result = nlop_chain2_swap_FF(result, 0, noir_join_create_s(model), 0);					//out: xn+1; in: y, coln, img0, l, coln
	result = nlop_dup_F(result, 1, 4);									//out: xn+1; in: y, coln, img0, l

	result = nlop_prepend_FF(nlop_del_out_F(noir_split_create_s(model), 0), result, 1);		
	result = nlop_prepend_FF(nlop_del_out_F(noir_split_create_s(model), 1), result, 2);
	result = nlop_prepend_FF(nlop_del_out_F(noir_split_create_s(model), 1), result, 3);			//out: xn+1; in: y, xn, x0, l

	return result;
}

const struct nlop_s* noir_sense_recon_create(struct noir2_net_s* model, const struct iter_conjgrad_conf* iter_conf)
{
	const struct nlop_s* nlops[model->Nb];

	for (int i = 0; i < model->Nb; i++){

		nlops[i] = noir_sense_recon_create_s(&(model->models[i]), iter_conf);
		nlops[i] = nlop_append_singleton_dim_in_F(nlops[i], 1);
		nlops[i] = nlop_append_singleton_dim_in_F(nlops[i], 2);
		nlops[i] = nlop_append_singleton_dim_in_F(nlops[i], 3);
		nlops[i] = nlop_append_singleton_dim_out_F(nlops[i], 0);
	}

	int istack_dims[] = { BATCH_DIM, 1, 1, 1 };
	int ostack_dims[] = { 1 };

	const struct nlop_s* ret = nlop_stack_multiple_F(model->Nb, nlops, 4, istack_dims, 1, ostack_dims, true, multigpu);

	return nlop_reshape2_in_F(ret, 0, model->config->N, model->config->batch_flag, model->config->cim_dims);
}

const struct nlop_s* noir_nlinv_regularization_create(struct noir2_net_s* model, unsigned long mask_flags)
{
	auto nlop_l2 = noir_split_create(model);

	auto dom_img = nlop_generic_codomain(nlop_l2, 0);
	auto dom_col = nlop_generic_codomain(nlop_l2, 1);
		
	int N = dom_img->N;
	assert(DIMS == N);

	long img_dims[N];
	long col_dims[N];
	long bat_dims[N];

	md_copy_dims(N, img_dims, dom_img->dims);
	md_copy_dims(N, col_dims, dom_col->dims);
	md_select_dims(N, BATCH_FLAG, bat_dims, img_dims);

	long msk_dims[N];
	md_select_dims(N, mask_flags, msk_dims, img_dims);

	nlop_l2 = nlop_chain2_swap_FF(nlop_l2, 0, nlop_tenmul_create(N, img_dims, img_dims, msk_dims), 0);
	nlop_l2 = nlop_chain2_swap_FF(nlop_l2, 1, nlop_tenmul_create(N, col_dims, col_dims, msk_dims), 0);
	nlop_l2 = nlop_dup_F(nlop_l2, 1, 2);
	
	nlop_l2 = nlop_append_FF(nlop_l2, 1, nlop_zss_create(N, img_dims, ~BATCH_FLAG));	
	nlop_l2 = nlop_append_FF(nlop_l2, 0, nlop_zss_create(N, col_dims, ~BATCH_FLAG));

	nlop_l2 = nlop_chain2_FF(nlop_l2, 0, nlop_zaxpbz_create(N, bat_dims, 1., 1.), 0);
	nlop_l2 = nlop_link_F(nlop_l2, 1, 0);										// OUT: (||m*img||^2 + ||Wm*col||^2); IN: x, m

	if (0 == mask_flags)
		nlop_l2 = nlop_set_input_scalar_F(nlop_l2, 1, 1.);

	return nlop_l2;
}

const struct nlop_s* noir_nlinv_average_coils_create(struct noir2_net_s* model, enum PADDING padding, int window)
{
	auto ret = noir_split_create(model);

	auto dom_col = nlop_generic_codomain(ret, 1);

	const struct linop_s* lop_avg = NULL;
		
	if (PAD_CAUSAL == padding)
		lop_avg = linop_padding_create_onedim(dom_col->N, dom_col->dims, padding, TIME_DIM, window - 1, 0);
	
	if (PAD_SAME == padding)
		lop_avg = linop_padding_create_onedim(dom_col->N, dom_col->dims, padding, TIME_DIM, window / 2, window / 2);
	
	auto cod = linop_codomain(lop_avg);
	lop_avg = linop_chain_FF(lop_avg, linop_hankelization_create(cod->N, cod->dims, TIME_DIM, TIME2_DIM, window));

	cod = linop_codomain(lop_avg);
	lop_avg = linop_chain_FF(lop_avg, linop_avg_create(cod->N, cod->dims, TIME2_FLAG));

	ret = nlop_append_FF(ret, 1, nlop_from_linop_F(lop_avg));
	ret = nlop_combine_FF(noir_join_create(model), ret);
	
	ret = nlop_link_F(ret, 1, 0);
	ret = nlop_link_F(ret, 1, 0);

	return ret;
}


struct noir_nlop_debug_s {

	nlop_data_t super;

	const struct nlop_s* frw;
	const struct nlop_s* der;
	const struct nlop_s* adj;
	long size;
};

DEF_TYPEID(noir_nlop_debug_s);

static void noir_nlop_debug_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(noir_nlop_debug_s, _data);

	md_copy(1, MD_DIMS(data->size), dst, src, CFL_SIZE);
	
	nlop_generic_apply_unchecked(data->frw, 1, (void*[1]) { (void*)src });
}

static void noir_nlop_debug_der(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(noir_nlop_debug_s, _data);

	md_copy(1, MD_DIMS(data->size), dst, src, CFL_SIZE);
	
	nlop_generic_apply_unchecked(data->der, 1, (void*[1]) { (void*)src });
}

static void noir_nlop_debug_adj(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(noir_nlop_debug_s, _data);

	md_copy(1, MD_DIMS(data->size), dst, src, CFL_SIZE);
	
	nlop_generic_apply_unchecked(data->adj, 1, (void*[1]) { (void*)src });
}

static void dump_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(noir_nlop_debug_s, _data);

	nlop_free(data->frw);
	nlop_free(data->der);
	nlop_free(data->adj);

	xfree(data);
}

/**
 * Operator dumping its input to a filename_%d_frw/der/adj.cfl file
 * @param N
 * @param dims
 * @param filename
 * @param frw - store frw input
 * @param der - store der input
 * @param adj - store adj input
 */

const struct nlop_s* noir_nlop_dump_create(struct noir2_net_s* model, const char* filename)
{
	PTR_ALLOC(struct noir_nlop_debug_s, data);
	SET_TYPEID(noir_nlop_debug_s, data);

	auto decomp = noir_decomp_create(model);

	auto iov_img = nlop_generic_codomain(decomp, 0);
	auto iov_col = nlop_generic_codomain(decomp, 1);

	const char* ni = ptr_printf("%s_img", filename);
	const char* nc = ptr_printf("%s_col", filename);
	
	auto dump_img = nlop_dump_create(iov_img->N, iov_img->dims, ni, true, true, true);
	auto dump_col = nlop_dump_create(iov_col->N, iov_col->dims, nc, true, true, true);

	xfree(ni);
	xfree(nc);

	data->frw = nlop_clone(decomp);
	data->frw = nlop_append_FF(data->frw, 0, nlop_clone(dump_img));
	data->frw = nlop_append_FF(data->frw, 1, nlop_clone(dump_col));
	data->frw = nlop_del_out_F(data->frw, 0);
	data->frw = nlop_del_out_F(data->frw, 0);

	data->der = nlop_clone(decomp);
	data->der = nlop_append_FF(data->der, 0, nlop_from_linop(nlop_get_derivative(dump_img, 0, 0)));
	data->der = nlop_append_FF(data->der, 1, nlop_from_linop(nlop_get_derivative(dump_col, 0, 0)));
	data->der = nlop_del_out_F(data->der, 0);
	data->der = nlop_del_out_F(data->der, 0);

	data->adj = nlop_clone(decomp);
	data->adj = nlop_append_FF(data->adj, 0, nlop_from_linop_F(linop_get_adjoint(nlop_get_derivative(dump_img, 0, 0))));
	data->adj = nlop_append_FF(data->adj, 1, nlop_from_linop_F(linop_get_adjoint(nlop_get_derivative(dump_col, 0, 0))));
	data->adj = nlop_del_out_F(data->adj, 0);
	data->adj = nlop_del_out_F(data->adj, 0);

	int N = nlop_generic_domain(decomp, 0)->N;
	long dims[N];
	md_copy_dims(N, dims, nlop_generic_domain(decomp, 0)->dims);

	data->size = md_calc_size(N, dims);

	nlop_free(decomp);
	nlop_free(dump_img);
	nlop_free(dump_col);

	return nlop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), noir_nlop_debug_fun, noir_nlop_debug_der, noir_nlop_debug_adj, NULL, NULL, dump_del);
}




