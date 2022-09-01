/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2019. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Xiaoqing Wang, Nick Scholand, Martin Uecker
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <math.h>

#include "misc/types.h"
#include "misc/mri.h"
#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/version.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops_p.h"
#include "num/rand.h"

#include "num/ops.h"
#include "num/iovec.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "wavelet/wavthresh.h"

#include "nlops/nlop.h"

#include "iter/prox.h"
#include "iter/prox2.h"
#include "iter/vec.h"
#include "iter/italgos.h"
#include "iter/iter2.h"
#include "iter/iter3.h"
#include "iter/iter4.h"
#include "iter/lsqr.h"
#include "iter/prox.h"
#include "iter/thresh.h"
#include "iter/vec.h"
#include "iter/admm.h"


#include "linops/someops.h"

#include "grecon/optreg.h"
#include "grecon/italgo.h"
#include "moba/optreg.h"
#include "moba/T1fun.h"

#include "iter_l1.h"



struct T1inv_s {

	INTERFACE(iter_op_data);

	const struct nlop_s* nlop;
	const struct mdb_irgnm_l1_conf* conf;
    
	long size_x;
	long size_y;

	float alpha;
    
	const long* dims;

	bool first_iter;
	int outer_iter;

	const struct operator_p_s* prox1;
	const struct operator_p_s* prox2;
};

DEF_TYPEID(T1inv_s);




static void normal(iter_op_data* _data, float* dst, const float* src)
{
	auto data = CAST_DOWN(T1inv_s, _data);

	linop_normal_unchecked(nlop_get_derivative(data->nlop, 0, 0), (complex float*)dst, (const complex float*)src);


// We do not enforce this for now, for backwards compatibility
#if 0
	if (0. == data->alpha)
		return;
 
	assert(dst != src);
#endif

	long img_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, data->dims);

	long img_strs[DIMS];
	md_calc_strides(DIMS, img_strs, img_dims, CFL_SIZE);

	long map_dims[DIMS];
	md_select_dims(DIMS, ~COEFF_FLAG, map_dims, img_dims);
	
	long pos[DIMS] = { 0 };
	for (pos[COEFF_DIM] = 0; pos[COEFF_DIM] < img_dims[COEFF_DIM]; pos[COEFF_DIM]++) {

		complex float* map_dst = &MD_ACCESS(DIMS, img_strs, pos, (complex float*)dst);
		const complex float* map_src = &MD_ACCESS(DIMS, img_strs, pos, (const complex float*)src);

		if (MD_IS_SET(data->conf->l2flags, pos[COEFF_DIM]))
			md_zaxpy2(DIMS, map_dims, img_strs, map_dst, data->alpha, img_strs, map_src);
	}

	complex float* col_dst = ((complex float*)dst) + md_calc_size(DIMS, img_dims);
	const complex float* col_src = ((const complex float*)src) + md_calc_size(DIMS, img_dims);
	long col_size = data->size_x / 2 - md_calc_size(DIMS, img_dims);

	md_zaxpy(1, MD_DIMS(col_size), col_dst, data->alpha, col_src);
}

static void pos_value(iter_op_data* _data, float* dst, const float* src)
{
	auto data = CAST_DOWN(T1inv_s, _data);


	// filter coils here, as we want to leave the coil sensitivity part untouched
	long img_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, data->dims);

	long strs[DIMS];
	md_calc_strides(DIMS, strs, img_dims, CFL_SIZE);

	long dims1[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, dims1, img_dims);
	
	long pos[DIMS] = { 0 };

	do {

		if ((1UL << pos[COEFF_DIM]) & data->conf->constrained_maps) {

			md_zsmax2(DIMS, dims1,
				strs, &MD_ACCESS(DIMS, strs, pos, (complex float*)dst),
				strs, &MD_ACCESS(DIMS, strs, pos, (const complex float*)src),
				data->conf->lower_bound);
		}

	} while(md_next(DIMS, img_dims, ~FFT_FLAGS, pos));
}



static void combined_prox(iter_op_data* _data, float rho, float* dst, const float* src)
{
	struct T1inv_s* data = CAST_DOWN(T1inv_s, _data);

	// coil sensitivity part is left untouched

	assert(src == dst); 

	if (data->first_iter) {

		data->first_iter = false;

	} else {

		pos_value(_data, dst, src);
	}

	if (1 == data->conf->opt_reg)
		operator_p_apply_unchecked(data->prox2, rho, (_Complex float*)dst, (const _Complex float*)dst);

	pos_value(_data, dst, dst);
}



static void inverse_fista(iter_op_data* _data, float alpha, float* dst, const float* src)
{
	auto data = CAST_DOWN(T1inv_s, _data);

	double maxeigen = 0.;

	if (!use_compat_to_version("v0.6.00")) {

		data->alpha = 0.;
		maxeigen = alpha;
	} else {

		data->alpha = alpha;
	}

	void* x = md_alloc_sameplace(1, MD_DIMS(data->size_x), FL_SIZE, src);
	md_gaussian_rand(1, MD_DIMS(data->size_x / 2), x);

	maxeigen += power(20, data->size_x, select_vecops(src), (struct iter_op_s){ normal, CAST_UP(data) }, x);

	md_free(x);

	data->alpha = alpha;	// reset alpha

	assert(data->conf->step < 1.);

	double step = data->conf->step / maxeigen;

	debug_printf(DP_DEBUG3, "##max eigenv = %f, step = %f, alpha = %f\n", maxeigen, step, alpha);

	wavthresh_rand_state_set(data->prox1, 1);
    
	int maxiter = MIN(data->conf->c2->cgiter, 10 * powf(2, data->outer_iter));


	float eps = md_norm(1, MD_DIMS(data->size_x), src);

	data->first_iter = true;

	NESTED(void, continuation, (struct ist_data* itrdata))
	{
		itrdata->scale = data->alpha;
	};

	fista(maxiter, data->conf->c2->cgtol * alpha * eps, step,
		data->size_x,
		select_vecops(src),
		continuation,
		(struct iter_op_s){ normal, CAST_UP(data) },
		(struct iter_op_p_s){ combined_prox, CAST_UP(data) },
		dst, src, NULL);

	pos_value(CAST_UP(data), dst, dst);

	data->outer_iter++;
}

static void inverse_admm(iter_op_data* _data, float alpha, float* dst, const float* src)
{
	auto data = CAST_DOWN(T1inv_s, _data);

	data->alpha = alpha;	// update alpha for normal operator

	int maxiter = MIN(data->conf->c2->cgiter, 10 * powf(2, data->outer_iter));

	// initialize prox functions

	const struct operator_p_s* thresh_ops[NUM_REGS] = { NULL };
	const struct linop_s* trafos[NUM_REGS] = { NULL };

	debug_printf(DP_INFO, "##reg. alpha = %f\n", data->alpha);

	struct optreg_conf optreg_conf = optreg_defaults;
	optreg_conf.moba_model = IRLL;

	opt_reg_moba_configure(DIMS, data->dims, data->conf->ropts, thresh_ops, trafos, &optreg_conf);

	struct iter_admm_conf conf1 = iter_admm_defaults;

	conf1.maxiter = maxiter;
	conf1.rho = data->conf->rho;
	conf1.cg_eps = 0.01 * alpha;

	struct iter_admm_conf *conf = &conf1;
	unsigned int D = data->conf->ropts->r;;

	const struct operator_s* normaleq_op = NULL;

	UNUSED(normaleq_op);

	struct admm_plan_s admm_plan = {

		.maxiter = conf->maxiter,
		.maxitercg = conf->maxitercg,
		.cg_eps = conf->cg_eps,
		.rho = conf->rho,
		.num_funs = D,
		.do_warmstart = conf->do_warmstart,
		.dynamic_rho = conf->dynamic_rho,
		.dynamic_tau = conf->dynamic_tau,
		.relative_norm = conf->relative_norm,
		.hogwild = conf->hogwild,
		.ABSTOL = conf->ABSTOL,
		.RELTOL = conf->RELTOL,
		.alpha = conf->alpha,
		.lambda = alpha,
		.tau = conf->tau,
		.tau_max = conf->tau_max,
		.mu = conf->mu,
		.fast = conf->fast,
		.biases = NULL,
	};

	struct admm_op a_ops[D ?:1];
	struct iter_op_p_s a_prox_ops[D ?:1];


	for (unsigned int i = 0; i < D; i++) {

		a_ops[i].forward = OPERATOR2ITOP(trafos[i]->forward),
		a_ops[i].normal = OPERATOR2ITOP(trafos[i]->normal);
		a_ops[i].adjoint = OPERATOR2ITOP(trafos[i]->adjoint);

		a_prox_ops[i] = OPERATOR_P2ITOP(thresh_ops[i]);
	}

	admm_plan.ops = a_ops;
	admm_plan.prox_ops = a_prox_ops;


	long z_dims[D ?: 1];

	for (unsigned int i = 0; i < D; i++)
		z_dims[i] = 2 * md_calc_size(linop_codomain(trafos[i])->N, linop_codomain(trafos[i])->dims);


	admm(&admm_plan, admm_plan.num_funs,
		z_dims, data->size_x, (float*)dst, src,
		select_vecops(src),
		(struct iter_op_s){ normal, CAST_UP(data) }, NULL);


	opt_reg_free(data->conf->ropts, thresh_ops, trafos);

	data->outer_iter++;
}


static const struct operator_p_s* create_prox(const long img_dims[DIMS], unsigned long jflag, float lambda)
{
	bool randshift = true;
	long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
	unsigned int wflags = 0;

	for (unsigned int i = 0; i < DIMS; i++) {

		if ((1 < img_dims[i]) && MD_IS_SET(FFT_FLAGS, i)) {

			wflags = MD_SET(wflags, i);
			minsize[i] = MIN(img_dims[i], DIMS);
		}
	}

	return prox_wavelet_thresh_create(DIMS, img_dims, wflags, jflag, WAVELET_DAU2, minsize, lambda, randshift);
}


struct T1inv2_s {

	INTERFACE(operator_data_t);

	struct T1inv_s data;
};

DEF_TYPEID(T1inv2_s);





static void T1inv_apply(const operator_data_t* _data, float alpha, complex float* dst, const complex float* src)
{
	const auto data = &CAST_DOWN(T1inv2_s, _data)->data;


	switch (data->conf->algo) {
	
	case ALGO_FISTA:
		inverse_fista(CAST_UP(data), alpha, (float*)dst, (const float*)src);
		break;

	case ALGO_ADMM:
		inverse_admm(CAST_UP(data), alpha, (float*)dst, (const float*)src);
		break;
	
	default:
		break;
	}
}


static void T1inv_del(const operator_data_t* _data)
{
	auto data = CAST_DOWN(T1inv2_s, _data);

	operator_p_free(data->data.prox1);
	operator_p_free(data->data.prox2);

	nlop_free(data->data.nlop);

	xfree(data->data.dims);
	xfree(data);
}


static const struct operator_p_s* T1inv_p_create(const struct mdb_irgnm_l1_conf* conf, const long dims[DIMS], struct nlop_s* nlop)
{
	PTR_ALLOC(struct T1inv2_s, data);
	SET_TYPEID(T1inv2_s, data);
	SET_TYPEID(T1inv_s, &data->data);

	auto cd = nlop_codomain(nlop);
	auto dm = nlop_domain(nlop);

	int M = 2 * md_calc_size(cd->N, cd->dims);
	int N = 2 * md_calc_size(dm->N, dm->dims);

	long* ndims = *TYPE_ALLOC(long[DIMS]);
	md_copy_dims(DIMS, ndims, dims);

	long img_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, dims);

        // jointly penalize the first few maps
        long penalized_dims = img_dims[COEFF_DIM] - conf->not_wav_maps;

        debug_printf(DP_DEBUG2, "nr. of penalized maps: %d\n", penalized_dims);

        img_dims[COEFF_DIM] = penalized_dims;

	auto prox1 = create_prox(img_dims, COEFF_FLAG, 1.);
	auto prox2 = operator_p_ref(prox1);

	if (0 < conf->not_wav_maps) {

		long map_dims[DIMS];
		md_copy_dims(DIMS, map_dims, img_dims);
		map_dims[COEFF_DIM] = conf->not_wav_maps;

		auto prox3 = prox_zero_create(DIMS, map_dims);
		prox2 = operator_p_stack_FF(COEFF_DIM, COEFF_DIM, prox2, prox3);
	}

	if (conf->auto_norm) {

		auto prox3 = op_p_auto_normalize(prox2, ~(COEFF_FLAG | TIME_FLAG | TIME2_FLAG | SLICE_FLAG), NORM_L2);
		operator_p_free(prox2);
		prox2 = prox3;
	}

	struct T1inv_s idata = {

		{ &TYPEID(T1inv_s) }, nlop_clone(nlop), conf,
		N, M, 1.0, ndims, true, 0, prox1, prox2
	};

	data->data = idata;

	auto tmp = operator_p_create(dm->N, dm->dims, dm->N, dm->dims, CAST_UP(PTR_PASS(data)), T1inv_apply, T1inv_del);

#if 0
	if (0 < cuda_num_devices()) {	// FIXME: not a correct check for GPU mode

		auto tmp2 = tmp;

		tmp = operator_p_gpu_wrapper(tmp2);

		operator_p_free(tmp2);
	}
#endif

	auto result = operator_p_pre_chain(nlop_get_derivative(nlop, 0, 0)->adjoint, tmp);
	operator_p_free(tmp);

	return result;
}




void mdb_irgnm_l1(const struct mdb_irgnm_l1_conf* conf,
	const long dims[DIMS],
	struct nlop_s* nlop,
	long N, float* dst,
	long M, const float* src)
{
	auto cd = nlop_codomain(nlop);
	auto dm = nlop_domain(nlop);

	assert(M * sizeof(float) == md_calc_size(cd->N, cd->dims) * cd->size);
	assert(N * sizeof(float) == md_calc_size(dm->N, dm->dims) * dm->size);

	const struct operator_p_s* inv_op = NULL;

	// initialize prox functions
	const struct operator_p_s* thresh_ops[NUM_REGS] = { NULL };
	const struct linop_s* trafos[NUM_REGS] = { NULL };

	if (ALGO_FISTA == conf->algo) {

		inv_op = T1inv_p_create(conf, dims, nlop);

	} else if (ALGO_ADMM == conf->algo) {

		debug_printf(DP_DEBUG3, " >> linearized problem solved by ADMM ");

		/* use lsqr */
		debug_printf(DP_DEBUG3, "in lsqr\n");

		struct optreg_conf optreg_conf = optreg_defaults;

		optreg_conf.moba_model = IRLL;
		optreg_conf.tvscales_N = conf->tvscales_N;
		optreg_conf.tvscales = conf->tvscales;

		opt_reg_moba_configure(DIMS, dims, conf->ropts, thresh_ops, trafos, &optreg_conf);


		struct iter_admm_conf iadmm_conf = iter_admm_defaults;
		iadmm_conf.maxiter = conf->c2->cgiter;
		iadmm_conf.cg_eps = conf->c2->cgtol;
		iadmm_conf.rho = conf->rho;


		struct lsqr_conf lsqr_conf = lsqr_defaults;
		lsqr_conf.it_gpu = false;
#if 0
		if (0 < cuda_num_devices())	// FIXME: not correct check for GPU mode
			lsqr_conf.it_gpu = true;
#endif
		lsqr_conf.warmstart = true;

		NESTED(void, lsqr_cont, (iter_conf* iconf))
		{
			auto aconf = CAST_DOWN(iter_admm_conf, iconf);

			aconf->maxiter = MIN(iadmm_conf.maxiter, 10. * powf(2., ceil(logf(1. / iconf->alpha) / logf(conf->c2->redu))));
			aconf->cg_eps = iadmm_conf.cg_eps * iconf->alpha;
		};

		lsqr_conf.icont = lsqr_cont;

		inv_op = lsqr2_create(&lsqr_conf, iter2_admm, CAST_UP(&iadmm_conf), NULL, &nlop->derivative[0][0],
			 NULL, conf->ropts->r, thresh_ops, trafos, NULL);
	}

	iter4_irgnm2(CAST_UP(conf->c2), nlop,
		N, dst, NULL, M, src, inv_op,
		(struct iter_op_s){ NULL, NULL });

	operator_p_free(inv_op);

	if (ALGO_ADMM == conf->algo)
		opt_reg_free(conf->ropts, thresh_ops, trafos);
}

