/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2019. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Xiaoqing Wang, Martin Uecker
 */

#include <assert.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

#include "misc/types.h"
#include "misc/mri.h"
#include "misc/debug.h"
#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops_p.h"
#include "num/rand.h"

#include "num/ops.h"
#include "num/iovec.h"

#include "wavelet/wavthresh.h"

#include "nlops/nlop.h"

#include "iter/prox.h"
#include "iter/vec.h"
#include "iter/italgos.h"
#include "iter/iter3.h"
#include "iter/iter2.h"

#include "iter_l1.h"



struct T1inv_s {

	INTERFACE(iter_op_data);

	const struct nlop_s* nlop;
	const struct iter3_irgnm_conf* conf;
    
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




static void normal_fista(iter_op_data* _data, float* dst, const float* src)
{
	auto data = CAST_DOWN(T1inv_s, _data);

	float* tmp = md_alloc_sameplace(1, MD_DIMS(data->size_y), FL_SIZE, src);

	linop_forward_unchecked(nlop_get_derivative(data->nlop, 0, 0), (complex float*)tmp, (const complex float*)src);
	linop_adjoint_unchecked(nlop_get_derivative(data->nlop, 0, 0), (complex float*)dst, (const complex float*)tmp);

	md_free(tmp);

	long res = data->dims[0];
	long parameters = data->dims[COEFF_DIM];
	long coils = data->dims[COIL_DIM];

	md_axpy(1, MD_DIMS(data->size_x * coils / (coils + parameters)),
						 dst + res * res * 2 * parameters,
						 data->alpha,
						 src + res * res * 2 * parameters);
}

static void pos_value(iter_op_data* _data, float* dst, const float* src)
{
	auto data = CAST_DOWN(T1inv_s, _data);

	long res = data->dims[0];
	long parameters = data->dims[COEFF_DIM];

	long dims1[DIMS];

	md_select_dims(DIMS, FFT_FLAGS, dims1, data->dims);

	md_zsmax(DIMS, dims1, (_Complex float*)dst + (parameters - 1) * res * res,
			(const _Complex float*)src + (parameters - 1) * res * res, 0.1);
}



static void combined_prox(iter_op_data* _data, float rho, float* dst, const float* src)
{
	struct T1inv_s* data = CAST_DOWN(T1inv_s, _data);

	if (data->first_iter) {

		data->first_iter = false;

	} else {

		pos_value(_data, dst, src);
	}

	operator_p_apply_unchecked(data->prox2, rho, (_Complex float*)dst, (const _Complex float*)dst);

	pos_value(_data, dst, dst);
}



static void inverse_fista(iter_op_data* _data, float alpha, float* dst, const float* src)
{
	auto data = CAST_DOWN(T1inv_s, _data);

	data->alpha = alpha;	// update alpha for normal operator

    
	void* x = md_alloc_sameplace(1, MD_DIMS(data->size_x), FL_SIZE, src);
	md_gaussian_rand(1, MD_DIMS(data->size_x / 2), x);
	double maxeigen = power(20, data->size_x, select_vecops(src), (struct iter_op_s){ normal_fista, CAST_UP(data) }, x);
	md_free(x);

	double step = data->conf->step / maxeigen;

	debug_printf(DP_DEBUG3, "##reg. alpha = %f\n", alpha);

	wavthresh_rand_state_set(data->prox1, 1);
    
	int maxiter = MIN(data->conf->cgiter, 10 * powf(2, data->outer_iter));
    
	float* tmp = md_alloc_sameplace(1, MD_DIMS(data->size_y), FL_SIZE, src);

	linop_adjoint_unchecked(nlop_get_derivative(data->nlop, 0, 0), (complex float*)tmp, (const complex float*)src);

	float eps = md_norm(1, MD_DIMS(data->size_x), tmp);

	data->first_iter = true;

	NESTED(void, continuation, (struct ist_data* itrdata))
	{
		itrdata->scale = data->alpha;
	};

	fista(maxiter, data->conf->cgtol * alpha * eps, step,
		data->size_x,
		select_vecops(src),
		continuation,
		(struct iter_op_s){ normal_fista, CAST_UP(data) },
		(struct iter_op_p_s){ combined_prox, CAST_UP(data) },
		dst, tmp, NULL);

	pos_value(CAST_UP(data), dst, dst);

	md_free(tmp);

	data->outer_iter++;
}


static const struct operator_p_s* create_prox(const long img_dims[DIMS])
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

	return prox_wavelet_thresh_create(DIMS, img_dims, wflags, COEFF_FLAG, minsize, 1., randshift);
}


struct T1inv2_s {

	INTERFACE(operator_data_t);

	struct T1inv_s data;
};

DEF_TYPEID(T1inv2_s);





static void T1inv_apply(const operator_data_t* _data, float alpha, complex float* dst, const complex float* src)
{
	const auto data = &CAST_DOWN(T1inv2_s, _data)->data;
	inverse_fista(CAST_UP(data), alpha, (float*)dst, (const float*)src);
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


static const struct operator_p_s* T1inv_p_create(const struct iter3_irgnm_conf* conf, const long dims[DIMS], struct nlop_s* nlop)
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
	debug_print_dims(DP_INFO, DIMS, img_dims);

	auto prox1 = create_prox(img_dims);
	auto prox2 = op_p_auto_normalize(prox1, ~COEFF_FLAG);

	struct T1inv_s idata = {

		{ &TYPEID(T1inv_s) }, nlop_clone(nlop), conf,
		N, M, 1.0, ndims, true, 0, prox1, prox2
	};

	data->data = idata;

	return operator_p_create(dm->N, dm->dims, cd->N, cd->dims, CAST_UP(PTR_PASS(data)), T1inv_apply, T1inv_del);
}



struct iterT1_nlop_s {

	INTERFACE(iter_op_data);

	struct nlop_s nlop;
};

DEF_TYPEID(iterT1_nlop_s);


static void nlop_for_iter(iter_op_data* _o, float* _dst, const float* _src)
{
	const auto nlop = CAST_DOWN(iterT1_nlop_s, _o);

	operator_apply_unchecked(nlop->nlop.op, (complex float*)_dst, (const complex float*)_src);
}

static void nlop_der_iter(iter_op_data* _o, float* _dst, const float* _src)
{
	const auto nlop = CAST_DOWN(iterT1_nlop_s, _o);

	assert(2 == operator_nr_args(nlop->nlop.op));
	linop_forward_unchecked(nlop->nlop.derivative[0], (complex float*)_dst, (const complex float*)_src);
}



void mdb_irgnm_l1(const iter3_conf* _conf,
	const long dims[],
	struct nlop_s* nlop,
	long N, float* dst,
	long M, const float* src)
{
	auto cd = nlop_codomain(nlop);
	auto dm = nlop_domain(nlop);

	assert(M * sizeof(float) == md_calc_size(cd->N, cd->dims) * cd->size);
	assert(N * sizeof(float) == md_calc_size(dm->N, dm->dims) * dm->size);

	struct iter3_irgnm_conf* conf = CAST_DOWN(iter3_irgnm_conf, _conf);

	struct iterT1_nlop_s nl_data = { { &TYPEID(iterT1_nlop_s) }, *nlop };

	const struct operator_p_s* inv_op = T1inv_p_create(conf, dims, nlop);

	irgnm2(conf->iter, conf->alpha, 0., conf->alpha_min, conf->redu, N, M, select_vecops(src),
		(struct iter_op_s){ nlop_for_iter, CAST_UP(&nl_data) },
		(struct iter_op_s){ nlop_der_iter, CAST_UP(&nl_data) },
		OPERATOR_P2ITOP(inv_op),
		dst, NULL, src,
		(struct iter_op_s){ NULL, NULL },
		NULL);

	operator_p_free(inv_op);
}

