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

#include "iter/prox2.h"
#include "iter/vec.h"
#include "iter/italgos.h"
#include "iter/iter2.h"
#include "iter/iter3.h"
#include "iter/iter4.h"

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

	long res = data->dims[0];
	long parameters = data->dims[COEFF_DIM];
	long coils = data->dims[COIL_DIM];
	long slices = data->dims[SLICE_DIM];

	if (1 == data->conf->opt_reg) {

		md_axpy(1, MD_DIMS(data->size_x * coils / (coils + parameters)),
	                                        dst + res * res * 2 * parameters * slices,
						data->alpha,
	                                        src + res * res * 2 * parameters * slices);
	} else {

		md_axpy(1, MD_DIMS(data->size_x), dst, data->alpha, src);
	}
}

static void pos_value(iter_op_data* _data, float* dst, const float* src)
{
	auto data = CAST_DOWN(T1inv_s, _data);

	long res = data->dims[0];
	long parameters = data->dims[COEFF_DIM];
	long slices = data->dims[SLICE_DIM];

	long dims1[DIMS];

	md_select_dims(DIMS, FFT_FLAGS, dims1, data->dims);

	for (int i = 0; i < slices; i++) {

	        md_zsmax(DIMS, dims1, (_Complex float*)dst + (parameters - data->conf->constrained_maps) * res * res + i * res * res * parameters,
			(const _Complex float*)src + (parameters - data->conf->constrained_maps) * res * res + i * res * res * parameters, data->conf->lower_bound);

        }

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

	data->alpha = alpha;	// update alpha for normal operator


	void* x = md_alloc_sameplace(1, MD_DIMS(data->size_x), FL_SIZE, src);
	md_gaussian_rand(1, MD_DIMS(data->size_x / 2), x);
	double maxeigen = power(20, data->size_x, select_vecops(src), (struct iter_op_s){ normal, CAST_UP(data) }, x);
	md_free(x);

	double step = data->conf->step / maxeigen;

	debug_printf(DP_DEBUG3, "##reg. alpha = %f\n", alpha);

	wavthresh_rand_state_set(data->prox1, 1);
    
	int maxiter = MIN(data->conf->c2->cgiter, 10 * powf(2, data->outer_iter));
    
	float* tmp = md_alloc_sameplace(1, MD_DIMS(data->size_y), FL_SIZE, src);

	linop_adjoint_unchecked(nlop_get_derivative(data->nlop, 0, 0), (complex float*)tmp, (const complex float*)src);

	float eps = md_norm(1, MD_DIMS(data->size_x), tmp);

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
		dst, tmp, NULL);

	pos_value(CAST_UP(data), dst, dst);

	md_free(tmp);

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

	return prox_wavelet_thresh_create(DIMS, img_dims, wflags, jflag, minsize, lambda, randshift);
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
	debug_print_dims(DP_INFO, DIMS, img_dims);

	auto prox1 = create_prox(img_dims, COEFF_FLAG, 1.);
	auto prox2 = op_p_auto_normalize(prox1, ~(COEFF_FLAG | SLICE_FLAG), NORM_L2);

	struct T1inv_s idata = {

		{ &TYPEID(T1inv_s) }, nlop_clone(nlop), conf,
		N, M, 1.0, ndims, true, 0, prox1, conf->auto_norm_off ? prox1 : prox2
	};

	data->data = idata;

	return operator_p_create(dm->N, dm->dims, cd->N, cd->dims, CAST_UP(PTR_PASS(data)), T1inv_apply, T1inv_del);
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

	const struct operator_p_s* inv_op = T1inv_p_create(conf, dims, nlop);

	iter4_irgnm2(CAST_UP(conf->c2), nlop,
		N, dst, NULL, M, src, inv_op,
		(struct iter_op_s){ NULL, NULL });

	operator_p_free(inv_op);
}

