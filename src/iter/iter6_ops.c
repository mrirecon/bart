/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <assert.h>
#include <math.h>

#include "num/ops.h"
#include "num/ops_p.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "nlops/nlop.h"

#include "misc/misc.h"
#include "misc/types.h"

#include "iter/iter6.h"
#include "iter6_ops.h"




struct adadelta_update_s {

	INTERFACE(operator_data_t);

	const struct iovec_s* dom;
	float* floating_dx;
	float* floating_g;

	float rho;
	float epsilon;
};

static DEF_TYPEID(adadelta_update_s);

static void adadelta_update_apply(const operator_data_t* _data, float learning_rate, complex float* _dst, const complex float* _src)
{
	struct adadelta_update_s* d = CAST_DOWN(adadelta_update_s, _data);

	if ((NULL == d->floating_g) || (NULL == d->floating_dx)){

		d->floating_g = md_alloc_sameplace(d->dom->N, d->dom->dims, d->dom->size, _dst);
		d->floating_dx = md_alloc_sameplace(d->dom->N, d->dom->dims, d->dom->size, _dst);
		md_clear(d->dom->N, d->dom->dims, d->floating_g, d->dom->size);
		md_clear(d->dom->N, d->dom->dims, d->floating_dx, d->dom->size);
	}

	float* dst = (float*)_dst;
	float* src = (float*)_src;

	//Accumulate E[g²]
	md_smul(d->dom->N, d->dom->dims, d->floating_g, d->floating_g, d->rho / (1. - d->rho));
	md_fmac(d->dom->N, d->dom->dims, d->floating_g, src, src);
	md_smul(d->dom->N, d->dom->dims, d->floating_g, d->floating_g, 1. - d->rho);

	//Compute RMS[g]
	float* rmsg = md_alloc_sameplace(d->dom->N, d->dom->dims, d->dom->size, _dst);
	md_sadd(d->dom->N, d->dom->dims, rmsg, d->floating_g, d->epsilon);
	md_sqrt(d->dom->N, d->dom->dims, rmsg, rmsg);

	//Compute RMS[x]/RMS[g]
	float* rmsx_rmsg = md_alloc_sameplace(d->dom->N, d->dom->dims, d->dom->size, _dst);
	md_sadd(d->dom->N, d->dom->dims, rmsx_rmsg, d->floating_dx, d->epsilon);
	md_sqrt(d->dom->N, d->dom->dims, rmsx_rmsg, rmsx_rmsg);
	md_div(d->dom->N, d->dom->dims, rmsx_rmsg, rmsx_rmsg, rmsg);
	md_free(rmsg);

	//Compute dx
	md_clear(d->dom->N, d->dom->dims, dst, d->dom->size);
	md_fmac(d->dom->N, d->dom->dims, dst, rmsx_rmsg, src);
	md_smul(d->dom->N, d->dom->dims, dst, dst, -learning_rate);
	md_free(rmsx_rmsg);

	//Accumulate E[dx²]
	md_smul(d->dom->N, d->dom->dims, d->floating_dx, d->floating_dx, d->rho / (1. - d->rho));
	md_fmac(d->dom->N, d->dom->dims, d->floating_dx, dst, dst);
	md_smul(d->dom->N, d->dom->dims, d->floating_dx, d->floating_dx, (1. - d->rho));
}


static void adadelta_update_free(const operator_data_t* _data)
{
	const auto d = CAST_DOWN(adadelta_update_s, _data);
	iovec_free(d->dom);
	md_free(d->floating_g);
	md_free(d->floating_dx);
	xfree(d);
}

const struct operator_p_s* operator_adadelta_update_create(unsigned int N, const long dims[N], float rho, float epsilon)
{
	PTR_ALLOC(struct adadelta_update_s, data);
	SET_TYPEID(adadelta_update_s, data);

	long rdims[N + 1];
	rdims[0] = 2;
	md_copy_dims(N, rdims + 1, dims);

	data->dom = iovec_create(N + 1, rdims, FL_SIZE);
	data->floating_dx = NULL;
	data->floating_g = NULL;
	data->rho = rho;
	data->epsilon = epsilon;

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), adadelta_update_apply, adadelta_update_free);
}

struct adam_update_s {

	INTERFACE(operator_data_t);

	const struct iovec_s* dom;
	float* first_mom;
	float* second_mom;

	float beta1;
	float beta2;
	float epsilon;

	int t;
	int t_reset;
};

static DEF_TYPEID(adam_update_s);

static void adam_update_apply(const operator_data_t* _data, float learning_rate, complex float* _dst, const complex float* _src)
{
	struct adam_update_s* d = CAST_DOWN(adam_update_s, _data);

	if ((NULL == d->first_mom) || (NULL == d->second_mom)){

		d->first_mom = md_alloc_sameplace(d->dom->N, d->dom->dims, d->dom->size, _dst);
		d->second_mom = md_alloc_sameplace(d->dom->N, d->dom->dims, d->dom->size, _dst);
		md_clear(d->dom->N, d->dom->dims, d->first_mom, d->dom->size);
		md_clear(d->dom->N, d->dom->dims, d->second_mom, d->dom->size);
	}

	float* dst = (float*)_dst;
	float* src = (float*)_src;

	//Accumulate first momentum
	md_smul(d->dom->N, d->dom->dims, d->first_mom, d->first_mom, d->beta1);
	md_axpy(d->dom->N, d->dom->dims, d->first_mom, 1. - d->beta1, src);

	//Accumulate second momentum
	md_smul(d->dom->N, d->dom->dims, d->second_mom, d->second_mom, d->beta2);
	md_mul(d->dom->N, d->dom->dims, dst, src, src);
	md_axpy(d->dom->N, d->dom->dims, d->second_mom, 1. - d->beta2, dst);

	//Compute unbiased scales
	float scale = learning_rate * sqrtf(1. - powf(d->beta2, (float)d->t + 1.)) / (1. - powf(d->beta1, (float)d->t + 1.));
	float epsilon = d->epsilon * sqrtf(1. - powf(d->beta2, (float)d->t + 1.));
	d->t++;

	//Compute update
	md_sqrt(d->dom->N, d->dom->dims, dst, d->second_mom);
	md_sadd(d->dom->N, d->dom->dims, dst, dst, epsilon);
	md_div(d->dom->N, d->dom->dims, dst, d->first_mom, dst);
	md_smul(d->dom->N, d->dom->dims, dst, dst, -scale);

	if (d->t == d->t_reset) {

		d->t = 0;
		md_free(d->first_mom);
		md_free(d->second_mom);
		d->first_mom = NULL;
		d->second_mom = NULL;
	}
}


static void adam_update_free(const operator_data_t* _data)
{
	const auto d = CAST_DOWN(adam_update_s, _data);
	iovec_free(d->dom);
	md_free(d->first_mom);
	md_free(d->second_mom);
	xfree(d);
}

const struct operator_p_s* operator_adam_update_create(unsigned int N, const long dims[N], float beta1, float beta2, float epsilon, long reset_mod)
{
	PTR_ALLOC(struct adam_update_s, data);
	SET_TYPEID(adam_update_s, data);

	long rdims[N + 1];
	rdims[0] = 2;
	md_copy_dims(N, rdims + 1, dims);

	data->dom = iovec_create(N + 1, rdims, FL_SIZE);
	data->first_mom = NULL;
	data->second_mom = NULL;
	data->epsilon = epsilon;
	data->beta1 = beta1;
	data->beta2 = beta2;
	data->t = 0;
	data->t_reset = reset_mod;

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), adam_update_apply, adam_update_free);
}


struct clip_s {

	INTERFACE(operator_data_t);

	const struct iovec_s* dom;

	float clipval;
	float clipnorm;
};

static DEF_TYPEID(clip_s);

static void clip_apply(const operator_data_t* _data, unsigned int N, void* args[N])
{
	struct clip_s* d = CAST_DOWN(clip_s, _data);
	assert(2 == N);
	float* dst = (float*)args[0];
	float* src = (float*)args[1];

	if (d->clipnorm != 0.){

		float norm = md_norm(d->dom->N - 1, d->dom->dims + 1, src) / md_calc_size(d->dom->N, d->dom->dims);
		md_smul(d->dom->N, d->dom->dims, dst, src, (norm > d->clipnorm) ? (d->clipnorm / norm) : 1.);
	}

	if (d->clipval != 0.){

		md_smax(d->dom->N, d->dom->dims, dst, (d->clipnorm != 0.) ? dst : src, -d->clipval);
		md_smin(d->dom->N, d->dom->dims, dst, dst, d->clipval);
	}

}

static void clip_free(const operator_data_t* _data)
{
	const auto d = CAST_DOWN(clip_s, _data);
	iovec_free(d->dom);
	xfree(d);
}

const struct operator_s* operator_clip_create(unsigned int N, const long dims[N], float clipnorm, float clipval)
{
	PTR_ALLOC(struct clip_s, data);
	SET_TYPEID(clip_s, data);

	long rdims[N + 1];
	rdims[0] = 2;
	md_copy_dims(N, rdims + 1, dims);

	data->dom = iovec_create(N + 1, rdims, FL_SIZE);

	data->clipnorm = clipnorm;
	data->clipval = clipval;

	return operator_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), clip_apply, clip_free);
}

struct sgd_update_s {

	INTERFACE(operator_data_t);

	const struct iovec_s* dom;
};

static DEF_TYPEID(sgd_update_s);

static void sgd_update_apply(const operator_data_t* _data, float learning_rate, complex float* _dst, const complex float* _src)
{
	struct sgd_update_s* d = CAST_DOWN(sgd_update_s, _data);

	float* dst = (float*)_dst;
	float* src = (float*)_src;

	md_smul(d->dom->N, d->dom->dims, dst, src, -learning_rate);
}


static void sgd_update_free(const operator_data_t* _data)
{
	const auto d = CAST_DOWN(sgd_update_s, _data);
	iovec_free(d->dom);
	xfree(d);
}

const struct operator_p_s* operator_sgd_update_create(unsigned int N, const long dims[N])
{
	PTR_ALLOC(struct sgd_update_s, data);
	SET_TYPEID(sgd_update_s, data);

	long rdims[N + 1];
	rdims[0] = 2;
	md_copy_dims(N, rdims + 1, dims);

	data->dom = iovec_create(N + 1, rdims, FL_SIZE);

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), sgd_update_apply, sgd_update_free);
}