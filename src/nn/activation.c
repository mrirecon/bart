/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <assert.h>
#include <stdbool.h>
#include <math.h>

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"

#include "num/iovec.h"
#include "num/ops.h"
#include "num/multind.h"
#include "num/flpmath.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"
#include "nlops/someops.h"
#include "nlops/tenmul.h"
#include "nlops/zexp.h"
#include "nlops/nlop_jacobian.h"

#include "activation.h"

static const struct nlop_s* append_activation_bias_internal(const struct nlop_s* network, int o, enum ACTIVATION activation, unsigned long bflags, bool bias);


/**
 * Append activation to nlop, free input and return nlop with appended activation
 *
 * @param network operator to append the activation (this operator is freed)
 * @param o output index of network, the layer is appended
 * @param activation type of activation
 */
const struct nlop_s* append_activation(const struct nlop_s* network, int o, enum ACTIVATION activation)
{
	return append_activation_bias_internal(network, o, activation, 0, false);
}

/**
 * Append activation and bias to nlop, free input and return nlop with appended activation
 *
 * @param network operator to append the activation (this operator is freed)
 * @param o output index of network, the layer is appended
 * @param activation type of activation
 * @param bflags select the dims of the bias, i.e. the dims which are not shared. In case of ACT_SOFTMAX, ~bflags is interpreted as batchflags.
 */
const struct nlop_s* append_activation_bias(const struct nlop_s* network, int o, enum ACTIVATION activation, unsigned long bflags)
{
	return append_activation_bias_internal(network, o, activation, bflags, true);
}




static const struct nlop_s* append_activation_bias_internal(const struct nlop_s* network, int o, enum ACTIVATION activation, unsigned long bflags, bool bias)
{
	long NI = nlop_get_nr_in_args(network);
	long NO = nlop_get_nr_out_args(network);
	assert(o < NO);

	const struct nlop_s* nlop_act;

	long N = nlop_generic_codomain(network, o)->N;

	long dims[N];
	md_copy_dims(N, dims, nlop_generic_codomain(network, o)->dims);

	long bdims[N];
	md_select_dims(N, bflags, bdims, dims);

	switch (activation){

		case ACT_LIN:

			nlop_act = nlop_from_linop_F(linop_identity_create(N, dims));
			break;

		case ACT_RELU:

			nlop_act = nlop_relu_create(N, dims);
			break;

		case ACT_SOFTMAX:

			nlop_act = nlop_softmax_create(N, dims, ~bflags);
			break;

		case ACT_SIGMOID:

			nlop_act = nlop_sigmoid_create(N, dims);
			break;

		case ACT_SIGLOG:

			nlop_act = nlop_siglog_create(N, dims, 1, 1);
			break;

		case ACT_IGAUSSIAN:

			nlop_act = nlop_igaussian_create(N, dims, 1);
			break;

		case ACT_CARDIOID:

			nlop_act = nlop_cardioid_create(N, dims);
			break;

		default:

			nlop_act = NULL;
			assert(0);
	}

	if (bias)
		nlop_act = nlop_chain2_FF(nlop_bias_create(N, dims, bdims), 0, nlop_act, 0);


	network = nlop_chain2_swap_FF(network, o, nlop_act, 0);
	network = nlop_shift_output_F(network, o, 0);

	if (!bias)
		return network;

	long bdims_layer[N];
	int j = 0;
	for (int i = 0; i < N; i++)
		if (MD_IS_SET(bflags, i)){

			bdims_layer[j] = bdims[i];
			j += 1;
		}

	network = nlop_reshape_in_F(network, NI, j, bdims_layer);

	return network;
}

struct bias_op_s {

	INTERFACE(nlop_data_t);

	int N;
	const long* dims;
	const long* bdims;

};

DEF_TYPEID(bias_op_s);

static void bias_op_apply(const nlop_data_t* _data, int N, complex float* args[N])
{
	const struct bias_op_s* d = CAST_DOWN(bias_op_s, _data);
	assert(3 == N);

#ifdef USE_CUDA

	if (cuda_ondevice(args[0])) {

		md_copy2(d->N, d->dims, MD_STRIDES(d->N, d->dims, CFL_SIZE), args[0], MD_STRIDES(d->N, d->bdims, CFL_SIZE), args[2], CFL_SIZE);
		md_zadd(d->N, d->dims, args[0], args[1], args[0]);
	} else
#endif

	md_zadd2(d->N, d->dims, MD_STRIDES(d->N, d->dims, CFL_SIZE), args[0], MD_STRIDES(d->N, d->dims, CFL_SIZE), args[1], MD_STRIDES(d->N, d->bdims, CFL_SIZE), args[2]);
}

static void bias_op_deriv1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const struct bias_op_s* d = CAST_DOWN(bias_op_s, _data);
	md_copy2(d->N, d->dims, MD_STRIDES(d->N, d->dims, CFL_SIZE), dst, MD_STRIDES(d->N, d->dims, CFL_SIZE), src, CFL_SIZE);
}

static void bias_op_deriv2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const struct bias_op_s* d = CAST_DOWN(bias_op_s, _data);
	md_copy2(d->N, d->dims, MD_STRIDES(d->N, d->dims, CFL_SIZE), dst, MD_STRIDES(d->N, d->bdims, CFL_SIZE), src, CFL_SIZE);
}

static void bias_op_adj1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const struct bias_op_s* d = CAST_DOWN(bias_op_s, _data);
	md_copy2(d->N, d->dims, MD_STRIDES(d->N, d->dims, CFL_SIZE), dst, MD_STRIDES(d->N, d->dims, CFL_SIZE), src, CFL_SIZE);
}

static void bias_op_adj2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const struct bias_op_s* d = CAST_DOWN(bias_op_s, _data);

	md_clear(d->N, d->bdims, dst, CFL_SIZE);
	md_zsum(d->N, d->dims, ~md_nontriv_dims(d->N, d->bdims), dst, src);

}

static void bias_op_free(const nlop_data_t* _data)
{
	const struct bias_op_s* d = CAST_DOWN(bias_op_s, _data);

	xfree(d->dims);
	xfree(d->bdims);

	xfree(d);
}


const struct nlop_s* nlop_bias_create(unsigned int N, const long dims[N], const long bdims[N])
{
	PTR_ALLOC(struct bias_op_s, data);
	SET_TYPEID(bias_op_s, data);

	data->N = N;

	for (unsigned int i = 0; i < N; i++)
		assert((1 == bdims[i]) || (dims[i] == bdims[i]));

	PTR_ALLOC(long[N], tdims);
	md_copy_dims(N, *tdims, dims);
	data->dims = *PTR_PASS(tdims);

	PTR_ALLOC(long[N], tbdims);
	md_copy_dims(N, *tbdims, bdims);
	data->bdims = *PTR_PASS(tbdims);

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], dims);

	long nl_idims[2][N];
	md_copy_dims(N, nl_idims[0], dims);
	md_copy_dims(N, nl_idims[1], bdims);


	long nl_ostrs[1][N];
	md_copy_strides(N, nl_ostrs[0], MD_STRIDES(N, nl_odims[0], CFL_SIZE));


	long nl_istrs[2][N];
	md_copy_strides(N, nl_istrs[0], MD_STRIDES(N, nl_idims[0], CFL_SIZE));
	md_copy_strides(N, nl_istrs[1], MD_STRIDES(N, nl_idims[1], CFL_SIZE));



	return nlop_generic_create2(1, N, nl_odims, nl_ostrs, 2, N, nl_idims, nl_istrs, CAST_UP(PTR_PASS(data)),
				    bias_op_apply, (nlop_der_fun_t[2][1]){ { bias_op_deriv1}, {bias_op_deriv2} }, (nlop_der_fun_t[2][1]){ {bias_op_adj1}, {bias_op_adj2} }, NULL, NULL, bias_op_free);
}


struct relu_s {

	INTERFACE(nlop_data_t);

	int N;
	const long* rdims;

	void* der;

	float slope_param;
};

DEF_TYPEID(relu_s);


static void relu_apply(const nlop_data_t* _data, complex float* _dst, const complex float* _src)
{
	struct relu_s* d = CAST_DOWN(relu_s, _data);

	int N = d->N;
	const long* dims = d->rdims;
	
	float* dst = (float*)_dst;
	const float* src = (float*)_src;

	md_smax(N, dims, dst, src, 0.);

	float* der = md_alloc_sameplace(N, dims, FL_SIZE, src);

	if ((0 == d->slope_param) && (NULL != der))
		md_greatequal(N, dims, der, src, dst);

	md_free(d->der);
	d->der = md_compress(N, dims, der);

	md_free(der);

	// leaky RELU if slope parameter has been set
	if (0 != d->slope_param) {

		float* tmp = md_alloc_sameplace(N, dims, FL_SIZE, dst);
		float* tmp2 = md_alloc_sameplace(N, dims, FL_SIZE, dst);
		float* tder = md_alloc_sameplace(N, dims, FL_SIZE, dst);

		md_greatequal(N, dims, tder, src, dst);

		// eliminate ones in derivative, where input is zero to calculate tmp(x) = (0, if x >= 0; 1, if x < 0)
		md_lessequal(N, dims, tmp, src, dst);
		md_mul(N, dims, tmp2, tder, tmp);
		md_sub(N, dims, tmp, tmp, tmp2);

		// derivative der(x) = {1, if x >= 0; d->slope_param if x < 0}
		md_axpy(N, dims, tder, d->slope_param, tmp);

		md_mul(N, dims, dst, tder, src);

		md_free(tmp);
		md_free(tmp2);

		md_free(tder);

	}
}

static void relu_free(const nlop_data_t* _data)
{
	struct relu_s* d = CAST_DOWN(relu_s, _data);
	md_free(d->der);
	xfree(d->rdims);

	xfree(_data);
}

static void relu_deradj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* _dst, const complex float* _src)
{
	struct relu_s* d = CAST_DOWN(relu_s, _data);

	assert(0 == i);
	assert(0 == o);
	
	int N = d->N;
	const long* dims = d->rdims;
	
	float* dst = (float*)_dst;
	const float* src = (float*)_src;

	float* der = md_alloc_sameplace(N, dims, FL_SIZE, d->der);
	md_decompress(N, dims, der, d->der);

	md_mul(N, dims, dst, src, der);

	if (0 != d->slope_param) {

		md_smul(N, dims, dst, dst, 1. -d->slope_param);
		md_axpy(N, dims, dst, d->slope_param, src);	
	}

	md_free(der);
}

/**
 * Create leaky RELU nlop with slope control parameter a
 * f(x) = {x, if x >= 0; ax, if x < 0}
 */
const struct nlop_s* nlop_leaky_relu_create(unsigned int N, const long dims[N], float slope_parameter)
{
	PTR_ALLOC(struct relu_s, data);
	SET_TYPEID(relu_s, data);

	data->N = N + 1;
	long rdims[N + 1];
	rdims[0] = 2;
	md_copy_dims(N, rdims + 1, dims);

	data->rdims = ARR_CLONE(long[N + 1], rdims);

	data->slope_param = slope_parameter;
	data->der = NULL;

	return nlop_create(N, dims, N, dims,  CAST_UP(PTR_PASS(data)), relu_apply, relu_deradj, relu_deradj, NULL, NULL, relu_free);
}

const struct nlop_s* nlop_relu_create(unsigned int N, const long dims[N])
{
	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	return nlop_leaky_relu_create(N, dims, 0.);
}


struct softmax_s {

	INTERFACE(nlop_data_t);

	complex float* tmp;
	unsigned long batch_flag;

	unsigned long N;

	const struct iovec_s* dom;
	const struct iovec_s* batchdom;
};

DEF_TYPEID(softmax_s);

static void softmax_apply(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	//S_i = exp(x_i)/sum_k(exp(x_k)) = S_i = exp(x_i - m)/sum_k(exp(x_k - m))
	struct softmax_s* d = CAST_DOWN(softmax_s, _data);

	if (NULL == d->tmp)
		d->tmp = md_alloc_sameplace(d->N, d->dom->dims, d->dom->size, dst);

	complex float* tmp_real = md_alloc_sameplace(d->N, d->dom->dims, CFL_SIZE, src);
	md_zreal(d->N, d->dom->dims, tmp_real, src);

	complex float* max = md_alloc_sameplace(d->N, d->batchdom->dims, CFL_SIZE, src);
	md_zfill(d->N, d->batchdom->dims, max, (complex float)(-INFINITY));
	md_zmax2(d->N, d->dom->dims, d->batchdom->strs, max, d->batchdom->strs, max, d->dom->strs, tmp_real);

#if 1	//FIXME: Optimize md functions for these cases!!!
	complex float* tmp_gpu = md_alloc_sameplace(d->N, d->dom->dims, d->dom->size, dst);
	md_copy2(d->N, d->dom->dims, d->dom->strs, tmp_gpu, d->batchdom->strs, max, CFL_SIZE);
	md_zsub(d->N, d->dom->dims, tmp_real, tmp_real, tmp_gpu);
#else
	md_zsub2(d->N, d->dom->dims, d->dom->strs, tmp_real, d->dom->strs, tmp_real, d->batchdom->strs, max);
#endif
	md_free(max);

	complex float* tmp_exp = md_alloc_sameplace(d->dom->N, d->dom->dims, CFL_SIZE, src);
	md_zexp(d->N, d->dom->dims, tmp_exp, tmp_real);
	md_free(tmp_real);

	complex float* scale = md_alloc_sameplace(d->N, d->batchdom->dims, CFL_SIZE, src);
	md_zsum(d->N, d->dom->dims, ~d->batch_flag, scale, tmp_exp);
#if 1
	md_copy2(d->N, d->dom->dims, d->dom->strs, tmp_gpu, d->batchdom->strs, scale, CFL_SIZE);
	md_zdiv(d->N, d->dom->dims, d->tmp, tmp_exp, tmp_gpu);
	md_free(tmp_gpu);
#else
	md_zdiv2(d->N, d->dom->dims, d->dom->strs, d->tmp, d->dom->strs, tmp_exp, d->batchdom->strs, scale);
#endif
	md_free(scale);
	md_free(tmp_exp);

	md_copy(d->N, d->dom->dims, dst, d->tmp, CFL_SIZE);
}

static void softmax_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	//applying a_i -> a_j = sum_i D_jS_ia_i = sum_i S_i(\delta_ij - S_j)a_i
	const struct softmax_s* d = CAST_DOWN(softmax_s, _data);
	assert(NULL != d->tmp);

	complex float* tmp_real = md_alloc_sameplace(d->N, d->dom->dims, CFL_SIZE, src);
	md_zreal(d->N, d->dom->dims, tmp_real, src);

	//sum_i \delta_ji S_i a_i
	md_ztenmul(d->N, d->dom->dims, dst, d->dom->dims, d->tmp, d->dom->dims, tmp_real);
	md_free(tmp_real);

	//\sum_i S_ia_i
	complex float* tmp1 = md_alloc_sameplace(d->N, d->batchdom->dims, CFL_SIZE, src);
	md_zsum(d->N, d->dom->dims, ~d->batch_flag, tmp1, dst);

	//S_j\sum_i D_jS_ia_i
	complex float* tmp2 = md_alloc_sameplace(d->N, d->dom->dims, CFL_SIZE, src);
#if 1
	complex float* tmp_gpu = md_alloc_sameplace(d->N, d->dom->dims, d->dom->size, dst);
	md_copy2(d->N, d->dom->dims, d->dom->strs, tmp_gpu, d->batchdom->strs, tmp1, CFL_SIZE);
	md_ztenmul(d->N, d->dom->dims, tmp2, d->dom->dims, d->tmp, d->dom->dims, tmp_gpu);
	md_free(tmp_gpu);
#else
	md_ztenmul(d->N, d->dom->dims, tmp2, d->dom->dims, d->tmp, d->batchdom->dims, tmp1);
#endif
	md_free(tmp1);

	md_zsub(d->N, d->dom->dims, dst, dst, tmp2);

	md_free(tmp2);
}

static void softmax_free(const nlop_data_t* _data)
{
	const struct softmax_s* d = CAST_DOWN(softmax_s, _data);

	md_free(d->tmp);

	iovec_free(d->dom);
	iovec_free(d->batchdom);

	xfree(d);
}

const struct nlop_s* nlop_softmax_create(unsigned int N, const long dims[N], unsigned long batch_flag)
{
	PTR_ALLOC(struct softmax_s, data);
	SET_TYPEID(softmax_s, data);

	data->N = N;
	data->tmp = NULL;

	long batchdims[N];
	md_select_dims(N, batch_flag, batchdims, dims);
	data->dom = iovec_create(N, dims, CFL_SIZE);
	data->batchdom = iovec_create(N, batchdims, CFL_SIZE);
	data->batch_flag = batch_flag;

	return nlop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), softmax_apply, softmax_der, softmax_der, NULL, NULL, softmax_free);
}


struct sigmoid_s {
	INTERFACE(nlop_data_t);
};

DEF_TYPEID(sigmoid_s);

static void sigmoid_apply(const nlop_data_t* _data, int N, const long dims[N], float* dst, const float* src, float* der)
{

	UNUSED(_data);

	float one = 1.;
	float* ones = md_alloc_sameplace(N, dims, FL_SIZE, dst);
	md_fill(N, dims, ones, &one, FL_SIZE);

	md_smul(N, dims, dst, src, -1);
	md_exp(N, dims, dst, dst);
	md_add(N, dims, dst, dst, ones);
	md_div(N, dims, dst, ones, dst);

	if (NULL != der) {

		md_sub(N, dims, der, ones, dst);
		md_mul(N, dims, der, der, dst);
	}

	md_free(ones);
}


static void sigmoid_free(const nlop_data_t* _data)
{
	xfree(_data);
}

const struct nlop_s* nlop_sigmoid_create(unsigned int N, const long dims[N])
{
	PTR_ALLOC(struct sigmoid_s, data);
	SET_TYPEID(sigmoid_s, data);

	return nlop_rdiag_create(N, dims, CAST_UP(PTR_PASS(data)), sigmoid_apply, sigmoid_free);
}





/**
 * Create Cardioid nlop 
 * f(z) = 0.5(1+cos(arg(z)))z = (|z|+z)^2/(4|z|)
 * PHD thesis Patrick Virtue : https://www2.eecs.berkeley.edu/Pubs/TechRpts/2019/EECS-2019-126.pdf
 */
const struct nlop_s* nlop_cardioid_create(unsigned int N, const long dims[N])
{
	auto result = nlop_zabs_create(N, dims);
	result = nlop_chain2_FF(result, 0, nlop_zaxpbz_create(N, dims, 1., 1.), 0);
	result = nlop_dup_F(result, 0, 1);
	
	auto square = nlop_tenmul_create(N, dims, dims, dims);
	square = nlop_dup_F(square, 0, 1);

	result = nlop_chain2_FF(result, 0, square, 0);
	result = nlop_chain2_FF(result, 0, nlop_zdiv_create(N, dims), 0);
	result = nlop_chain2_FF(nlop_zabs_create(N, dims), 0, result, 0);
	result = nlop_chain2_FF(nlop_from_linop_F(linop_scale_create(N, dims, 4)), 0, result, 1);
	result = nlop_dup_F(result, 0, 1);

	return nlop_zrprecomp_jacobian_F(result);
}


/**
 * Create siglog nlop 
 * f(z) = z / (c + |z| / r)
 * PHD thesis Patrick Virtue : https://www2.eecs.berkeley.edu/Pubs/TechRpts/2019/EECS-2019-126.pdf
 */
const struct nlop_s* nlop_siglog_create(unsigned int N, const long dims[N], float c, float r)
{
	auto result = nlop_zdiv_reg_create(N, dims, c);
	result = nlop_chain2_FF(nlop_zabs_create(N, dims), 0, result, 0);
	result = nlop_chain2_FF(nlop_from_linop_F(linop_scale_create(N, dims, 1./ r)), 0, result, 0);

	return nlop_zrprecomp_jacobian_F(result);
}

/**
 * Create iGaussian nlop 
 * f(z) = (1 - exp(-|z|^2/(2s^2)))*z/|z|
 * PHD thesis Patrick Virtue : https://www2.eecs.berkeley.edu/Pubs/TechRpts/2019/EECS-2019-126.pdf
 */
const struct nlop_s* nlop_igaussian_create(unsigned int N, const long dims[N], float sigma)
{
	auto result = nlop_tenmul_create(N, dims, dims, dims);
	result = nlop_chain2_FF(nlop_from_linop_F(linop_zconj_create(N, dims)), 0, result, 0);
	result = nlop_dup_F(result, 0, 1);
	result = nlop_chain_FF(result, nlop_from_linop_F(linop_scale_create(N, dims, 1./(2*sigma*sigma))));
	result = nlop_chain_FF(result, nlop_zexp_create(N, dims));
	result = nlop_chain_FF(result, nlop_from_linop_F(linop_scale_create(N, dims, -1.)));
	result = nlop_chain2_FF(result, 0, nlop_tenmul_create(N, dims, dims, dims), 0);
	result = nlop_chain2_FF(nlop_zphsr_create(N, dims), 0, result, 0);
	result = nlop_dup_F(result, 0, 1);

	return nlop_zrprecomp_jacobian_F(result);
}