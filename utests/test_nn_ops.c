/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/iovec.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "linops/linop.h"
#include "linops/lintest.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/const.h"
#include "nlops/nltest.h"

#include "nn/activation.h"
#include "nn/rbf.h"
#include "nn/batchnorm.h"
#include "nn/layers.h"
#include "nn/nn_ops.h"
#include "nn/losses.h"

#include "utest.h"



static bool test_nlop_relu_der_adj(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	const struct nlop_s* relu = nlop_relu_create(N, dims);

	complex float* dst = md_alloc(N, dims, CFL_SIZE);
	complex float* src = md_alloc(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, src);

	nlop_apply(relu, N, dims, dst, N, dims, src);

	md_free(src);
	md_free(dst);

	float err = linop_test_adjoint_real(nlop_get_derivative(relu, 0, 0));

	nlop_free(relu);

	UT_ASSERT(err < 1.E-2);
}



UT_REGISTER_TEST(test_nlop_relu_der_adj);


static bool test_nlop_softmax_derivative(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	const struct nlop_s* softmax = nlop_softmax_create(N, dims, 4);

	double err = nlop_test_derivative(softmax);

	nlop_free(softmax);

	UT_ASSERT(err < 1.E-1);
}



UT_REGISTER_TEST(test_nlop_softmax_derivative);




static bool test_nlop_softmax_der_adj(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	const struct nlop_s* softmax = nlop_softmax_create(N, dims,4);

	complex float* dst = md_alloc(N, dims, CFL_SIZE);
	complex float* src = md_alloc(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, src);

	nlop_apply(softmax, N, dims, dst, N, dims, src);

	md_free(src);
	md_free(dst);

	float err = linop_test_adjoint_real(nlop_get_derivative(softmax, 0, 0));

	nlop_free(softmax);

	UT_ASSERT(err < 1.E-5);
}



UT_REGISTER_TEST(test_nlop_softmax_der_adj);

static bool test_nlop_sigmoid_derivative(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	const struct nlop_s* sigmoid = nlop_sigmoid_create(N, dims);

	double err = nlop_test_derivative(sigmoid);

	nlop_free(sigmoid);

	UT_ASSERT(err < 1.E-1);
}

UT_REGISTER_TEST(test_nlop_sigmoid_derivative);



static bool test_nlop_sigmoid_der_adj(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	const struct nlop_s* sigmoid = nlop_sigmoid_create(N, dims);

	complex float* dst = md_alloc(N, dims, CFL_SIZE);
	complex float* src = md_alloc(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, src);

	nlop_apply(sigmoid, N, dims, dst, N, dims, src);

	md_free(src);
	md_free(dst);

	float err = linop_test_adjoint_real(nlop_get_derivative(sigmoid, 0, 0));

	nlop_free(sigmoid);

	UT_ASSERT(err < 1.E-5);
}

UT_REGISTER_TEST(test_nlop_sigmoid_der_adj);


static bool test_nlop_stats(void)
{
	enum { N = 2 };
	long idims[N] = { 10, 3 };
	unsigned long flags = MD_BIT(0);
	long odims[N];
	md_select_dims(N, ~flags, odims, idims);

	complex float* src = md_alloc(N, idims, CFL_SIZE);
	complex float* mean = md_alloc(N, odims, CFL_SIZE);
	complex float* var = md_alloc(N, odims, CFL_SIZE);

	md_gaussian_rand(N, idims, src);
	auto nlop = nlop_stats_create(N, idims, MD_BIT(0));
	nlop_generic_apply_unchecked(nlop, 3, MAKE_ARRAY((void*)mean, (void*)var, (void*)src));

	complex float* mean2 = md_alloc(N, odims, CFL_SIZE);
	complex float* var2 = md_alloc(N, odims, CFL_SIZE);

	md_zavg(N, idims, flags, mean2, src);
	md_zvar(N, idims, flags, var2, src);

	float scale = md_calc_size(N, idims) - md_calc_size(N, odims);
	scale = scale / md_calc_size(N, idims);
	md_zsmul(N, odims, var2, var2, scale); // 1/N vs 1/(N-1);

	float err = md_znrmse(N, odims, mean2, mean);
	err += md_znrmse(N, odims, var2, var);

	float err_adj = nlop_test_adj_derivatives(nlop, true);
	float err_der = nlop_test_derivatives(nlop);

	debug_printf(DP_DEBUG1, "Stats: Error: mean: %.8f, var: %.8f, der: %.8f, adj: %.8f\n",md_znrmse(N, odims, mean2, mean), md_znrmse(N, odims, var2, var), err_der, err_adj);

	md_free(mean);
	md_free(var);
	md_free(mean2);
	md_free(var2);
	md_free(src);

	nlop_free(nlop);


	UT_ASSERT((err < 1.e-6) && (err_der < 5.e-3) && (err_adj < 1.e-6));
}



UT_REGISTER_TEST(test_nlop_stats);

static bool test_nlop_normalize(void)
{
	enum { N = 2 };
	long idims[N] = { 10, 3 };
	unsigned long flags = MD_BIT(0);
	long odims[N];
	md_select_dims(N, ~flags, odims, idims);

	complex float* src = md_alloc(N, idims, CFL_SIZE);
	complex float* dst = md_alloc(N, idims, CFL_SIZE);
	complex float* mean = md_alloc(N, odims, CFL_SIZE);
	complex float* var = md_alloc(N, odims, CFL_SIZE);
	complex float* var2 = md_alloc(N, odims, CFL_SIZE);
	md_zfill(N, odims, var2, 1.);

	md_gaussian_rand(N, idims, src);

	auto nlop_stats = nlop_stats_create(N, idims, MD_BIT(0));
	nlop_generic_apply_unchecked(nlop_stats, 3, MAKE_ARRAY((void*)mean, (void*)var, (void*)src));

	auto nlop_normalize = nlop_normalize_create(N, idims, MD_BIT(0), 0.);
	nlop_generic_apply_unchecked(nlop_normalize, 4, MAKE_ARRAY((void*)dst, (void*)src, (void*)mean, (void*)var));

	//test mean / var after normalization
	nlop_generic_apply_unchecked(nlop_stats, 3, MAKE_ARRAY((void*)mean, (void*)var, (void*)dst));
	float err = md_zrms(N, odims, mean);
	err += md_znrmse(N, odims, var, var2);

	auto nlop = nlop_chain2_FF(nlop_stats, 1, nlop_normalize, 2); // the variance input of nlop_normalize must be positive
	float err_der = nlop_test_derivatives(nlop);
	float err_adj = nlop_test_adj_derivatives(nlop, true);

	debug_printf(DP_DEBUG1, "Normalize: mean: %.8f, var: %.8f, der: %.8f, adj: %.8f\n",
			md_zrms(N, odims, mean), md_znrmse(N, odims, var, var2), err_der, err_adj);

	md_free(mean);
	md_free(var);
	md_free(var2);
	md_free(src);
	md_free(dst);

	nlop_free(nlop);

	UT_ASSERT((err < 5.e-7) && (err_der < 5.e-3) && (err_adj < 1.e-6));
}

UT_REGISTER_TEST(test_nlop_normalize);


static bool test_nlop_bn(void)
{
	enum { N = 2 };
	long idims[N] = { 10, 3 };

	auto nlop = nlop_batchnorm_create(N, idims, MD_BIT(0), 0, STAT_TRAIN);
	const long* statdims = nlop_generic_codomain(nlop, 1)->dims;
	complex float* tmp = md_alloc(N + 1, statdims, CFL_SIZE);

	nlop = nlop_set_input_const_F(nlop, 1, N + 1, statdims, true, tmp);
	nlop = nlop_del_out_F(nlop, 1);

	float err_adj = nlop_test_adj_derivatives(nlop, true);
	float err_der = nlop_test_derivatives(nlop);

	debug_printf(DP_DEBUG1, "Batchnorm: Error: der: %.8f, adj: %.8f\n", err_der, err_adj);

	md_free(tmp);

	nlop_free(nlop);


	UT_ASSERT((err_der < 5.e-3) && (err_adj < 1.e-6));
}

UT_REGISTER_TEST(test_nlop_bn);


static bool test_nlop_conv_derivative(void)
{
	enum { N = 6 };
	long dims_image[N] = { 6, 1, 2, 5, 1, 2};
	long dims_kernel[N] = { 3, 4, 2, 2, 1, 2};
	long dims_output[N] = { 4, 4, 2, 4, 1, 1};
	unsigned long conv_flags = 9; //100100

	const struct nlop_s* conv_geom = nlop_convcorr_geom_create(N, conv_flags, dims_output, dims_image, dims_kernel, PAD_VALID, true, NULL, NULL, 'N');

	float err_adj_geom = nlop_test_adj_derivatives(conv_geom, false);
	float err_der_geom = nlop_test_derivatives(conv_geom);

	nlop_free(conv_geom);

	_Bool test = (err_der_geom < 1.E-1) && (err_adj_geom < 1.E-6);
	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_nlop_conv_derivative);



static bool test_padding(void)
{
	enum { N = 2 };
	long dims_in[N] = { 3, 2};
	long dims_out[N] = {7, 4};

	long pad[] = {2, 1};

	complex float in[] = {	1, 2, 3,
				4, 5, 6};

	complex float exp_valid[] = {	1, 2, 3,
					4, 5, 6};
	complex float exp_same[] = {	0, 0, 0, 0, 0, 0, 0,
                                 	0, 0, 1, 2, 3, 0, 0,
                                 	0, 0, 4, 5, 6, 0, 0,
                                 	0, 0, 0, 0, 0, 0, 0};
	complex float exp_reflect[] = {	6, 5, 4, 5, 6, 5, 4,
                                	3, 2, 1, 2, 3, 2, 1,
                                	6, 5, 4, 5, 6, 5, 4,
                                	3, 2, 1, 2, 3, 2, 1};
	complex float exp_sym[] = {	2, 1, 1, 2, 3, 3, 2,
                                  	2, 1, 1, 2, 3, 3, 2,
                                  	5, 4, 4, 5, 6, 6, 5,
                                  	5, 4, 4, 5, 6, 6, 5};
	complex float exp_cyc[] = {	5, 6, 4, 5, 6, 4, 5,
                                  	2, 3, 1, 2, 3, 1, 2,
                                  	5, 6, 4, 5, 6, 4, 5,
                                  	2, 3, 1, 2, 3, 1, 2};


	complex float* out = md_alloc(2, dims_out, CFL_SIZE);

	const struct linop_s* lin_pad;
	float err = 0;

	lin_pad = linop_padding_create(2, dims_in, PAD_SAME, pad, pad);
	linop_forward_unchecked(lin_pad, out, in);
	linop_free(lin_pad);
	err += md_zrmse(2, dims_out, exp_same, out);

	lin_pad = linop_padding_create(2, dims_in, PAD_REFLECT, pad, pad);
	linop_forward_unchecked(lin_pad, out, in);
	linop_free(lin_pad);
	err += md_zrmse(2, dims_out, exp_reflect, out);

	lin_pad = linop_padding_create(2, dims_in, PAD_SYMMETRIC, pad, pad);
	linop_forward_unchecked(lin_pad, out, in);
	linop_free(lin_pad);
	err += md_zrmse(2, dims_out, exp_sym, out);

	lin_pad = linop_padding_create(2, dims_in, PAD_CYCLIC, pad, pad);
	linop_forward_unchecked(lin_pad, out, in);
	linop_free(lin_pad);
	err += md_zrmse(2, dims_out, exp_cyc, out);

	long pad_down[] = {-2, -1};
	lin_pad = linop_padding_create(2, dims_out, PAD_VALID, pad_down, pad_down);
	linop_forward_unchecked(lin_pad, in, out);
	linop_free(lin_pad);
	err += md_zrmse(2, dims_in, in, exp_valid);

	md_free(out);

	UT_ASSERT(1.e-7 > err);
}

UT_REGISTER_TEST(test_padding);


static bool test_padding_adjoint(void)
{
	enum { N = 2 };
	long dims_in[N] = { 3, 2};
	long dims_out[N] = {7, 4};

	long pad[] = {2, 1};

	const struct linop_s* lin_pad;
	float err = 0;

	lin_pad = linop_padding_create(2, dims_in, PAD_SAME, pad, pad);
	err += linop_test_adjoint(lin_pad);
	linop_free(lin_pad);

	lin_pad = linop_padding_create(2, dims_in, PAD_REFLECT, pad, pad);
	linop_free(lin_pad);

	lin_pad = linop_padding_create(2, dims_in, PAD_SYMMETRIC, pad, pad);
	err += linop_test_adjoint(lin_pad);
	linop_free(lin_pad);

	lin_pad = linop_padding_create(2, dims_in, PAD_CYCLIC, pad, pad);
	err += linop_test_adjoint(lin_pad);
	linop_free(lin_pad);

	long pad_down[] = {-2, -1};
	lin_pad = linop_padding_create(2, dims_out, PAD_VALID, pad_down, pad_down);
	err += linop_test_adjoint(lin_pad);
	linop_free(lin_pad);

	debug_printf(DP_DEBUG1, "err: %.8f\n", err);

	UT_ASSERT(1.e-6 > err);
}

UT_REGISTER_TEST(test_padding_adjoint);



static bool test_dense_der(void)
{
	unsigned int N = 2;
	long indims[] = {210, 18};

	const struct linop_s* id = linop_identity_create(N, indims);
	const struct nlop_s* network = nlop_from_linop(id);
	linop_free(id);

	network = append_dense_layer(network, 0, 128);

	float err_adj = nlop_test_adj_derivatives(network, true);
	float err_der = nlop_test_derivatives(network);

	nlop_free(network);

	debug_printf(DP_DEBUG1, "dense errors der, adj: %.8f, %.8f\n", err_der, err_adj);
	_Bool test = (err_adj < 3.E-5) && (err_der < 1.E-1);
	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_dense_der);

static bool test_conv_der(void)
{
	unsigned int N = 5;
	long indims[] = {5, 7, 6, 3, 5};

	const struct linop_s* id = linop_identity_create(N, indims);
	const struct nlop_s* network = nlop_from_linop(id);
	linop_free(id);

	long kernel_size[] = {3, 3, 1};
	long ones[] = {1, 1, 1};

	network = append_convcorr_layer(network, 0, 4, kernel_size, true, PAD_VALID, true, ones, ones);

	float err_adj = nlop_test_adj_derivatives(network, true);
	float err_der = nlop_test_derivatives(network);

	nlop_free(network);

	debug_printf(DP_DEBUG1, "conv errors der, adj: %.8f, %.8f\n", err_der, err_adj);
	_Bool test = (err_adj < 1.E-5) && (err_der < 1.E-1);
	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_conv_der);

static bool test_conv_transp(void)
{
	unsigned int N = 5;
	long indims[] = {5, 7, 6, 3, 5};
	long outdims[] = {4, 5, 4, 3, 5};
	long kernel_size[] = {3, 3, 1};
	long kdims[] = {4, 5, 3, 3 ,1};

	complex float* kernel = md_alloc(N, kdims, CFL_SIZE);
	md_gaussian_rand(N, kdims, kernel);

	auto forward = append_convcorr_layer(nlop_from_linop_F(linop_identity_create(N, indims)), 0, 4, kernel_size, true, PAD_VALID, true, NULL, NULL);
	forward = nlop_set_input_const_F(forward, 1, N, kdims, true, kernel);
	auto adjoint = append_transposed_convcorr_layer(nlop_from_linop_F(linop_identity_create(N, outdims)), 0, 5, kernel_size, true, true, PAD_VALID, true, NULL, NULL);
	adjoint = nlop_set_input_const_F(adjoint, 1, N, kdims, true, kernel);

	PTR_ALLOC(struct linop_s, c);
	c->forward = forward->op;
	c->adjoint = adjoint->op;
	c->normal = NULL;
	c->norm_inv = NULL;

	float err = linop_test_adjoint(c);
	XFREE(c);

	md_free(kernel);

	nlop_free(forward);
	nlop_free(adjoint);
	UT_ASSERT(err < 1.e-5);
}

UT_REGISTER_TEST(test_conv_transp);



static bool test_mpool_der(void)
{
	unsigned int N = 5;
	long indims[] = {2, 6, 1, 1, 2}; //channel, x, y, z, batch
	long outdims[] = {2, 2, 1, 1, 2}; //channel, x, y, z, batch

	//digits reference, e.g. 1204.: batch(1), channel(2), count(04)
	complex float in[] = {	1101., 1202., 1103., 1204., 1105., 1206., 1107., 1208., 1109., 1210., 1111., 1212.,
				2103., 2204., 2101., 2202., 2107., 2208., 2105., 2206., 2109., 2210., 2111., 2212. };

	complex float adj_exp[] = {	0., 0., 0., 0., 1105., 1206., 0., 0., 0., 0., 1111., 1212.,
					0., 0., 0., 0., 2107., 2208., 0., 0., 0., 0., 2111., 2212. };

	complex float out_exp[] = {	1105., 1206., 1111., 1212.,
					2107., 2208., 2111., 2212.};
	complex float* out = md_alloc(N, indims, CFL_SIZE);

	const struct nlop_s* network = nlop_from_linop_F(linop_identity_create(N, indims));
	network = append_maxpool_layer(network, 0, MAKE_ARRAY(3l, 1l, 1l), PAD_VALID, true);
	nlop_apply(network, 5, outdims, out, N, indims, in);
	nlop_adjoint(network, N, indims, in, N, outdims, out);

	nlop_free(network);

	bool err =  md_zrmse(N, outdims, out, out_exp) + md_zrmse(N, indims, in, adj_exp);
	md_free(out);

	UT_ASSERT(1.e-8 > err);
}

UT_REGISTER_TEST(test_mpool_der);



static bool test_bias_der(void)
{
	unsigned int N = 4;
	long dims[] = { 4, 1, 3, 4};
	long bdims[] = { 1, 1, 3, 4};

	const struct nlop_s* network = nlop_bias_create(N, dims, bdims);

	float err_adj = nlop_test_adj_derivatives(network, true);
	float err_der = nlop_test_derivatives(network);

	nlop_free(network);

	debug_printf(DP_DEBUG1, "bias errors der, adj: %.8f, %.8f\n", err_der, err_adj);
	_Bool test = (err_adj < 1.E-5) && (err_der < 1.E-1);
	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_bias_der);

static bool test_relu_der(void)
{
	unsigned int N = 4;
	long dims[] = { 30, 78, 3, 25};

	const struct linop_s* id = linop_identity_create(N, dims);
	const struct nlop_s* network = nlop_from_linop(id);
	linop_free(id);

	network = append_activation(network, 0, ACT_RELU);

	float err_adj = nlop_test_adj_derivatives(network, true);
	float err_der = nlop_test_derivatives(network);

	nlop_free(network);

	debug_printf(DP_DEBUG1, "relu errors der, adj: %.8f, %.8f\n", err_der, err_adj);
	_Bool test = (err_adj < 1.E-6) && (err_der < 1.E1);
	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_relu_der);

static bool test_nlop_rbf(void)
{
 	enum { N = 3 };
 	long dims[N] = { 4, 3, 5};

	auto op = nlop_activation_rbf_create(dims, 1., -1., false);

	float err_adj = nlop_test_adj_derivatives(op, true);
	float err_der = nlop_test_derivatives(op);

	nlop_free(op);

	debug_printf(DP_DEBUG1, "rbf errors der: %.8f, adj %.8f\n", err_der, err_adj);
	UT_ASSERT((err_der < 1.E-2) && (err_adj < 1.E-6));
}

UT_REGISTER_TEST(test_nlop_rbf);

static bool test_nlop_rbf2(void)
{
 	enum { N = 3 };
 	long dims[N] = { 4, 3, 5};

	auto op = nlop_activation_rbf_create(dims, 1., -1., true);

	float err_adj = nlop_test_adj_derivatives(op, true);
	float err_der = nlop_test_derivatives(op);

	nlop_free(op);

	debug_printf(DP_DEBUG1, "rbf errors der: %.8f, adj %.8f\n", err_der, err_adj);
	UT_ASSERT((err_der < 1.E-2) && (err_adj < 1.E-6));
}

UT_REGISTER_TEST(test_nlop_rbf2);


/**
 * Test append_convcorr_layer for implementation
 * of strides and dilations.
 * Test dilations with PAD_SAME and strides with PAD_VALID.
 **/
static bool test_nlop_conv_strs_dil(void)
{
	unsigned int N = 5;
	long idims[] = {3, 7, 5, 1, 1};
	long odims[] = {3, 7, 5, 1, 1};

	long dilations[] = {2, 2, 1};
	long strides[] = {2, 2, 1};
	long kernel_size[] = {3, 3, 1};
	long kernel_size_no_dil[] = {5, 5, 1};

	long kdims[] = {odims[0], idims[0], kernel_size[0], kernel_size[1], kernel_size[2]};
	long kdims_no_dil[] = {odims[0], idims[0], kernel_size_no_dil[0], kernel_size_no_dil[1], kernel_size_no_dil[2]};

	// calculation of outdims for PAD_VALID convolution with strides
	long odims_pad_valid[N];
	long odims_strided[N];
	md_copy_dims(N, odims_pad_valid, odims);
	md_copy_dims(N, odims_strided, odims);
	for (int i = 0; i < 3; i++){

		odims_pad_valid[i+1] = odims[i+1] - (kernel_size[i] - 1);
		odims_strided[i+1] = (odims_pad_valid[i+1] - 1) / strides[i] + 1;
	}

	complex float* kernel = md_alloc(N, kdims, CFL_SIZE); // kernel for conv with dilations
	complex float* kernel_no_dil = md_alloc(N, kdims_no_dil, CFL_SIZE); // kernel for conv without dilations

	// test dilations
	// calculate strides strs_dil to manually create kernel equivalent to kernel with dilations
	long strs_kdims[N];
	md_calc_strides(N, strs_kdims, kdims, CFL_SIZE);

	long strs_dil[N];
	md_copy_strides(N, strs_dil, strs_kdims);
	long prod_dil = 1;
	long prod_no_dil = 1;
	for (int i = 0; i < 3; i++){

		strs_dil[2 + i] /= prod_dil;
		strs_dil[2 + i] *= dilations[i] * prod_no_dil;
		prod_no_dil *= kernel_size_no_dil[i];
		prod_dil *= kernel_size[i];
	}

	md_gaussian_rand(N, kdims, kernel);
	md_zfill(N, kdims_no_dil, kernel_no_dil, 0.);
	md_copy2(N, kdims, strs_dil, kernel_no_dil, strs_kdims, kernel, CFL_SIZE); // equivalent to dilations option of convcorr function

	auto forward_dil = append_convcorr_layer(nlop_from_linop_F(linop_identity_create(N, idims)), 0, odims[0], kernel_size, true, PAD_SAME, true, NULL, dilations);
	forward_dil = nlop_set_input_const_F(forward_dil, 1, N, kdims, true, kernel);

	auto forward_no_dil = append_convcorr_layer(nlop_from_linop_F(linop_identity_create(N, idims)), 0, odims[0], kernel_size_no_dil, true, PAD_SAME, true, NULL, NULL);
	forward_no_dil = nlop_set_input_const_F(forward_no_dil, 1, N, kdims_no_dil, true, kernel_no_dil);

	complex float* input = md_alloc(N, idims, CFL_SIZE);
	complex float* output = md_alloc(N, odims, CFL_SIZE);
	complex float* output_no_dil = md_alloc(N, odims, CFL_SIZE);

	md_gaussian_rand(N, idims, input);

	nlop_apply(forward_dil, N, odims, output, N, idims, input);
	nlop_apply(forward_no_dil, N, odims, output_no_dil, N, idims, input);

	float err = md_znrmse(N, odims, output, output_no_dil);

	// test strides
	auto forward_strs = append_convcorr_layer(nlop_from_linop_F(linop_identity_create(N, idims)), 0, odims[0], kernel_size, true, PAD_VALID, true, strides, NULL);
	forward_strs = nlop_set_input_const_F(forward_strs, 1, N, kdims, true, kernel);

	auto forward_pad = append_convcorr_layer(nlop_from_linop_F(linop_identity_create(N, idims)), 0, odims[0], kernel_size, true, PAD_VALID, true, NULL, NULL);
	forward_pad = nlop_set_input_const_F(forward_pad, 1, N, kdims, true, kernel);

	complex float* output_strs = md_alloc(N, odims_strided, CFL_SIZE);
	complex float* output_pad = md_alloc(N, odims_pad_valid, CFL_SIZE);

	nlop_apply(forward_strs, N, odims_strided, output_strs, N, idims, input);
	nlop_apply(forward_pad, N, odims_pad_valid, output_pad, N, idims, input);

	// calculate strides strs_strs to manually create equivalent of strided convolution
	long strs_pad[N];
	md_calc_strides(N, strs_pad, odims_strided, CFL_SIZE);

	long strs_strs[N];
	md_copy_strides(N, strs_strs, strs_pad);
	long prod_strs = 1;
	long prod_no_strs = 1;
	strs_strs[4] = odims[0] * 8;
	for (int i = 0; i < 3; i++){

		strs_strs[1 + i] /= prod_strs;
		strs_strs[1 + i] *= strides[i] * prod_no_strs;
		prod_no_strs *= odims_pad_valid[i+1];
		prod_strs *= odims_strided[i+1];
		strs_strs[4] *= odims_pad_valid[i+1];
	}

	// manually create equivalent output to strides option of convcorr
	complex float* pad_strided = md_alloc(N, odims_strided, CFL_SIZE);
	md_copy2(N, odims_strided, strs_pad, pad_strided, strs_strs, output_pad, CFL_SIZE);

	err += md_znrmse(N, odims_strided, output_strs, pad_strided);

	md_free(kernel);
	md_free(kernel_no_dil);

	md_free(input);
	md_free(output);
	md_free(output_no_dil);
	md_free(output_strs);
	md_free(output_pad);
	md_free(pad_strided);

	nlop_free(forward_dil);
	nlop_free(forward_no_dil);
	nlop_free(forward_strs);
	nlop_free(forward_pad);

	UT_ASSERT(err < 1.e-2);
}
UT_REGISTER_TEST(test_nlop_conv_strs_dil);


static bool test_dice(void)
{
	long dims[] = {4, 12};

	auto nlop = nlop_dice_create(ARRAY_SIZE(dims), dims, MD_BIT(0), 0, -1., false);
	nlop = nlop_chain2_FF(nlop_softmax_create(ARRAY_SIZE(dims), dims, 1) , 0, nlop, 0);
	nlop = nlop_chain2_FF(nlop_softmax_create(ARRAY_SIZE(dims), dims, 1) , 0, nlop, 0);

	bool der = nlop_test_derivatives_reduce(nlop, 10, 5, 0.01);
	float adj_err = nlop_test_adj_derivatives(nlop, true);

	nlop_free(nlop);

	debug_printf(DP_DEBUG1, "%d %f\n", der, adj_err);
	UT_ASSERT(der && (UT_TOL > adj_err));
}

UT_REGISTER_TEST(test_dice);

static bool test_dice2(void)
{
	long dims[] = {4, 12, 5};

	auto nlop = nlop_dice_create(ARRAY_SIZE(dims), dims, MD_BIT(0), MD_BIT(2), -2., false);
	nlop = nlop_chain2_FF(nlop_softmax_create(ARRAY_SIZE(dims), dims, 1) , 0, nlop, 0);
	nlop = nlop_chain2_FF(nlop_softmax_create(ARRAY_SIZE(dims), dims, 1) , 0, nlop, 0);

	bool der = nlop_test_derivatives_reduce(nlop, 10, 4, 0.01);
	float adj_err = nlop_test_adj_derivatives(nlop, true);

	nlop_free(nlop);

	debug_printf(DP_DEBUG1, "%d %f\n", der, adj_err);
	UT_ASSERT(der && (UT_TOL > adj_err));
}

UT_REGISTER_TEST(test_dice2);

static bool test_dice3(void)
{
	long dims[] = {4, 12, 5};

	auto nlop = nlop_dice_create(ARRAY_SIZE(dims), dims, MD_BIT(0), 0, 0, true);
	nlop = nlop_chain2_FF(nlop_softmax_create(ARRAY_SIZE(dims), dims, 1) , 0, nlop, 0);
	nlop = nlop_chain2_FF(nlop_softmax_create(ARRAY_SIZE(dims), dims, 1) , 0, nlop, 0);

	bool der = nlop_test_derivatives_reduce(nlop, 10, 5, 0.01);
	float adj_err = nlop_test_adj_derivatives(nlop, true);

	nlop_free(nlop);

	debug_printf(DP_DEBUG1, "%d %f\n", der, adj_err);
	UT_ASSERT(der && (2 * UT_TOL > adj_err));
}

UT_REGISTER_TEST(test_dice3);


static bool test_nlop_cardioid(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	const struct nlop_s* cardioid = nlop_cardioid_create(N, dims);
	cardioid = nlop_chain_FF(nlop_from_linop_F(linop_zreal_create(N, dims)), cardioid);

	const struct nlop_s* relu = nlop_relu_create(N, dims);
	relu = nlop_chain_FF(nlop_from_linop_F(linop_zreal_create(N, dims)), relu);

	bool test = compare_nlops(relu, cardioid, true, true, true, UT_TOL);

	nlop_free(relu);
	nlop_free(cardioid);

	UT_ASSERT(test);
}


UT_REGISTER_TEST(test_nlop_cardioid);