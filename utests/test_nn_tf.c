/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/zexp.h"
#include "nlops/nltest.h"

#include "nn/tf_wrapper.h"

#include "utest.h"


static bool test_nn_tf_forward(void)
{
	const struct nlop_s* nlop = nlop_tf_create("./utests/test_nn_tf");

	nlop_debug(DP_DEBUG1, nlop);

	const struct iovec_s* dom0 = nlop_generic_domain(nlop, 0);
	const struct iovec_s* dom1 = nlop_generic_domain(nlop, 1);

	complex float* in0 = md_alloc(dom0->N, dom0->dims, dom0->size);
	complex float* in1 = md_alloc(dom1->N, dom1->dims, dom1->size);

	md_zfill(dom0->N, dom0->dims, in0, 1.);
	md_zfill(dom1->N, dom1->dims, in1, 0. + 1.i);

	auto cod = nlop_generic_codomain(nlop, 0);

	assert(1 == md_calc_size(cod->N, cod->dims));

	complex float* out = md_alloc(cod->N, cod->dims, cod->size);

	nlop_generic_apply_unchecked(nlop, 3, (void*[3]){ out, in0, in1 });

	debug_printf(DP_DEBUG1, "Loss: %f + %f i\n", crealf(out[0]), cimagf(out[0]));

	float err = (cabsf(out[0] - 0.5));

	md_free(out);
	md_free(in0);
	md_free(in1);

	nlop_free(nlop);

	return (err < UT_TOL);
}

UT_REGISTER_TEST(test_nn_tf_forward);



static bool test_nn_tf_adjoint(void)
{
	const struct nlop_s* nlop = nlop_tf_create("./utests/test_nn_tf");

	nlop_debug(DP_DEBUG1, nlop);

	const struct iovec_s* dom0 = nlop_generic_domain(nlop, 0);
	const struct iovec_s* dom1 = nlop_generic_domain(nlop, 1);

	complex float* in0 = md_alloc(dom0->N, dom0->dims, dom0->size);
	complex float* in1 = md_alloc(dom1->N, dom1->dims, dom1->size);

	md_zfill(dom0->N, dom0->dims, in0,  1.0);
	md_zfill(dom1->N, dom1->dims, in1, +1.0i);

	auto cod = nlop_generic_codomain(nlop, 0);

	assert(1 == md_calc_size(cod->N, cod->dims));

	complex float* out = md_alloc(cod->N, cod->dims, cod->size);

	nlop_generic_apply_unchecked(nlop, 3, (void*[3]){ out, in0, in1 });

	debug_printf(DP_DEBUG1, "Loss: %f + %f i\n", crealf(out[0]), cimagf(out[0]));

#if 1
	complex float* grad1 = md_alloc(dom0->N, dom0->dims, dom0->size);
	complex float* grad2 = md_alloc(dom0->N, dom0->dims, dom0->size);

	complex float grad_ys[1] = { 1. + 0.i };

	auto gradop_1 = nlop_get_derivative(nlop, 0, 0);
	linop_adjoint(gradop_1, dom0->N, dom0->dims, grad1, cod->N, cod->dims, grad_ys);

	grad_ys[0] = 2;

	auto gradop_2 = nlop_get_derivative(nlop, 0, 1);
	linop_adjoint(gradop_2, dom0->N, dom0->dims, grad2, cod->N, cod->dims, grad_ys);

	// y = (x_1 - x_2)^2 -> dx/dx1 = - dy/dx2
	// factor of 0.5 for grad_ys[0] = 2 (test linearity)

	md_zaxpy(dom0->N, dom0->dims, grad1, 0.5, grad2);

	if (UT_TOL < md_zrms(dom0->N, dom0->dims, grad1))
		UT_ASSERT(false);

	md_free(grad1);
	md_free(grad2);
#endif
	md_free(out);
	md_free(in0);
	md_free(in1);

	nlop_free(nlop);

	return true;
}

UT_REGISTER_TEST(test_nn_tf_adjoint);

/*
generate graphs for the following tests:

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import bart_tf


g = tf.Graph()
with g.as_default():
    i = tf.placeholder(tf.complex64, shape = [4, 2], name="input_0")
    o = tf.exp(i, name="output_0")

bart_tf.tf1_export_graph(g, "utests", "exp_grad")


g = tf.Graph()
with g.as_default():
    i = tf.placeholder(tf.complex64, shape = [4, 2], name="input_0")
    o = tf.exp(i, name="output_0")
    j = tf.identity(o, name="jacobian_0_0")

bart_tf.tf1_export_graph(g, "utests", "exp_grad_jac")


g = tf.Graph()
with g.as_default():
    i = tf.placeholder(tf.complex64, shape = [4, 2], name="input_0")
    o = tf.conj(tf.exp(i), name="output_0")

    real = tf.reshape(tf.gradients(tf.math.real(o), i), [4, 2])
    imag = tf.reshape(tf.gradients(tf.math.imag(o), i), [4, 2])

    jac = tf.stack([real, imag], axis=2, name="jacobian_real_0_0")

bart_tf.tf1_export_graph(g, "utests", "exp_grad_jac_real")
*/

static bool test_nn_tf_exp_v1(void)
{
	const struct nlop_s* nlop_tf = nlop_tf_create("./utests/exp_grad");
	auto iov = nlop_domain(nlop_tf);

	const struct nlop_s* nlop_bart = nlop_zexp_create(iov->N, iov->dims);

	bool ok = compare_nlops(nlop_bart, nlop_tf, true, false, true, UT_TOL);

	nlop_free(nlop_tf);
	nlop_free(nlop_bart);

	return (ok);
}

UT_REGISTER_TEST(test_nn_tf_exp_v1);


static bool test_nn_tf_exp_jacobian_v1(void)
{
	const struct nlop_s* nlop_tf = nlop_tf_create("./utests/exp_grad_jac");
	auto iov = nlop_domain(nlop_tf);

	const struct nlop_s* nlop_bart = nlop_zexp_create(iov->N, iov->dims);

	bool ok = compare_nlops(nlop_bart, nlop_tf, true, true, true, UT_TOL);

	nlop_free(nlop_tf);
	nlop_free(nlop_bart);

	return (ok);
}

UT_REGISTER_TEST(test_nn_tf_exp_jacobian_v1);

static bool test_nn_tf_exp_jacobian_real_v1(void)
{
	const struct nlop_s* nlop_tf = nlop_tf_create("./utests/exp_grad_jac_real");
	auto iov = nlop_domain(nlop_tf);

	const struct nlop_s* nlop_bart = nlop_chain_FF(nlop_zexp_create(iov->N, iov->dims), nlop_from_linop_F(linop_zconj_create(iov->N, iov->dims)));

	bool ok = compare_nlops(nlop_bart, nlop_tf, true, true, true, UT_TOL);

	nlop_free(nlop_tf);
	nlop_free(nlop_bart);

	return (ok);
}

UT_REGISTER_TEST(test_nn_tf_exp_jacobian_real_v1);