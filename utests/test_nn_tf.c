/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "nlops/nlop.h"

#include "nn/tf_wrapper.h"

#include "utest.h"


static bool test_nn_tf_forward(void)
{
	const struct nlop_s* nlop = nlop_tf_create(1, 2, "./utests/test_nn_tf", false);

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
	const struct nlop_s* nlop = nlop_tf_create(1, 2, "./utests/test_nn_tf", false);

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
	complex float* grad = md_alloc(dom0->N, dom0->dims, dom0->size);
	complex float grad_ys[1] = { 1. + 1.i };

	auto gradop = nlop_get_derivative(nlop, 0, 0);
	linop_adjoint(gradop, dom0->N, dom0->dims, grad, cod->N, cod->dims, grad_ys);

	md_free(grad);
#endif
	md_free(out);
	md_free(in0);
	md_free(in1);

	nlop_free(nlop);

	return true;
}

UT_REGISTER_TEST(test_nn_tf_adjoint);

