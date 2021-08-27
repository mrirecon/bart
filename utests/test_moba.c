/* Copyright 2019. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Xiaoqing Wang, Martin Uecker
 */

#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/iovec.h"
#include "num/ops_p.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"

#include "linops/linop.h"
#include "linops/lintest.h"
#include "linops/someops.h"

#include "nlops/nlop.h"

#include "moba/T1fun.h"
#include "moba/optreg.h"

#include "utest.h"






static bool test_nlop_T1fun(void) 
{
	enum { N = 16 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long in_dims[N] = { 16, 16, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long TI_dims[N] = { 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	complex float* dst = md_alloc(N, out_dims, CFL_SIZE);
	complex float* src = md_alloc(N, in_dims, CFL_SIZE);

	complex float TI[4] = { 0., 1., 2., 3. };

	md_zfill(N, in_dims, src, 1.0);

	struct nlop_s* T1 = nlop_T1_create(N, map_dims, out_dims, in_dims, TI_dims, TI, false);

	nlop_apply(T1, N, out_dims, dst, N, in_dims, src);
	
	float err = linop_test_adjoint(nlop_get_derivative(T1, 0, 0));

	nlop_free(T1);

	md_free(src);
	md_free(dst);

	UT_ASSERT(err < 1.E-3);
}

UT_REGISTER_TEST(test_nlop_T1fun);

static bool test_op_p_stack_moba_nonneg(void)
{
	enum { N = 5 };
	long dims[N] = { 2, 4, 7, 5, 6};

	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	long s_dim = 2;

	long p_pos = 3;
	unsigned int s_flag = MD_BIT(p_pos);

	const struct operator_p_s* p = moba_nonneg_prox_create(N, dims, s_dim, s_flag, 0.);

	complex float* in  = md_alloc(N, dims, CFL_SIZE);
	complex float* out = md_alloc(N, dims, CFL_SIZE);

	md_zfill(N, dims, in, -1.);
	md_zfill(N, dims, out, 100.);

	operator_p_apply(p, 0., N, dims, out, N, dims, in);
	operator_p_free(p);

	long dims1[N];
	md_select_dims(N, ~MD_BIT(s_dim), dims1, dims);

	complex float* in1 = md_alloc(N, dims1, CFL_SIZE);

	long pos[N];
	for (int i = 0; i < N; i++)
		pos[i] = 0;

	pos[s_dim] = p_pos;

	md_copy_block(N, pos, dims1, in1, dims, in, CFL_SIZE);
	md_clear(N, dims1, in1, CFL_SIZE);
	md_copy_block(N, pos, dims, in, dims1, in1, CFL_SIZE);

	float err = md_znrmse(N, dims, out, in);

	md_free(in);
	md_free(in1);
	md_free(out);

	UT_ASSERT(err < UT_TOL);
}

UT_REGISTER_TEST(test_op_p_stack_moba_nonneg);
