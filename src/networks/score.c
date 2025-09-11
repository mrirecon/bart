/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include "iter/italgos.h"
#include "num/multind.h"
#include "num/iovec.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/const.h"
#include "nlops/cast.h"
#include "nlops/zexp.h"
#include "nlops/tenmul.h"
#include "nlops/someops.h"
#include "nlops/mri_ops.h"
#include "nlops/losses.h"

#include "score.h"


//D(x, s) = s^2 * score(x, s) + x
const struct nlop_s* nlop_score_to_expectation(const struct nlop_s* score)
{
	auto dom = nlop_generic_domain(score, 0);

	int N = dom->N;
	long dims[N];
	md_copy_dims(N, dims, dom->dims);

	auto sdom = nlop_generic_domain(score, 1);

	long sdims[N];
	md_singleton_dims(N, sdims);
	sdims[N - 1] = md_calc_size(sdom->N, sdom->dims);

	const struct nlop_s* nlop_scale = nlop_tenmul_create(N, dims, dims, sdims);
	nlop_scale = nlop_prepend_FF(nlop_zspow_create(N, sdims, 2), nlop_scale, 1);
	nlop_scale = nlop_reshape_in_F(nlop_scale, 1, sdom->N, sdom->dims);

	const struct nlop_s* nlop_skip = nlop_zaxpbz_create(N, dims, 1, 1);

	auto ret = nlop_chain2_swap_FF(score, 0, nlop_scale, 0);
	ret = nlop_dup_F(ret, 1, 2);

	ret = nlop_chain2_FF(ret, 0, nlop_skip, 0);
	ret = nlop_dup_F(ret, 0, 1);

	return ret;
}

//score = (D(x, s) - x) / s^2
const struct nlop_s* nlop_expectation_to_score(const struct nlop_s* Dx)
{
	auto dom = nlop_generic_domain(Dx, 0);

	int N = dom->N;
	long dims[N];
	md_copy_dims(N, dims, dom->dims);

	auto sdom = nlop_generic_domain(Dx, 1);

	long sdims[N];
	md_singleton_dims(N, sdims);
	sdims[N - 1] = md_calc_size(sdom->N, sdom->dims);

	const struct nlop_s* nlop_scale = nlop_tenmul_create(N, dims, dims, sdims);
	nlop_scale = nlop_prepend_FF(nlop_zspow_create(N, sdims, -2), nlop_scale, 1);
	nlop_scale = nlop_reshape_in_F(nlop_scale, 1, sdom->N, sdom->dims);

	const struct nlop_s* nlop_skip = nlop_zaxpbz_create(N, dims, -1, 1);

	auto ret = nlop_chain2_FF(Dx, 0, nlop_skip, 1);
	ret = nlop_dup_F(ret, 0, 1);

	ret = nlop_chain2_swap_FF(ret, 0, nlop_scale, 0);
	ret = nlop_dup_F(ret, 1, 2);

	return ret;
}