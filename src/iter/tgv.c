/* Copyright 2022. Institute of Medical Engineering. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 **/

#include "num/ops.h"
#include "num/iovec.h"
#include "num/multind.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/grad.h"

#include "iter/thresh.h"


#include "tgv.h"



/* TGV
 *
 * min x 0.5 \|Ix - y\|_2^2 + min z \alpha \|grad x - z \|_1 + \beta \|Eps z \|_1
 *
 * min x,z 0.5 \| Ix - y \|_2^2 + \alpha \|grad x - z\|_1 + \beta \|Eps z\|_1
 *
 * \alpha = 1, \beta = 2
 *
 * */
struct reg2 tgv_reg(unsigned long flags, unsigned long jflags, float lambda, int N, const long in_dims[N], long isize, long* ext_shift)
{
	assert(1 <= N);

	struct reg2 reg2;

	const struct linop_s* grad1 = linop_grad_create(N, in_dims, N, flags);
	const struct linop_s* grad2x = linop_grad_create(N + 1, linop_codomain(grad1)->dims, N + 1, flags);


	auto grad2a = linop_transpose_create(N + 2, N + 0, N + 1, linop_codomain(grad2x)->dims);
	auto grad2b = linop_identity_create(N + 2, linop_codomain(grad2x)->dims);
	auto grad2 = linop_chain_FF(grad2x, linop_plus_FF(grad2a, grad2b));


	long grd_dims[N + 1];
	md_copy_dims(N + 1, grd_dims, linop_codomain(grad1)->dims);


	auto iov = linop_domain(grad1);
	auto grad1b = linop_extract_create(1, MD_DIMS(0), MD_DIMS(md_calc_size(N, in_dims)), MD_DIMS(isize));
	auto grad1c = linop_reshape_out_F(grad1b, iov->N, iov->dims);
	auto grad1d = linop_chain_FF(grad1c, grad1);

	auto grad1e = linop_extract_create(1, MD_DIMS(*ext_shift), MD_DIMS(md_calc_size(N + 1, grd_dims)), MD_DIMS(isize));
	grad1e = linop_reshape_out_F(grad1e, N + 1, grd_dims);
	reg2.linop[0] = linop_plus_FF(grad1e, grad1d);

	auto grad2e = linop_extract_create(1, MD_DIMS(*ext_shift), MD_DIMS(md_calc_size(N + 1, grd_dims)), MD_DIMS(isize));
	grad2e = linop_reshape_out_F(grad2e, N + 1, grd_dims);
	reg2.linop[1] = linop_chain_FF(grad2e, grad2);

	reg2.prox[0] = prox_thresh_create(N + 1, linop_codomain(reg2.linop[0])->dims, lambda, jflags);
	reg2.prox[1] = prox_thresh_create(N + 2, linop_codomain(reg2.linop[1])->dims, lambda, jflags);

	*ext_shift += md_calc_size(N + 1, grd_dims);


	return reg2;
}



/*
 *	\| \Delta (x - z) \| + \| \Delta z \|
 *
 * */

struct reg2 ictv_reg(unsigned long flags1, unsigned long flags2, unsigned long jflags, float lambda, int N, const long in_dims[N], long isize, long* ext_shift)
{
	struct reg2 reg2;

	assert(0 != flags1);
	assert(0 != flags2);

	auto grad1b = linop_extract_create(1, MD_DIMS(0), MD_DIMS(md_calc_size(N, in_dims)), MD_DIMS(isize));
	grad1b = linop_reshape_out_F(grad1b, N, in_dims);

	auto grad1c = linop_extract_create(1, MD_DIMS(*ext_shift), MD_DIMS(md_calc_size(N, in_dims)), MD_DIMS(isize));
	grad1c = linop_reshape_out_F(grad1c, N, in_dims);

	auto grad1d = linop_plus_FF(grad1b, grad1c);


	const struct linop_s* grad1 = linop_grad_create(N, in_dims, N, flags1);

	// \Delta (x + z)

	reg2.linop[0] = linop_chain_FF(grad1d, grad1);

	const struct linop_s* grad2 = linop_grad_create(N, in_dims, N, flags2);


	auto grad2e = linop_extract_create(1, MD_DIMS(*ext_shift), MD_DIMS(md_calc_size(N, in_dims)), MD_DIMS(isize));
	grad2e = linop_reshape_out_F(grad2e, N, in_dims);

	reg2.linop[1] = linop_chain_FF(grad2e, grad2);

	reg2.prox[0] = prox_thresh_create(N + 1, linop_codomain(reg2.linop[0])->dims, lambda, jflags);
	reg2.prox[1] = prox_thresh_create(N + 1, linop_codomain(reg2.linop[1])->dims, lambda, jflags);

	*ext_shift += md_calc_size(N, in_dims);

	return reg2;
}


