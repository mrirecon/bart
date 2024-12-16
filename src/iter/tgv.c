/* Copyright 2022. Institute of Medical Engineering. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Publications:
 * 
 * Rudin LI, Osher S, Fatemi E.
 * Nonlinear total variation based noise removal algorithms,
 * Physica D: Nonlinear Phenomena 1992; 60:259-268.
 * 
 * Bredies K, Kunisch K, Pock T.
 * Total generalized variation.
 * SIAM Journal on Imaging Sciences 2010; 3:492-526.
 * 
 * Knoll F, Bredies K, Pock T, Stollberger R.
 * Second order total generalized variation (TGV) for MRI.
 * Magn Reson Med 2010; 65:480-491.
 **/

#include "num/ops.h"
#include "num/iovec.h"
#include "num/multind.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/grad.h"

#include "iter/thresh.h"
#include "misc/debug.h"

#include "tgv.h"


/**
 * This function creates a Total Variation (TV) regularization operator with the specified parameters.
 * It minimizes the following objective:
 * 
 * \f[
 * \min_{x} \alpha_1 \| \Delta(u) \|_1
 * \f]
 *
 * @param flags       Bitmask specifying the dimensions for the regularization.
 * @param jflags      Bitmask for joint thresholding operation.
 * @param lambda      Regularization parameter.
 * @param N           Number of dimensions.
 * @param img_dims    Array of size N specifying the input dimensions.
 * @param tvscales_N  Number of TV scales.
 * @param tvscales    Array of size tvscales_N specifying the scaling of the derivatives.
 * @return            A structure containing the TV regularization operator, which contains
 * 					  a linear operator for the gradient and a proximal operator for the thresholding.
 */
struct reg tv_reg(unsigned long flags, unsigned long jflags, float lambda, int N, const long img_dims[N], int tvscales_N, const float tvscales[tvscales_N])
{
	struct reg reg;

	while ((0 < tvscales_N) && (0. == tvscales[tvscales_N - 1]))
		tvscales_N--;

	reg.linop = linop_grad_create(N, img_dims, N, flags);

	if (0 < tvscales_N) {

		debug_printf(DP_INFO, "TV anisotropic scaling: %d\n", tvscales_N);

		assert(tvscales_N == linop_codomain(reg.linop)->dims[N]);

		complex float ztvscales[tvscales_N];
		for (int i = 0; i < tvscales_N; i++)
			ztvscales[i] = tvscales[i];

		reg.linop = linop_chain_FF(reg.linop,
			linop_cdiag_create(N + 1, linop_codomain(reg.linop)->dims, MD_BIT(N), ztvscales));
	}

	reg.prox = prox_thresh_create(N + 1,
			linop_codomain(reg.linop)->dims,
			lambda, jflags | MD_BIT(N));
	
	return reg;
}


/**
 * This function creates a TGV regularization operator that minimizes the following objective:
 * 
 * \f[
 * \min_{x,z} \alpha_1 \| \Delta(x) + z \|_1 + \alpha_0 \| \text{Eps}(z) \|_1
 * \f]
 * 
 * where \f$ \text{Eps}(z) = \Delta(z + z^T) \f$.
 *
 * @param flags       Bitmask specifying the dimensions for the regularization.
 * @param jflags      Bitmask for joint thresholding operation.
 * @param lambda      Regularization parameter.
 * @param N           Number of dimensions.
 * @param in_dims     Array of size N specifying the input dimensions.
 * @param isize       Size of the image including supporting variables.
 * @param ext_shift   Pointer to an integer specifying the external shift.
 * @param alpha       Array of size 2 specifying the regularization parameters \f$ \alpha_1 \f$ and \f$ \alpha_0 \f$.
 * @param tvscales_N  Number of TV scales.
 * @param tvscales    Array of size tvscales_N specifying the scaling of the derivatives.
 * @return            A structure containing the TGV regularization operator, which contains
 * 		      two linear operators for the gradient and the symmetric gradient
 * 		      and two proximal operators for the thresholding.
 */
struct reg2 tgv_reg(unsigned long flags, unsigned long jflags, float lambda, int N, const long in_dims[N], long isize, long* ext_shift, const float alpha[2], int tvscales_N, const float tvscales[tvscales_N])
{
	assert(1 <= N);

	struct reg2 reg2;

	while ((0 < tvscales_N) && (0. == tvscales[tvscales_N - 1]))
		tvscales_N--;

	const struct linop_s* grad1 = linop_grad_create(N, in_dims, N, flags);

	long grd_dims[N + 2];
	md_copy_dims(N + 1, grd_dims, linop_codomain(grad1)->dims);
	grd_dims[N + 1] = 1;
	
	
	const struct linop_s* grad2x1 = linop_grad_create(N + 2, grd_dims, N + 1, flags);
	const struct linop_s* grad2x2 = linop_transpose_create(N + 2, N + 0, N + 1, grd_dims);
	grad2x2 = linop_chain_FF(grad2x2, linop_grad_create(N + 2, linop_codomain(grad2x2)->dims, N + 0, flags));
	auto grad2 = linop_plus_FF(grad2x1, grad2x2);

	if (0 < tvscales_N) {

		debug_printf(DP_INFO, "TGV anisotropic scaling: %d\n", tvscales_N);

		assert(tvscales_N == linop_codomain(grad1)->dims[N]);

		complex float ztvscales[tvscales_N];
		for (int i = 0; i < tvscales_N; i++)
			ztvscales[i] = tvscales[i];

		grad1 = linop_chain_FF(grad1,
			linop_cdiag_create(N + 1, linop_codomain(grad1)->dims, MD_BIT(N), ztvscales));

		grad2 = linop_chain_FF(grad2,
			linop_cdiag_create(N + 2, linop_codomain(grad2)->dims, MD_BIT(N + 1), ztvscales));
	}

	auto iov = linop_domain(grad1);
	auto grad1b = linop_extract_create(1, MD_DIMS(0), MD_DIMS(md_calc_size(N, in_dims)), MD_DIMS(isize));
	auto grad1c = linop_reshape_out_F(grad1b, iov->N, iov->dims);
	auto grad1d = linop_chain_FF(grad1c, grad1);

	auto grad1e = linop_extract_create(1, MD_DIMS(*ext_shift), MD_DIMS(md_calc_size(N + 1, grd_dims)), MD_DIMS(isize));
	grad1e = linop_reshape_out_F(grad1e, N + 1, grd_dims);
	reg2.linop[0] = linop_plus_FF(grad1e, grad1d);

	auto grad2e = linop_extract_create(1, MD_DIMS(*ext_shift), MD_DIMS(md_calc_size(N + 1, grd_dims)), MD_DIMS(isize));
	grad2e = linop_reshape_out_F(grad2e, N + 2, grd_dims);
	reg2.linop[1] = linop_chain_FF(grad2e, grad2);

	reg2.prox[0] = prox_thresh_create(N + 1, linop_codomain(reg2.linop[0])->dims, lambda*alpha[0], jflags);
	reg2.prox[1] = prox_thresh_create(N + 2, linop_codomain(reg2.linop[1])->dims, lambda*alpha[1], jflags);

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


