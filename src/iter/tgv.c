/* Copyright 2022-2025. Institute of Medical Engineering. TU Graz.
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
 *
 * Holler M, Kunisch K.
 * On Infimal Convolution of TV-Type Functionals and Applications
 * to Video and Image Reconstruction.
 * SIAM J. Imaging Sci. 2014; 7, 2258-2300.
 *
 * Schloegl M, Holler M, Schwarzl A, Bredies K, Stollberger R.
 * Infimal convolution of total generalized variation functionals for dynamic MRI.
 * Magn Reson Med 2017;78(1):142-155.
 **/

#include "num/ops.h"
#include "num/iovec.h"
#include "num/multind.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/grad.h"

#include "iter/thresh.h"
#include "misc/debug.h"
#include "misc/mri.h"

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
 * @param lop_trafo   The linear operator that transforms the input data.
 * @return            A structure containing the TV regularization operator, which contains
 * 					  a linear operator for the gradient and a proximal operator for the thresholding.
 */
struct reg tv_reg(unsigned long flags, unsigned long jflags, float lambda, int N, const long img_dims[N], int tvscales_N, const float tvscales[tvscales_N], const struct linop_s* lop_trafo)
{
	struct reg reg;

	while ((0 < tvscales_N) && (0. == tvscales[tvscales_N - 1]))
		tvscales_N--;

	long in2_dims[N];

	if (NULL != lop_trafo) {

		assert(N == linop_domain(lop_trafo)->N);
		assert(md_check_equal_dims(N, img_dims, linop_domain(lop_trafo)->dims, ~0UL));
		assert(N == linop_codomain(lop_trafo)->N);

		md_copy_dims(N, in2_dims, linop_codomain(lop_trafo)->dims);

	} else {

		md_copy_dims(N, in2_dims, img_dims);
	}

	reg.linop = linop_grad_create(N, in2_dims, N, flags);

	if (NULL != lop_trafo)
		reg.linop = linop_chain_FF(lop_trafo, reg.linop);

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

static struct reg2 tgv_reg_int(unsigned long flags, unsigned long jflags, float lambda, int N, const long in_dims[N], long isize, long img_shift, long* ext_shift, const float alpha[2],
		    int tvscales_N, const float tvscales[tvscales_N], const struct linop_s* lop_trafo)
{
	assert(1 <= N);

	struct reg2 reg2;

	while ((0 < tvscales_N) && (0. == tvscales[tvscales_N - 1]))
		tvscales_N--;

	long in2_dims[N];

	if (NULL != lop_trafo) {

		assert(N == linop_domain(lop_trafo)->N);
		assert(md_check_equal_dims(N, in_dims, linop_domain(lop_trafo)->dims, ~0UL));
		assert(N == linop_codomain(lop_trafo)->N);

		md_copy_dims(N, in2_dims, linop_codomain(lop_trafo)->dims);

	} else {

		md_copy_dims(N, in2_dims, in_dims);
	}

	const struct linop_s* grad1 = linop_grad_create(N, in2_dims, N, flags);

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

	grad2 = linop_chain_FF(grad2, linop_scale_create(N + 2, linop_codomain(grad2)->dims, 0.5f));

	auto iov = linop_domain(grad1);
	auto grad1b = linop_extract_create(1, MD_DIMS(img_shift), MD_DIMS(md_calc_size(N, in_dims)), MD_DIMS(isize));

	if (NULL != lop_trafo) {

		auto iov = linop_codomain(grad1b);
		auto trafo_flat = linop_reshape_in(lop_trafo, iov->N, iov->dims);

		grad1b = linop_chain_FF(grad1b, trafo_flat);
	}

	auto grad1c = linop_reshape_out_F(grad1b, iov->N, iov->dims);
	auto grad1d = linop_chain_FF(grad1c, grad1);

	auto grad1e = linop_extract_create(1, MD_DIMS(*ext_shift), MD_DIMS(md_calc_size(N + 1, grd_dims)), MD_DIMS(isize));
	grad1e = linop_reshape_out_F(grad1e, N + 1, grd_dims);
	reg2.linop[0] = linop_plus_FF(grad1e, grad1d);

	auto grad2e = linop_extract_create(1, MD_DIMS(*ext_shift), MD_DIMS(md_calc_size(N + 1, grd_dims)), MD_DIMS(isize));
	grad2e = linop_reshape_out_F(grad2e, N + 2, grd_dims);
	reg2.linop[1] = linop_chain_FF(grad2e, grad2);

	reg2.prox[0] = prox_thresh_create(N + 1, linop_codomain(reg2.linop[0])->dims, lambda * alpha[0], jflags);
	reg2.prox[1] = prox_thresh_create(N + 2, linop_codomain(reg2.linop[1])->dims, lambda * alpha[1], jflags);

	*ext_shift += md_calc_size(N + 1, grd_dims);


	return reg2;
}

/**
 * This function creates a TGV regularization operator that minimizes the following objective:
 *
 * \f[
 * \min_{x,z} \alpha_1 \| \Delta(x) + z \|_1 + \alpha_0 \| \text{Eps}(z) \|_1
 * \f]
 *
 * where \f$ \text{Eps}(z) = 0.5 \cdot \Delta(z + z^T) \f$.
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
 * @param lop_trafo   The linear operator that transforms the input data.
 * @return            A structure containing the TGV regularization operator, which contains
 * 		      two linear operators for the gradient and the symmetric gradient
 * 		      and two proximal operators for the thresholding.
 */
struct reg2 tgv_reg(unsigned long flags, unsigned long jflags, float lambda, int N, const long in_dims[N], long isize, long* ext_shift, const float alpha[2],
	int tvscales_N, const float tvscales[tvscales_N], const struct linop_s* lop_trafo)
{
	return tgv_reg_int(flags, jflags, lambda, N, in_dims, isize, 0, ext_shift, alpha,
		tvscales_N, tvscales, lop_trafo);
}



/**
 * This function creates an ICTV (infimal convolution of total variation) operator with the specified parameters.
 * The ICTV regularization is applied to the image dimensions specified by `out_dims`.
 *
 * The regularization minimizes the following expression:
 * \f[
 * \min_{x,z} \gamma_1 \| \Delta (x + z) \| + \gamma_2 \| \Delta(z) \|
 * \f]
 *
 * @param flags        Bitmask specifying the dimensions for the regularization.
 * @param jflags       Bitmask for joint thresholding operation.
 * @param lambda       Regularization parameter.
 * @param N            Number of dimensions.
 * @param in_dims      Array of size N specifying the input dimensions.
 * @param isize        Size of the image including supporting variables.
 * @param ext_shift    Pointer to an array specifying the external shift.
 * @param gamma        Array of size 2 specifying gamma_1 and gamma_2, the weighting parameters between the TV terms.
 * @param tvscales_N   Number of TV scales for the first gradient.
 * @param tvscales     Array of size tvscales_N specifying the scaling of the derivatives
 * 		       of \f$ | \Delta(z) \| \f$.
 * @param tvscales2_N  Number of TV scales for the second gradient.
 * @param tvscales2    Array of size tvscales2_N specifying the scaling of the derivatives
 * 		       of \f$ | \Delta(x + z) \| \f$.
 * @param lop_trafo    The linear operator that transforms the input data.
 * @return             A structure containing the ICTV regularization operator, which contains
 * 		       two linear operators for the gradients and two proximal operators for the thresholding.
 */

struct reg2 ictv_reg(unsigned long flags, unsigned long jflags, float lambda, int N, const long in_dims[N], long isize, long* ext_shift, const float gamma[2],
		     int tvscales_N, const float tvscales[tvscales_N], int tvscales2_N, const float tvscales2[tvscales2_N], const struct linop_s* lop_trafo)
{
	struct reg2 reg2;

	while ((0 < tvscales_N) && (0. == tvscales[tvscales_N - 1]))
		tvscales_N--;

	while ((0 < tvscales2_N) && (0. == tvscales2[tvscales2_N - 1]))
		tvscales2_N--;

	long in2_dims[N];

	if (NULL != lop_trafo) {

		assert(N == linop_domain(lop_trafo)->N);
		assert(md_check_equal_dims(N, in_dims, linop_domain(lop_trafo)->dims, ~0UL));
		assert(N == linop_codomain(lop_trafo)->N);

		md_copy_dims(N, in2_dims, linop_codomain(lop_trafo)->dims);

	} else {

		md_copy_dims(N, in2_dims, in_dims);
	}

	assert(0 != (flags & FFT_FLAGS));
	assert(0 != (flags & ~FFT_FLAGS));

	auto grad1b = linop_extract_create(1, MD_DIMS(0), MD_DIMS(md_calc_size(N, in_dims)), MD_DIMS(isize));

	if (NULL != lop_trafo) {

		auto iov = linop_codomain(grad1b);
		auto trafo_flat = linop_reshape_in(lop_trafo, iov->N, iov->dims);

		grad1b = linop_chain_FF(grad1b, trafo_flat);
	}

	grad1b = linop_reshape_out_F(grad1b, N, in2_dims);

	auto grad1c = linop_extract_create(1, MD_DIMS(*ext_shift), MD_DIMS(md_calc_size(N, in2_dims)), MD_DIMS(isize));
	grad1c = linop_reshape_out_F(grad1c, N, in2_dims);

	auto grad1d = linop_plus_FF(grad1b, grad1c);

	const struct linop_s* grad1 = linop_grad_create(N, in2_dims, N, flags);

	if (0 < tvscales_N) {

		debug_printf(DP_INFO, "ICTV anisotropic scaling of first gradient: %d\n", tvscales_N);

		assert(tvscales_N == linop_codomain(grad1)->dims[N]);

		complex float ztvscales[tvscales_N];

		for (int i = 0; i < tvscales_N; i++)
			ztvscales[i] = tvscales[i];

		grad1 = linop_chain_FF(grad1,
			linop_cdiag_create(N + 1, linop_codomain(grad1)->dims, MD_BIT(N), ztvscales));
	}

	// \Delta (x + z)

	reg2.linop[0] = linop_chain_FF(grad1d, grad1);

	const struct linop_s* grad2 = linop_grad_create(N, in2_dims, N, flags);

	if (0 < tvscales2_N) {

		debug_printf(DP_INFO, "ICTV anisotropic scaling of second gradient: %d\n", tvscales2_N);

		assert(tvscales2_N == linop_codomain(grad2)->dims[N]);

		complex float ztvscales2[tvscales2_N];

		for (int i = 0; i < tvscales2_N; i++)
			ztvscales2[i] = tvscales2[i];

		grad2 = linop_chain_FF(grad2,
				linop_cdiag_create(N + 1, linop_codomain(grad2)->dims, MD_BIT(N), ztvscales2));
	}

	auto grad2e = linop_extract_create(1, MD_DIMS(*ext_shift), MD_DIMS(md_calc_size(N, in2_dims)), MD_DIMS(isize));
	grad2e = linop_reshape_out_F(grad2e, N, in2_dims);

	reg2.linop[1] = linop_chain_FF(grad2e, grad2);

	reg2.prox[0] = prox_thresh_create(N + 1, linop_codomain(reg2.linop[0])->dims, lambda*gamma[0], jflags);
	reg2.prox[1] = prox_thresh_create(N + 1, linop_codomain(reg2.linop[1])->dims, lambda*gamma[1], jflags);

	*ext_shift += md_calc_size(N, in2_dims);

	return reg2;
}


/**
 * This function creates an ICTGV (Infimal Convolution of Total Generalized Variation) regularization operator
 * with the specified parameters. The ICTGV regularization is applied to the image dimensions specified by `out_dims`.
 *
 * The regularization term is given by:
 * \f[
 * \gamma_1 \| \text{TGV} (x + z) \| + \gamma_2 \| \text{TGV} (z) \|
 * \]
 *
 * The minimization problem is:
 * \f[
 * \min_{x,z,u,w} \gamma_1 (\alpha_1 \|\Delta (x + z) + u\|_1 + \alpha_0 \|\text{Eps} (u)\|_1)
 * + \gamma_2 (\alpha_1 \|\Delta (z) + w\|_1 + \alpha_0 \|\text{Eps} (w)\|_1)
 * \]
 *
 * where \f$ \text{Eps} (u) = 0.5 \Delta (u + u^T) \f$.
 *
 * @param flags        Bitmask specifying the dimensions for the regularization.
 * @param jflags       Bitmask for joint thresholding operation.
 * @param lambda       Regularization parameter.
 * @param N            Number of dimensions.
 * @param in_dims      Array of size N specifying the input dimensions.
 * @param isize        Size of the image including supporting variables.
 * @param ext_shift    Pointer to an integer specifying the external shift.
 * @param alpha        Array of size 2 specifying alpha_1 and alpha_0, the regularization parameters for each TGV regularization.
 * @param gamma        Array of size 2 specifying gamma_1 and gamma_2, the weighting factors between TGV terms.
 * @param tvscales_N   Number of TV scales for the first TGV regularization.
 * @param tvscales     Array of size tvscales_N specifying the scaling of the derivative in
 * 		       \f$ \| \text{TGV} (x + z) \| \f$.
 * @param tvscales2_N  Number of TV scales for the second TGV regularization.
 * @param tvscales2    Array of size tvscales2_N specifying the scaling of the derivative in
 * 		       \f$ \text{TGV} (z) \f$.
 * @param lop_trafo    The linear operator that transforms the input data.
 * @return             A structure containing the ICTGV regularization operator, which contains
 * 		       four linear operators for the two gradients and two symmetric gradients,
 * 		       and four proximal operators for the thresholding.
 */

struct reg4 ictgv_reg(unsigned long flags, unsigned long jflags, float lambda, int N, const long in_dims[N], long isize, long* ext_shift, const float alpha[2],
		      const float gamma[2], int tvscales_N, const float tvscales[tvscales_N], int tvscales2_N, const float tvscales2[tvscales2_N], const struct linop_s* lop_trafo)
{
	struct reg4 reg4;

	assert(0 != (flags & FFT_FLAGS));
	assert(0 != (flags & ~FFT_FLAGS));

	while ((0 < tvscales_N) && (0. == tvscales[tvscales_N - 1]))
		tvscales_N--;

	while ((0 < tvscales2_N) && (0. == tvscales2[tvscales2_N - 1]))
		tvscales2_N--;

	long in2_dims[N];

	if (NULL != lop_trafo) {

		assert(N == linop_domain(lop_trafo)->N);
		assert(md_check_equal_dims(N, in_dims, linop_domain(lop_trafo)->dims, ~0UL));

		assert(N == linop_codomain(lop_trafo)->N);

		md_copy_dims(N, in2_dims, linop_codomain(lop_trafo)->dims);

	} else {

		md_copy_dims(N, in2_dims, in_dims);
	}

	const struct linop_s* grad1 = linop_grad_create(N, in2_dims, N, flags);

	if (0 < tvscales_N) {

		debug_printf(DP_INFO, "TGV anisotropic scaling: %d\n", tvscales_N);

		assert(tvscales_N == linop_codomain(grad1)->dims[N]);

		complex float ztvscales[tvscales_N];

		for (int i = 0; i < tvscales_N; i++)
			ztvscales[i] = tvscales[i];

		grad1 = linop_chain_FF(grad1,
				linop_cdiag_create(N + 1, linop_codomain(grad1)->dims, MD_BIT(N), ztvscales));
	}

	long grd_dims[N + 2];
	md_copy_dims(N + 1, grd_dims, linop_codomain(grad1)->dims);
	grd_dims[N + 1] = 1;

	auto iov = linop_domain(grad1);
	auto grad1b = linop_extract_create(1, MD_DIMS(*ext_shift), MD_DIMS(md_calc_size(N, grd_dims)), MD_DIMS(isize));
	auto grad1c = linop_reshape_out_F(grad1b, iov->N, iov->dims);

	// \Delta ( z )
	auto grad1d = linop_chain_FF(grad1c, grad1);

	long shift_conv = *ext_shift;
	*ext_shift += md_calc_size(N, grd_dims);

	// \Delta ( x ) + u
	struct reg2 reg_tgv1 = tgv_reg(flags, jflags, lambda*gamma[0], N, in_dims, isize, ext_shift, alpha, tvscales_N, tvscales, lop_trafo);

	// \Delta ( z + x ) + u
	reg4.linop[0] = linop_plus_FF(grad1d, reg_tgv1.linop[0]);

	// \Eps ( u )
	reg4.linop[1] = reg_tgv1.linop[1];

	reg4.prox[0] = reg_tgv1.prox[0];
	reg4.prox[1] = reg_tgv1.prox[1];

	struct reg2 reg_tgv2 = tgv_reg_int(flags, jflags, lambda*gamma[1], N, in2_dims, isize, shift_conv, ext_shift, alpha, tvscales2_N, tvscales2, NULL);

	// \Delta ( z )
	reg4.linop[2] = reg_tgv2.linop[0];

	// \Eps ( w )
	reg4.linop[3] = reg_tgv2.linop[1];

	reg4.prox[2] = reg_tgv2.prox[0];
	reg4.prox[3] = reg_tgv2.prox[1];

	return reg4;
}

