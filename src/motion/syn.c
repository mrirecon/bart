/* Copyright 2024. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 *
 *
 * Avants BB, Epstein CL, Grossman M, Gee JC.
 * Symmetric diffeomorphic image registration with cross-correlation:
 * evaluating automated labeling of elderly and neurodegenerative brain.
 * Med Image Anal 2008;12:26-41.
 *
 * Avants BB, Tustison NJ, Song G, Cook PA, Klein A, Gee JC.
 * A reproducible evaluation of ANTs similarity metric performance in
 * brain image registration. Neuroimage 2011;54:2033-2044.
 */

#include <complex.h>
#include <math.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/loop.h"

#include "linops/linop.h"
#include "linops/sum.h"
#include "linops/someops.h"
#include "linops/grad.h"

#include "nlops/nlop.h"
#include "nlops/const.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/conv.h"
#include "nlops/someops.h"
#include "nlops/tenmul.h"

#include "nn/losses.h"

#include "motion/interpolate.h"
#include "motion/displacement.h"
#include "motion/pyramide.h"

#include "syn.h"


static void zentral_differences(int D, const long dims[D], int d, unsigned long flags, complex float* out, const complex float* in)
{
	long idims[D];
	md_select_dims(D, ~MD_BIT(d), idims, dims);

	const struct linop_s* lop = linop_grad_zentral_create(D, idims, d, flags);

	linop_forward(lop, D, dims, out, D, idims, in);

	linop_free(lop);
}


//static void demons(float reg, int N, const long dims[N], const long img_dims[N], complex float* scl, const complex float* grad)
//{
//	complex float* den = md_alloc_sameplace(N, img_dims, CFL_SIZE, scl);
//	md_zmulc(N, img_dims, den, scl, scl);
//	md_zsmul(N, img_dims, den, den, 1. / reg);
//	md_zfmacc2(N, dims, MD_STRIDES(N, img_dims, CFL_SIZE), den, MD_STRIDES(N, dims, CFL_SIZE), grad, MD_STRIDES(N, dims, CFL_SIZE), grad);
//	md_zdiv(N, img_dims, scl, scl, den);
//	md_free(den);
//}


static void _iterate(int iter, int d, unsigned long flags, int N, const long dims[N],
			const struct nlop_s* nlop_metric, const struct linop_s* lop_gaussian,
			complex float* static_to_ref_fw, complex float* static_to_ref_bw, const complex float* static_img,
			complex float* moving_to_ref_fw, complex float* moving_to_ref_bw, const complex float* moving_img)
{
	float step_length = 0.25;

	long img_dims[N];
	md_select_dims(N, ~MD_BIT(d), img_dims, dims);

	complex float* current_static = md_alloc_sameplace(N, img_dims, CFL_SIZE, static_img);
	complex float* current_moving = md_alloc_sameplace(N, img_dims, CFL_SIZE, moving_img);

	const struct linop_s* lop_static = linop_interpolate_displacement_create(d, flags, 3, N, img_dims, dims, static_to_ref_bw, img_dims);
	const struct linop_s* lop_moving = linop_interpolate_displacement_create(d, flags, 3, N, img_dims, dims, moving_to_ref_bw, img_dims);

	linop_forward_unchecked(lop_static, current_static, static_img);
	linop_forward_unchecked(lop_moving, current_moving, moving_img);

	linop_free(lop_static);
	linop_free(lop_moving);

	complex float* scalar = md_alloc_sameplace(1, MD_DIMS(1), CFL_SIZE, static_to_ref_bw);
	nlop_generic_apply_unchecked(nlop_metric, 3, (void*[3]) { scalar, (void*)current_static, (void*)current_moving });

	complex float loss;
	md_copy(1, MD_DIMS(1), &loss, scalar, CFL_SIZE);

	debug_printf(DP_DEBUG1, "Loss[%d]: %e\n", iter, crealf(loss));

	complex float* bw_stp = md_alloc_sameplace(N, dims, CFL_SIZE, scalar);
	complex float* fw_stp = md_alloc_sameplace(N, dims, CFL_SIZE, scalar);

	zentral_differences(N, dims, d, flags, fw_stp, current_static);
	zentral_differences(N, dims, d, flags, bw_stp, current_moving);

	md_zfill(1, MD_DIMS(1), scalar, 1.);
	linop_adjoint_unchecked(nlop_get_derivative(nlop_metric, 0, 0), current_static, scalar);
	linop_adjoint_unchecked(nlop_get_derivative(nlop_metric, 0, 1), current_moving, scalar);

	nlop_clear_derivatives(nlop_metric);

	//float sigma_sq1_x = 4. * factor * factor;
	//demons(sigma_sq1_x, N, dims, img_dims, current_static, fw_stp);
	//demons(sigma_sq1_x, N, dims, img_dims, current_moving, bw_stp);

	md_zmul2(N, dims, MD_STRIDES(N, dims, CFL_SIZE), fw_stp, MD_STRIDES(N, dims, CFL_SIZE), fw_stp, MD_STRIDES(N, img_dims, CFL_SIZE), current_static);
	md_zmul2(N, dims, MD_STRIDES(N, dims, CFL_SIZE), bw_stp, MD_STRIDES(N, dims, CFL_SIZE), bw_stp, MD_STRIDES(N, img_dims, CFL_SIZE), current_moving);

	md_free(current_static);
	md_free(current_moving);

	linop_forward(lop_gaussian, N, dims, fw_stp, N, dims, fw_stp);
	linop_forward(lop_gaussian, N, dims, bw_stp, N, dims, bw_stp);

	long rdims[N];
	md_copy_dims(N, rdims, dims);

	for (int i = 0; i < N; i++)
		if (MD_IS_SET(flags, i))
			rdims[i] -= 2;

	// Set boundary to zero
	complex float* tmpr = md_alloc_sameplace(N, rdims, CFL_SIZE, scalar);
	md_resize_center(N, rdims, tmpr, dims, fw_stp, CFL_SIZE);
	md_resize_center(N, dims, fw_stp, rdims, tmpr, CFL_SIZE);
	md_resize_center(N, rdims, tmpr, dims, bw_stp, CFL_SIZE);
	md_resize_center(N, dims, bw_stp, rdims, tmpr, CFL_SIZE);
	md_free(tmpr);

	complex float* tmp = md_alloc_sameplace(N, dims, CFL_SIZE, scalar);
	complex float* tmp_img = md_alloc_sameplace(N, img_dims, CFL_SIZE, scalar);

	// Normalize to max norm 1
	md_ztenmulc(N, img_dims, tmp_img, dims, fw_stp, dims, fw_stp);
	md_zreal(N, img_dims, tmp_img, tmp_img);

	md_zfill(1, MD_DIMS(1), scalar, 0.);
	md_zmax2(N, img_dims, MD_SINGLETON_STRS(N), scalar, MD_SINGLETON_STRS(N), scalar, MD_STRIDES(N, img_dims, CFL_SIZE), tmp_img);
	float fw_iscale;
	md_copy(1, MD_DIMS(1), &fw_iscale, scalar, FL_SIZE);
	md_zsmul(N, dims, fw_stp, fw_stp, step_length / sqrtf(fw_iscale));

	md_ztenmulc(N, img_dims, tmp_img, dims, bw_stp, dims, bw_stp);
	md_zreal(N, img_dims, tmp_img, tmp_img);
	md_zfill(1, MD_DIMS(1), scalar, 0.);
	md_zmax2(N, img_dims, MD_SINGLETON_STRS(N), scalar, MD_SINGLETON_STRS(N), scalar, MD_STRIDES(N, img_dims, CFL_SIZE), tmp_img);
	float bw_iscale;
	md_copy(1, MD_DIMS(1), &bw_iscale, scalar, FL_SIZE);
	md_zsmul(N, dims, bw_stp, bw_stp, step_length / sqrtf(bw_iscale));

	// Update displacement
	compose_displacement(N, d, flags, dims, tmp, static_to_ref_fw, fw_stp);
	md_copy(N, dims, static_to_ref_fw, tmp, CFL_SIZE);

	invert_displacement(N, d, flags, dims, static_to_ref_bw, static_to_ref_fw);
	invert_displacement(N, d, flags, dims, static_to_ref_fw, static_to_ref_bw);

	compose_displacement(N, d, flags, dims, tmp, moving_to_ref_fw, bw_stp);
	md_copy(N, dims, moving_to_ref_fw, tmp, CFL_SIZE);

	invert_displacement(N, d, flags, dims, moving_to_ref_bw, moving_to_ref_fw);
	invert_displacement(N, d, flags, dims, moving_to_ref_fw, moving_to_ref_bw);

	md_free(fw_stp);
	md_free(bw_stp);
	md_free(tmp);
	md_free(tmp_img);
	md_free(scalar);
}





void syn(int levels, float sigma[levels], float factors[levels], int nwarps[levels],
	int d, unsigned long flags, int N, const long _dims[N],
	complex float* disp, complex float* idisp,
	const complex float* static_img, const complex float* moving_img)
{
	assert(_dims[d] == bitcount(flags));

	long tdims[N];
	md_select_dims(N, ~MD_BIT(d), tdims, _dims);

	long dims[levels][N];
	complex float* img_static[levels];
	complex float* img_moved[levels];

	debug_gaussian_pyramide(levels, factors, sigma, N, flags, tdims);

	gaussian_pyramide(levels, factors, sigma, 3, N, flags, tdims, moving_img, dims, img_moved);
	gaussian_pyramide(levels, factors, sigma, 3, N, flags, tdims, static_img, dims, img_static);

	long udims[N];
	md_copy_dims(N, udims, dims[levels - 1]);
	udims[d] = bitcount(flags);

	complex float* static_to_ref_fw = md_alloc_sameplace(N, udims, CFL_SIZE, disp);
	complex float* static_to_ref_bw = md_alloc_sameplace(N, udims, CFL_SIZE, disp);
	complex float* moving_to_ref_fw = md_alloc_sameplace(N, udims, CFL_SIZE, disp);
	complex float* moving_to_ref_bw = md_alloc_sameplace(N, udims, CFL_SIZE, disp);

	md_clear(N, udims, static_to_ref_fw, CFL_SIZE);
	md_clear(N, udims, static_to_ref_bw, CFL_SIZE);
	md_clear(N, udims, moving_to_ref_fw, CFL_SIZE);
	md_clear(N, udims, moving_to_ref_bw, CFL_SIZE);

	for (int i = levels - 1; i >= 0; i--) {

		debug_printf(DP_DEBUG1, "Optimizing level %d\n", i);

		long img_dims[N];
		md_copy_dims(N, img_dims, dims[i]);

		md_copy_dims(N, udims, dims[i]);
		udims[d] = bitcount(flags);

		long kdims[N];
		md_set_dims(N, kdims, 5);
		md_select_dims(N, flags, kdims, kdims);

		float sigma[N];
		for (int j = 0; j < N; j++)
			sigma[j] = MD_IS_SET(flags, j) ? 2. : 0.;

		const struct linop_s* lop_gaussian = linop_conv_gaussian_create(N, CONV_TRUNCATED, udims, sigma);

		const struct nlop_s* nlop_metric = nlop_patched_cross_correlation_create(N, img_dims, kdims, flags, 1.e-18);
		//const struct nlop_s* nlop_metric = nlop_mse_create(N, img_dims, 0ul);

		for (int j = 0; j < nwarps[i]; j++)
			_iterate(j, d, flags, N, udims, nlop_metric, lop_gaussian,
				static_to_ref_fw, static_to_ref_bw, img_static[i],
				moving_to_ref_fw, moving_to_ref_bw, img_moved[i]);

		linop_free(lop_gaussian);
		nlop_free(nlop_metric);

		md_free(img_moved[i]);
		md_free(img_static[i]);

		if (0 == i)
			break;

		long nudims[N];
		md_copy_dims(N, nudims, dims[i - 1]);
		nudims[d] = _dims[d];

		complex float* n_static_to_ref_fw = md_alloc_sameplace(N, nudims, CFL_SIZE, disp);
		complex float* n_static_to_ref_bw = md_alloc_sameplace(N, nudims, CFL_SIZE, disp);
		complex float* n_moving_to_ref_fw = md_alloc_sameplace(N, nudims, CFL_SIZE, disp);
		complex float* n_moving_to_ref_bw = md_alloc_sameplace(N, nudims, CFL_SIZE, disp);

		upscale_displacement(N, d, flags, nudims, n_static_to_ref_fw, udims, static_to_ref_fw);
		upscale_displacement(N, d, flags, nudims, n_static_to_ref_bw, udims, static_to_ref_bw);
		upscale_displacement(N, d, flags, nudims, n_moving_to_ref_fw, udims, moving_to_ref_fw);
		upscale_displacement(N, d, flags, nudims, n_moving_to_ref_bw, udims, moving_to_ref_bw);

		md_free(static_to_ref_fw);
		md_free(static_to_ref_bw);
		md_free(moving_to_ref_fw);
		md_free(moving_to_ref_bw);

		static_to_ref_fw = n_static_to_ref_fw;
		static_to_ref_bw = n_static_to_ref_bw;
		moving_to_ref_fw = n_moving_to_ref_fw;
		moving_to_ref_bw = n_moving_to_ref_bw;
	}

	compose_displacement(N, d, flags, udims, disp, static_to_ref_fw, moving_to_ref_bw);

	if (NULL != idisp)
		compose_displacement(N, d, flags, udims, idisp, moving_to_ref_fw, static_to_ref_bw);

	md_free(static_to_ref_fw);
	md_free(static_to_ref_bw);
	md_free(moving_to_ref_fw);
	md_free(moving_to_ref_bw);
}
