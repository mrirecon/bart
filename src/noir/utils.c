/* Copyright 2013. The Regents of the University of California.
 * Copyright 2017-2022. Martin Uecker.
 * Copyright 2023. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 *
 * Uecker M, Hohage T, Block KT, Frahm J. Image reconstruction by regularized nonlinear
 * inversion – Joint estimation of coil sensitivities and image content.
 * Magn Reson Med 2008; 60:674-682.
 */



#include <complex.h>
#include <math.h>

#include "misc/debug.h"

#include "misc/misc.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/filter.h"
#include "num/fft.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "utils.h"


void noir_calc_weights(double a, double b, const long dims[3], complex float* dst)
{
	unsigned long flags = 0UL;

	for (int i = 0; i < 3; i++)
		if (1 != dims[i])
			flags = MD_SET(flags, i);

	klaplace(3, dims, flags, dst);
	md_zsmul(3, dims, dst, dst, a);
	md_zsadd(3, dims, dst, dst, 1.);
	md_zspow(3, dims, dst, dst, -b / 2.);	// 1 + 220. \Laplace^16
}

static struct linop_s* linop_ifft_resize_create(int N, unsigned long flags, const long osdims[N], const long idims[N], const long kdims[N])
{

	auto lop_ret = linop_ifft_create(N, osdims, flags);

	if (!md_check_equal_dims(N, osdims, kdims, flags))
		lop_ret = linop_chain_FF(linop_resize_center_create(N, osdims, kdims), lop_ret);

	if (!md_check_equal_dims(N, osdims, idims, flags))
		lop_ret = linop_chain_FF(lop_ret, linop_resize_center_create(N, idims, osdims));

	return lop_ret;
}


/**
 * Create linear operator computing R(WF)^(-1)R with
 * W - c * (1 + a||k||^2)^(b/2)
 * F - Centered, unitary Fourier Transform
 * R - Centered resizing operators
 *
 * The grid, W^-1 and F^-1 are applied on, is oversized by a factor factor_fov with respect to img_dims
 *
 * @param N
 * @param img_dims
 * @param ksp_dims
 * @param ref_dims for computing laplacian
 * @param flags
 * @param factor_fov fft is applied on grid with size factor_fov * img_dims
 * @param a parameter a of W
 * @param b parameter b of W
 * @param c parameter b of W
 * img_spacing = dx = fov / img_dims
 * ksp_spacing = dk = img_dims / fov
 */
struct linop_s* linop_noir_weights_create(int N, const long img_dims[N], const long ksp_dims[N], const long ref_dims[N], unsigned long flags, double factor_fov, double a, double b, double c)
{
	flags &= md_nontriv_dims(N, img_dims);

	assert(md_nontriv_dims(N, img_dims) == md_nontriv_dims(N, ksp_dims));
	assert(md_check_equal_dims(N, img_dims, ksp_dims, ~flags));

	long os_dims[N];

	for (int i = 0; i < N; i++) {

		os_dims[i] = lround(img_dims[i] * (MD_IS_SET(flags, i) ? fabs(factor_fov) : 1.));
		if (fabs(img_dims[i] * (MD_IS_SET(flags, i) ? fabs(factor_fov) : 1.) - os_dims[i]) > 0.0001)
			debug_printf(DP_WARN, "Sobolev oversampling factor %f is incompatible with grid size %ld!\n", factor_fov, img_dims[i]);
	}

	long wgh_dims[N];
	md_select_dims(N, flags, wgh_dims, os_dims);

	complex float* wgh = md_alloc(N, wgh_dims, CFL_SIZE);

	float sc[N];
	for (int i = 0; i < N; i++)
		sc[i] = (NULL == ref_dims) ? 1. / wgh_dims[i] :  1. / ref_dims[i];

	klaplace_scaled(N, wgh_dims, flags, sc, wgh);
	md_zsmul(N, wgh_dims, wgh, wgh, a);
	md_zsadd(N, wgh_dims, wgh, wgh, 1.);
	md_zspow(N, wgh_dims, wgh, wgh, -b / 2.);
	md_zsmul(N, wgh_dims, wgh, wgh, c);

	ifftmod(N, wgh_dims, flags, wgh, wgh);
	fftscale(N, wgh_dims, flags, wgh, wgh);

	struct linop_s* lop_ret = NULL;

	lop_ret = linop_ifft_resize_create(N, flags, os_dims, img_dims, ksp_dims);

	long wgh_ksp_dims[N];
	long wgh_img_dims[N];

	md_select_dims(N, flags, wgh_ksp_dims, ksp_dims);
	md_select_dims(N, flags, wgh_img_dims, img_dims);

	complex float* wgh_res = md_alloc(N, wgh_ksp_dims, CFL_SIZE);
	md_resize_center(N, wgh_ksp_dims, wgh_res, wgh_dims, wgh, CFL_SIZE);
	lop_ret = linop_chain_FF(linop_cdiag_create(N, ksp_dims, flags, wgh_res), lop_ret);
	md_free(wgh_res);

	md_zfill(N, wgh_dims, wgh, 1);
	ifftmod(N, wgh_dims, flags, wgh, wgh);

	wgh_res = md_alloc(N, wgh_img_dims, CFL_SIZE);
	md_resize_center(N, wgh_img_dims, wgh_res, wgh_dims, wgh, CFL_SIZE);
	lop_ret = linop_chain_FF(lop_ret, linop_cdiag_create(N, img_dims, flags, wgh_res));
	md_free(wgh_res);

	md_free(wgh);

	return lop_ret;
}

