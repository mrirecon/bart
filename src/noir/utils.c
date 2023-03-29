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

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/filter.h"
#include "num/fft.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "utils.h"


void noir_calc_weights(double a, double b, const long dims[3], complex float* dst)
{
	unsigned int flags = 0;

	for (int i = 0; i < 3; i++)
		if (1 != dims[i])
			flags = MD_SET(flags, i);

	klaplace(3, dims, flags, dst);
	md_zsmul(3, dims, dst, dst, a);
	md_zsadd(3, dims, dst, dst, 1.);
	md_zspow(3, dims, dst, dst, -b / 2.);	// 1 + 220. \Laplace^16
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
 * @param flags
 * @param factor_fov fft is applied on grid with size factor_fov * img_dims or -factor_fov * img_dims
 * @param a parameter a of W
 * @param b parameter b of W
 * @param c parameter b of W
 * img_spacing = dx = fov / img_dims
 * ksp_spacing = dk = img_dims / fov
 */
struct linop_s* linop_noir_weights_create(int N, const long img_dims[N], const long ksp_dims[N], unsigned long flags, double factor_fov, double a, double b, double c)
{
	flags &= md_nontriv_dims(N, img_dims);

	assert(md_nontriv_dims(N, img_dims) == md_nontriv_dims(N, ksp_dims));
	assert(md_check_equal_dims(N, img_dims, ksp_dims, ~flags));

	long tmp_dims[N];

	for (int i = 0; i < N; i++)
		tmp_dims[i] = lround((0 < factor_fov ? img_dims[i] : ksp_dims[i]) * (MD_IS_SET(flags, i) ? fabs(factor_fov) : 1.));

	long wgh_dims[N];
	md_select_dims(N, flags, wgh_dims, tmp_dims);

	complex float* wgh = md_alloc(N, wgh_dims, CFL_SIZE);
	complex float* wgh_res = md_alloc(N, wgh_dims, CFL_SIZE);

	klaplace(N, wgh_dims, flags, wgh);
	md_zsmul(N, wgh_dims, wgh, wgh, a);
	md_zsadd(N, wgh_dims, wgh, wgh, 1.);
	md_zspow(N, wgh_dims, wgh, wgh, -b / 2.);
	md_zsmul(N, wgh_dims, wgh, wgh, c);

	ifftmod(N, wgh_dims, flags, wgh, wgh);
	fftscale(N, wgh_dims, flags, wgh, wgh);

	struct linop_s* lop_ret = linop_ifft_create(N, tmp_dims, flags);

	if (!md_check_equal_dims(N, tmp_dims, ksp_dims, flags))
		lop_ret = linop_chain_FF(linop_resize_center_create(N, tmp_dims, ksp_dims), lop_ret);

	if (!md_check_equal_dims(N, tmp_dims, img_dims, flags))
		lop_ret = linop_chain_FF(lop_ret, linop_resize_center_create(N, img_dims, tmp_dims));

	long wgh_ksp_dims[N];
	long wgh_img_dims[N];

	md_select_dims(N, flags, wgh_ksp_dims, ksp_dims);
	md_select_dims(N, flags, wgh_img_dims, img_dims);

	md_resize_center(N, wgh_ksp_dims, wgh_res, wgh_dims, wgh, CFL_SIZE);
	lop_ret = linop_chain_FF(linop_cdiag_create(N, ksp_dims, flags, wgh_res), lop_ret);

	md_zfill(N, wgh_dims, wgh, 1);
	ifftmod(N, wgh_dims, flags, wgh, wgh);

	md_resize_center(N, wgh_img_dims, wgh_res, wgh_dims, wgh, CFL_SIZE);
	lop_ret = linop_chain_FF(lop_ret, linop_cdiag_create(N, img_dims, flags, wgh_res));

	md_free(wgh);
	md_free(wgh_res);

	return lop_ret;
}

