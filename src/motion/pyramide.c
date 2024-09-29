/* Copyright 2024. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <complex.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "motion/interpolate.h"

#include "pyramide.h"

void gaussian_pyramide(int levels, float factors[levels], float sigma[levels], int ord,
			int N, unsigned long flags, const long idims[N], const complex float* img,
			long dims[levels][N], complex float* imgs[levels])
{
	complex float* tmp = md_alloc_sameplace(N, idims, CFL_SIZE, img);

	complex float* max = md_alloc_sameplace(N, MD_SINGLETON_DIMS(N), CFL_SIZE, img);
	md_zfill(N, MD_SINGLETON_DIMS(N), max, 0.);
	
	md_zmax2(N, idims, MD_SINGLETON_STRS(N), max, MD_SINGLETON_STRS(N), max, MD_STRIDES(N, idims, CFL_SIZE), img);
	
	complex float max_val;
	md_copy(N, MD_SINGLETON_DIMS(N), &max_val, max, CFL_SIZE);

	md_free(max);

	for (int i = 0; i < levels; i++) {

		if (0. == sigma[i]) {
		
			md_copy(N, idims, tmp, img, CFL_SIZE);

		} else {

			float sigmas[N];
			for (int j = 0; j < N; j++)
				sigmas[j] = MD_IS_SET(flags, j) ? sigma[i] : 0;

			auto lop_gauss = linop_conv_gaussian_create(N, CONV_TRUNCATED, idims, sigmas);

			linop_forward(lop_gauss, N, idims, tmp, N, idims, img);

			linop_free(lop_gauss);
		}

		md_copy_dims(N, dims[i], idims);

		for (int j = 0; j < N; j++) {

			if (!MD_IS_SET(flags, j))
				continue;

			dims[i][j] = MAX(MIN(4, idims[j]), (long)(idims[j] *  factors[i]));
		}

		imgs[i] = md_alloc_sameplace(N, dims[i], CFL_SIZE, img);
		md_resample(flags, ord, N, dims[i], imgs[i], idims, tmp);

		md_zsmul(N, dims[i], imgs[i], imgs[i], 1. / max_val);
	}

	md_free(tmp);
}


void debug_gaussian_pyramide(int levels, float factors[levels], float sigma[levels], int N, unsigned long flags, const long idims[N])
{
	long dims[levels][N];
	
	debug_printf(DP_DEBUG1, "Generate Gaussian Pyramide:\n");

	for (int i = 0; i < levels; i++) {

		md_copy_dims(N, dims[i], idims);

		for (int j = 0; j < N; j++) {

			if (!MD_IS_SET(flags, j))
				continue;

			dims[i][j] = MAX(MIN(4, idims[j]), (long)(idims[j] *  factors[i]));
		}

		debug_printf(DP_DEBUG1, "\tLevel %d: sigma=%f, dims=", i, sigma[i]);
		debug_print_dims(DP_DEBUG1, N, dims[i]);
	}
}




void upscale_displacement(int N, int d, unsigned long flags,
			  const long odims[N], complex float* out,
			  const long idims[N], const complex float* in)
{
	md_resample(flags, 3, N, odims, out, idims, in);

	long sdims[N];
	md_select_dims(N, MD_BIT(d), sdims, odims);

	complex float* scale = md_alloc_sameplace(N, sdims, CFL_SIZE, in);

	for (int i = 0, ip = 0; i < N; i++)
		if (MD_IS_SET(flags, i))
			md_zfill(N, MD_SINGLETON_DIMS(N), scale + ip++, (float)odims[i] / idims[i]);

	md_zmul2(N, odims, MD_STRIDES(N, odims, CFL_SIZE), out, MD_STRIDES(N, odims, CFL_SIZE), out, MD_STRIDES(N, sdims, CFL_SIZE), scale);

	md_free(scale);
}







