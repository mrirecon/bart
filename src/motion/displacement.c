/* Copyright 2024. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */


#include <complex.h>

#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "motion/interpolate.h"

#include "displacement.h"


static void compose_displacement_internal(int N, int d, unsigned long flags, const long dims[N], complex float* composed, const complex float* d1, const complex float* d2, const complex float* pos)
{
	assert(d + 1< N);
	assert(dims[d] == bitcount(flags));

	complex float* tmp = md_alloc_sameplace(N, dims, CFL_SIZE, d1);

	if (NULL == pos) {

		md_positions(N, d, flags, dims, dims, tmp);
		md_zadd(N, dims, tmp, tmp, d1);

	} else {

		md_zadd(N, dims, tmp, pos, d1);
	}

	long cdims[N];
	md_transpose_dims(N, d, d + 1, cdims, dims);

	md_interpolate(d + 1, flags, 1, N, dims, composed, cdims, tmp, dims, d2);
	md_zadd(N, dims, composed, composed, d1);

	md_free(tmp);
}

void compose_displacement(int N, int d, unsigned long flags, const long dims[N], complex float* composed, const complex float* d1, const complex float* d2)
{
	compose_displacement_internal(N, d, flags, dims, composed, d1, d2, NULL);
}

/**
 * Chen, M., Lu, W., Chen, Q., Ruchala, K. J., & Olivera, G. H. (2008).
 * A simple fixed-point approach to invert a deformation field.
 * Medical Physics, 35(1), 81. doi:10.1118/1.2816107
**/
void invert_displacement(int N, int d, unsigned long flags, const long dims[N], complex float* inv_disp, const complex float* disp)
{
	int max_iter = 20;
	float tol = 0.001;

	assert(d + 1 < N);
	assert(1 == dims[d + 1]);

	complex float* pos = md_alloc_sameplace(N, dims, CFL_SIZE, disp);

	md_positions(N, d, flags, dims, dims, pos);

	long img_dims[N];
	md_select_dims(N, ~MD_BIT(d), img_dims, dims);

	complex float* tmp = md_alloc_sameplace(N, dims, CFL_SIZE, disp);
	complex float* mag = md_alloc_sameplace(N, img_dims, CFL_SIZE, disp);
	complex float* err = md_alloc_sameplace(N, MD_SINGLETON_DIMS(N), CFL_SIZE, disp);

	for (int i = 0; i < max_iter; i++) {

		// tmp = inv_disp + disp(pos + inv_disp)
		compose_displacement_internal(N, d, flags, dims, tmp, inv_disp, disp, pos);

		md_ztenmulc(N, img_dims, mag, dims, tmp, dims, tmp);
		md_zreal(N, img_dims, mag, mag);
		md_zsqrt(N, img_dims, mag, mag);
		md_zavg(N, img_dims, ~0ul, err, mag);

		// inv_disp = - disp(pos + inv_disp)
		md_zaxpy(N, dims, inv_disp, -0.25, tmp);

		if (md_znorm(N, MD_SINGLETON_DIMS(N), err) < tol)
			break;
	}

	md_free(pos);
	md_free(tmp);
	md_free(mag);
	md_free(err);
}


const struct linop_s* linop_interpolate_displacement_create(int d, unsigned long flags, int ord, int N, const long idims[N], const long mdims[N], const complex float* motion, const long gdims[N])
{
	long sdims[N];
	md_select_dims(N, MD_BIT(d), sdims, mdims);

	complex float* scale = md_alloc_sameplace(N, sdims, CFL_SIZE, motion);

	for (int i = 0, ip = 0; i < N; i++)
		if (MD_IS_SET(flags, i))
			md_zfill(N, MD_SINGLETON_DIMS(N), scale + (ip++), (float)gdims[i] / idims[i]);

	complex float* pos = md_alloc_sameplace(N, mdims, CFL_SIZE, motion);

	md_positions(N, d, flags, gdims, mdims, pos);

	md_zfmac2(N, mdims, MD_STRIDES(N, mdims, CFL_SIZE), pos, MD_STRIDES(N, mdims, CFL_SIZE), motion, MD_STRIDES(N, sdims, CFL_SIZE), scale);

	const struct linop_s* ret = linop_interpolate_create(d, MD_BIT(mdims[d]) - 1, ord, N, idims, mdims, pos, gdims);

	md_free(pos);
	md_free(scale);

	return ret;
}


