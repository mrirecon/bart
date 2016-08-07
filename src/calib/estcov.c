/* Copyright 2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2016 Siddharth Iyer <sid8795@gmail.com>
 */

#include <assert.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"

#include "estcov.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static long pointCov(complex float* out_data, long kx, long ky, long kz, long nc, long xdx, long ydx, long zdx, const complex float* kspace) {

	long flag = 0;
	complex float tmp1;
	complex float tmp2;

	// cdx is the column index of the covarience matrix
	for (long cdx = 0; cdx < nc; ++ cdx) {
		for (long ddx = 0; ddx <= cdx; ++ ddx) {
			// Indexing kspace[xdx, ydx, zdx, cdx]
			tmp1 = kspace[((cdx * kz + zdx) * ky + ydx) * kx + xdx];
			// Indexing kspace[xdx, ydx, zdx, ddx]
			tmp2 = kspace[((ddx * kz + zdx) * ky + ydx) * kx + xdx];
			// Indexing out_data[cdx, ddx]

			if (tmp1 != 0 && tmp2 != 0) {
				flag = 1;
				out_data[cdx * nc + ddx] += tmp1 * conj(tmp2);
				if (ddx != cdx)
					out_data[ddx * nc + cdx] += conj(tmp1) * tmp2;
			}

		}
	}

	return flag;
}

extern void estcov(long out_dims[2], complex float* out_data, float p, long N, const long kspace_dims[N], const complex float* kspace) {

	long kx = kspace_dims[0];
	long ky = kspace_dims[1];
	long kz = kspace_dims[2];
	long nc = kspace_dims[3];

	long sx = (long) ceil(kx * p);
	long sy = (long) ceil(ky * p);
	long sz = (long) ceil(kz * p);

	// Asserts that dim(covMat) = [nc x nc]
	assert(out_dims[0] == nc && out_dims[1] == nc);

	// Zero out out_data
	for (long idx = 0; idx < nc * nc; idx ++)
		out_data[idx] = 0;

	float numel = 0;

	for (long zdx = 0; zdx < kz; ++ zdx) {
		for (long ydx = 0; ydx < ky; ++ ydx) {
			for (long xdx = 0; xdx < kx; ++ xdx) {
				if (((xdx >= 0 && xdx < sx) || (xdx >= kx-sx && xdx < kx)) &&
				    ((ydx >= 0 && ydx < sy) || (ydx >= ky-sy && ydx < ky)) &&
				    ((zdx >= 0 && zdx < sz) || (zdx >= kz-sz && zdx < kz))) {
					numel += pointCov(out_data, kx, ky, kz, nc, xdx, ydx, zdx, kspace);
				}
			}
		}
	}

	for (long idx = 0; idx < nc * nc; idx ++)
		out_data[idx] /= numel;

}

extern float estcov_var(float p, long N, const long kspace_dims[N], const complex float* kspace) {

	long nc = kspace_dims[3];

	long cov_dims[2] = {nc, nc};
	complex float* cov = md_alloc(2, cov_dims, CFL_SIZE);

	estcov(cov_dims, cov, p, N, kspace_dims, kspace);

	float var = 0;
	for (int idx = 0; idx < nc; ++ idx)
		var += cov[idx * nc + idx];
	
	return var/nc;
}

