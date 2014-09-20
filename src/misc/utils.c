/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2014 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/loop.h"

#include "misc/misc.h"

#include "utils.h"


void linear_phase(unsigned int N, const long dims[N], const float pos[N], complex float* out)
{
	complex float grad[N];

	for (unsigned int n = 0; n < N; n++)
		grad[n] = 2.i * M_PI * (float)(pos[n]) / ((float)dims[n]);

	md_zgradient(N, dims, out, grad); // (x * p - x0 * p

	long dims0[N];
	md_singleton_dims(N, dims0);

	long strs0[N];
	md_calc_strides(N, strs0, dims0, CFL_SIZE);

	complex float cn = 0.;

	for (unsigned int n = 0; n < N; n++)
		 cn -= grad[n] * (float)dims[n] / 2.;

	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	md_zadd2(N, dims, strs, out, strs, out, strs0, &cn);
	md_zmap(N, dims, out, out, cexpf);
}


complex float* compute_mask(unsigned int N, const long msk_dims[N], const float restrict_fov[N])
{
	complex float* mask = md_alloc(N, msk_dims, CFL_SIZE);

	long small_dims[N];

	for (unsigned int i = 0; i < N; i++)
		small_dims[i] = (1 == msk_dims[i]) ? 1 : (msk_dims[i] * restrict_fov[i]);

	complex float* small_mask = md_alloc(N, small_dims, CFL_SIZE);

	md_fill(N, small_dims, small_mask, &(complex float){ 1. }, CFL_SIZE);
	md_resizec(N, msk_dims, mask, small_dims, small_mask, CFL_SIZE);

	md_free(small_mask);

	return mask;
}


void apply_mask(unsigned int N, const long dims[N], complex float* x, const float restrict_fov[N])
{
	unsigned int flags = 0;
	for (unsigned int i = 0; i < N; i++)
		if (1. != restrict_fov[i])
			flags |= (1 << i);

	long msk_dims[N];
	md_select_dims(N, flags, msk_dims, dims);

	long msk_strs[N];
	md_calc_strides(N, msk_strs, msk_dims, CFL_SIZE);

	complex float* mask = compute_mask(N, msk_dims, restrict_fov);

	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);
	md_zmul2(N, dims, strs, x, strs, x, msk_strs, mask);
	md_free(mask);
}


void normalize(int N, unsigned int flags, const long dims[N], complex float* maps)
{
	long dims_img[N];
	md_select_dims(N, ~flags, dims_img, dims);

	complex float* maps_norm = md_alloc(N, dims_img, CFL_SIZE);

	md_zrss(N, dims, flags, maps_norm, maps);

	long str[N];
	long str_img[N];

	md_calc_strides(N, str, dims, CFL_SIZE);
	md_calc_strides(N, str_img, dims_img, CFL_SIZE);

	md_zdiv2(N, dims, str, maps, str, maps, str_img, maps_norm);
	md_free(maps_norm);
}



