/* Copyright 2023. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <assert.h>
#include <math.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/loop.h"

#include "nlmeans.h"


/* non-local means filter
 *
 * Buades, A, Coll B, Morel J-M. A non-local algorithm for image denoising.
 * IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05); 2005.
 */
void md_znlmeans2(int D, const long dim[D], unsigned long flags,
		const long ostrs[D], complex float* optr,
		const long istrs[D], const complex float* iptr,
		long patch_length, long patch_dist, float h, float a)
{
	assert(1 == patch_length % 2);

	bool multi = true;
	bool parallel = true;

	int reflect_dist = patch_dist + patch_length / 2;

	int flag_count = bitcount(flags);
	int xD = D + flag_count;
	unsigned long xflags = 0; // ! unsigned ! because long could be only 32 bit -> plusminus 2 ^ 16

	long r_patchcenter_offset[D];

	long fdim[D];
	long rdim[D];
	long xdim[xD];

	long weight_full_dim[xD];
	long weight_1_dim[xD];
	long w_weight_dim[xD];
	long n_dim[D];

	long patch_dim[xD];

	long rstrs[D];
	long n_str[D];
	long n_full_str[D];
	long ostr2[D];

	long xstr[xD];
	long weight_full_str[xD];
	long weight_1_str[xD];
	long w_weight_str[xD];
	long patch_str[xD];

	md_select_dims(D, flags, fdim, dim);
	md_copy_dims(D, rdim, dim);
	md_copy_dims(D, xdim, dim);
	md_copy_dims(D, weight_full_dim, dim);
	md_set_dims(xD, w_weight_dim, 1);

	md_set_dims(xD, weight_1_dim, 1);
	md_copy_dims(D, weight_1_dim, dim);

	md_copy_dims(D, patch_dim, dim);
	md_set_dims(D, r_patchcenter_offset, 0);

	md_select_dims(D, ~flags, n_dim, dim);


	for (int i = 0; i < D; i++) {

		if (MD_IS_SET(flags, i)) {

			rdim[i] += 2 * reflect_dist;

			r_patchcenter_offset[i] = patch_length / 2;

			patch_dim[i] = 2 * patch_dist + 1;
			patch_dim[D + i] = patch_length;

			weight_full_dim[i + D] = patch_dim[i];
			weight_1_dim[i] = patch_dim[i];

			w_weight_dim[D + i] = patch_dim[D + i];

			xdim[i] = dim[i] + 2 * (patch_length / 2);
			xdim[D + i] = 2 * patch_dist + 1;

			xflags = MD_SET(xflags, D + i);
		}
	}

	md_calc_strides(xD, xstr, xdim, CFL_SIZE);
	md_calc_strides(xD, weight_full_str, weight_full_dim, CFL_SIZE);
	md_copy_strides(xD, patch_str, xstr);
	md_copy_strides(xD, weight_1_str, weight_full_str);
	md_copy_strides(D, ostr2, ostrs);

	md_calc_strides(D, n_full_str, dim, CFL_SIZE);
	md_copy_strides(D, n_str, n_full_str);

	md_calc_strides(xD, w_weight_str, w_weight_dim, CFL_SIZE);
	md_calc_strides(D, rstrs, rdim, CFL_SIZE);

	for (int i = 0; i < D; i++) {

		if (MD_IS_SET(flags, i)) {

			n_str[i] = 0;
			ostr2[i] = 0;

			patch_str[i] = xstr[i + D];
			patch_str[i + D] = xstr[i];

			// weight_1 'reuses' the flag dimensions, as they become free from summing over a patch,
			// for the patch distance. weight_full has these in D+[0 .. flag_count].
			weight_1_str[i] = weight_full_str[i + D];
			weight_1_str[i + D] = 0;
		}
	}



	complex float* r_in = md_calloc(D, rdim, CFL_SIZE);
	complex float* x = md_calloc(xD, xdim, CFL_SIZE);
	complex float* weight_full_vec = md_calloc(xD, multi ? weight_full_dim : weight_1_dim, CFL_SIZE);
	complex float* n_full_vec = md_calloc(D, dim, CFL_SIZE);
	complex float* w_weight = md_alloc(xD, w_weight_dim, CFL_SIZE);



	// Create extended input: reflectpad
	md_reflectpad_center2(D, rdim, rstrs, r_in, dim, istrs, iptr, CFL_SIZE);
	complex float* r_patchcenter = &MD_ACCESS(D, rstrs, r_patchcenter_offset, r_in);

	// 'precompute' x(N(i)) - x(N(j)): xdim[0:D] stores i,  xdim[D:xD] stores (i-j)
	md_znlmeans_distance2(D, rdim, xD, xdim, flags, xstr, x, rstrs, r_in);
	md_zmulc2(xD, xdim, xstr, x, xstr, x, xstr, x);

	// gauss kernel for distance-weighted euclidean norm
	md_zgausspdf(xD, w_weight_dim, w_weight, powf(a, 2));
	md_zsmul(xD, w_weight_dim, w_weight, w_weight, 1. / md_znorm(xD, w_weight_dim, w_weight)); // don't interfere with h_factor.

	md_clear2(D, dim, ostrs, optr, CFL_SIZE);


	// needed for clang
	const long *a_weight_full_str = &weight_full_str[0];
	const long *a_weight_1_dim = &weight_1_dim[0];
	const long *a_weight_1_str = &weight_1_str[0];
	const long *a_w_weight_str = &w_weight_str[0];
	const long *a_patch_dim = &patch_dim[0];
	const long *a_patch_str = &patch_str[0];
	const long *a_n_dim = &n_dim[0];
	const long *a_n_str = &n_str[0];
	const long *a_n_full_str = &n_full_str[0];
	const long *a_ostrs = &ostrs[0];
	const long *a_ostr2 = &ostr2[0];
	const long *a_rstrs = &rstrs[0];
	const long *a_xstr = &xstr[0];

	// loop over pixels
	NESTED(void, znlmeans_core, (const long im_pos[]))
	{
		complex float* weight_vec = multi ? &MD_ACCESS(D, a_weight_full_str, im_pos, weight_full_vec) : weight_full_vec;
		complex float* n_vec = multi ? &MD_ACCESS(D, a_n_full_str, im_pos, n_full_vec) : n_full_vec;

		/* Creative use of strides:
		 * flag dim:
		 *   - patch_dim and weight_1_dim have (2 x patch distance + 1) size in flag dimensions
		 *   - weight_1_str is 'normal': -> loop over image position
		 *   - patch_str is 'swapped'  : -> loop over patch location (in D+... dimensions)
		 * xflag dim:
		 *   - patch_dim has patch_length in xflag dimensions
		 *   - weight_1_str is 0    -> accumulate
		 *   - patch_str is swapped -> loop over image position, i.e. patch (difference) content.
		 */
		// weighted (!) euclidean distance ^ 2. (convolution in the original paper; but w_weights is symmetric, thus equivalent to pointwise multiplication & summation)

		md_zfmac2(xD, a_patch_dim, a_weight_1_str, weight_vec,
				a_patch_str, &MD_ACCESS(D, a_xstr, im_pos, x),
				a_w_weight_str, w_weight);

		// exponential
		md_zsmul2(xD, a_weight_1_dim, a_weight_1_str, weight_vec, a_weight_1_str, weight_vec, -1. / (2. * h * h));
		md_zexp2(D, a_weight_1_dim, a_weight_1_str, weight_vec, a_weight_1_str, weight_vec);

		// normalize
		md_clear(D, a_n_dim, n_vec, CFL_SIZE);
		md_zadd2(D, a_weight_1_dim, a_n_str, n_vec, a_n_str, n_vec, a_weight_1_str, weight_vec);
		md_zdiv2(D, a_weight_1_dim, a_weight_1_str, weight_vec, a_weight_1_str, weight_vec, a_n_str, n_vec);

		// patch center x weight
		md_zfmac2(D, a_weight_1_dim, a_ostr2, &MD_ACCESS(D, a_ostrs, im_pos, optr), a_weight_1_str, weight_vec, a_rstrs, &MD_ACCESS(D, a_rstrs, im_pos, r_patchcenter));
	};

	md_parallel_loop(D, fdim, parallel ? ~0UL : 0UL, znlmeans_core);


	md_free(x);
	md_free(r_in);
	md_free(weight_full_vec);
	md_free(n_full_vec);
	md_free(w_weight);
}

void md_znlmeans(int D, const long dim[D], unsigned long flags,
		complex float* optr, const complex float* iptr,
		long patch_length, long patch_dist, float h, float a)
{
	md_znlmeans2(D, dim, flags,
			MD_STRIDES(D, dim, CFL_SIZE), optr,
			MD_STRIDES(D, dim, CFL_SIZE), iptr,
			patch_length, patch_dist, h, a);
}

/*
 * optr[i,...,2d-1] = iptr[i] - iptr[i-d]
 */
void md_znlmeans_distance2(int D, const long idim[D], int xD,
		const long odim[xD], unsigned long flags,
		const long ostrs[xD], complex float* optr,
		const long istrs[D], const complex float* iptr)
{
	assert(xD > D);

	int flag_count = 0;
	long loop_idx[D];
	long r_patchstart_offset[D];
	long ipos[D];
	long xpos[xD];
	unsigned long xflags = 0;

	for (int i = 0; i < D; i++) {

		loop_idx[i] = 0;
		r_patchstart_offset[i] = 0;

		if (MD_IS_SET(flags, i)) {

			assert(xD > D + flag_count);
			assert(1 == odim[D + flag_count] % 2);

			long patch_dist = (odim[D + flag_count] - 1) / 2;

			assert(2 * patch_dist == idim[i] - odim[i]);

			loop_idx[flag_count] = i;
			r_patchstart_offset[i] = patch_dist;

			xflags = MD_SET(xflags, D + flag_count);
			++flag_count;

		} else {

			assert(odim[i] == idim[i]);
		}
	}

	assert(flag_count == (xD - D));

	const complex float* r_patchstart = &MD_ACCESS(D, istrs, r_patchstart_offset, iptr);

	md_set_dims(D, ipos, 0);
	md_set_dims(xD, xpos, 0);

	do {

		for (int i = 0; i < flag_count; i++)
			ipos[loop_idx[i]] = xpos[i + D] - (odim[D + i] - 1) / 2;

		md_zsub2(D, odim, ostrs, &MD_ACCESS(xD, ostrs, xpos, optr),
				istrs, r_patchstart,
				istrs, &MD_ACCESS(D, istrs, ipos, r_patchstart));

	} while (md_next(xD, odim, xflags, xpos));
}


void md_znlmeans_distance(int D, const long idim[D], int xD,
		const long odim[xD], unsigned long flags,
		complex float* optr, const complex float* iptr)
{
	md_znlmeans_distance2(D, idim, xD, odim, flags,
			MD_STRIDES(xD, odim, CFL_SIZE), optr,
			MD_STRIDES(D, idim, CFL_SIZE), iptr);
}




