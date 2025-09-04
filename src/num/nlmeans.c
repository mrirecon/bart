/* Copyright 2023 - 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <assert.h>

#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlmeans.h"


/* non-local means filter
 *
 * Buades, A, Coll B, Morel J-M. A non-local algorithm for image denoising.
 * IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05); 2005.
 */
void md_znlmeans2(int D, const long dims[D], unsigned long flags,
		const long ostrs[D], complex float* optr,
		const long istrs[D], const complex float* iptr,
		long patch_length, long patch_dist, float h, float a)
{
	assert(1 == patch_length % 2);

	int flag_count = bitcount(flags);
	int xD = D + flag_count;
	long dist_size = 2 * patch_dist + 1; // number of pixels along one dim that will be used for the mean
	long padding_img = 2 * (patch_dist + patch_length / 2); // size difference input img vs input for sliding differences, along flagged dims
	long padding_diff = 2 * (patch_length / 2); // size difference input img vs result of sliding differences


	long weight_dims[xD + flag_count]; // size of weight-space: values with which pixels will be weighted.
					   // [0..D] pixel for which mean is calculated (output pixel)
					   // [D..D+flag_count] pixel to which this weight applies. Relative offset to output pixel in flagged dims. (input pixel)
					   // [D+flag_count..D+2*flag_count] neighborhood / patch around input pixel, in flagged dims. Used with patch_strides, not allocated
	md_copy_dims(D, weight_dims, dims);
	md_set_dims(flag_count, weight_dims + D, dist_size);
	md_set_dims(flag_count, weight_dims + xD, patch_length);

	long weight_strs[xD + flag_count];
	md_calc_strides(xD, weight_strs, weight_dims, CFL_SIZE);
	md_set_dims(flag_count, weight_strs + xD, 0); // weights = accumulated patches, thus stride = 0 in the last dims


	long* kernel_dims = weight_dims + xD; // size of gaussian pdf for weighted euclidean norm of patches
	long kernel_strs[xD + flag_count];
	md_set_dims(xD, kernel_strs, 0);
	md_calc_strides(flag_count, kernel_strs + xD, kernel_dims, CFL_SIZE);


	long pad_dims[D]; // size of padded image
	for (int i = 0; i < D; i++)
		pad_dims[i] = MD_IS_SET(flags, i) ? dims[i] + padding_img : dims[i];

	long pad_strs[xD];
	md_set_dims(xD, pad_strs, 0);
	md_calc_strides(D, pad_strs, pad_dims, CFL_SIZE);

	for (int i = 0, j = 0; i < D; i++)
		if (MD_IS_SET(flags, i))
			pad_strs[D + (j++)] = pad_strs[i];


	long diff_dims[xD]; // size of difference images
	for (int i = 0; i < D; i++)
		diff_dims[i] = MD_IS_SET(flags, i) ? dims[i] + padding_diff : dims[i] ;

	md_set_dims(flag_count, diff_dims + D, dist_size);

	long diff_strs[xD];
	md_calc_strides(xD, diff_strs, diff_dims, CFL_SIZE);


	long patch_strs[xD + flag_count]; // another view on difference images: difference of patches of own and neighbouring pixels
	md_copy_strides(xD, patch_strs, diff_strs);
	for (int i = 0, j = 0; i < D; i++)
		if (MD_IS_SET(flags, i))
			patch_strs[xD + (j++)] = diff_strs[i];


	long r_patchcenter_offset[D]; // index to center of first patch
	for (int i = 0; i < D; i++)
		r_patchcenter_offset[i] = MD_IS_SET(flags, i) ?  patch_length / 2 : 0;


	long ostr2[xD]; // extended output strides for accumulation into output
	md_set_dims(xD, ostr2, 0);
	md_copy_strides(D, ostr2, ostrs);


	complex float* diff = md_alloc_sameplace(xD, diff_dims, CFL_SIZE, iptr);
	complex float* weight = md_alloc_sameplace(xD, weight_dims, CFL_SIZE, iptr);
	complex float* kernel = md_alloc_sameplace(flag_count, kernel_dims, CFL_SIZE, iptr);
	complex float* padded = md_alloc_sameplace(D, pad_dims, CFL_SIZE, iptr);

	// pad input
	md_reflectpad_center2(D, pad_dims, pad_strs, padded, dims, istrs, iptr, CFL_SIZE);

	// gauss kernel for distance-weighted euclidean norm
	md_zgausspdf(flag_count, kernel_dims, kernel, a * a);

	// compute difference images: diff(N(i)) - diff(N(j)), diff_dims[0:D] stores i,  diff_dims[D:xD] stores (i-j)
	md_znlmeans_distance2(D, pad_dims, xD, diff_dims, flags, diff_strs, diff, pad_strs, padded);
	// square differences
	md_zmulc2(xD, diff_dims, diff_strs, diff, diff_strs, diff, diff_strs, diff);

	// weighted sum of squared differences (convolution in original paper; but kernel is symmetric, thus equivalent to pointwise multiplication & summation)
	md_clear(xD, weight_dims, weight, CFL_SIZE);
	md_zfmac2(xD + flag_count, weight_dims, weight_strs, weight, patch_strs, diff, kernel_strs, kernel);

	// exponential
	float var = -1. / (2. * h * h) / md_znorm(flag_count, kernel_dims, kernel);
	md_zsmul2(xD, weight_dims, weight_strs, weight, weight_strs, weight, var);
	md_zexp2(xD, weight_dims, weight_strs, weight, weight_strs, weight);

	// normalize
	md_clear2(D, dims, ostrs, optr, CFL_SIZE);
	md_zadd2(xD, weight_dims, ostr2, optr, ostr2, optr, weight_strs, weight);
	md_zdiv2(xD, weight_dims, weight_strs, weight, weight_strs, weight, ostr2, optr);

	// weighted average of patch centers
	md_clear2(D, dims, ostrs, optr, CFL_SIZE);
	md_zfmac2(xD, weight_dims, ostr2, optr, weight_strs, weight, pad_strs, &MD_ACCESS(D, pad_strs, r_patchcenter_offset, padded));


	md_free(diff);
	md_free(padded);
	md_free(weight);
	md_free(kernel);
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
	int flag_count = bitcount(flags);

	assert(xD > D);
	assert(xD - D == flag_count);
	assert(flags < MD_BIT(D));

	long ioffset[D];
	md_set_dims(D, ioffset, 0);

	long istrs_moving[xD];
	md_copy_strides(D, istrs_moving, istrs);

	long istrs_fix[xD];
	md_copy_strides(D, istrs_fix, istrs);

	for (int i = 0, j = 0; i < D; i++) {

		if (MD_IS_SET(flags, i)) {

			istrs_moving[D + j] = istrs[i];
			istrs_fix[D + j] = 0;

			int diff = idim[i] - odim[i];
			assert(diff + 1 == odim[D + i]);
			assert(0 == diff % 2);

			ioffset[i] = diff / 2;
			j++;
		}
	}

	md_zsub2(xD, odim, ostrs, optr, istrs_fix, &MD_ACCESS(D, istrs, ioffset, iptr), istrs_moving, iptr);
}

void md_znlmeans_distance(int D, const long idim[D], int xD,
		const long odim[xD], unsigned long flags,
		complex float* optr, const complex float* iptr)
{
	md_znlmeans_distance2(D, idim, xD, odim, flags,
			MD_STRIDES(xD, odim, CFL_SIZE), optr,
			MD_STRIDES(D, idim, CFL_SIZE), iptr);
}

