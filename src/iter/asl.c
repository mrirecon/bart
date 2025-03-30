/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "num/ops.h"
#include "num/iovec.h"
#include "num/multind.h"

#include "linops/linop.h"
#include "linops/fmac.h"
#include "linops/someops.h"

#include "iter/tgv.h"

#include "misc/debug.h"
#include "misc/misc.h"

#include "asl.h"

/**
 * This function extracts the size of the separated control and label image
 * from an input array of dimensions.
 * 
 * @param N        Number of dimensions.
 * @param asl_dim  Dimension along which to separate control and label images.
 * @param asl_dims Array of size N to store the separated dimensions.
 * @param in_dims  Array of size N specifying the size of each input dimension.
 */
void get_asl_dims(int N, int asl_dim, long asl_dims[N], const long in_dims[N])
{
	assert(in_dims[asl_dim] == 2);
	md_copy_dims(N, asl_dims, in_dims);
	asl_dims[asl_dim] = 1;
}

/**
 * This function creates a linear operator that separates control and label images along the ASL
 * dimension specified by `asl_dim` and computes the difference image.
 * The ASL dimension must have a size of 2 (control and label image).
 * The returned operator has a size of 1 along the ASL dimension.
 *
 * @param N        Number of dimensions.
 * @param img_dims Array of size N specifying the size of each dimension.
 * @param asl_dim  Dimension along which to separate control and label images.
 * @return         Pointer to the created linear operator structure.
 */
const struct linop_s* linop_asl_create(int N, const long img_dims[N], int asl_dim)
{
	assert(img_dims[asl_dim] == 2);
	
	long asl_img_dims[N];
	get_asl_dims(N, asl_dim, asl_img_dims, img_dims);

	long tdims[N];
	md_singleton_dims(N, tdims);
	tdims[asl_dim] = 2;

	complex float tensor[2] = { 1.f, -1.f };

	return linop_fmac_dims_create(N, asl_img_dims, img_dims, tdims, tensor);
}


/**
 * This function extracts the size of the label image for TE-ASL
 * from an input array of dimensions.
 * 
 * @param N                Number of dimensions.
 * @param teasl_dim        Dimension along which to separate label and PWI images.
 * @param teasl_label_dims Array of size N to store the dimensions of the label image.
 * @param in_dims          Array of size N specifying the size of each input dimension.
 */
void get_teasl_label_dims(int N, int teasl_dim, long teasl_label_dims[N], const long in_dims[N])
{
	assert(1 < in_dims[teasl_dim]);

	md_copy_dims(N, teasl_label_dims, in_dims);
	teasl_label_dims[teasl_dim] = 1;
}


/**
 * This function extracts the size of the perfusion-weighted images (PWIs) for TE-ASL
 * from an input array of dimensions.
 * 
 * @param N                Number of dimensions.
 * @param teasl_dim        Dimension along which to separate label and PWI.
 * @param teasl_label_dims Array of size N to store the dimensions of the PWI.
 * @param in_dims          Array of size N specifying the size of each input dimension.
 */
void get_teasl_pwi_dims(int N, int teasl_dim, long teasl_pwi_dims[N], const long in_dims[N])
{
	assert(1 < in_dims[teasl_dim]);

	md_copy_dims(N, teasl_pwi_dims, in_dims);
	teasl_pwi_dims[teasl_dim]--;
}


/**
 * This function creates a linear operator that extracts the label image from the first position
 * of the specified teasl_dim dimension.
 * 
 * For regularization, the label needs to be separated from the PWI images. This is because the 
 * gradient from the label image to the first PWI image is very high and would influence the
 * regularization of the PWI images.
 *
 * @param N         Number of dimensions.
 * @param img_dims  Array of size N specifying the size of each dimension.
 * @param teasl_dim Dimension from which to extract the label image.
 * @return          Pointer to the created linear operator structure.
 */
const struct linop_s* linop_teasl_extract_label(int N, const long img_dims[N], int teasl_dim)
{
	assert(1 < img_dims[teasl_dim]);
	
	long label_img_dims[N];
	get_teasl_label_dims(N, teasl_dim, label_img_dims, img_dims);

	debug_print_dims(DP_DEBUG3, N, label_img_dims);

	long pos0[N];
	for (int i = 0; i < N; i++)
		pos0[i] = 0;
	pos0[teasl_dim] = 0;

	return linop_extract_create(N, pos0, label_img_dims, img_dims);
}


/**
 * This function creates a linear operator that extracts the perfusion-weighted image from the
 * 1-Nth positions of the specified teasl_dim dimension.
 * 
 * For regularization, the label needs to be separated from the PWI images. This is because the 
 * gradient from the label image to the first PWI image is very high and would influence the
 * regularization of the PWI images.
 *
 * @param N         Number of dimensions.
 * @param img_dims  Array of size N specifying the size of each dimension.
 * @param teasl_dim Dimension from which to extract the label image.
 * @return          Pointer to the created linear operator structure.
 */
const struct linop_s* linop_teasl_extract_pwi(int N, const long img_dims[N], int teasl_dim)
{
	assert(1 < img_dims[teasl_dim]);

	long pwi_img_dims[N];
	get_teasl_pwi_dims(N, teasl_dim, pwi_img_dims, img_dims);

	debug_print_dims(DP_DEBUG3, N, pwi_img_dims);

	long pos1[N];
	for (int i = 0; i < N; i++)
		pos1[i] = 0;
	pos1[teasl_dim] = 1;

	return linop_extract_create(N, pos1, pwi_img_dims, img_dims);
}

