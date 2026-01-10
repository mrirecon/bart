/* Copyright 2014. The Regents of the University of California.
 * Copyright 2025. Graz University of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013 Martin Uecker
 * 2025 Moritz BLumenthal
 *
 * Inati SJ, Hansen MS, Kellman P.
 * A solution to the phase problem in adaptive coil combination.
 * In: ISMRM proceeding; April; Salt Lake City, Utah, USA; 2013. 2672.
 */


#include <complex.h>
#include <math.h>
#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/linalg.h"

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "calib/calmat.h"

#include "walsh.h"


void walsh(const long bsize[3], const long dims[DIMS], complex float* sens, const long caldims[DIMS], const complex float* data)
{
	assert(1 == caldims[MAPS_DIM]);
	assert(1 == dims[MAPS_DIM]);

	int channels = caldims[COIL_DIM];
	int cosize = channels * (channels + 1) / 2;
	assert(dims[COIL_DIM] == cosize);

	long dims1[DIMS];
	md_copy_dims(DIMS, dims1, dims);
	dims1[COIL_DIM] = channels;

	long kdims[4];
	kdims[0] = MIN(bsize[0], dims[0]);
	kdims[1] = MIN(bsize[1], dims[1]);
	kdims[2] = MIN(bsize[2], dims[2]);

	md_resize_center(DIMS, dims1, sens, caldims, data, CFL_SIZE);
	ifftc(DIMS, dims1, FFT_FLAGS, sens, sens);

	long odims[DIMS];
	md_copy_dims(DIMS, odims, dims1);

	for (int i = 0; i < 3; i++)
		odims[i] = dims[i] + kdims[i] - 1;

	complex float* tmp = md_alloc(DIMS, odims, CFL_SIZE);
#if 0
	md_resizec(DIMS, odims, tmp, dims1, sens, CFL_SIZE);
#else
	long cen[DIMS] = { };

	for (int i = 0; i < 3; i++)
		cen[i] = (odims[i] - dims[i] + 1) / 2;

	complex float* tmp1 = md_alloc(DIMS, odims, CFL_SIZE);
	md_circ_ext(DIMS, odims, tmp1, dims1, sens, CFL_SIZE);
//	md_resize(DIMS, odims, tmp1, dims1, sens, CFL_SIZE);
	md_circ_shift(DIMS, odims, cen, tmp, tmp1, CFL_SIZE);
	md_free(tmp1);
#endif

	long calmat_dims[2];
	complex float* cm = calibration_matrix(calmat_dims, kdims, odims, tmp);
	md_free(tmp);

	int xh = dims[0];
	int yh = dims[1];
	int zh = dims[2];

	int pixels = calmat_dims[1] / channels;

#pragma omp parallel for
	for (int k = 0; k < zh; k++) {

		complex float in[channels][pixels];
		complex float cov[cosize];

		for (int j = 0; j < yh; j++) {
			for (int i = 0; i < xh; i++) {

				for (int c = 0; c < channels; c++)
					for (int p = 0; p < pixels; p++)
						in[c][p] = cm[((((c * pixels + p) * zh) + k) * yh + j) * xh + i];

				gram_matrix2(channels, cov, pixels, in);

				for (int l = 0; l < cosize; l++)
					sens[(((l * zh) + k) * yh + j) * xh + i] = cov[l];
			}
		}
	}
}


/**
 * Computes walsh calibration matrix but sums over the shifted blocks
 */
static void avg_walsh_matrix(const long bsize[3], const long dims[DIMS], complex float* sens, const long caldims[DIMS], const complex float* data)
{
	assert(1 == dims[MAPS_DIM]);
	int channels = caldims[COIL_DIM];

	long dims1[DIMS];
	md_copy_dims(DIMS, dims1, dims);
	dims1[COIL_DIM] = channels;

	long kdims[4];
	kdims[0] = MIN(bsize[0], dims[0]);
	kdims[1] = MIN(bsize[1], dims[1]);
	kdims[2] = MIN(bsize[2], dims[2]);

	md_resize_center(DIMS, dims1, sens, caldims, data, CFL_SIZE);
	ifftc(DIMS, dims1, FFT_FLAGS, sens, sens);

	long odims[DIMS];
	md_copy_dims(DIMS, odims, dims1);

	for (int i = 0; i < 3; i++)
		odims[i] = dims[i] + kdims[i] - 1;

	complex float* tmp = md_alloc(DIMS, odims, CFL_SIZE);

	long cen[DIMS] = { };

	for (int i = 0; i < 3; i++)
		cen[i] = (odims[i] - dims[i] + 1) / 2;

	complex float* tmp1 = md_alloc(DIMS, odims, CFL_SIZE);
	md_circ_ext(DIMS, odims, tmp1, dims1, sens, CFL_SIZE);
	md_circ_shift(DIMS, odims, cen, tmp, tmp1, CFL_SIZE);
	md_free(tmp1);

	long tdims[2 * DIMS];
	md_singleton_dims(2 * DIMS, tdims);
	md_copy_dims(DIMS, tdims, dims);
	md_copy_dims(3, tdims + DIMS, kdims);

	long tostrs[2 * DIMS];
	long tistrs[2 * DIMS];

	md_calc_strides(DIMS, tostrs, tdims, CFL_SIZE);
	md_singleton_strides(DIMS, tostrs + DIMS);
	md_calc_strides(DIMS, tistrs, odims, CFL_SIZE);
	md_calc_strides(DIMS, tistrs + DIMS, odims, CFL_SIZE);

	md_clear(DIMS, tdims, sens, CFL_SIZE);
	md_zadd2(2 * DIMS, tdims, tostrs, sens, tostrs, sens, tistrs, tmp);
	md_free(tmp);
}

/**
 * Phase normalization is achieved using the average of the left singular vector of the walsh calibration matrix (Inati SJ et al.).
 * We assume the coils are the right singular vector, so we obtain left singular vectors by multiplication of coils with the
 * calibration matrix. We only need the average of the blocks (which can be precomputed).
 */
void phase_normalization(const long bsize[3], const long dims[DIMS], complex float* sens, const long caldims[DIMS], const complex float* data)
{
	long ndims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, ndims, dims);

	complex float* tmp = md_alloc(DIMS, ndims, CFL_SIZE);
	avg_walsh_matrix(bsize, ndims, tmp, caldims, data);

	long pdims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, pdims, dims);
	complex float* phase = md_alloc(DIMS, ndims, CFL_SIZE);

	md_ztenmulc(DIMS, pdims, phase, ndims, tmp, dims, sens);
	md_free(tmp);

	md_zphsr(DIMS, pdims, phase, phase);
	md_zmul2(DIMS, dims, MD_STRIDES(DIMS, dims, CFL_SIZE), sens, MD_STRIDES(DIMS, dims, CFL_SIZE), sens, MD_STRIDES(DIMS, pdims, CFL_SIZE), phase);

	md_free(phase);
}


