/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <complex.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "sense/optcom.h"

#include "mri.h"




void data_consistency(const long dims[KSPACE_DIMS], complex float* dst, const complex float* pattern, const complex float* kspace1, const complex float* kspace2)
{
	assert(1 == dims[MAPS_DIM]);

	long strs[KSPACE_DIMS];
	long dims1[KSPACE_DIMS];
	long strs1[KSPACE_DIMS];

	md_select_dims(KSPACE_DIMS, ~COIL_FLAG, dims1, dims);
	md_calc_strides(KSPACE_DIMS, strs1, dims1, sizeof(complex float));
	md_calc_strides(KSPACE_DIMS, strs, dims, sizeof(complex float));

	complex float* tmp = md_alloc_sameplace(KSPACE_DIMS, dims, sizeof(complex float), dst);
	md_zmul2(KSPACE_DIMS, dims, strs, tmp, strs, kspace2, strs1, pattern);
	md_zsub(KSPACE_DIMS, dims, tmp, kspace2, tmp);
	md_zfmac2(KSPACE_DIMS, dims, strs, tmp, strs, kspace1, strs1, pattern);
	md_copy(KSPACE_DIMS, dims, dst, tmp, sizeof(complex float));
	md_free(tmp);
}





/**
 * Default transfer function. dst = src .* pattern
 *
 * @param _data transfer function data
 * @param pattern sampling pattern
 * @param dst destination pointer
 * @param src source pointer
 */
void transfer_function(void* _data, const complex float* pattern, complex float* dst, const complex float* src)
{
	struct transfer_data_s* data = _data;
	md_zmul2(DIMS, data->dims, data->strs, dst, data->strs, src, data->strs_tf, pattern);
}













void estimate_pattern(unsigned int D, const long dims[D], unsigned int dim, complex float* pattern, const complex float* kspace_data)
{
	md_rss(D, dims, (1u << dim), pattern, kspace_data);

	long dims2[D];
	long strs2[D];
	assert(dim < D);
	md_select_dims(D, ~(1u << dim), dims2, dims);
	md_calc_strides(D, strs2, dims2, sizeof(complex float));

	long strs1[D];
	md_singleton_strides(D, strs1);
	complex float val = 0.;

	md_zcmp2(D, dims2, strs2, pattern, strs2, pattern, strs1, &val);

	val = 1.;
	md_zsub2(D, dims2, strs2, pattern, strs1, &val, strs2, pattern);
}




void calib_geom(long caldims[KSPACE_DIMS], long calpos[KSPACE_DIMS], const long calsize[3], const long in_dims[KSPACE_DIMS], const complex float* in_data)
{
	long pat_dims[KSPACE_DIMS];

	assert(1 == in_dims[MAPS_DIM]);

	md_select_dims(KSPACE_DIMS, ~COIL_FLAG, pat_dims, in_dims);
	
	complex float* pattern = md_alloc(KSPACE_DIMS, pat_dims, sizeof(complex float));
	estimate_pattern(KSPACE_DIMS, in_dims, COIL_DIM, pattern, in_data);

	for (unsigned int i = 0; i < KSPACE_DIMS; i++)
		caldims[i] = 1;

	for (unsigned int i = 0; i < KSPACE_DIMS; i++)
		calpos[i] = 0;

	calpos[0] = (in_dims[0] - caldims[0]) / 2;
	calpos[1] = (in_dims[1] - caldims[1]) / 2;
	calpos[2] = (in_dims[2] - caldims[2]) / 2;



	long pat_strs[KSPACE_DIMS];
	md_calc_strides(KSPACE_DIMS, pat_strs, pat_dims, sizeof(complex float));

	bool stop[3] = { false, false, false };

	// increase in diagonals first

	while (!(stop[0] && stop[1] & stop[2])) {

		for (int i = 0; i < 3; i++) {

	 		if (caldims[i] == in_dims[i])
				stop[i] = true;

			if (caldims[i] >= calsize[i])
				stop[i] = true;

			if (stop[i])
				continue;

			caldims[i] += 1;
			calpos[i] = (in_dims[i] - caldims[i]) / 2;

		//	printf("Try: %ld %ld %ld %ld\n", caldims[1], caldims[2], calpos[1], calpos[2]);

			long offset = md_calc_offset(KSPACE_DIMS, calpos, pat_strs);
			float si = sqrtf((float)caldims[0] * (float)caldims[1] * (float)caldims[2]);
		
			if (si != md_znorm2(KSPACE_DIMS, caldims, pat_strs, pattern + offset / sizeof(complex float))) {
		
				caldims[i]--;
				calpos[i] = (in_dims[i] - caldims[i]) / 2;
				stop[i] = true;
			}
		}
	}

	caldims[COIL_DIM] = in_dims[COIL_DIM];
	md_free(pattern);

#if 1
	// now move along readout to find maximum energy

	long in_strs[KSPACE_DIMS];
	md_calc_strides(KSPACE_DIMS, in_strs, in_dims, sizeof(complex float));

	int maxind = 0;
	float maxeng = 0.;

	for (int r = 0; r < in_dims[READ_DIM] - caldims[READ_DIM] + 1; r++) {

		calpos[READ_DIM] = r;

		long offset = md_calc_offset(KSPACE_DIMS, calpos, in_strs);
		float energy = md_znorm2(KSPACE_DIMS, caldims, in_strs, in_data + offset / sizeof(complex float));

		if (energy > maxeng) {

			maxind = r;
			maxeng = energy;
		}
	}

	calpos[READ_DIM] = maxind;
#endif
}



complex float* extract_calib2(long caldims[KSPACE_DIMS], const long calsize[3], const long in_dims[KSPACE_DIMS], const long in_strs[KSPACE_DIMS], const complex float* in_data, bool fixed)
{
	// first extract center of size in_dims[0], calsize[1], calsize[2], and then process further to save time

	long tmp_dims[KSPACE_DIMS];
	long tmp_pos[KSPACE_DIMS];
	long tmp_strs[KSPACE_DIMS];

	md_copy_dims(KSPACE_DIMS, tmp_dims, in_dims);
	md_set_dims(KSPACE_DIMS, tmp_pos, 0);

	for (unsigned int i = 0; i < 3; i++) {

		//tmp_dims[i] = MIN(calsize[i], in_dims[i]);
		tmp_dims[i] = (READ_DIM == i) ? in_dims[i] : MIN(calsize[i], in_dims[i]);
		tmp_pos[i] = (in_dims[i] - tmp_dims[i]) / 2.; // what about odd sizes?
	}

	complex float* tmp_data = md_alloc(KSPACE_DIMS, tmp_dims, sizeof(complex float));

	md_calc_strides(KSPACE_DIMS, tmp_strs, tmp_dims, sizeof(complex float));

	md_copy_block2(KSPACE_DIMS, tmp_pos, tmp_dims, tmp_strs, tmp_data, in_dims, in_strs, in_data, sizeof(complex float));

	long calpos[KSPACE_DIMS];
	calib_geom(caldims, calpos, calsize, tmp_dims, tmp_data);

	if (fixed) { // we should probably change calib_geom instead

		for (unsigned int i = 0; i < 3; i++) {

			caldims[i] = MIN(calsize[i], tmp_dims[i]);

			if (i != READ_DIM)
				calpos[i] = (tmp_dims[i] - caldims[i]) / 2;
		}
	}

	debug_printf(DP_DEBUG1, "Calibration region...  (size: %ldx%ldx%ld, pos: %ldx%ldx%ld)\n", 
				caldims[0], caldims[1], caldims[2], calpos[0] + tmp_pos[0], calpos[1] + tmp_pos[1], calpos[2] + tmp_pos[2]);

	complex float* cal_data = md_alloc(KSPACE_DIMS, caldims, sizeof(complex float));

	md_copy_block(KSPACE_DIMS, calpos, caldims, cal_data, tmp_dims, tmp_data, sizeof(complex float));
	md_free(tmp_data);

	return cal_data;
}


complex float* extract_calib(long caldims[KSPACE_DIMS], const long calsize[3], const long in_dims[KSPACE_DIMS], const complex float* in_data, bool fixed)
{
	long in_strs[KSPACE_DIMS];
	md_calc_strides(KSPACE_DIMS, in_strs, in_dims, sizeof(complex float));
	return extract_calib2(caldims, calsize, in_dims, in_strs, in_data, fixed);
}


complex float* compute_mask(unsigned int N, const long msk_dims[N], float restrict_fov)
{
	complex float* mask = md_alloc(KSPACE_DIMS, msk_dims, sizeof(complex float));

	long small_dims[KSPACE_DIMS] = { [0 ... KSPACE_DIMS - 1] = 1 };
	small_dims[0] = (1 == msk_dims[0]) ? 1 : (msk_dims[0] * restrict_fov);
	small_dims[1] = (1 == msk_dims[1]) ? 1 : (msk_dims[1] * restrict_fov);
	small_dims[2] = (1 == msk_dims[2]) ? 1 : (msk_dims[2] * restrict_fov);

	complex float* small_mask = md_alloc(KSPACE_DIMS, small_dims, sizeof(complex float));
	complex float one = 1.;

	md_fill(KSPACE_DIMS, small_dims, small_mask, &one, sizeof(complex float));
	md_resizec(KSPACE_DIMS, msk_dims, mask, small_dims, small_mask, sizeof(complex float));

	md_free(small_mask);

	return mask;
}



