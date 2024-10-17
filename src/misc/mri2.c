/* Copyright 2013. The Regents of the University of California.
 * Copyright 2022. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013-2022 Martin Uecker <uecker@tugraz.at>
 */

#include <complex.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/loop.h"
#include "num/vptr.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "sense/optcom.h"

#include "mri2.h"



void data_consistency(const long dims[DIMS], complex float* dst, const complex float* pattern, const complex float* kspace1, const complex float* kspace2)
{
	assert(1 == dims[MAPS_DIM]);

	long strs[DIMS];
	long dims1[DIMS];
	long strs1[DIMS];

	md_select_dims(DIMS, ~COIL_FLAG, dims1, dims);
	md_calc_strides(DIMS, strs1, dims1, CFL_SIZE);
	md_calc_strides(DIMS, strs, dims, CFL_SIZE);

	complex float* tmp = md_alloc_sameplace(DIMS, dims, CFL_SIZE, dst);
	md_zmul2(DIMS, dims, strs, tmp, strs, kspace2, strs1, pattern);
	md_zsub(DIMS, dims, tmp, kspace2, tmp);
	md_zfmac2(DIMS, dims, strs, tmp, strs, kspace1, strs1, pattern);
	md_copy(DIMS, dims, dst, tmp, CFL_SIZE);
	md_free(tmp);
}




void estimate_pattern(int D, const long dims[D], unsigned long flags, complex float* pattern, const complex float* kspace_data)
{
	md_zrss(D, dims, flags, pattern, kspace_data);

	long dims2[D];
	long strs2[D];
	md_select_dims(D, ~flags, dims2, dims);
	md_calc_strides(D, strs2, dims2, CFL_SIZE);

	long strs1[D];
	md_singleton_strides(D, strs1);

	complex float* tmp = md_alloc_sameplace(D, dims2, CFL_SIZE, kspace_data);
	md_zfill(D, dims2, tmp, 0.);
	md_zcmp2(D, dims2, strs2, pattern, strs2, pattern, strs2, tmp);

	md_zfill(D, dims2, tmp, 1.);
	md_zsub2(D, dims2, strs2, pattern, strs2, tmp, strs2, pattern);

	md_free(tmp);
}


static void calib_readout_pos(const long caldims[DIMS], long calpos[DIMS], const long in_dims[DIMS], const complex float* in_data)
{
	// now move along readout to find maximum energy

	long in_strs[DIMS];
	md_calc_strides(DIMS, in_strs, in_dims, CFL_SIZE);

	int maxind = 0;
	float maxeng = 0.;

	for (int r = 0; r < in_dims[READ_DIM] - caldims[READ_DIM] + 1; r++) {

		calpos[READ_DIM] = r;

		long offset = md_calc_offset(DIMS, calpos, in_strs);
		float energy = md_znorm2(DIMS, caldims, in_strs, in_data + offset / (long)CFL_SIZE);

		if (energy > maxeng) {

			maxind = r;
			maxeng = energy;
		}
	}

	calpos[READ_DIM] = maxind;
}


void calib_geom(long caldims[DIMS], long calpos[DIMS], const long calsize[3], const long in_dims[DIMS], const complex float* in_data)
{
	long pat_dims[DIMS];

	assert(1 == in_dims[MAPS_DIM]);

	md_select_dims(DIMS, ~COIL_FLAG, pat_dims, in_dims);

	complex float* pattern = md_alloc_sameplace(DIMS, pat_dims, CFL_SIZE, in_data);
	estimate_pattern(DIMS, in_dims, COIL_FLAG, pattern, in_data);

	for (int i = 0; i < DIMS; i++)
		caldims[i] = 1;

	for (int i = 0; i < DIMS; i++)
		calpos[i] = 0;

	calpos[0] = (in_dims[0] - caldims[0]) / 2;
	calpos[1] = (in_dims[1] - caldims[1]) / 2;
	calpos[2] = (in_dims[2] - caldims[2]) / 2;



	long pat_strs[DIMS];
	md_calc_strides(DIMS, pat_strs, pat_dims, CFL_SIZE);

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

			long offset = md_calc_offset(DIMS, calpos, pat_strs);
			float si = sqrtf((float)caldims[0] * (float)caldims[1] * (float)caldims[2]);

			if (si != md_znorm2(DIMS, caldims, pat_strs, pattern + offset / (long)CFL_SIZE)) {

				caldims[i]--;
				calpos[i] = (in_dims[i] - caldims[i]) / 2;
				stop[i] = true;
			}
		}
	}

	caldims[COIL_DIM] = in_dims[COIL_DIM];
	md_free(pattern);

#if 1
	calib_readout_pos(caldims, calpos, in_dims, in_data);
#endif
}



complex float* extract_calib2(long caldims[DIMS], const long calsize[3], const long in_dims[DIMS], const long in_strs[DIMS], const complex float* in_data, bool fixed)
{
	// first extract center of size in_dims[0], calsize[1], calsize[2], and then process further to save time

	long tmp_dims[DIMS];
	long tmp_pos[DIMS];
	long tmp_strs[DIMS];

	md_copy_dims(DIMS, tmp_dims, in_dims);
	md_set_dims(DIMS, tmp_pos, 0);

	for (int i = 0; i < 3; i++) {

		//tmp_dims[i] = MIN(calsize[i], in_dims[i]);
		tmp_dims[i] = (READ_DIM == i) ? in_dims[i] : MIN(calsize[i], in_dims[i]);
		tmp_pos[i] = (in_dims[i] - tmp_dims[i]) / 2.; // what about odd sizes?
	}

	complex float* tmp_data = md_alloc_sameplace(DIMS, tmp_dims, CFL_SIZE, in_data);

	md_calc_strides(DIMS, tmp_strs, tmp_dims, CFL_SIZE);

	md_copy_block2(DIMS, tmp_pos, tmp_dims, tmp_strs, tmp_data, in_dims, in_strs, in_data, CFL_SIZE);

	long calpos[DIMS];
	calib_geom(caldims, calpos, calsize, tmp_dims, tmp_data);

	if (fixed) { // we should probably change calib_geom instead

		for (int i = 0; i < 3; i++) {

			caldims[i] = MIN(calsize[i], tmp_dims[i]);

				calpos[i] = (tmp_dims[i] - caldims[i]) / 2;
		}
	}

	debug_printf(DP_DEBUG1, "Calibration region...  (size: %ldx%ldx%ld, pos: %ldx%ldx%ld)\n",
				caldims[0], caldims[1], caldims[2], calpos[0] + tmp_pos[0], calpos[1] + tmp_pos[1], calpos[2] + tmp_pos[2]);

	complex float* cal_data = md_alloc_sameplace(DIMS, caldims, CFL_SIZE, tmp_data);

	md_copy_block(DIMS, calpos, caldims, cal_data, tmp_dims, tmp_data, CFL_SIZE);
	md_free(tmp_data);

	return cal_data;
}


complex float* extract_calib(long caldims[DIMS], const long calsize[3], const long in_dims[DIMS], const complex float* in_data, bool fixed)
{
	long in_strs[DIMS];
	md_calc_strides(DIMS, in_strs, in_dims, CFL_SIZE);
	return extract_calib2(caldims, calsize, in_dims, in_strs, in_data, fixed);
}


/**
 * Estimate image dimensions from trajectory
 */
void estimate_im_dims(int N, unsigned long flags, long dims[N], const long tdims[N], const complex float* traj)
{
	if (is_vptr(traj)) {

		complex float* traj_cpu = md_alloc(N, tdims, CFL_SIZE);
		md_copy(N, tdims, traj_cpu, traj, CFL_SIZE);
		estimate_im_dims(N, flags, dims, tdims, traj_cpu);
		md_free(traj_cpu);
		return;
	}

	int T = tdims[0];

	assert(T == bitcount(flags));

	float max_dims[T];
	for (int i = 0; i < T; i++)
		max_dims[i] = 0.;

	for (long i = 0; i < md_calc_size(N - 1, tdims + 1); i++)
		for(int j = 0; j < tdims[0]; j++)
			max_dims[j] = MAX(cabsf(traj[j + tdims[0] * i]), max_dims[j]);

	for (int j = 0, t = 0; j < N; j++) {

		dims[j] = 1;

		if (MD_IS_SET(flags, j)) {

			dims[t] = (0. == max_dims[t]) ? 1 : (2 * ceilf(max_dims[t]));
			t++;
		}
	}
}

/**
 * Estimate fast square image dimensions from trajectory
 */
void estimate_fast_sq_im_dims(int N, long dims[3], const long tdims[N], const complex float* traj)
{
	float max_dims[3] = { 0., 0., 0. };

	for (long i = 0; i < md_calc_size(N - 1, tdims + 1); i++)
		for(int j = 0; j < 3; j++)
			max_dims[j] = MAX(cabsf(traj[j + tdims[0] * i]), max_dims[j]);


	// 2* is needed since we take the absolute value of the trajectory above, and it is scaled from
	// -DIM/2 to DIM/2
	long max_square = 2 * MAX(MAX(max_dims[0], max_dims[1]), max_dims[2]);


	// compute next fast size for Fourier transform.
	// That is the next number only composed of small prime factors,
	// i.e. 2, 3, 5 (and possibly 7?)

	// to avoid an infinite loop here, we constrain our search
	long fast_size = max_square;

	for ( ; fast_size <= 4 * max_square; ++fast_size) {

		long n = fast_size;

		while (0 == n % 2l) { n /= 2l; }
		while (0 == n % 3l) { n /= 3l; }
		while (0 == n % 5l) { n /= 5l; }
		while (0 == n % 7l) { n /= 7l; }

		if (n <= 1)
			break;
	}

	for (int j = 0; j < 3; j++)
		dims[j] = (0. == max_dims[j]) ? 1 : fast_size;
}

