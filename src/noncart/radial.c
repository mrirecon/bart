/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <assert.h>
#include <complex.h>
#include <math.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "radial.h"

void traj_radial_angles(int N, float angles[N], const long tdims[DIMS], const complex float* traj)
{
	assert(N == tdims[2]);

	long tdims1[DIMS];
	md_select_dims(DIMS, ~MD_BIT(1), tdims1, tdims);

	complex float* traj1 = md_alloc(DIMS, tdims1, CFL_SIZE);

	md_slice(DIMS, MD_BIT(1), (long[DIMS]){ }, tdims, traj1, traj, CFL_SIZE);

	for (int i = 0; i < N; i++)
		angles[i] = M_PI + atan2f(crealf(traj1[3 * i + 0]), crealf(traj1[3 * i + 1]));

	md_free(traj1);
}



float traj_radial_dcshift(const long tdims[DIMS], const complex float* traj)
{
	long tdims1[DIMS];
	md_select_dims(DIMS, ~MD_BIT(1), tdims1, tdims);

	complex float* traj1 = md_alloc(DIMS, tdims1, CFL_SIZE);
	// Extract what would be the DC component in Cartesian sampling

	md_slice(DIMS, MD_BIT(1), (long[DIMS]){ [1] = tdims[1] / 2 }, tdims, traj1, traj, CFL_SIZE);

	NESTED(float, dist, (int i))
	{
		return sqrtf(powf(crealf(traj1[3 * i + 0]), 2.) + powf(crealf(traj1[3 * i + 1]), 2.));
	};

	float dc_shift = dist(0);

	for (int i = 0; i < tdims[2]; i++)
		if (fabsf(dc_shift - dist(i)) > 0.0001)
			debug_printf(DP_WARN, "Inconsistently shifted spoke: %d %f != %f\n", i, dist(i), dc_shift);

	md_free(traj1);

	return dc_shift;
}


float traj_radial_dk(const long tdims[DIMS], const complex float* traj)
{
	long tdims1[DIMS];
	md_select_dims(DIMS, ~MD_BIT(1), tdims1, tdims);

	complex float* traj1 = md_alloc(DIMS, tdims1, CFL_SIZE);
	// Extract what would be the DC component in Cartesian sampling

	md_slice(DIMS, MD_BIT(1), (long[DIMS]){ [1] = tdims[1] / 2 }, tdims, traj1, traj, CFL_SIZE);

	NESTED(float, dist, (int i))
	{
		return sqrtf(powf(crealf(traj1[3 * i + 0]), 2.) + powf(crealf(traj1[3 * i + 1]), 2.));
	};

	float dc_shift = dist(0);

	md_slice(DIMS, MD_BIT(1), (long[DIMS]){ [1] = tdims[1] / 2 + 1 }, tdims, traj1, traj, CFL_SIZE);

	float shift1 = dist(0) - dc_shift;

	md_free(traj1);

	return shift1;
}

