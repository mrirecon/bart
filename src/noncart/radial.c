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


void traj_radial_angles(int N, const long adims[N], float* angles, const long tdims[N], const complex float* traj)
{
	assert(md_check_compat(N, MD_BIT(0) | MD_BIT(1), adims, tdims));
	long tdims1[N];
	md_select_dims(N, ~(MD_BIT(0) | MD_BIT(1)) , tdims1, tdims);

	complex float* x = md_alloc_sameplace(N, tdims1, CFL_SIZE, traj);
	complex float* y = md_alloc_sameplace(N, tdims1, CFL_SIZE, traj);

	long pos[N];
	md_set_dims(N, pos, 0);

	md_slice(N, MD_BIT(0) | MD_BIT(1), pos, tdims, x, traj, CFL_SIZE);

	pos[0] = 1;
	md_slice(N, MD_BIT(0) | MD_BIT(1), pos, tdims, y, traj, CFL_SIZE);

	complex float* cangles = md_alloc_sameplace(N, tdims1, CFL_SIZE, traj);
	md_zatan2r(N, tdims1, cangles, x, y);

	md_free(x);
	md_free(y);

#if 1
	//FIXME: float point precission leads to failing test tests/test-estdelay-ring
	for (int i = 0; i < md_calc_size(N, adims); i++)
		angles[i] = M_PI + crealf(cangles[i]);
#else
	md_zsadd(N, tdims1, cangles, cangles, M_PI);
	md_real(N, tdims1, angles, cangles);
#endif

	md_free(cangles);
}



float traj_radial_dcshift(int N, const long tdims[N], const complex float* traj)
{
	long tdims1[N];
	md_select_dims(N, ~MD_BIT(1), tdims1, tdims);

	complex float* traj1 = md_alloc_sameplace(N, tdims1, CFL_SIZE, traj);
	// Extract what would be the DC component in Cartesian sampling

	md_resize_center(N, tdims1, traj1, tdims, traj, CFL_SIZE);

	long sdims[N];
	md_select_dims(N, ~MD_BIT(0), sdims, tdims1);

	complex float* shift = md_alloc_sameplace(N, sdims, CFL_SIZE, traj);
	md_zrss(N, tdims1, MD_BIT(0), shift, traj1);
	md_free(traj1);

	float dc_shift = crealf(shift[0]);

	for (int i = 0; i < md_calc_size(N, sdims); i++)
		if (fabsf(dc_shift - crealf(shift[i])) > 0.0001)
			debug_printf(DP_WARN, "Inconsistently shifted spoke: %d %f != %f\n", i, crealf(shift[i]), dc_shift);

	md_free(shift);

	return dc_shift;
}


float traj_radial_deltak(int N, const long tdims[N], const complex float* traj)
{
	assert(1 < N);

	long dtdims[N];
	md_select_dims(N, ~MD_BIT(1), dtdims, tdims);

	complex float* dtraj = md_alloc_sameplace(N, dtdims, CFL_SIZE, traj);
	// Extract what would be the DC component in Cartesian sampling

	long pos1[N];
	long pos2[N];

	md_set_dims(N, pos1, 0);
	md_set_dims(N, pos2, 0);

	pos1[1] = tdims[1] / 2;
	pos2[1] = tdims[1] / 2 + 1;

	long tstrs[N];
	long dtstrs[N];

	md_calc_strides(N, tstrs, tdims, CFL_SIZE);
	md_calc_strides(N, dtstrs, dtdims, CFL_SIZE);

	md_zsub2(N, dtdims, dtstrs, dtraj, tstrs, &(MD_ACCESS(N, tstrs, pos1, traj)), tstrs, &(MD_ACCESS(N, tstrs, pos2, traj)));

	long adtdims[N];
	md_select_dims(N, ~MD_BIT(0), adtdims, dtdims);

	complex float* adtraj = md_alloc_sameplace(N, adtdims, CFL_SIZE, traj);
	md_zrss(N, dtdims, MD_BIT(0), adtraj, dtraj);
	md_free(dtraj);

	float dk = crealf(adtraj[0]);
	md_free(adtraj);

	return dk;
}

