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

	long tdims1[N];
	md_select_dims(N, ~MD_BIT(1), tdims1, tdims);

	long pos[N];
	md_set_dims(N, pos, 0);


	complex float* traj1 = md_alloc_sameplace(N, tdims1, CFL_SIZE, traj);
	// Extract what would be the DC component in Cartesian sampling

	pos[1] = tdims[1] / 2;
	md_slice(N, MD_BIT(1), pos, tdims, traj1, traj, CFL_SIZE);

	NESTED(float, dist, (int i))
	{
		return sqrtf(powf(crealf(traj1[3 * i + 0]), 2.) + powf(crealf(traj1[3 * i + 1]), 2.));
	};

	float dc_shift = dist(0);

	pos[1] = tdims[1] / 2 + 1;
	md_slice(N, MD_BIT(1), pos, tdims, traj1, traj, CFL_SIZE);

	float shift1 = dist(0) - dc_shift;

	md_free(traj1);

	return shift1;
}

bool traj_radial_same_dk(int N, const long tdims[N], const complex float* traj)
{
	assert(1 < N);

	long rdims[N];
	md_copy_dims(N, rdims, tdims);
	rdims[1] -= 2;

	complex float* tmp1 = md_alloc_sameplace(N, rdims, CFL_SIZE, traj);
	complex float* tmp2 = md_alloc_sameplace(N, rdims, CFL_SIZE, traj);

	long pos[N];
	md_set_dims(N, pos, 0);
	pos[1] = 1;

	md_copy_block(N, pos, rdims, tmp1, tdims, traj, CFL_SIZE);
	md_zsmul(N, rdims, tmp1, tmp1, -2.);

	pos[1] = 0;
	md_copy_block(N, pos, rdims, tmp2, tdims, traj, CFL_SIZE);
	md_zadd(N, rdims, tmp1, tmp1, tmp2);

	pos[1] = 2;
	md_copy_block(N, pos, rdims, tmp2, tdims, traj, CFL_SIZE);
	md_zadd(N, rdims, tmp1, tmp1, tmp2);

	md_free(tmp2);

	float err = md_znorm(N, rdims, tmp1) / (float)md_calc_size(N, rdims);
	md_free(tmp1);

	return (1.e-5 > err);
}

static void traj_radial_direction_int(int N, long idx, const long ddims[N], complex float* dir, const long tdims[N], const complex float* traj)
{
	long pos[N];
	md_set_dims(N, pos, 0);
	pos[1] = idx;

	md_slice(N, MD_BIT(1), pos, tdims, dir, traj, CFL_SIZE);

	long ndims[N];
	md_select_dims(N, ~MD_BIT(0), ndims, ddims);

	complex float* nrm = md_alloc_sameplace(N, ndims, CFL_SIZE, traj);
	md_zrss(N, ddims, MD_BIT(0), nrm, dir);

	md_zdiv2(N, ddims, MD_STRIDES(N, ddims, CFL_SIZE), dir, MD_STRIDES(N, ddims, CFL_SIZE), dir, MD_STRIDES(N, ndims, CFL_SIZE), nrm);
	md_free(nrm);
}

bool traj_radial_through_center(int N, const long tdims[N], const complex float* traj)
{
	long ddims[N];
	md_select_dims(N, ~MD_BIT(1), ddims, tdims);

	complex float* dir1 = md_alloc_sameplace(N, ddims, CFL_SIZE, traj);
	complex float* dir2 = md_alloc_sameplace(N, ddims, CFL_SIZE, traj);

	traj_radial_direction_int(N, tdims[1] - 1, ddims, dir1, tdims, traj);
	traj_radial_direction_int(N, tdims[1] - 2, ddims, dir2, tdims, traj);

	float err = md_znrmse(N, ddims, dir1, dir2);

	md_free(dir1);
	md_free(dir2);

	return (1.e-5 > err);
}

bool traj_is_radial(int N, const long tdims[N], const _Complex float* traj)
{
	return traj_radial_same_dk(N, tdims, traj) && traj_radial_through_center(N, tdims, traj);
}

void traj_radial_direction(int N, const long ddims[N], complex float* dir, const long tdims[N], const complex float* traj)
{
	traj_radial_direction_int(N, tdims[1] - 1, ddims, dir, tdims, traj);
}


