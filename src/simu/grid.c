/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2025 Martin Heide
 */

#include <complex.h>
#include <math.h>

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/linalg.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/misc.h"

#include "grid.h"


struct grid_opts grid_opts_init = {

	.dims = { [0 ... 2] = -1, [3 ... TIME_DIM - 1] = 1, -1, [TIME_DIM + 1 ... DIMS - 1] = 1 },

	.kspace = false,

	.b0 = { 0., 0., 0. },
	.b1 = { 0., 0., 0. },
	.b2 = { 0., 0., 0. },
	.bt = 0.,
};

struct grid_opts grid_opts_defaults = {

	.dims = { 128, 128, [2 ... DIMS - 1] = 1 },

	.kspace = false,

	.b0 = { 0.5, 0., 0. },
	.b1 = { 0., 0.5, 0. },
	.b2 = { 0., 0., 0. },
	.bt = 0.,
};

struct grid_opts grid_opts_coilcoeff = {

	.dims = { 5, 5, [2 ... DIMS - 1] = 1 },

	.kspace = true,

	.b0 = { 1., 0., 0. },
	.b1 = { 0., 1., 0. },
	.b2 = { 0., 0., 0. },
	.bt = 0.,
};


float* compute_grid(int D, long gdims[D], struct grid_opts* go, const long tdims[D], const complex float* traj)
{
	// minimum: 1d coord indices x 3d space x 1d coils x 1x coeff x 1d time
	assert(D >= 7);

	float veclen[4];
	veclen[0] = vecf_norm(3, go->b0);
	veclen[1] = vecf_norm(3, go->b1);
	veclen[2] = vecf_norm(3, go->b2);
	veclen[3] = fabsf(go->bt);

	for (int i = 0; i < 3; i++) {

		if (0. == veclen[i]) {

			if (1 < go->dims[i])
				debug_printf(DP_INFO, "The %dth basis vector has length zero, dim set to one\n", i + 1);

			go->dims[i] = 1;
		}
	}

	if (0 == veclen[3]) {

		if (1 < go->dims[TIME_DIM]) {

			debug_printf(DP_INFO, "The time basis vector has length zero, dim set to one\n");
			go->dims[TIME_DIM] = 1;
		}
	}

	if (   (0. != vecf_sdot(3, go->b0, go->b1))
	    || (0. != vecf_sdot(3, go->b0, go->b2))
	    || (0. != vecf_sdot(3, go->b1, go->b2)))
		debug_printf(DP_WARN, "Basis vectors are not orthogonal.\n");

	// initialize the sampling trajectory
	if (NULL != traj) {

		md_copy_dims(D, gdims, tdims);
		gdims[0] = 4;

	} else {

		md_set_dims(D, gdims, 1);
		gdims[0] = 4;

		for (int i = 0; i < 3; i++)
			gdims[i + 1] = go->dims[i];
	}

	gdims[TIME_DIM] = go->dims[TIME_DIM];

	float* grid = md_alloc(D, gdims, FL_SIZE);

	long gstrs[D];
	long tstrs[D];
	long pos[D];
	long ppos[D];

	if (NULL != traj)
		md_calc_strides(D, tstrs, tdims, CFL_SIZE);

	md_calc_strides(D, gstrs, gdims, FL_SIZE);
	md_set_dims(D, pos, 0);


	do {
		md_copy_dims(D, ppos, pos);

		if (NULL != traj) {

			for (ppos[0] = 0; ppos[0] < 3; ppos[0]++)
				MD_ACCESS(D, gstrs, ppos, grid) = creal(MD_ACCESS(D, tstrs, ppos, traj));

		} else {

			float c[3] = { };

			for (int i = 0; i < 3; i++) {

				if (1 == gdims[i + 1])
					continue;

				float s = floorf(gdims[i + 1] / 2.);
				// division by 1/(2*veclen*veclen). 1/(2*veclen) for kspace sampling density and 1/veclen for
				// normalization of basis vector during multiplication

				if (go->kspace)
					c[i] = (pos[i + 1] - s) / (2. * powf(veclen[i], 2.));
				else
					c[i] = (pos[i + 1] - s) / (0.5 * (double)gdims[i + 1]);
			}

			for (ppos[0] = 0; ppos[0] < 3; ppos[0]++)
				MD_ACCESS(D, gstrs, ppos, grid) = c[0] * go->b0[ppos[0]] + c[1] * go->b1[ppos[0]] + c[2] * go->b2[ppos[0]];
		}

		float co = 0.;

		if (1 != gdims[TIME_DIM])
			co = ppos[TIME_DIM] / (gdims[TIME_DIM] - 1);

		ppos[0] = 3;
		MD_ACCESS(D, gstrs, ppos, grid) = go->bt * co;

	} while (md_next(D, gdims, ~VEC_FLAG_S, pos));

	return grid;
}

