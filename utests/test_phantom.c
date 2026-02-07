/* Copyright 2025-2026. TU Graz. Institute of Biomedical Imaging
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2025 Martin Heide
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"

#include "stl/misc.h"
#include "stl/models.h"

#include "simu/phantom.h"

#include "utest.h"

static bool test_stl_kspace(void)
{
        bool b = true;
        long stldims[3];

	struct phantom_opts popts;
	popts.kspace = true;
        double* model = stl_internal_tetrahedron(stldims);
	stl_compute_normals(stldims, model);
	phantom_stl_init(&popts, 3, stldims, model);

	float pos[3] = {0, 0, 0};

	complex double c = stl_fun_k(&popts, 0, pos);

	if (1E-10 < fabs(creal(c) - 0.243))
		b = false;

	pos[0] = 0.1;
	pos[1] = 0.1;
	pos[1] = 0.1;
	c = stl_fun_k(&popts, 0, pos);

	if (1E-10 < fabs(creal(c) - 0.0597860454167))
		b = false;

	if (1E-10 < fabs(cimag(c)))
		b = false;

	popts.dstr(&popts);
        md_free(model);

        return b;
}

UT_REGISTER_TEST(test_stl_kspace);
