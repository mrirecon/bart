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
#include "num/specfun.h"

#include "stl/misc.h"
#include "stl/models.h"

#include "simu/shape.h"
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


static bool test_kpolygon(void)
{
	const double pg[4][2] = {
		{ -0.5, -0.5 },
		{ +0.5, -0.5 },
		{ +0.5, +0.5 },
		{ -0.5, +0.5 },
	};

	double test[][3] = {
		{  0.,  0.,  0. },
		{ +0.1, 0.,  0. },
		{ -0.2, 0.,  0. },
		{ +0.1, 0.3, 0. },
		{ -0.2, 0.2, 0. },
	};

	for (int i = 0; i < 5; i++) {

		double diff1 = sinc(M_PI * test[i][0]) * sinc(M_PI * test[i][1]);
		double diff2 = 4. * creal(kpolygon(4, pg, test[i]));

		if (1.E-10 < fabs(diff1 - diff2))
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_kpolygon);



static bool test_kpolygon2(void)
{
	const double pg[4][2] = {

		{ -0.5, -0.5 },
		{ +0.5, -0.5 },
		{ +0.5, +0.5 },
		{ -0.5, +0.5 },
	};

	for (int i = 3; i < 9; i++) {

		double x = pow(10., -i);	// close to 0.
		double y = 0.;

		double pos[3] = { x, y, 0. };

		double value = 4. * kpolygon(4, pg, pos);

		if (1.E-5 < fabs(value - 1.))
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_kpolygon2);



