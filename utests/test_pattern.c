/* Copyright 2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <complex.h>

#include "num/flpmath.h"
#include "num/multind.h"

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mri.h"

#include "utest.h"


static bool test_pattern_flags(unsigned int D, const long dims[D], unsigned int flags, const complex float* in, const complex float* ref)
{
	long odims[D];
	md_select_dims(D, ~flags, odims, dims);

	complex float* out = md_alloc(D, odims, CFL_SIZE);

	estimate_pattern(D, dims, flags, out, in);

	UT_ASSERT(md_znrmse(D, odims, ref, out) < UT_TOL);

	md_free(out);

	return true;
}


static bool test_pattern(void)
{
	const complex float in[1][5][3] = { {

		{ 3., 0., 0. },
		{ 0., 2., 0. },
		{ .2, 0., 0. },
		{ 0., 0., 0. },
		{ 0., 2., 0. },
	} };


	const complex float ref0[1][5][3] = { {

		{ 1., 0., 0. },
		{ 0., 1., 0. },
		{ 1., 0., 0. },
		{ 0., 0., 0. },
		{ 0., 1., 0. },
	} };

	const complex float ref2[1][1][3] = { {

		{ 1., 1., 0. },
	} };

	const complex float ref3[1][1][1] = { {

		{ 1. },
	} };

	long idims[3] = { 3, 5, 1 };


	return (test_pattern_flags(3, idims, 0, &in[0][0][0], &ref0[0][0][0]) && 
		test_pattern_flags(3, idims, 2, &in[0][0][0], &ref2[0][0][0]) &&
		test_pattern_flags(3, idims, 3, &in[0][0][0], &ref3[0][0][0]));
}


UT_REGISTER_TEST(test_pattern);

