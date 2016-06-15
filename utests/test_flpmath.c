/* Copyright 2016. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>

#include "num/flpmath.h"
#include "num/multind.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "utest.h"


// include test data
#include "test_flpmath_data.h"




static bool test_md_zfmacc2_flags(unsigned int D, const long idims[D], unsigned int flags, const complex float* in1, const complex float* in2, const complex float* out_ref)
{
	long odims[D];
	md_select_dims(D, ~flags, odims, idims);

	complex float* out = md_calloc(D, odims, CFL_SIZE);

	long istr[D];
	long ostr[D];

	md_calc_strides(D, istr, idims, CFL_SIZE);
	md_calc_strides(D, ostr, odims, CFL_SIZE);

	md_zfmacc2(D, idims, ostr, out, istr, in1, istr, in2);

	float err = md_znrmse(D, odims, out_ref, out);

	md_free(out);

	UT_ASSERT(err < UT_TOL);

	return true;
}


/*
 * Tests based on previously generated data included in the header file
 */
static bool test_md_zfmacc2(void)
{
	long idims[4] = { 3, 3, 3, 3 };

	bool ret = true;

	for (unsigned int flags = 0u; flags < 16u; flags++) {

		debug_printf(DP_DEBUG1, "Testing md_zfmacc2_flags with flags=%d\n", flags);

		ret &= test_md_zfmacc2_flags(4, idims, flags, test_md_in0, test_md_in1, test_md_zfmacc2_out[flags]);
	}

	return ret;
}


static bool test_md_zavg_flags(unsigned int D, const long idims[D], unsigned int flags, const complex float* in, const complex float* out_ref, bool wavg)
{
	long odims[D];
	md_select_dims(D, ~flags, odims, idims);

	complex float* out = md_alloc(D, odims, CFL_SIZE);

	(wavg ? md_zwavg : md_zavg)(D, idims, flags, out, in);

	float err = md_znrmse(D, odims, out_ref, out);

	md_free(out);

	UT_ASSERT(err < UT_TOL);

	return true;
}


/*
 * Tests based on previously generated data included in the header file
 */
static bool test_md_zwavg(void)
{
	long idims[4] = { 3, 3, 3, 3 };

	bool wavg = true;
	bool ret = true;

	for (unsigned int flags = 0u; flags < 16u; flags++) {

		debug_printf(DP_DEBUG1, "Testing md_zwavg_flags with flags=%d\n", flags);

		ret &= test_md_zavg_flags(4, idims, flags, test_md_in0, test_md_zwavg_out[flags], wavg);
	}

	return ret;
}


static bool test_md_zavg(void)
{
	long idims[4] = { 3, 3, 3, 3 };

	bool wavg = false;
	bool ret = true;

	for (unsigned int flags = 0u; flags < 16u; flags++) {

		debug_printf(DP_DEBUG1, "Testing md_zavg_flags with flags=%d\n", flags);

		ret &= test_md_zavg_flags(4, idims, flags, test_md_in0, test_md_zavg_out[flags], wavg);
	}

	return ret;
}


UT_REGISTER_TEST(test_md_zfmacc2);
UT_REGISTER_TEST(test_md_zwavg);
UT_REGISTER_TEST(test_md_zavg);

