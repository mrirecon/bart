/* Copyright 2016. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2016	Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>

#include "num/flpmath.h"
#include "num/multind.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/utest.h"

#include "test_flpmath.h"




static bool test_md_zfmac2_flags(unsigned int D, const long idims[D], unsigned int flags, const complex float* in1, const complex float* in2, const complex float* out_ref)
{
	long odims[D];
	md_select_dims(D, ~flags, odims, idims);

	complex float* out = md_calloc(D, odims, CFL_SIZE);

	long istr[D];
	long ostr[D];

	md_calc_strides(D, istr, idims, CFL_SIZE);
	md_calc_strides(D, ostr, odims, CFL_SIZE);

	md_zfmac2(D, idims, ostr, out, istr, in1, istr, in2);

	float err = md_znrmse(D, odims, out_ref, out);

	md_free(out);

	UT_ASSERT(err < UT_TOL);

	return true;
}


/*
 * Tests based on previously generated data included in the header file
 */
static bool test_md_zfmac2(void)
{
	long idims[4] = { 3, 3, 3, 3 };

	bool ret = true;

	for (unsigned int flags = 0u; flags < 16u; flags++) {

		debug_printf(DP_DEBUG1, "Testing md_zfmac2_flags with flags=%d\n", flags);

		ret &= test_md_zfmac2_flags(4, idims, flags, test_md_in0, test_md_in1, test_md_zfmac2_out[flags]);
	}

	return ret;
}


static bool test_md_zwavg_flags(unsigned int D, const long idims[D], unsigned int flags, const complex float* in, const complex float* out_ref)
{
	long odims[D];
	md_select_dims(D, ~flags, odims, idims);

	complex float* out = md_alloc(D, odims, CFL_SIZE);

	md_zwavg(D, idims, flags, out, in);

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

	bool ret = true;

	for (unsigned int flags = 0u; flags < 16u; flags++) {

		debug_printf(DP_DEBUG1, "Testing md_zwavg_flags with flags=%d\n", flags);

		ret &= test_md_zwavg_flags(4, idims, flags, test_md_in0, test_md_zwavg_out[flags]);
	}

	return ret;
}


ut_test_f* tests[] = {

	test_md_zwavg,
	test_md_zfmac2,
};


int main()
{
	ut_run_tests(ARRAY_SIZE(tests), tests);
}


