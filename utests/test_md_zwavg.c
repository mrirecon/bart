/* Copyright 2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2016	Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

/*
 *  Unit test based on the CUnit sample code:
 *  http://cunit.sourceforge.net/example.html
 */



#include <stdio.h>
#include <string.h>
#include <complex.h>
#include "CUnit/Basic.h"

#include "num/flpmath.h"
#include "num/multind.h"

#include "misc/mmio.h"
#include "misc/debug.h"

#include "test_md_zwavg.h"


#ifndef DIMS
#define DIMS 16
#endif

#define TOL 1E-6


/* The suite initialization function.
 * Opens the temporary file used by the tests.
 * Returns zero on success, non-zero otherwise.
 */
int init_suite1(void)
{
	return 0;
}

/* The suite cleanup function.
 * Closes the temporary file used by the tests.
 * Returns zero on success, non-zero otherwise.
 */
int clean_suite1(void)
{
	return 0;
}


static void test_md_zwavg_flags(unsigned int D, const long idims[D], unsigned int flags, const complex float* in, const complex float* out_ref)
{
	long odims[D];
	md_select_dims(D, ~flags, odims, idims);

	complex float* out = md_alloc(D, odims, CFL_SIZE);

	md_zwavg(D, idims, flags, out, in);

	float err = md_znrmse(D, odims, out_ref, out);

	md_free(out);

	CU_TEST(err < TOL);

}


/*
 * Test of md_zwavg.
 * Manually tests based on previously generated data included in the header file
 */
void test_md_zwavg(void)
{
	long idims[4] = {3, 3, 3, 3};

	test_md_zwavg_flags(4, idims, 0u, test_md_zwavg_in, test_md_zwavg_in);
	test_md_zwavg_flags(4, idims, 8u, test_md_zwavg_in, test_md_zwavg_8_out);
	test_md_zwavg_flags(4, idims, 10u, test_md_zwavg_in, test_md_zwavg_10_out);
	test_md_zwavg_flags(4, idims, 15u, test_md_zwavg_in, test_md_zwavg_15_out);

}


/* The main() function for setting up and running the tests.
 * Returns a CUE_SUCCESS on successful running, another
 * CUnit error code on failure.
 */
int main()
{
	CU_pSuite pSuite = NULL;

	/* initialize the CUnit test registry */
	if (CUE_SUCCESS != CU_initialize_registry())
		return CU_get_error();

	/* add a suite to the registry */
	pSuite = CU_add_suite("Suite_1", init_suite1, clean_suite1);
	if (NULL == pSuite) {
		CU_cleanup_registry();
		return CU_get_error();
	}

	/* add the tests to the suite */
	/* NOTE - ORDER IS IMPORTANT - MUST TEST fread() AFTER fprintf() */
	if (NULL == CU_add_test(pSuite, "test of md_zwavg()", test_md_zwavg))
	{
		CU_cleanup_registry();
		return CU_get_error();
	}

	/* Run all tests using the CUnit Basic interface */
	CU_basic_set_mode(CU_BRM_VERBOSE);
	CU_basic_run_tests();
	CU_cleanup_registry();
	return CU_get_error();
}


