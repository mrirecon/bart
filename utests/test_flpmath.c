/* Copyright 2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2016	Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

/*
 *  Unit test based on the MinUnit sample code:
 *  http://www.jera.com/techinfo/jtns/jtn002.html
 */



#include <stdio.h>
#include <string.h>
#include <complex.h>

#include "num/flpmath.h"
#include "num/multind.h"

#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/debug.h"

#include "minunit.h"

#include "test_flpmath.h"


#ifndef DIMS
#define DIMS 16
#endif



static char* test_md_zfmac2_flags(unsigned int D, const long idims[D], unsigned int flags, const complex float* in1, const complex float* in2, const complex float* out_ref)
{
    long odims[D];
    md_select_dims(D, ~flags, odims, idims);

    complex float* out = md_alloc(D, odims, CFL_SIZE);
    md_clear(D, odims, out, CFL_SIZE);

    long istr[D];
    long ostr[D];

    md_calc_strides(D, istr, idims, CFL_SIZE);
    md_calc_strides(D, ostr, odims, CFL_SIZE);

    md_zfmac2(D, idims, ostr, out, istr, in1, istr, in2);

    float err = md_znrmse(D, odims, out_ref, out);

    md_free(out);

    MU_ASSERT("Error: test_md_zfmac2_flags failed!\n", err < TOL);

    return NULL;

}


/*
 * Test of md_zfmac2
 * Tests based on previously generated data included in the header file
 */
static char* test_md_zfmac2()
{
    long idims[4] = {3, 3, 3, 3};

    char* msg = NULL;

    for (unsigned int flags = 0u; flags < 16u; flags++) {

        debug_printf(DP_DEBUG1, "Testing md_zfmac2_flags with flags=%d\n", flags);

        msg = test_md_zfmac2_flags(4, idims, flags, test_md_in0, test_md_in1, test_md_zfmac2_out[flags]);

        if (NULL != msg)
            break;
    }

    return msg;
}


static char* test_md_zwavg_flags(unsigned int D, const long idims[D], unsigned int flags, const complex float* in, const complex float* out_ref)
{
    long odims[D];
    md_select_dims(D, ~flags, odims, idims);

    complex float* out = md_alloc(D, odims, CFL_SIZE);

    md_zwavg(D, idims, flags, out, in);

    float err = md_znrmse(D, odims, out_ref, out);

    md_free(out);

    MU_ASSERT("Error: test_md_zwavg_flags failed!\n", err < TOL);

    return NULL;

}


/*
 * Test of md_zwavg.
 * Tests based on previously generated data included in the header file
 */
static char* test_md_zwavg()
{
    long idims[4] = {3, 3, 3, 3};

    char* msg = NULL;

    for (unsigned int flags = 0u; flags < 16u; flags++) {

        debug_printf(DP_DEBUG1, "Testing md_zwavg_flags with flags=%d\n", flags);

        msg = test_md_zwavg_flags(4, idims, flags, test_md_in0, test_md_zwavg_out[flags]);

        if (NULL != msg)
            break;
    }

    return msg;
}

static char * run_all_tests()
{
    MU_RUN_TEST(test_md_zwavg);
    MU_RUN_TEST(test_md_zfmac2);
    return NULL;
}


int main()
{

    char* msg = run_all_tests();

    if (NULL != msg)
        debug_printf(DP_ERROR, msg);
    else
        debug_printf(DP_INFO, "ALL TESTS PASSED\n");

    return NULL != msg;
}


