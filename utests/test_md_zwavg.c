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

#include "test_md_zwavg.h"


#ifndef DIMS
#define DIMS 16
#endif



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

        msg = test_md_zwavg_flags(4, idims, flags, test_md_zwavg_out[0], test_md_zwavg_out[flags]);

        if (NULL != msg)
            break;
    }

    return msg;
}

static char * run_all_tests()
{
    MU_RUN_TEST(test_md_zwavg);
    return NULL;
}


int main()
{

    char* msg = run_all_tests();

    if (NULL != msg)
        debug_printf(DP_ERROR, msg);
    else
        debug_printf(DP_DEBUG1, "ALL TESTS PASSED\n");

    return NULL != msg;
}


