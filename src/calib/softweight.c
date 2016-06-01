/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2015-2016 Siddharth Iyer <sid8795@gmail.com>
 *
 * Iyer S, Ong F, Lustig M.
 * Towards a Parameter­Free ESPIRiT: Soft­Weighting for Robust Coil Sensitivity Estimation
 * In Proceedings of ISMRM 2016.
 * 
 * Candès E, Long C, Trzasko J. 
 * Unbiased Risk Estimates for Singular Value Thresholding and Spectral Estimators.
 * IEEE Transactions on Signal Processing 61, no. 19 (2013): 4643­657.
 *
 */

#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "num/rand.h"
#include "num/multind.h"
#include "num/lapack.h"

#include "misc/debug.h"

#include "calib/estvar.h"

#include "softweight.h"

/*
 * divergence - Calculates the divergence of the spectral estimator for use in SURE as
 *              proposed by Candès et al.
 *
 * Parameters:
 *  N            - Number of singular values.
 *  S            - Array of singular values.
 *  calmat_dims  - Dimension of the calibration matrix.
 *  lambda       - Soft-threshold to test.
 */
static float divergence(long N, const float S[N], const long calmat_dims[2], float lambda) {

    int idx, jdx;

    float div = 0;
    float abs_diff_bw_calmat_dims = labs(calmat_dims[0] - calmat_dims[1]);

    float s, s1, s2, t;
    for (idx = 0; idx < N; idx ++) {
        s  = S[idx];

	if (s == 0)
		continue;

        t  = 1 - lambda/s;
        s1 = (s > lambda ? 1: 0) + 2 * abs_diff_bw_calmat_dims * (t > 0? t: 0);
        s2 = 0;
        for (jdx = 0; jdx < N; jdx++) {
            if (idx == jdx)
                continue;
            t = s - lambda;
            s2 += s * (t > 0? t: 0)/(s * s - S[jdx] * S[jdx]);
        }
        div += s1 + 4 * s2;
    }

    return div;

}

extern void soft_weight_singular_vectors(long N, float variance, const long kernel_dims[3], const long calreg_dims[4], const float S[N], float W[N]) {

    int idx = 0, jdx = 0;

    float t;

    long calmat_dims[2] = {(calreg_dims[0] - kernel_dims[0] + 1) * (calreg_dims[1] - kernel_dims[1] + 1) * 
                (calreg_dims[2] - kernel_dims[2] + 1),
            kernel_dims[0] * kernel_dims[1] * kernel_dims[2] * calreg_dims[3]};

    float Y = calmat_dims[0] * calmat_dims[1] * variance;
    float G = 0;
    for (jdx = 0; jdx < N; jdx++) {
        G += S[jdx] * S[jdx];
    }

    debug_printf(DP_DEBUG1, "Using estimated variance: : %f\n", variance);

    float lambda = S[N-1];
    float testMSE = 0;
    float testLambda = 0;
    float MSE = -Y + G + variance * divergence(N, S, calmat_dims, lambda);

    for (idx = 1; idx < N; idx++) {

        G = 0;
        testLambda = S[N-idx-1];
        for (jdx = 0; jdx < N; jdx++) {
            t = S[jdx];
            G += (t < testLambda? t * t : testLambda * testLambda);
        }

        testMSE = -Y + G + variance * divergence(N, S, calmat_dims, testLambda);

        if (testMSE < MSE) {
            MSE    = testMSE;
            lambda = testLambda;
        }

    }

    debug_printf(DP_DEBUG1, "Soft threshold (Lambda): %f\n", lambda);

    for (int idx = 0; idx < N; idx++) {
        t = (S[idx] > 0) ?(S[idx] - lambda)/S[idx] : 0;
        W[idx] = (t > 0)? t : 0;
    }

}
