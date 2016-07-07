/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2015 Siddharth Iyer <sid8795@gmail.com>
 */

#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "num/rand.h"
#include "num/multind.h"
#include "num/lapack.h"

#include "misc/debug.h"
#include "misc/misc.h"

#include "calib/calib.h"
#include "calib/calmat.h"

#include "estvar.h"

/**
 * noise_calreg - This uses the dimension of the calibration 
 *                region to create a new "calibration region"
 *                with each entry being iid standard normal
 *                (or Gaussian) samples.
 * 
 * Parameters:
 *  T       - Product of all the dimensions of the calibration
 *            region.
 *  ncalreg - Pointer to store the new calibration region to.
 */
static void noise_calreg(long T, complex float* ncalreg)
{

    float spike = 1;
    float stdev = 1.f/sqrtf(2.f); 

    for (long idx = 0; idx < T; idx++) {
        if (spike >= uniform_rand()) {
            ncalreg[idx] = stdev * gaussian_rand();
        }
    }

}

/**
 * file_name - This returns the name of file to read
               or write the simulated noise singular values to.
 *
 * Parameters:
 *  kernel_dims - kernel dimensions.
 *  calreg_dims - calibration region dimensions.
 */
static char* file_name(const long kernel_dims[3], const long calreg_dims[4]) {

    char PATH[]     = "/save/nsv/";
    char KERNEL[]   = "KERNEL_";
    char CALREG[]   = "CALREG_";
    char DAT[]      = ".dat";

    int space[] = {strlen(TOOLBOX_PATH), strlen(PATH), strlen(KERNEL), 
        floor(log10(kernel_dims[0])) + 2,
        floor(log10(kernel_dims[1])) + 2,
        floor(log10(kernel_dims[2])) + 2, strlen(CALREG),
        floor(log10(calreg_dims[0])) + 2,
        floor(log10(calreg_dims[1])) + 2,
        floor(log10(calreg_dims[2])) + 2,
        floor(log10(calreg_dims[3])) + 1, strlen(DAT) + 1};

    size_t total = 0;
    for (size_t idx = 0; idx < sizeof(space)/sizeof(int); idx ++) {
        total += space[idx];
    }

    char* name = calloc(total, sizeof(char));

    assert(NULL != name);

    sprintf(name, "%s%s%s%ldx%ldx%ld_%s%ldx%ldx%ldx%ld%s", 
        TOOLBOX_PATH, 
        PATH, 
        KERNEL,
        kernel_dims[0], kernel_dims[1], kernel_dims[2],
        CALREG,
        calreg_dims[0], calreg_dims[1], calreg_dims[2], calreg_dims[3],
        DAT);
    

    return name;

}

/**
 * load_noise_sv - This loads the noise singular values if
 *                 previously simulated.
 *
 * Parameters:
 *  kernel_dims[3] - kernel dimensions
 *  calreg_dims[3] - calibration region dimensions
 *  L              - Number of elements in E.
 *  E              - Load simulated noise singular values to.
 */
static int load_noise_sv(const long kernel_dims[3], const long calreg_dims[4], long L, float* E) {

    char* name = file_name(kernel_dims, calreg_dims);
    FILE* fp   = fopen(name, "rb");

    if (!fp) {
        free(name);
        return 0;
    }

    int c = fread(E, sizeof(float), L, fp);
    assert(c == L);
    
    free(name);
    fclose(fp);

    return 1;

}

/**
 * save_noise_sv - This saves the simulated noise singular
 *                 values to use it again should the same
 *                 parameters are encountered again.
 *
 * Parameters:
 *  kernel_dims[3] - kernel dimensions
 *  calreg_dims[4] - calibration region dimensions
 *  L              - Number of elements in E.
 *  E              - Load simulated noise singular values to.
 */
static void save_noise_sv(const long kernel_dims[3], const long calreg_dims[4], long L, float* E) {

    char* name = file_name(kernel_dims, calreg_dims);
    FILE* fp   = fopen(name, "wb");

    if (!fp) {
        free(name);
        return;
    }

    fwrite(E, sizeof(float), L, fp);

    free(name);
    fclose(fp);

}

/**
 * nsv - This takes the singular value 
 *       decomposition of the Hankel matrix 
 *       constructed from a noise-only
 *       calibration region. The noise is 
 *       distributed as zero-mean unit-variance
 *       Gaussian noise.
 *
 * Parameters:
 *  kernel_dims - The dimensions of the window that sweeps through the
 *                calibration matrix.
 *  calreg_dims - The calibration region dimensions.
 *  L           - The number of singular values.
 *  E           - Array to save noise singular values to.
 *  num_iters   - The number of iterations in order to get a better 
 *                estimate of the noise singular values.
 */
static void nsv(const long kernel_dims[3], const long calreg_dims[4], long L, float* E, long num_iters)
{
    

    if (NULL != getenv("TOOLBOX_PATH") && 1 == load_noise_sv(kernel_dims, calreg_dims, L, E)) {
        return;
    }

    debug_printf(DP_DEBUG1, "NOTE: Running simulations to figure out noise singular values.\n");
    debug_printf(DP_DEBUG1, "      The simulation results are saved if TOOLBOX_PATH is set.\n");

    long N = kernel_dims[0] * kernel_dims[1] * kernel_dims[2] * calreg_dims[3]; 

    float tmpE[N];
    long T = md_calc_size(4, calreg_dims) * sizeof(complex float);

    complex float ncalreg[T];
    noise_calreg(T, ncalreg);

    PTR_ALLOC(complex float[N][N], vec);
    covariance_function(kernel_dims, N, *vec, calreg_dims, ncalreg);
    lapack_eig(N, tmpE, *vec);

    for (int idx = 0; idx < L; idx ++)
        E[idx] = sqrtf(tmpE[N-idx-1]);
    
    for (long idx = 0; idx < num_iters - 1; idx ++) {

        noise_calreg(T, ncalreg);
        covariance_function(kernel_dims, N, *vec, calreg_dims, ncalreg);
        lapack_eig(N, tmpE, *vec);

        for (long jdx = 0; jdx < L; jdx ++) {
            E[jdx] += sqrtf(tmpE[N-jdx-1]);
        }

    }

    for (long idx = 0; idx < L; idx++) {
        E[idx] /= num_iters;
    }

    if (NULL != getenv("TOOLBOX_PATH"))
        save_noise_sv(kernel_dims, calreg_dims, L, E);

    PTR_FREE(vec);

}

/**
 * estimate_noise_variance - This function estimates the variance
 *                           of noise present in the calibration
 *                           region by fitting the last s^th singular
 *                           values of the noise simulation to the
 *                           Calibration matrix's singular values.
 * 
 * Parameters:
 *  L - This is the number of singular values, or the length 
 *      of S and E.
 *  S - This is the singular values obtained from the singular 
 *      value decomposition of the Hankel matrix constructed
 *      from the calibration data.
 *  E - This is the noise singular values as constructed by
 *      function: standard_normal_noise_sv
 */
static float estimate_noise_variance(long L, const float* S, const float* E)
{

    float t = 0.f;
    float c = 0.f; // Counter to avoid zero singular values.
    long  s = 4;   // We fit the last one s^th singular values.

    int num   = L/s;
    int start = L - num;

    for (long idx = 0; idx < num; idx ++) {

        if (isnan(S[start + idx]) || S[start + idx] <= 0 || isnan(E[start + idx]) || E[start + idx] <= 0) {
            break;
        }

        t += ((float)S[start + idx])/((float)E[start + idx]);
        c += 1.f;

    }

    return ((t * t)/(c * c))/1.21; //Scaling down since it works well in practice.

}

extern float estvar_sv(long L, const float S[L], const long kernel_dims[3], const long calreg_dims[4]) {

    float E[L];
    nsv(kernel_dims, calreg_dims, L, E, 10); // Number of iterations set to 5.
    return estimate_noise_variance(L, S, E);

}

extern float estvar_calreg(const long kernel_dims[3], const long calreg_dims[4], const complex float* calreg) {

    // Calibration/Hankel matrix dimension.
    long calmat_dims[2] = {(calreg_dims[0] - kernel_dims[0] + 1) *
            (calreg_dims[1] - kernel_dims[1] + 1) *
            (calreg_dims[2] - kernel_dims[2] + 1),
        calreg_dims[3] * kernel_dims[0] * kernel_dims[1] * kernel_dims[2]};

    long L = calmat_dims[0] > calmat_dims[1] ? calmat_dims[1] : calmat_dims[0];
    long N = calmat_dims[1]; //Number of columns.

    float tmpE[N];
    float S[L];

    PTR_ALLOC(complex float[N][N], vec);
    covariance_function(kernel_dims, N, *vec, calreg_dims, calreg);
    lapack_eig(N, tmpE, *vec);

    for (int idx = 0; idx < L; idx ++)
        S[idx] = sqrtf(tmpE[N-idx-1]);
    
    return estvar_sv(L, S, kernel_dims, calreg_dims);

}

extern float estvar_kspace(long N, const long kernel_dims[3], const long calib_size[3], const long kspace_dims[N], const complex float* kspace) {

    long calreg_dims[N];
    complex float* calreg = NULL;
    calreg = extract_calib(calreg_dims, calib_size, kspace_dims, kspace, false);
    float variance = estvar_calreg(kernel_dims, calreg_dims, calreg);
    md_free(calreg);
    return variance;

}
