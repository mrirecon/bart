/* Copyright 2015. The Regents of the University of California.
 * Copyright 2021. Martin Uecker.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2015 Siddharth Iyer <sid8795@gmail.com>
 */

#include <math.h>
#include <stdio.h>

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
	float stdev = 1.f / sqrtf(2.f);

	for (long idx = 0; idx < T; idx++)
		if (spike >= uniform_rand())
			ncalreg[idx] = stdev * gaussian_rand();
}


/**
 * file_name - This returns the name of file to read
 *             or write the simulated noise singular values to.
 *
 * Parameters:
 *  kernel_dims - kernel dimensions.
 *  calreg_dims - calibration region dimensions.
 */
static char* file_name(const char* toolbox, const long kernel_dims[3], const long calreg_dims[4])
{
	int size = 0;

	size = snprintf(NULL, size, "%s/save/nsv/KERNEL_%ldx%ldx%ld_CAL_REG%ldx%ldx%ldx%ld.dat",
		toolbox, kernel_dims[0], kernel_dims[1], kernel_dims[2],
		calreg_dims[0], calreg_dims[1], calreg_dims[2], calreg_dims[3]);

	assert(size > 0);
    
	char* name = calloc(size, sizeof(char));

	if (NULL == name)
		error("Memory out\n");

	size = snprintf(name, size, "%s/save/nsv/KERNEL_%ldx%ldx%ld_CAL_REG%ldx%ldx%ldx%ld.dat",
		toolbox, kernel_dims[0], kernel_dims[1], kernel_dims[2],
		calreg_dims[0], calreg_dims[1], calreg_dims[2], calreg_dims[3]);

	if (size < 0) {

		xfree(name);
		return NULL;
	}

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
static int load_noise_sv(const char* toolbox, const long kernel_dims[3], const long calreg_dims[4], long L, float* E)
{
	int ok = 0;

	if (NULL == toolbox)
		goto out0;

	char* name = file_name(toolbox, kernel_dims, calreg_dims);

	FILE* fp = fopen(name, "rb");

	if (!fp)
		goto out1;

	int c = fread(E, sizeof(float), L, fp);

	if (c != L)
		goto out2;
 
	ok = 1;
out2:
	fclose(fp);
out1:
	xfree(name);
out0:
	return ok;
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
static void save_noise_sv(const char* toolbox, const long kernel_dims[3], const long calreg_dims[4], long L, float* E)
{
	char* name = file_name(toolbox, kernel_dims, calreg_dims);

	FILE* fp = fopen(name, "wb");

	if (!fp)
		goto out;

	fwrite(E, sizeof(float), L, fp);

	fclose(fp);
out:
	xfree(name);
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
static void nsv(const char* toolbox, const long kernel_dims[3], const long calreg_dims[4], long L, float* E, long num_iters)
{
	if (1 == load_noise_sv(toolbox, kernel_dims, calreg_dims, L, E))
		return;

	debug_printf(DP_DEBUG1, "NOTE: Running simulations to figure out noise singular values.\n");
	debug_printf(DP_DEBUG1, "      The simulation results are saved if TOOLBOX_PATH is set.\n");

	long N = kernel_dims[0] * kernel_dims[1] * kernel_dims[2] * calreg_dims[3]; 

	float tmpE[N];

	long T = md_calc_size(4, calreg_dims) * sizeof(complex float);

	long ncalreg_dims[] = { T };

	complex float* ncalreg = md_calloc(1, ncalreg_dims, sizeof(complex float));

	noise_calreg(T, ncalreg);

	PTR_ALLOC(complex float[N][N], vec);

	covariance_function(kernel_dims, N, *vec, calreg_dims, ncalreg);
	lapack_eig(N, tmpE, *vec);

	for (int idx = 0; idx < L; idx++)
		E[idx] = sqrtf(tmpE[N - idx - 1]);
    
	for (long idx = 0; idx < num_iters - 1; idx++) {

		noise_calreg(T, ncalreg);
		covariance_function(kernel_dims, N, *vec, calreg_dims, ncalreg);
		lapack_eig(N, tmpE, *vec);

		for (long jdx = 0; jdx < L; jdx++)
			E[jdx] += sqrtf(tmpE[N- jdx - 1]);
	}

	for (long idx = 0; idx < L; idx++)
		E[idx] /= num_iters;

	if (NULL != toolbox)
		save_noise_sv(toolbox, kernel_dims, calreg_dims, L, E);

	PTR_FREE(vec);
	md_free(ncalreg);
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

	int num = L / s;
	int start = L - num;

	for (long idx = 0; idx < num; idx++) {

		if (isnan(S[start + idx]) || (S[start + idx] <= 0) || isnan(E[start + idx]) || (E[start + idx] <= 0))
			break;

		t += ((float)S[start + idx]) / ((float)E[start + idx]);
		c += 1.f;
	}

	return ((t * t) / (c * c)) / 1.21; // Scaling down since it works well in practice.
}


extern float estvar_sv(const char* toolbox, long L, const float S[L], const long kernel_dims[3], const long calreg_dims[4])
{
	float E[L];

	nsv(toolbox, kernel_dims, calreg_dims, L, E, 10); // Number of iterations set to 5.

	return estimate_noise_variance(L, S, E);
}

extern float estvar_calreg(const char* toolbox, const long kernel_dims[3], const long calreg_dims[4], const complex float* calreg)
{
	// Calibration/Hankel matrix dimension.
	long calmat_dims[2] = {
		(calreg_dims[0] - kernel_dims[0] + 1)
		* (calreg_dims[1] - kernel_dims[1] + 1)
		* (calreg_dims[2] - kernel_dims[2] + 1),
		calreg_dims[3] * kernel_dims[0] * kernel_dims[1] * kernel_dims[2]
	};

	long L = (calmat_dims[0] > calmat_dims[1]) ? calmat_dims[1] : calmat_dims[0];
	long N = calmat_dims[1]; //Number of columns.

	float tmpE[N];
	float S[L];

	complex float vec[N][N];
	covariance_function(kernel_dims, N, vec, calreg_dims, calreg);
	lapack_eig(N, tmpE, vec);

	for (int idx = 0; idx < L; idx++)
		S[idx] = sqrtf(tmpE[N - idx - 1]);
    
	return estvar_sv(toolbox, L, S, kernel_dims, calreg_dims);
}

extern float estvar_kspace(const char* toolbox, int N, const long kernel_dims[3], const long calib_size[3], const long kspace_dims[N], const complex float* kspace)
{
	long calreg_dims[N];
	complex float* calreg = NULL;

	calreg = extract_calib(calreg_dims, calib_size, kspace_dims, kspace, false);

	float variance = estvar_calreg(toolbox, kernel_dims, calreg_dims, calreg);

	md_free(calreg);

	return variance;
}

