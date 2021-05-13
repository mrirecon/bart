/* Copyright 2013. The Regents of the University of California.
 * Copyright 2021. Uecker Lab. University Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Martin Uecker, Dara Bahri, Moritz Blumenthal
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#ifdef _WIN32
#include "win/rand_r.h"
#endif

#include "num/multind.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "rand.h"

unsigned int num_rand_seed = 123;


void num_rand_init(unsigned int seed)
{
	num_rand_seed = seed;
}


double uniform_rand(void)
{
	double ret;

	#pragma omp critical
	ret = rand_r(&num_rand_seed) / (double)RAND_MAX;

	return ret;
}


/**
 * Box-Muller
 */
complex double gaussian_rand(void)
{
	double u1, u2, s;

 	do {

		u1 = 2. * uniform_rand() - 1.;
		u2 = 2. * uniform_rand() - 1.;
   		s = u1 * u1 + u2 * u2;

   	} while (s > 1.);

	double re = sqrt(-2. * log(s) / s) * u1;
	double im = sqrt(-2. * log(s) / s) * u2;

	return re + 1.i * im;
}

void md_gaussian_rand(unsigned int D, const long dims[D], complex float* dst)
{
#ifdef  USE_CUDA
	if (cuda_ondevice(dst)) {

		complex float* tmp = md_alloc(D, dims, sizeof(complex float));
		md_gaussian_rand(D, dims, tmp);
		md_copy(D, dims, dst, tmp, sizeof(complex float));
		md_free(tmp);
		return;
	}
#endif
//#pragma omp parallel for
	for (long i = 0; i < md_calc_size(D, dims); i++)
		dst[i] = (complex float)gaussian_rand();
}

void md_uniform_rand(unsigned int D, const long dims[D], complex float* dst)
{
#ifdef  USE_CUDA
	if (cuda_ondevice(dst)) {

		complex float* tmp = md_alloc(D, dims, sizeof(complex float));
		md_uniform_rand(D, dims, tmp);
		md_copy(D, dims, dst, tmp, sizeof(complex float));
		md_free(tmp);
		return;
	}
#endif
	for (long i = 0; i < md_calc_size(D, dims); i++)
		dst[i] = (complex float)uniform_rand();
}

void md_rand_one(unsigned int D, const long dims[D], complex float* dst, double p)
{
#ifdef  USE_CUDA
	if (cuda_ondevice(dst)) {

		complex float* tmp = md_alloc(D, dims, sizeof(complex float));
		md_rand_one(D, dims, tmp, p);
		md_copy(D, dims, dst, tmp, sizeof(complex float));
		md_free(tmp);
		return;
	}
#endif
	for (long i = 0; i < md_calc_size(D, dims); i++)
		dst[i] = (complex float)(uniform_rand() < p);
}
