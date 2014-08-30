/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 *
 *
 * Initialization routines. 
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <fenv.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "num/fft.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#ifdef USE_CULA
#include <cula_lapack_device.h>
#endif

#include "init.h"


void num_init(void)
{
#ifdef __linux__
//	feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW);
#endif

#ifdef _OPENMP
	int p = omp_get_num_procs();

	if (NULL == getenv("OMP_NUM_THREADS"))
		omp_set_num_threads(p);

	p = omp_get_max_threads();
	fft_set_num_threads(p);
#else
	fft_set_num_threads(2);
#endif
}

void num_init_gpu(void) {
  num_init_gpu_device(0);
}

void num_init_gpu_device(int device) 
{
	num_init();

#ifdef USE_CUDA
	cuda_init(device);
#else
	assert(0);
#endif

#ifdef USE_CULA
	culaInitialize();
#endif

}

void num_init_gpu_memopt(void)
{
	num_init();

#ifdef USE_CUDA
	cuda_init_memopt();
#else
	assert(0);
#endif

#ifdef USE_CULA
	culaInitialize();
#endif
}


void num_set_num_threads(int n)
{
#ifdef _OPENMP
	omp_set_num_threads(n);
#endif
	fft_set_num_threads(n);
}


