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
#include <sys/resource.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "misc/debug.h"
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

#if 0
	// set stack limit
	if (-1 == setrlimit(RLIMIT_STACK, &(struct rlimit){ 500000000, 500000000 }))
		debug_printf(DP_WARN, "error setting stack size\n");

	// FIXME: should also set openmp stack size
#endif

#ifdef _OPENMP
	int p = omp_get_num_procs();

	if (NULL == getenv("OMP_NUM_THREADS"))
		omp_set_num_threads(p);

	p = omp_get_max_threads();

	// omp_set_nested(1);
#else
	int p = 2;
#endif
#ifdef FFTWTHREADS
	fft_set_num_threads(p);
#endif
}


void num_init_gpu(void)
{
	num_init();

	// don't call cuda_init so that GPU can get assigned by driver

#ifdef USE_CULA
	culaInitialize();
#endif
}


void num_init_gpu_device(int device) 
{
	num_init();

#ifdef USE_CUDA
	cuda_init(device);
#else
	(void)device;
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
#ifdef FFTWTHREADS
	fft_set_num_threads(n);
#endif
}


