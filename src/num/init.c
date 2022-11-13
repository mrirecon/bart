/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2016-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013-2018 Martin Uecker <martin.uecker@med-uni-goettingen.de>
 *
 *
 * Initialization routines. 
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <fenv.h>
#if 0
#include <sys/resource.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "misc/debug.h"
#include "misc/misc.h"

#include "num/fft.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#ifdef USE_CULA
#include <cula_lapack_device.h>
#endif

#include "init.h"

extern unsigned long num_chunk_size;	// num/optimize.c


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

	const char* wisdom_str;

	if (NULL != (wisdom_str = getenv("BART_USE_FFTW_WISDOM"))) {
		
		long wisdom = strtoul(wisdom_str, NULL, 10);

		if ((1 != wisdom) && (0 != wisdom))
			error("BART_USE_FFTW_WISDOM environment variable must be 0 or 1!\n");
		
		use_fftw_wisdom = (1 == wisdom);
	}
		

	const char* chunk_str;

	if (NULL != (chunk_str = getenv("BART_PARALLEL_CHUNK_SIZE"))) {

		long chunk_size = strtoul(chunk_str, NULL, 10);

		if (0 < chunk_size) {

			num_chunk_size = chunk_size;

		} else {

			debug_printf(DP_WARN, "invalid chunk size\n");
		}
	}
}


void num_init_gpu(void)
{
	num_init();

#ifdef USE_CUDA
	cuda_init();
#else
	error("BART compiled without GPU support.\n");
#endif

#ifdef USE_CULA
	culaInitialize();
#endif
}


void num_init_multigpu_select(unsigned long requested_gpus)
{
	num_init();

#ifdef USE_CUDA
	cuda_init_multigpu_select(requested_gpus);
#else
	UNUSED(requested_gpus);
	error("BART compiled without GPU support.\n");
#endif

#ifdef USE_CULA
	culaInitialize();
#endif
}

void num_init_multigpu(int requested_gpus)
{
	num_init();

#ifdef USE_CUDA
	cuda_init_multigpu_number(requested_gpus);
#else
	UNUSED(requested_gpus);
	error("BART compiled without GPU support.\n");
#endif

#ifdef USE_CULA
	culaInitialize();
#endif
}


void num_init_gpu_device(int device)
{
	num_init();

#ifdef USE_CUDA
	if (!cuda_try_init(device))
		error("Could not allocate selected GPU device!");
#else
	UNUSED(device);
	error("BART compiled without GPU support.\n");
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
	error("BART compiled without GPU support.\n");
#endif

#ifdef USE_CULA
	culaInitialize();
#endif
}

void num_deinit_gpu(void)
{

#ifdef USE_CUDA
	cuda_exit();
#else
	error("BART compiled without GPU support.\n");
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


