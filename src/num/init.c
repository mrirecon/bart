/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2016-2018. Martin Uecker.
 * Copyright 2023-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013-2018 Martin Uecker
 *
 *
 * Initialization routines.
 */

#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <fenv.h>
#if 0
#include <sys/resource.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mmio.h"

#include "num/fft_plan.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "init.h"

extern long num_chunk_size;	// num/optimize.c

static bool bart_gpu_support = false;
bool bart_use_gpu = false;
unsigned long bart_mpi_split_flags = 0;
unsigned long bart_delayed_loop_flags = 0;
long bart_delayed_loop_dims[16] = { [0 ... 15] = -1 };
bool bart_delayed_computations = false;

static void num_init_internal(void)
{
	const char* wisdom_str;

	if (NULL != (wisdom_str = getenv("BART_USE_FFTW_WISDOM"))) {

		long wisdom = strtol(wisdom_str, NULL, 10);

		if ((1 != wisdom) && (0 != wisdom))
			error("BART_USE_FFTW_WISDOM environment variable must be 0 or 1!\n");

		use_fftw_wisdom = (1 == wisdom);
	}


	const char* chunk_str;

	if (NULL != (chunk_str = getenv("BART_PARALLEL_CHUNK_SIZE"))) {

		long chunk_size = strtol(chunk_str, NULL, 10);

		if (0 < chunk_size) {

			num_chunk_size = chunk_size;

		} else {

			debug_printf(DP_WARN, "invalid chunk size\n");
		}
	}

#ifdef USE_CUDA
	const char* gpu_str;

	if (NULL != (gpu_str = getenv("BART_GPU"))) {

		int bart_num_gpus = strtol(gpu_str, NULL, 10);

		if (0 < bart_num_gpus)
			bart_use_gpu = true;
	}

	if (NULL != (gpu_str = getenv("BART_GPU_STREAMS"))) {

		int bart_num_streams = strtol(gpu_str, NULL, 10);

		if (0 < bart_num_streams)
			cuda_num_streams = bart_num_streams;
	}

	const char* mem_str;

	if (NULL != (mem_str = getenv("BART_GPU_GLOBAL_MEMORY"))) {

		long mem = strtol(mem_str, NULL, 10);

		if ((1 != mem) && (0 != mem))
			error("BART_GPU_GLOBAL_MEMORY environment variable must be 0 or 1!\n");

		if (1 == mem)
			cuda_use_global_memory();
	}
#endif

	int p = 2;

#ifdef _OPENMP
	p = omp_get_num_procs();

	if (NULL == getenv("OMP_NUM_THREADS"))
		omp_set_num_threads(p);

	p = omp_get_max_threads();

	int running_thr = omp_get_num_threads(); //get number of running top level threads
	p = MAX(1, p / running_thr);
#endif

#ifdef FFTWTHREADS
	fft_set_num_threads(p);
#endif

}


void num_init(void)
{
	static int initialized = false;

#pragma omp critical (bart_num_init)
	if (!initialized) {

		num_init_internal();
		initialized = true;
	}

#ifdef USE_CUDA
	if (bart_gpu_support && bart_use_gpu)
			cuda_init();
#else
	if (bart_use_gpu)
		error("BART compiled without GPU support.\n");
#endif

}

void num_init_gpu_support(void)
{
	bart_gpu_support = true;
	num_init();
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


#define MD_BIT(x) (1ul << (x))
#define MD_IS_SET(x, y)	((x) & MD_BIT(y))

void num_init_delayed(void)
{
	bart_delayed_loop_flags = 0;

	for (int i = 0, ip = 0; i < 16; i++) {

		if (-1 == bart_delayed_loop_dims[i])
			break;

		int dim = bart_delayed_loop_dims[i];
		bart_delayed_loop_dims[i] = -1;

		assert(dim < 8 * (long)sizeof(unsigned long));

		if (!MD_IS_SET(bart_delayed_loop_flags, dim)) {

			bart_delayed_loop_dims[ip++] = dim;
			bart_delayed_loop_flags |= MD_BIT(dim);
		}
	}

	if (0 != bart_delayed_loop_flags)
		bart_delayed_computations = true;
}

void num_delayed_add_loop_dims(int dim)
{
	num_init_delayed();

	if (!MD_IS_SET(bart_delayed_loop_flags, dim)) {

		for (int i = 0; i < 16; i++) {

			if (-1 == bart_delayed_loop_dims[i]) {

				bart_delayed_loop_dims[i] = dim;
				bart_delayed_loop_flags |= MD_BIT(dim);
				break;
			}
		}
	}

	bart_delayed_computations = true;
}




