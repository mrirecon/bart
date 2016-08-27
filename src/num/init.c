/* Copyright 2013. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
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
#include "misc/misc.h"
#include "num/fft.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#ifdef USE_CULA
#include <cula_lapack_device.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "init.h"


#ifdef USE_MPI
// for use in optimize.c
#define MPI_MAX_DIMS (sizeof(mpi_flags) * 8)
unsigned long mpi_flags = 0;
int mpi_rank = 0;
int mpi_size = 0;
long mpi_dims[MPI_MAX_DIMS] = { [0 ... MPI_MAX_DIMS - 1] = 1 };
long mpi_position[MPI_MAX_DIMS] = { 0 };
#endif


static void num_init2(void)
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
#else
	int p = 2;
#endif
#ifdef FFTWTHREADS
	fft_set_num_threads(p);
#endif
}

static void mpi_exit(void)
{
	MPI_Finalize();
}

bool num_root_node_p(void)
{
#ifdef USE_MPI
	return (0 == mpi_rank);
#else
	return true;
#endif	
}

void num_init(void)
{
	num_init2();
#ifdef USE_MPI
	const char* mpi = getenv("BART_MPI_FLAGS");

	if (NULL != mpi) {

		MPI_Init(NULL, NULL);
		atexit(mpi_exit);

		mpi_flags = atoi(mpi);
		
		assert(0u == (mpi_flags & (mpi_flags - 1)));

		MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
		MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

		debug_printf(DP_INFO, "MPI Rank: %d/%d\n", mpi_rank, mpi_size);

		for (unsigned int i = 0; i < ARRAY_SIZE(mpi_position); i++) {

			if (mpi_flags & (1 << i)) {

				mpi_position[i] = mpi_rank;
				mpi_dims[i] = mpi_size;
			}
		}
	}
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


