/* Copyright 2023. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * Bernhard Rapp
 * Moritz Blumenthal
 */


#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef OPEN_MPI
#include <mpi-ext.h>
#endif

#include <complex.h>
#include <assert.h>
#include <stdbool.h>
#include <limits.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "mpi_ops.h"

static int mpi_rank = -1;  //ranks are the process ID of MPI
static int mpi_nprocs = 1; // number of processes
bool cuda_aware_mpi = false;

#ifdef USE_MPI
static MPI_Comm comm = MPI_COMM_NULL;

static MPI_Comm mpi_get_comm(void)
{
	return comm;
}

#endif


void init_mpi(int* argc, char*** argv)
{
#ifdef USE_MPI
	int initialized = 0;
	MPI_Initialized(&initialized);
	if (!initialized) {

		MPI_Init(argc, argv);
		MPI_Comm_dup(MPI_COMM_WORLD, &comm);
		MPI_Comm_rank(comm, &mpi_rank);
		MPI_Comm_size(comm, &mpi_nprocs);
		
		if (1 == mpi_nprocs)
			return;

#ifdef MPIX_CUDA_AWARE_SUPPORT
		if (1 == MPIX_Query_cuda_support())
			cuda_aware_mpi = true;
#endif

		MPI_Comm node_comm;
		MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, mpi_get_num_procs(),MPI_INFO_NULL, &node_comm);

		int rank_on_node;
		MPI_Comm_rank(node_comm, &rank_on_node);
		
		int number_of_nodes = (rank_on_node == 0);
		MPI_Allreduce(MPI_IN_PLACE, &number_of_nodes, 1, MPI_INT, MPI_SUM, comm);

		if ((1 == number_of_nodes) && (!mpi_shared_files)) {

			if (mpi_is_main_proc())
				debug_printf(DP_DEBUG1, "Activate shared file system support due to single node execution!\n");

			mpi_shared_files = true;
		}

		MPI_Comm_free(&node_comm);
	}
		
#else
	UNUSED(argc);
	UNUSED(argv);
#endif
}

void deinit_mpi(void)
{
#ifdef USE_MPI
	int finalized = 0;
	MPI_Finalized(&finalized);
	if (!finalized)
		MPI_Finalize();
#endif
}

int mpi_get_rank(void)
{
	return MAX(0, mpi_rank);
}

int mpi_get_num_procs(void)
{
	return mpi_nprocs;
}

bool mpi_is_main_proc(void)
{
	return 0 == mpi_get_rank();
}

/**
* Drops process from communication
 */
void mpi_signoff_proc(bool signoff)
{
#ifdef USE_MPI
	if (1 >= mpi_get_num_procs())
		return;

	int tag = signoff ? 0 : 1;

	MPI_Comm new_comm;	
	MPI_Comm_split(comm, tag, mpi_get_rank(), &new_comm);
	MPI_Comm_free(&comm);
	comm = new_comm;

	if (tag) {

		MPI_Comm_rank(comm, &mpi_rank);
		MPI_Comm_size(comm, &mpi_nprocs);
	} else {

		mpi_rank = 0;
		mpi_nprocs = 1;
	}

#else
	UNUSED(signoff);
#endif
}


void mpi_sync(void)
{
#ifdef USE_MPI
	if(1 < mpi_get_num_procs())
		MPI_Barrier(mpi_get_comm());
#endif
}

void mpi_sync_val(void* pval, long size)
{
#ifdef USE_MPI
	if(1 < mpi_get_num_procs())
		MPI_Bcast((char*)pval, size, MPI_CHAR, 0, mpi_get_comm());
#else
	UNUSED(pval);
	UNUSED(size);
#endif
}

/**
 * Inplace scatter src to dest in block of size N
 * Copies N elements from src buffer (rank = 0) to dst buffers
 * (rank != 0). For rank == 0, dst == src, evently over communicator
 * 
 * This function requires Communicator handling!
 *
 * @param dst destination buffer of ranks != 0
 * @param count elements which should be copied
 * @param src buffer that holds enough data to spread to buffers
 * @param type_size of a single element
 */
void mpi_scatter_batch(void* dst, long count, const void* src, size_t type_size)
{
#ifdef USE_MPI
	count *= type_size;
	assert(count < INT_MAX);

	MPI_Scatter(src, count, MPI_BYTE, (0 == mpi_get_rank() && dst == src) ? MPI_IN_PLACE : dst, count, MPI_BYTE, 0, mpi_get_comm());
#else
	UNUSED(src);
	UNUSED(count);
	UNUSED(dst);
	UNUSED(type_size);
#endif
}



/**
 * Copies N elements from src buffer (rank = 0) to dst buffers
 * (rank != 0). For rank == 0, dst == src, evently over communicator
 * 
 * This function requires Communicator handling!
 *
 * @param dst destination buffer of ranks != 0
 * @param count elements which should be copied
 * @param src buffer that holds enough data to spread to buffers
 * @param type_size of a single element
 * @param to_all distribute values to all
 */
void mpi_gather_batch(void* dst, long count, const void* src, size_t type_size)
{
#ifdef USE_MPI
	count *= type_size;
	assert(count < INT_MAX);

	MPI_Gather((0 == mpi_get_rank() && dst == src) ? MPI_IN_PLACE : src, count, MPI_BYTE, dst, count, MPI_BYTE, 0, mpi_get_comm());
#else
	UNUSED(dst);
	UNUSED(count);
	UNUSED(src);
	UNUSED(type_size);
#endif
}


