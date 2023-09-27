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
#include "num/multind.h"
#include "num/optimize.h"

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

#ifdef USE_CUDA
static void print_cuda_aware_warning(void)
{
	static bool printed = false;
	
	if (!printed && !cuda_aware_mpi)
		debug_printf(DP_WARN, "CUDA aware MPI is not activated. This may decrease performance for multi-GPU operations significantly!.\n");

	printed = true;
}
#endif


void mpi_sync(void)
{
#ifdef USE_MPI
	if(1 < mpi_get_num_procs())
		MPI_Barrier(mpi_get_comm());
#endif
}

#ifdef USE_CUDA
static void mpi_bcast_selected_gpu(bool _tag, void* ptr, long size, int root)
{
	if(1 == mpi_get_num_procs())
		return;

	print_cuda_aware_warning();

	void* tmp = xmalloc(size);
	
	if (mpi_get_rank() == root)
		cuda_memcpy(size, tmp, ptr);

	mpi_bcast_selected(_tag, tmp, size, root);

	if (_tag)
		cuda_memcpy(size, ptr, tmp);

	xfree(tmp);
}
#endif

void mpi_bcast_selected(bool _tag, void* ptr, long size, int root)
{
#ifdef USE_MPI
	if(1 == mpi_get_num_procs())
		return;

#ifdef USE_CUDA
	if (!cuda_aware_mpi && cuda_ondevice(ptr)) {

		mpi_bcast_selected_gpu(_tag, ptr, size, root);
		return;
	}
#endif

	int tag = _tag ? 1 : 0;
	MPI_Comm comm_sub;
	MPI_Comm_split(mpi_get_comm(), tag, (mpi_get_rank() == root) ? 0 : 1, &comm_sub);

	if (_tag) {

		void* end = ptr + size;
		while (ptr < end) {

			MPI_Bcast(ptr, MIN(end - ptr, INT_MAX / 2), MPI_BYTE, 0, comm_sub);
			ptr += MIN(end - ptr, INT_MAX / 2);
		}
	}

	MPI_Comm_free(&comm_sub);
#else
	UNUSED(_tag);
	UNUSED(ptr);
	UNUSED(size);
	UNUSED(root);
#endif
}

void mpi_bcast(void* ptr, long size, int root)
{
	mpi_bcast_selected(true, ptr, size, root);
}

void mpi_bcast2(int N, const long dims[N], const long strs[N], void* ptr, long size, int root)
{
	long tdims[N];
	md_copy_dims(N, tdims, dims);

	for (int i = 0; i < N; i++) {

		if (strs[i] == size) {

			size *= tdims[i];
			tdims[i] = 1;
		}
	}

	NESTED(void, nary_bcast, (void* ptr[]))
	{
		mpi_bcast(ptr[0], size, root);
	};
	
	md_nary(1, N, tdims, &strs, &ptr, nary_bcast);
}


/**
* Data transfer API
**/

void mpi_copy(void* dst, long size, const void* src, int sender_rank, int recv_rank)
{
	if (sender_rank == recv_rank) {

		if ((mpi_get_rank() == sender_rank) && dst != src)
			memcpy(dst, src, size);

		return;
	}

#ifdef USE_MPI
	if (mpi_get_rank() == sender_rank) {

		const void* end = src + size;

		while (src < end) {

			int tsize = MIN(end - src, INT_MAX / 2);
			MPI_Send(src, tsize, MPI_BYTE, recv_rank, 0, mpi_get_comm());
			src += tsize;
		}
	}

	if (mpi_get_rank() == recv_rank) {

		void* end = dst + size;

		while (dst < end) {

			int tsize = MIN(end - src, INT_MAX / 2);
			MPI_Recv(dst, size, MPI_BYTE, sender_rank, 0, mpi_get_comm(), MPI_STATUS_IGNORE);
			dst += tsize;
		}
	}
#else
	UNUSED(dst);
	UNUSED(size);
	UNUSED(src);
	UNUSED(sender_rank);
	UNUSED(recv_rank);
#endif
}

void mpi_copy2(int N, const long dim[N], const long ostr[N], void* optr, const long istr[N], const void* iptr, long size, int sender_rank, int recv_rank)
{
	const long (*nstr[2])[N] = { (const long (*)[N])ostr, (const long (*)[N])istr };

	NESTED(void, nary_copy_mpi, (struct nary_opt_data_s* opt_data, void* ptr[]))
	{
		size_t size2 = size * opt_data->size;

		mpi_copy(ptr[0], size2, ptr[1], sender_rank, recv_rank);
	};

	optimized_nop(2, MD_BIT(0), N, dim, nstr, (void*[2]){ optr, (void*)iptr }, (size_t[2]){ size, size }, nary_copy_mpi);
}

/**
 * Syncronise pval to all processes (take part in calculation)
 * 
 * This function requires Communicator handling!
 *
 * @param pval source (rank == 0) /destination (rank != 0) buffer
 * @param size size in bytes which should be copied
 */
void mpi_sync_val(void* pval, long size)
{
	mpi_bcast(pval, size, 0);
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


