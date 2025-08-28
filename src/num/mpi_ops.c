/* Copyright 2023-2024. Institute of Biomedical Imaging. TU Graz.
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
#include "num/flpmath.h"
#include "num/vptr.h"

#include "mpi_ops.h"

#define UNUSED(x) (void)x


static int mpi_rank = -1;  //ranks are the process ID of MPI
static int mpi_nprocs = 1; // number of processes
bool cuda_aware_mpi = false;

#ifdef USE_MPI
static bool mpi_initialized = false;
static MPI_Comm comm = MPI_COMM_NULL;

static MPI_Comm mpi_get_comm(void)
{
	return comm;
}
#endif


void init_mpi(int* argc, char*** argv)
{
#ifdef USE_MPI
	if (!mpi_initialized) {

		mpi_initialized = true;

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
		MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, mpi_get_num_procs(), MPI_INFO_NULL, &node_comm);

		int rank_on_node;
		MPI_Comm_rank(node_comm, &rank_on_node);

		int number_of_nodes = (rank_on_node == 0);

		MPI_Allreduce(MPI_IN_PLACE, &number_of_nodes, 1, MPI_INT, MPI_SUM, comm);

		if ((1 == number_of_nodes) && !mpi_shared_files) {

			if (mpi_is_main_proc())
				debug_printf(DP_DEBUG1, "Activate shared file system support due to single node execution!\n");

			mpi_shared_files = true;
		}

		MPI_Comm_free(&node_comm);
	}
#else
	error("BART was compiled without MPI support!\n");
	UNUSED(argc);
	UNUSED(argv);
#endif
}

void deinit_mpi(void)
{
#ifdef USE_MPI
	if (mpi_initialized)
		MPI_Finalize();
#endif
}

void abort_mpi(int err_code)
{
#ifdef USE_MPI
	if (1 < mpi_get_num_procs())
		MPI_Abort(comm, err_code);
#else
	UNUSED(err_code);
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

	MPI_Comm new_comm;
	MPI_Comm_split(comm, !signoff, mpi_get_rank(), &new_comm);
	MPI_Comm_free(&comm);

	comm = new_comm;

	if (!signoff) {

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

#ifdef USE_MPI
#ifdef USE_CUDA
static void print_cuda_aware_warning(void)
{
	static bool printed = false;

	if (!printed && !cuda_aware_mpi)
		debug_printf(DP_WARN, "CUDA aware MPI is not activated. This may decrease performance for multi-GPU operations significantly!.\n");

	printed = true;
}
#endif
#endif

void mpi_sync(void)
{
#ifdef USE_MPI
	if (1 < mpi_get_num_procs())
		MPI_Barrier(mpi_get_comm());
#endif
}

#ifdef USE_MPI
#ifdef USE_CUDA
static void mpi_bcast_selected_gpu(bool tag, void* ptr, long size, int root)
{
	if (1 == mpi_get_num_procs())
		return;

	print_cuda_aware_warning();

	void* tmp = xmalloc((size_t)size);

	if (mpi_get_rank() == root)
		cuda_memcpy(size, tmp, ptr);

	mpi_bcast_selected(tag, tmp, size, root);

	if (tag)
		cuda_memcpy(size, ptr, tmp);

	xfree(tmp);
}
#endif
#endif

void mpi_bcast_selected(bool tag, void* ptr, long size, int root)
{
#ifdef USE_MPI
	if (1 == mpi_get_num_procs())
		return;

#ifdef USE_CUDA
	if (!cuda_aware_mpi && cuda_ondevice(ptr)) {

		mpi_bcast_selected_gpu(tag, ptr, size, root);
		return;
	}

	if (cuda_ondevice(ptr))
		cuda_sync_stream();
#endif

	MPI_Comm comm_sub;
	MPI_Comm_split(mpi_get_comm(), tag, (mpi_get_rank() != root), &comm_sub);

	if (tag) {

		for (long n = 0; n < size; n += INT_MAX / 2)
			MPI_Bcast(ptr + n, MIN(size - n, INT_MAX / 2), MPI_BYTE, 0, comm_sub);
	}

	MPI_Comm_free(&comm_sub);
#else
	UNUSED(tag);
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

		if ((mpi_get_rank() == sender_rank) && (dst != src)) {

#ifdef USE_CUDA
			if (cuda_ondevice(dst) || cuda_ondevice(src))
				cuda_memcpy(size, dst, src);
			else
#endif
				memcpy(dst, src, (size_t)size);
		}

		return;
	}

#ifdef USE_MPI
	if (mpi_get_rank() == sender_rank) {

		void* src2 = (void*)src;

#ifdef USE_CUDA
		if (cuda_ondevice(src) && !cuda_aware_mpi) {

			print_cuda_aware_warning();

			src2 = xmalloc((size_t)size);
			cuda_memcpy(size, src2, src);
		}

		if (cuda_ondevice(src2))
			cuda_sync_stream();
#endif

		for (long n = 0; n < size; n += INT_MAX / 2)
			MPI_Send(src2 + n, MIN(size - n, INT_MAX / 2), MPI_BYTE, recv_rank, 0, mpi_get_comm());

#ifdef USE_CUDA
		if (cuda_ondevice(src) && !cuda_aware_mpi)
			xfree(src2);
#endif
	}

	if (mpi_get_rank() == recv_rank) {

		void* dst2 = dst;

#ifdef USE_CUDA
		if (cuda_ondevice(dst) && !cuda_aware_mpi) {

			print_cuda_aware_warning();
			dst2 = xmalloc((size_t)size);
		}
#endif

		for (long n = 0; n < size; n += INT_MAX / 2)
			MPI_Recv(dst2 + n, MIN(size - n, INT_MAX / 2), MPI_BYTE, sender_rank, 0, mpi_get_comm(), MPI_STATUS_IGNORE);

#ifdef USE_CUDA
		if (cuda_ondevice(dst) && !cuda_aware_mpi) {

			cuda_memcpy(size, dst, dst2);
			xfree(dst2);
		}
#endif
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
		long size2 = size * opt_data->size;

		mpi_copy(ptr[0], size2, ptr[1], sender_rank, recv_rank);
	};

	optimized_nop(2, MD_BIT(0), N, dim, nstr, (void*[2]){ optr, (void*)iptr }, (size_t[2]){ (size_t)size, (size_t)size }, nary_copy_mpi);
}


/**
 * Synchronise pval to all processes (take part in calculation)
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
 * (rank != 0). For rank == 0, dst == src, evenly over communicator
 *
 * This function requires Communicator handling!
 *
 * @param dst destination buffer of ranks != 0
 * @param count elements which should be copied
 * @param src buffer that holds enough data to spread to buffers
 * @param size size of a single element
 */
void mpi_scatter_batch(void* dst, long count, const void* src, size_t size)
{
#ifdef USE_MPI
	count *= (long)size;
	assert(count < INT_MAX);

	MPI_Scatter(src, count, MPI_BYTE, ((0 == mpi_get_rank()) && (dst == src)) ? MPI_IN_PLACE : dst,
			count, MPI_BYTE, 0, mpi_get_comm());
#else
	UNUSED(src);
	UNUSED(count);
	UNUSED(dst);
	UNUSED(size);
#endif
}



/**
 * Copies N elements from src buffer (rank = 0) to dst buffers
 * (rank != 0). For rank == 0, dst == src, evenly over communicator
 *
 * This function requires Communicator handling!
 *
 * @param dst destination buffer of ranks != 0
 * @param count elements which should be copied
 * @param src buffer that holds enough data to spread to buffers
 * @param size size of a single element
 * @param to_all distribute values to all
 */
void mpi_gather_batch(void* dst, long count, const void* src, size_t size)
{
#ifdef USE_MPI
	count *= (long)size;
	assert(count < INT_MAX);

	MPI_Gather(((0 == mpi_get_rank()) && (dst == src)) ? MPI_IN_PLACE : src, count,
			MPI_BYTE, dst, count, MPI_BYTE, 0, mpi_get_comm());
#else
	UNUSED(dst);
	UNUSED(count);
	UNUSED(src);
	UNUSED(size);
#endif
}


/*
* Reduction kernels
*/

#ifdef USE_MPI
#ifdef USE_CUDA
static void mpi_reduce_land_gpu(long N, bool vec[N])
{
	print_cuda_aware_warning();

	long size = (long)sizeof(bool[N]);

	bool* tmp = xmalloc((size_t)size);
	cuda_memcpy(size, tmp, vec);

	mpi_reduce_land(N, tmp);

	cuda_memcpy(size, vec, tmp);
	xfree(tmp);
}
#endif
#endif

void mpi_reduce_land(long N, bool vec[__VLA(N)])
{
	if (1 == mpi_get_num_procs())
		error("MPI reduction requested but only run by one process!\n");

#ifdef USE_MPI
#ifdef USE_CUDA
	if (!cuda_aware_mpi && cuda_ondevice(vec)) {

		mpi_reduce_land_gpu(N, vec);
		return;
	}

	if (cuda_ondevice(vec))
		cuda_sync_stream();
#endif

	for (long n = 0; n < N; n += INT_MAX / 2)
		MPI_Allreduce(MPI_IN_PLACE, vec + n, MIN(N - n, INT_MAX / 2), MPI_C_BOOL, MPI_LAND, mpi_get_comm());
#else
	(void)vec;
#endif
}


#ifdef USE_MPI
static void mpi_allreduce_sum_gpu(int N, float vec[N], MPI_Comm comm)
{
#ifdef USE_CUDA
	if (!cuda_aware_mpi && cuda_ondevice(vec)) {

		print_cuda_aware_warning();

		long size = (long)sizeof(float[N]);

		float* tmp = xmalloc((size_t)size);
		cuda_memcpy(size, tmp, vec);

		MPI_Allreduce(MPI_IN_PLACE, tmp, N, MPI_FLOAT, MPI_SUM, comm);

		cuda_memcpy(size, vec, tmp);
		xfree(tmp);

		return;
	}

	if (cuda_ondevice(vec))
		cuda_sync_stream();
#endif

	MPI_Allreduce(MPI_IN_PLACE, vec, N, MPI_FLOAT, MPI_SUM, comm);
}
#endif

#ifdef USE_MPI
static void mpi_reduce_sum_kernel(unsigned long reduce_flags, long N, float vec[N])
{
	if (1 == mpi_get_num_procs())
		error("MPI reduction requested but only run by one process!\n");

	int tag = mpi_reduce_color(reduce_flags, vec);

	MPI_Comm comm_sub;
	MPI_Comm_split(mpi_get_comm(), tag, 0, &comm_sub);

	if (0 < tag) {

		vec = vptr_resolve(vec);

		for (long n = 0; n < N; n += INT_MAX / 2)
			mpi_allreduce_sum_gpu(MIN(N - n, INT_MAX / 2), vec + n, comm_sub);
	}

	MPI_Comm_free(&comm_sub);
}
#endif


#ifdef USE_MPI
void mpi_reduce_sum_vector(long N, float vec[N])
{
	if (1 == mpi_get_num_procs())
		error("MPI reduction requested but only run by one process!\n");

	for (long n = 0; n < N; n += INT_MAX / 2)
		mpi_allreduce_sum_gpu(MIN(N - n, INT_MAX / 2), vec + n, mpi_get_comm());
}
#endif

void mpi_reduce_sum(int N, unsigned long reduce_flags, const long dims[N], float* ptr)
{
	long tdims[N];
	md_copy_dims(N, tdims, dims);

	long strs[N];
	md_calc_strides(N, strs, dims, FL_SIZE);

	unsigned long block_flags = vptr_block_loop_flags(N, dims, strs, ptr, FL_SIZE);

	long size = 1;

	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(block_flags, i))
			break;

		if (strs[i] == size * (long)FL_SIZE) {

			size *= tdims[i];
			tdims[i] = 1;
		}
	}

	long pos[N];
	md_singleton_strides(N, pos);

	do {
#ifdef USE_MPI
		mpi_reduce_sum_kernel(reduce_flags, size, &MD_ACCESS(N, strs, pos, ptr));
#else
		(void)reduce_flags;
#endif
	} while (md_next(N, tdims, ~0UL, pos));
}

void mpi_reduce_zsum(int N, unsigned long reduce_flags, const long dims[N], complex float* ptr)
{
	mpi_reduce_sum(N + 1, reduce_flags, MD_REAL_DIMS(N, dims), (float*)ptr);
}

void mpi_reduce_zsum_vector(long N, complex float ptr[N])
{
#ifdef USE_MPI
	mpi_reduce_sum_vector(2 * N, (float*)ptr);
#else
	(void)N;
	(void)ptr;
#endif
}

#ifdef USE_MPI
static void mpi_allreduce_sumD_gpu(int N, double vec[N], MPI_Comm comm)
{
#ifdef USE_CUDA
	if (!cuda_aware_mpi && cuda_ondevice(vec)) {

		print_cuda_aware_warning();

		long size = (long)sizeof(double[N]);

		float* tmp = xmalloc((size_t)size);
		cuda_memcpy(size, tmp, vec);

		MPI_Allreduce(MPI_IN_PLACE, tmp, N, MPI_DOUBLE, MPI_SUM, comm);

		cuda_memcpy(size, vec, tmp);
		xfree(tmp);

		return;
	}

	if (cuda_ondevice(vec))
		cuda_sync_stream();
#endif
	MPI_Allreduce(MPI_IN_PLACE, vec, N, MPI_DOUBLE, MPI_SUM, comm);
}
#endif

#ifdef USE_MPI
static void mpi_reduce_sumD_kernel(unsigned long reduce_flags, long N, double vec[N])
{
	if (1 == mpi_get_num_procs())
		error("MPI reduction requested but only run by one process!\n");

	int tag = mpi_reduce_color(reduce_flags, vec);

	MPI_Comm comm_sub;
	MPI_Comm_split(mpi_get_comm(), tag, 0, &comm_sub);

	if (0 < tag) {

		double *vec2 = vptr_resolve(vec);

		for (long n = 0; n < N; n += INT_MAX / 2)
			mpi_allreduce_sumD_gpu(MIN(N - n, INT_MAX / 2), vec2 + n, comm_sub);
	}

	MPI_Comm_free(&comm_sub);
}
#endif

void mpi_reduce_sumD(int N, unsigned long reduce_flags, const long dims[N], double* ptr)
{
	long tdims[N];
	md_copy_dims(N, tdims, dims);

	long strs[N];
	md_calc_strides(N, strs, dims, DL_SIZE);

	unsigned long block_flags = vptr_block_loop_flags(N, dims, strs, ptr, DL_SIZE);

	long size = 1;

	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(block_flags, i))
			break;

		if (strs[i] == size * (long)DL_SIZE) {

			size *= tdims[i];
			tdims[i] = 1;
		}
	}

	long pos[N];
	md_singleton_strides(N, pos);

	do {
#ifdef USE_MPI
		mpi_reduce_sumD_kernel(reduce_flags, size, &MD_ACCESS(N, strs, pos, ptr));
#else
		(void)reduce_flags;
#endif

	} while (md_next(N, tdims, ~0UL, pos));
}

void mpi_reduce_zsumD(int N, unsigned long reduce_flags, const long dims[N], complex double* ptr)
{
	mpi_reduce_sumD(N + 1, reduce_flags, MD_REAL_DIMS(N, dims), (double*)ptr);
}


