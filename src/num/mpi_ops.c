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

#include <stdbool.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "mpi_ops.h"

static int mpi_rank = -1;  //ranks are the process ID of MPI
static int mpi_nprocs = 1; // number of processes


void init_mpi(int* argc, char*** argv)
{
#ifdef USE_MPI
	if (-1 == mpi_rank) {
		MPI_Init(argc, argv);
		MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
		MPI_Comm_size(MPI_COMM_WORLD, &mpi_nprocs);
	}
#else
	UNUSED(argc);
	UNUSED(argv);
#endif
}

void deinit_mpi(void)
{
#ifdef USE_MPI
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
	return 0 == mpi_get_num_procs();
}

#ifdef USE_MPI

static MPI_Comm comm = MPI_COMM_NULL;
static bool use_comm_world = true;;
MPI_Comm mpi_get_comm(void)
{
	if ((MPI_COMM_NULL == comm) || use_comm_world)
		return MPI_COMM_WORLD;

	return comm;
}

void mpi_split_comm(MPI_Comm base_comm, int tag)
{
	if (comm != MPI_COMM_NULL) {

		debug_printf(DP_DEBUG4, "Reset MPI comm\n");
		MPI_Comm_free(&comm);
	}
	MPI_Comm_split(base_comm, tag, mpi_get_rank(), &comm);
}

void mpi_comm_subset_activate(void)
{
	use_comm_world = false;
}

#endif

