/* Copyright 2023. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Bernhard Rapp
 *
*/

#include <complex.h>
#include <assert.h>
#include <stdio.h>

#include "misc/debug.h"
#include "num/mpi_ops.h"
#include "num/vptr.h"
#include "num/multind.h"

#include "utest.h"

#define N 10
#define D 16

//shift flags according to complex to real conversion
static bool test_mpi_get_flags_C2R(void)
{
	const size_t CFL_SIZE = sizeof(complex float);
	const size_t FL_SIZE = sizeof(float);

	/* so far dims is ignored in mpi_get_flags */

	const unsigned long flags = 8UL;
	const long cdims[N] = { 32, 32, 1, 8, 1, 1, 1, 1, 1, 1};

	complex float* ptr = md_alloc_mpi(N, flags, cdims, CFL_SIZE);

	long cstrs[N];
	md_calc_strides(N, cstrs, cdims, CFL_SIZE);

	const long rdims[N + 1] = { 2, 32, 32, 1, 8, 1, 1, 1, 1, 1, 1};
	long rstrs[N + 1];
	md_calc_strides(N, rstrs, rdims, FL_SIZE);

	//return
	const unsigned long complex_flags = vptr_block_loop_flags(N, cdims, cstrs, ptr, CFL_SIZE, false);
	const unsigned long real_flags = vptr_block_loop_flags(N, rdims, rstrs, ptr, FL_SIZE, false);

	md_free(ptr);

#ifdef USE_MPI
	UT_RETURN_ASSERT((flags == complex_flags) && (complex_flags == (real_flags >> 1UL)));
#else
	UT_RETURN_ASSERT(0UL == (complex_flags | real_flags));
#endif
}
UT_REGISTER_TEST(test_mpi_get_flags_C2R);

//Clear flags on sliced dims
static bool test_mpi_get_flags_slice(void)
{
	const size_t CFL_SIZE = sizeof(complex float);

	/* so far dims is ignored in mpi_get_flags */

	const unsigned long flags = 8UL;
	const long dims[N] = { 32, 32, 1, 8, 1, 1, 1, 1, 1, 1};

	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	complex float* ptr = md_alloc_mpi(N, flags, dims, CFL_SIZE);

	long sstrs[N];
	md_select_strides(N, ~flags, sstrs, strs);

	unsigned long f = vptr_block_loop_flags(N, dims, sstrs, ptr, CFL_SIZE, false);

	md_free(ptr);

	UT_RETURN_ASSERT(0UL == f);
}

UT_REGISTER_TEST(test_mpi_get_flags_slice);

//keep dims if not reshaped on distributed dimensions
static bool test_mpi_get_flags_reshape(void)
{
	const size_t CFL_SIZE = sizeof(complex float);

	const unsigned long flags = 8UL;
	const long dims[N] = { 32, 32, 1, 8, 1, 1, 1, 1, 1, 1};

	complex float* ptr = md_alloc_mpi(N, flags, dims, CFL_SIZE);

	const long reshape1_dims[N] = { 32, 16, 2, 8, 1, 1, 1, 1, 1};
	long reshape1_strs[N];
	md_calc_strides(N, reshape1_strs, reshape1_dims, CFL_SIZE);

	unsigned long f = vptr_block_loop_flags(N, dims, reshape1_strs, ptr, CFL_SIZE, false);

	md_free(ptr);

#ifdef USE_MPI
	UT_RETURN_ASSERT(flags == f);
#else
	UT_RETURN_ASSERT(0UL == f);
#endif
}
UT_REGISTER_TEST(test_mpi_get_flags_reshape);


static bool test_mpi_get_flags_roi(void)
{
	const size_t CFL_SIZE = sizeof(complex float);

	const unsigned long flags = 8UL;

	long dims[N] = { 128, 128, 1, 8, 1, 1, 1, 1, 1, 1};
	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	complex float* ptr = md_alloc_mpi(N, flags, dims, CFL_SIZE);

	long roi[N] = { 128, 32, 1, 8, 1, 1, 1, 1, 1, 1};


	unsigned long f1 = vptr_block_loop_flags(N, roi, strs, ptr, CFL_SIZE, false);

	md_free(ptr);
#ifdef USE_MPI
	UT_RETURN_ASSERT(f1 == flags);
#else
	UT_RETURN_ASSERT(0UL == f1);
#endif
}

UT_REGISTER_TEST(test_mpi_get_flags_roi);

