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
#include <unistd.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "num/vptr.h"
#include "num/mpi_ops.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"

#include "utest.h"

//Merge distributed data to local
static bool test_mpi_copy_from_distributed(unsigned long flags, int ndim1, int ndim2, int ndim3, int ndim4)
{
	enum { N = 10 };

	const long dims[N] = { 32, 32, 1, ndim1, ndim2, ndim3, ndim4, 1, 1, 1};
	long strs[N];
	md_calc_strides(N, strs, dims, 1);

	long sdims[N];
	md_select_dims(N, 3UL, sdims, dims);

	complex float* ref = md_alloc(N, dims, CFL_SIZE);
	complex float* dist_ptr = md_alloc_mpi(N, flags, dims, CFL_SIZE);
	for(int i = 0; i < (ndim1 * ndim2 * ndim3 * ndim4); i++) {
		complex float val = i + 5;

		long pos[N];
		md_singleton_strides(N, pos);
		md_unravel_index(N, pos, ~3UL, dims, i);
		long offset = md_calc_offset(N, strs, pos);

		md_zfill(N, sdims, dist_ptr + offset, val);
		md_zfill(N, sdims, ref + offset, val);
	}


	complex float* ptr = md_alloc(N, dims, CFL_SIZE);
	md_copy(N, dims, ptr, dist_ptr, CFL_SIZE);

	float err = md_znrmse(N, dims, ref, ptr);

	md_free(dist_ptr);
	md_free(ptr);
	md_free(ref);

	UT_RETURN_ASSERT(err < UT_TOL);
}

static bool test_mpi_copy_from_distributed_3_3(void) 	{ return test_mpi_copy_from_distributed(MD_BIT(3),  3, 1,1,1); }
static bool test_mpi_copy_from_distributed_36_2_6(void) { return test_mpi_copy_from_distributed(MD_BIT(3)|MD_BIT(6), 2, 2,3,6); }

UT_REGISTER_TEST(test_mpi_copy_from_distributed_3_3);
UT_REGISTER_TEST(test_mpi_copy_from_distributed_36_2_6);

static bool test_mpi_copy2_roi_with_shift(void)
{
	enum { N = 10 };

	long dims[N] = { 4, 4, 1, 4, 1, 1, 1, 1, 1, 1};
	long roi_dims[N] = { 2, 2, 1, 4, 1, 1, 1, 1, 1, 1};
	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	complex float* ref_in = md_alloc(N, dims, CFL_SIZE);
	md_gaussian_rand(N, dims, ref_in);
	complex float* ptr_in = md_mpi_move(N, MD_BIT(3), dims, ref_in, CFL_SIZE);

	complex float val = 1;

	complex float* ref = md_alloc(N, dims, CFL_SIZE);
	complex float* ptr = md_alloc_mpi(N, 8, dims, CFL_SIZE);
	
	md_zfill(N, dims, ref, val);
	md_zfill(N, dims, ptr, val);

	md_copy2(N, roi_dims, strs, ref, strs, ref_in + 5, CFL_SIZE);
	md_copy2(N, roi_dims, strs, ptr, strs, ptr_in + 5, CFL_SIZE);

	complex float* ptr_cpu = md_alloc(N, dims, CFL_SIZE);
	md_copy(N, dims, ptr_cpu, ptr, CFL_SIZE);

	bool equal = md_compare(N, roi_dims, ref, ptr_cpu, CFL_SIZE);

	UT_RETURN_ASSERT(equal);
}

//Deatdlock!!!!!!!!
UT_UNUSED_TEST(test_mpi_copy2_roi_with_shift);

static bool test_mpi_copy_roi_with_shift(void)
{
	enum { N = 10 };
	long dims[N] = { 4, 4, 1, 4, 1, 1, 1, 1, 1, 1};
	long roi_dims[N] = { 2, 2, 1, 4, 1, 1, 1, 1, 1, 1};

	complex float* ref_in = md_alloc(N, dims, CFL_SIZE);
	md_gaussian_rand(N, dims, ref_in);
	complex float* ptr_in = md_mpi_move(N, 8UL, dims, ref_in, CFL_SIZE);

	complex float* ref = md_alloc(N, dims, CFL_SIZE);
	complex float* ptr = md_alloc_mpi(N, 8, dims, CFL_SIZE);
	
	md_copy(N, roi_dims, ref, ref_in + 1, CFL_SIZE);
	md_copy(N, roi_dims, ptr, ptr_in + 1, CFL_SIZE);

	complex float* ptr_cpu = md_alloc(N, dims, CFL_SIZE);
	md_copy(N, dims, ptr_cpu, ptr, CFL_SIZE);

	bool equal = md_compare(N, roi_dims, ref, ptr_cpu, CFL_SIZE);

	UT_RETURN_ASSERT(equal);
}

UT_UNUSED_TEST(test_mpi_copy_roi_with_shift);

static bool test_mpi_transpose(void)
{
	enum { N = 4 };
	long dims[N] = { 10, 10, 10, 10 };

	complex float* a = md_alloc(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, a);
	complex float* a_dist = md_mpi_move(N, 3, dims, a, CFL_SIZE);

	complex float* b = md_alloc_mpi(N, 3, dims, CFL_SIZE);
	complex float* c = md_alloc_mpi(N, 3, dims, CFL_SIZE);

	md_transpose(N, 0, 2, dims, b, dims, a_dist, CFL_SIZE);
	md_transpose(N, 0, 2, dims, c, dims, b, CFL_SIZE);

	bool eq = md_compare(N, dims, a_dist, c, CFL_SIZE);

	md_free(a);
	md_free(a_dist);
	md_free(b);
	md_free(c);

	return eq;
}


UT_UNUSED_TEST(test_mpi_transpose);


static bool test_mpi_reshape(void)
{
	enum { N = 4 };
	long dims1[N] = { 10, 10, 10, 10 };
	long dims2[N] = { 10, 20, 10,  5 };
	long dims3[N] = {  5, 20, 20,  5 };
	long dims4[N] = {  5, 10, 20, 10 };

	complex float* a = md_alloc(N, dims1, CFL_SIZE);
	complex float* b = md_alloc_mpi(N, 4, dims1, CFL_SIZE);
	complex float* c = md_alloc_mpi(N, 4, dims1, CFL_SIZE);

	md_gaussian_rand(N, dims1, a);
	complex float* a_dist = md_mpi_move(N, 4, dims1, a, CFL_SIZE);

	md_reshape(N, MD_BIT(1)|MD_BIT(3), dims2, b, dims1, a_dist, CFL_SIZE);
	md_reshape(N, MD_BIT(0)|MD_BIT(2), dims3, c, dims2, b, CFL_SIZE);
	md_reshape(N, MD_BIT(1)|MD_BIT(3), dims4, b, dims3, c, CFL_SIZE);
	md_reshape(N, MD_BIT(0)|MD_BIT(2), dims1, c, dims4, b, CFL_SIZE);

	bool eq = md_compare(N, dims1, a_dist, c, CFL_SIZE);

	md_free(a);
	md_free(a_dist);
	md_free(b);
	md_free(c);

	return eq;
}

UT_UNUSED_TEST(test_mpi_reshape);

static bool test_mpi_flip(void)
{
	enum { N = 4 };
	long dims[N] = { 10, 10, 10, 10 };

	complex float* a = md_alloc(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, a);
	complex float* a_dist = md_mpi_move(N, 4, dims, a, CFL_SIZE);

	complex float* b = md_alloc_mpi(N, 4, dims, CFL_SIZE);
	complex float* c = md_alloc_mpi(N, 4, dims, CFL_SIZE);

	md_flip(N, dims, MD_BIT(0) | MD_BIT(2), b, a_dist, CFL_SIZE);
	md_flip(N, dims, MD_BIT(0) | MD_BIT(2), c, b, CFL_SIZE);

	bool eq = md_compare(N, dims, a_dist, c, CFL_SIZE);

	md_free(a);
	md_free(a_dist);
	md_free(b);
	md_free(c);

	return eq;
}

UT_UNUSED_TEST(test_mpi_flip);