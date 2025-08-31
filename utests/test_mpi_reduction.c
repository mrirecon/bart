/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 *
*/

#include <complex.h>
#include <assert.h>
#include <math.h>

#include "num/flpmath.h"
#include "num/mpi_ops.h"
#include "num/rand.h"
#include "num/vptr.h"
#include "num/multind.h"

#include "utest.h"

static bool test_mpi_znorm(void)
{
	long dims[3] = { 4, 4, 4 };

	complex float* dat_cpu = md_alloc(3, dims, CFL_SIZE);
	md_gaussian_rand(3, dims, dat_cpu);

	complex float* dat_mpi = md_alloc_mpi(3, MD_BIT(1), dims, CFL_SIZE);
	md_copy(3, dims, dat_mpi, dat_cpu, CFL_SIZE);

	float nrm_cpu = md_znorm(3, dims, dat_cpu);
	float nrm_mpi = md_znorm(3, dims, dat_mpi);

	md_free(dat_cpu);
	md_free(dat_mpi);

	UT_RETURN_ASSERT(UT_TOL > fabsf(nrm_cpu - nrm_mpi));

}

UT_UNUSED_TEST(test_mpi_znorm);


static bool test_mpi_znorm_flat(void)
{
	long dims[3] = { 4, 4, 4 };

	complex float* dat_cpu = md_alloc(3, dims, CFL_SIZE);
	md_gaussian_rand(3, dims, dat_cpu);

	complex float* dat_mpi = md_alloc_mpi(3, MD_BIT(1), dims, CFL_SIZE);
	md_copy(3, dims, dat_mpi, dat_cpu, CFL_SIZE);

	float nrm_cpu = md_znorm(1, MD_DIMS(md_calc_size(3, dims)), dat_cpu);
	float nrm_mpi = md_znorm(1, MD_DIMS(md_calc_size(3, dims)), dat_mpi);

	md_free(dat_cpu);
	md_free(dat_mpi);

	UT_RETURN_ASSERT(UT_TOL > fabsf(nrm_cpu - nrm_mpi));
}

UT_UNUSED_TEST(test_mpi_znorm_flat);

static bool test_mpi_scalar_flat(void)
{
	long dims[3] = { 4, 4, 4 };

	complex float* dat_cpu = md_alloc(3, dims, CFL_SIZE);
	md_gaussian_rand(3, dims, dat_cpu);

	complex float* dat_mpi = md_alloc_mpi(3, MD_BIT(1), dims, CFL_SIZE);
	md_copy(3, dims, dat_mpi, dat_cpu, CFL_SIZE);

	float nrm_cpu = md_scalar(1, MD_DIMS(2 * md_calc_size(3, dims)), (float*)dat_cpu, (float*)dat_cpu);
	float nrm_mpi = md_scalar(1, MD_DIMS(2 * md_calc_size(3, dims)), (float*)dat_mpi, (float*)dat_mpi);

	md_free(dat_cpu);
	md_free(dat_mpi);

	UT_RETURN_ASSERT(UT_TOL > fabsf(nrm_cpu - nrm_mpi));
}

UT_UNUSED_TEST(test_mpi_scalar_flat);


static bool test_mpi_znorm_slice(void)
{
	long dims[3] = { 4, 4, 4 };

	complex float* dat_cpu = md_alloc(3, dims, CFL_SIZE);
	md_gaussian_rand(3, dims, dat_cpu);

	complex float* dat_mpi = md_alloc_mpi(3, MD_BIT(1), dims, CFL_SIZE);
	md_copy(3, dims, dat_mpi, dat_cpu, CFL_SIZE);

	long strs[3];
	md_calc_strides(3, strs, dims, CFL_SIZE);
	md_select_dims(3, ~MD_BIT(1), dims, dims);

	float nrm_cpu = md_znorm2(3, dims, strs, dat_cpu);
	float nrm_mpi = md_znorm2(3, dims, strs, dat_mpi);

	md_free(dat_cpu);
	md_free(dat_mpi);

	UT_RETURN_ASSERT(UT_TOL > fabsf(nrm_cpu - nrm_mpi));
}

UT_UNUSED_TEST(test_mpi_znorm_slice);


static bool test_mpi_znorm_slice2(void)
{
	long dims[3] = { 4, 4, 4 };

	complex float* dat_cpu = md_alloc(3, dims, CFL_SIZE);
	md_gaussian_rand(3, dims, dat_cpu);

	complex float* dat_mpi = md_alloc_mpi(3, MD_BIT(1), dims, CFL_SIZE);
	md_copy(3, dims, dat_mpi, dat_cpu, CFL_SIZE);

	long sdims[3];
	md_select_dims(3, ~MD_BIT(1), sdims, dims);

	complex float* dat_cpu2 = md_alloc_sameplace(3, sdims, CFL_SIZE, dat_cpu);
	complex float* dat_mpi2 = md_alloc_sameplace(3, sdims, CFL_SIZE, dat_mpi);

	md_slice(3, MD_BIT(1), MD_SINGLETON_STRS(3), dims, dat_cpu2, dat_cpu, CFL_SIZE);
	md_slice(3, MD_BIT(1), MD_SINGLETON_STRS(3), dims, dat_mpi2, dat_mpi, CFL_SIZE);

	float nrm_cpu = md_znorm(3, sdims, dat_cpu2);
	float nrm_mpi = md_znorm(3, sdims, dat_mpi2);

	md_free(dat_cpu2);
	md_free(dat_mpi2);
	md_free(dat_cpu);
	md_free(dat_mpi);

	UT_RETURN_ASSERT(UT_TOL > fabsf(nrm_cpu - nrm_mpi));
}

UT_UNUSED_TEST(test_mpi_znorm_slice2);

static bool test_mpi_znorm_slice3(void)
{
	long dims[3] = { 4, 2, 2 };

	complex float* dat_cpu = md_alloc(3, dims, CFL_SIZE);
	md_gaussian_rand(3, dims, dat_cpu);

	complex float* dat_mpi = md_alloc_mpi(3, 6UL, dims, CFL_SIZE);
	md_copy(3, dims, dat_mpi, dat_cpu, CFL_SIZE);

	long sdims[3];
	md_select_dims(3, ~MD_BIT(1), sdims, dims);

	complex float* dat_cpu2 = md_alloc_sameplace(3, sdims, CFL_SIZE, dat_cpu);
	complex float* dat_mpi2 = md_alloc_sameplace(3, sdims, CFL_SIZE, dat_mpi);

	md_slice(3, MD_BIT(1), MD_SINGLETON_STRS(3), dims, dat_cpu2, dat_cpu, CFL_SIZE);
	md_slice(3, MD_BIT(1), MD_SINGLETON_STRS(3), dims, dat_mpi2, dat_mpi, CFL_SIZE);

	md_free(dat_cpu);
	md_free(dat_mpi);

	float nrm_cpu = md_znorm(3, sdims, dat_cpu2);
	float nrm_mpi = md_znorm(3, sdims, dat_mpi2);

	md_free(dat_cpu2);
	md_free(dat_mpi2);

	UT_RETURN_ASSERT(UT_TOL > fabsf(nrm_cpu - nrm_mpi));
}

UT_REGISTER_TEST(test_mpi_znorm_slice3);

