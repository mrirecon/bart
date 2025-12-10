/* Copyright 2023. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Bernhard Rapp
 *
*/

#include <complex.h>
#include <assert.h>
#include <math.h>

#include "num/mpi_ops.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "utest.h"

// include test data
#include "test_flpmath_data.h"

typedef void (*z3opd_t)(int D, const long dims[D], const long ostrs[D], complex double* optr, const long istrs1[D], const complex float* iptr1, const long istrs2[D], const complex float* iptr2);
typedef void (*z3op_t)(int D, const long dims[D], const long ostrs[D], complex float* optr, const long istrs1[D], const complex float* iptr1, const long istrs2[D], const complex float* iptr2);
typedef void (*r3op_t)(int D, const long dims[D], const long ostrs[D], float* optr, const long istrs1[D], const float* iptr1, const long istrs2[D], const float* iptr2);
typedef void (*r3opd_t)(int D, const long dims[D], const long ostrs[D], double* optr, const long istrs1[D], const float* iptr1, const long istrs2[D], const float* iptr2);

static bool test_mpi_zscalar2(unsigned long mpi_flags)
{
	enum { N = 5};
	long dims[N] = { 2, 3, 3, 3, 3 };
	long strs[N ];
	md_calc_strides(N, strs, dims, FL_SIZE);

	float* ptr_dist = md_mpi_move(N, mpi_flags, dims, (float*)test_md_in0, FL_SIZE);

	float zscalar_dist = md_scalar2(N, dims, strs, ptr_dist, strs, ptr_dist);
	float zscalar_ref = md_scalar2(N, dims, strs, (float*)test_md_in0, strs, (float*)test_md_in0);

	md_free(ptr_dist);

	float err = zscalar_ref - zscalar_dist;
	UT_RETURN_ASSERT(err < UT_TOL);
}

static bool test_mpi_scalar2_8(void)	{ return test_mpi_zscalar2(8UL); }
static bool test_mpi_scalar2_12(void)	{ return test_mpi_zscalar2(12UL); }

UT_REGISTER_TEST(test_mpi_scalar2_8);
UT_REGISTER_TEST(test_mpi_scalar2_12);


static bool test_mpi_z3opd(z3opd_t test_fun, unsigned long mpi_flags)
{
	enum { N = 4};
	long dims[N] = { 3, 3, 3, 3 };
	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	complex float* in1 = md_mpi_move(N, mpi_flags, dims, test_md_in0, CFL_SIZE);
	complex float* in2 = md_mpi_move(N, mpi_flags, dims, test_md_in1, CFL_SIZE);

	long dstrs[N];
	md_calc_strides(N, dstrs, dims, CDL_SIZE);
	complex double* ret = md_alloc_mpi(N, mpi_flags, dims, CDL_SIZE);
	md_clear(N, dims, ret, CDL_SIZE);

	test_fun(N, dims, dstrs, ret, strs, in1, strs, in2);

	complex double* ret_copy = md_alloc(N, dims, CDL_SIZE);
	md_copy(N, dims, ret_copy, ret, CDL_SIZE);

	complex double* ref = md_alloc(N, dims, CDL_SIZE);
	md_clear(N, dims, ref, CDL_SIZE);
	
	test_fun(N, dims, dstrs, ref, strs, test_md_in0, strs, test_md_in1);

	//Hacky but should work for this purpose
	long rdims[N + 1];
	rdims[0] = DL_SIZE;
	md_copy_dims(N, rdims + 1, dims);

	float err = md_znrmse(N, dims, (complex float*)ref, (complex float*)ret_copy);

	md_free(in1);
	md_free(in2);
	md_free(ret);
	md_free(ref);
	md_free(ret_copy);

	UT_RETURN_ASSERT(err < UT_TOL);
}

static bool test_mpi_zfmaccD2_3(void)		{ return test_mpi_z3opd(md_zfmaccD2,	MD_BIT(3))		; }
static bool test_mpi_zfmaccD2_34(void)		{ return test_mpi_z3opd(md_zfmaccD2,	MD_BIT(3)|MD_BIT(4))	; }

UT_REGISTER_TEST(test_mpi_zfmaccD2_3);
UT_REGISTER_TEST(test_mpi_zfmaccD2_34);

static bool test_mpi_zfmacD2_3(void)	{ return test_mpi_z3opd(md_zfmacD2,	MD_BIT(3))		; }
static bool test_mpi_zfmacD2_34(void)	{ return test_mpi_z3opd(md_zfmacD2,	MD_BIT(3)|MD_BIT(4))	; }

UT_REGISTER_TEST(test_mpi_zfmacD2_3);
UT_REGISTER_TEST(test_mpi_zfmacD2_34);

static bool test_mpi_z3op(z3op_t test_fun, unsigned long mpi_flags)
{
	enum { N = 4};
	long dims[N] = { 3, 3, 3, 3 };
	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	complex float* in1 = md_mpi_move(N, mpi_flags, dims, test_md_in0, CFL_SIZE);
	complex float* in2 = md_mpi_move(N, mpi_flags, dims, test_md_in1, CFL_SIZE);

	complex float* ret = md_alloc_mpi(N, mpi_flags, dims, CFL_SIZE);
	md_clear(N, dims, ret, CFL_SIZE);

	test_fun(N, dims, strs, ret, strs, in1, strs, in2);


	complex float* ref = md_alloc(N, dims, CFL_SIZE);
	md_clear(N, dims, ref, CFL_SIZE);

	test_fun(N, dims, strs, ref, strs, test_md_in0, strs, test_md_in1);

	complex float* ret_copy = md_alloc(N, dims, CFL_SIZE);
	md_copy(N, dims, ret_copy, ret, CFL_SIZE);

	float err = md_znrmse(N, dims, ref, ret_copy);

	md_free(in1);
	md_free(in2);
	md_free(ret);
	md_free(ref);
	md_free(ret_copy);
	
	UT_RETURN_ASSERT(err < UT_TOL * 10); //because single precision
}

static bool test_mpi_zfmac2_3(void)	{ return test_mpi_z3op(md_zfmac2,	MD_BIT(3))		; }
static bool test_mpi_zfmac2_34(void)	{ return test_mpi_z3op(md_zfmac2,	MD_BIT(3)|MD_BIT(4))	; }

UT_REGISTER_TEST(test_mpi_zfmac2_3);
UT_REGISTER_TEST(test_mpi_zfmac2_34);

static bool test_mpi_zfmacc2_3(void)	{ return test_mpi_z3op(md_zfmacc2,	MD_BIT(3))		; }
static bool test_mpi_zfmacc2_34(void)	{ return test_mpi_z3op(md_zfmacc2,	MD_BIT(3)|MD_BIT(4))	; }

UT_REGISTER_TEST(test_mpi_zfmacc2_3);
UT_REGISTER_TEST(test_mpi_zfmacc2_34);

static bool test_mpi_zadd2_3(void)		{ return test_mpi_z3op(md_zadd2,	MD_BIT(3))		; }
static bool test_mpi_zadd2_34(void)		{ return test_mpi_z3op(md_zadd2,	MD_BIT(3)|MD_BIT(4))	; }

UT_REGISTER_TEST(test_mpi_zadd2_3);
UT_REGISTER_TEST(test_mpi_zadd2_34);

static bool test_mpi_r3op(r3op_t test_fun, unsigned long mpi_flags)
{
	enum { N = 5};
	long dims[N] = { 2, 3, 3, 3, 3 };
	long strs[N];
	md_calc_strides(N, strs, dims, FL_SIZE);
	
	float* in1 = md_mpi_move(N, mpi_flags, dims, test_md_in0, FL_SIZE);
	float* in2 = md_mpi_move(N, mpi_flags, dims, test_md_in1, FL_SIZE);

	float* ret = md_alloc_mpi(N, mpi_flags, dims, FL_SIZE);
	md_clear(N, dims, ret, FL_SIZE);

	test_fun(N, dims, strs, ret, strs, in1, strs, in2);


	float* ref = md_alloc(N, dims, FL_SIZE);
	md_clear(N, dims, ref, FL_SIZE);

	test_fun(N, dims, strs, ref, strs, (float*)test_md_in0, strs, (float*)test_md_in1);

	float* ret_copy = md_alloc(N, dims, FL_SIZE);
	md_copy(N, dims, ret_copy, ret, FL_SIZE);

	float err = md_nrmse(N, dims, ref, ret_copy);

	md_free(in1);
	md_free(in2);
	md_free(ret);
	md_free(ref);
	md_free(ret_copy);

	UT_RETURN_ASSERT(err < UT_TOL);
}

static bool test_mpi_fmac2_4(void)	{ return test_mpi_r3op(md_fmac2,	MD_BIT(4)); }
static bool test_mpi_fmac2_23(void)	{ return test_mpi_r3op(md_fmac2,	MD_BIT(2)|MD_BIT(3)); }

UT_REGISTER_TEST(test_mpi_fmac2_4);
UT_REGISTER_TEST(test_mpi_fmac2_23);


static bool test_mpi_r3opd(r3opd_t test_fun, unsigned long mpi_flags)
{
	enum { N = 5};
	long dims[N] = { 2, 3, 3, 3, 3 };
	long strs[N];
	md_calc_strides(N, strs, dims, FL_SIZE);

	float* in1 = md_mpi_move(N, mpi_flags, dims, test_md_in0, FL_SIZE);
	float* in2 = md_mpi_move(N, mpi_flags, dims, test_md_in1, FL_SIZE);

	long dstrs[N];
	md_calc_strides(N, dstrs, dims, DL_SIZE);

	double* ret = md_alloc_mpi(N, mpi_flags, dims, DL_SIZE);
	md_clear(N, dims, ret, DL_SIZE);

	test_fun(N, dims, dstrs, ret, strs, in1, strs, in2);


	double* ref = md_alloc(N, dims, DL_SIZE);
	md_clear(N, dims, ref, DL_SIZE);

	test_fun(N, dims, dstrs, ref, strs, (float*)test_md_in0, strs, (float*)test_md_in1);
	
	double* ret_copy = md_alloc(N, dims, DL_SIZE);
	md_copy(N, dims, ret_copy, ret, DL_SIZE);
	
	//Because their memory layout it the same
	float err = md_znrmse(N, dims, (complex float*)ref, (complex float*)ret_copy);

	md_free(in1);
	md_free(in2);
	md_free(ret);
	md_free(ref);
	md_free(ret_copy);

	UT_RETURN_ASSERT(err < UT_TOL);
}

static bool test_mpi_fmacD2_4(void)	{ return test_mpi_r3opd(md_fmacD2,	MD_BIT(4)); }
static bool test_mpi_fmacD2_34(void)	{ return test_mpi_r3opd(md_fmacD2,	MD_BIT(3)|MD_BIT(4)); }

UT_REGISTER_TEST(test_mpi_fmacD2_4);
UT_REGISTER_TEST(test_mpi_fmacD2_34);

