/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */


/**
 * In this file, we check if md_calls with specific strides can be efficiently
 * computed by external libraries / specialized functions.
 *
 * The strategy is as follows:
 * 1.) A check_* function checks if the strides have a specific form.  If this
 * is the case - optimized dimensions and strides are copied to
 * ndims/nostrs/etc.  - the number of dimensions captured by the optimized
 * function call is returned else, -1 is returned
 *
 * 2.) We loop over the other dimensions and apply the inner kernel
 * (c.f. simple_* functions)
 *
 * To combine the check functions and the inner kernels, we use structs (e.g.
 * simple_z3op_check), to hold the check function, the corresponding kernel,
 * and a flag if the optimization should be applied on cpu/gpu
 *
 * In the documentation, of each check function, we use the symbols s - for the
 * size of one element x - for a integer, not necessarily the same for
 * different x's. We only print the optimized dims used for the strided kernels
 * and not the dims looped over by nested
 **/


#include <stdbool.h>
#include <complex.h>
#include <limits.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/optimize.h"
#include "num/blas_md_wrapper.h"
#include "num/reduce_md_wrapper.h"
#include "num/md_wrapper.h"
#include "num/convcorr.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "vecops_strided.h"

//not all BLAS libraries seem to be safe when called from omp region
#ifdef BLAS_THREADSAFE
bool blas_threadsafe = true;
#else
bool blas_threadsafe = false;
#endif

static bool use_strided_vecops = true;

void activate_strided_vecops(void)
{
	use_strided_vecops = true;
}

void deactivate_strided_vecops(void)
{
	use_strided_vecops = false;
}

typedef int (*md_check_3op_t)(int N, long ndims[N], long nostrs[N], long nistrs1[N], long nistrs2[N], const long dims[N], const long ostrs[N], const long istrs1[N], const long istrs2[N], long size);
typedef int (*md_check_2op_t)(int N, long ndims[N], long nostrs[N], long nistrs[N], const long dims[N], const long ostrs[N], const long istrs[N], long size);

typedef void (*md_3op_t)(int D, const long dims[D], const long ostrs[D], float* optr, const long istrs1[D], const float* iptr1, const long istrs2[D], const float* iptr2);
typedef void (*md_s2op_t)(int D, const long dims[D], const long ostrs[D], float* optr, const long istrs[D], const float* iptr, float val);
typedef void (*md_z3op_t)(int D, const long dims[D], const long ostrs[D], complex float* optr, const long istrs1[D], const complex float* iptr1, const long istrs2[D], const complex float* iptr2);

static void perm_z3op(	unsigned int D, const long dims[D], int order[D],
			unsigned long oflag, complex float* out,
			unsigned long iflag1, const complex float* in1,
			unsigned long iflag2, const complex float* in2,
			md_z3op_t fun, bool ignore_out)
{
	long dims_p[D];
	md_permute_dims(D, order, dims_p, dims);

	int order_p[D];

	unsigned oflag_p = 0;
	unsigned iflag1_p = 0;
	unsigned iflag2_p = 0;

	for (unsigned int i = 0; i < D; i++) {

		order_p[order[i]] = i;

		if (MD_IS_SET(oflag, i))
			oflag_p = MD_SET(oflag_p, order[i]);
		if (MD_IS_SET(iflag1, i))
			iflag1_p = MD_SET(iflag1_p, order[i]);
		if (MD_IS_SET(iflag2, i))
			iflag2_p = MD_SET(iflag2_p, order[i]);
	}

	long odims[D];
	long odims_p[D];
	long idims1[D];
	long idims1_p[D];
	long idims2[D];
	long idims2_p[D];

	md_select_dims(D, oflag_p, odims_p, dims_p);
	md_select_dims(D, oflag, odims, dims);
	md_select_dims(D, iflag1_p, idims1_p, dims_p);
	md_select_dims(D, iflag1, idims1, dims);
	md_select_dims(D, iflag2_p, idims2_p, dims_p);
	md_select_dims(D, iflag2, idims2, dims);

	complex float* out_p = md_alloc_sameplace(D, odims_p, CFL_SIZE, out);
	complex float* in1_p = md_alloc_sameplace(D, idims1_p, CFL_SIZE, in1);
	complex float* in2_p = md_alloc_sameplace(D, idims2_p, CFL_SIZE, in2);

	if (!ignore_out)
		md_permute(D, order, odims_p, out_p, odims, out, CFL_SIZE);

	md_permute(D, order, idims1_p, in1_p, idims1, in1, CFL_SIZE);
	md_permute(D, order, idims2_p, in2_p, idims2, in2, CFL_SIZE);

	fun(D, dims_p, MD_STRIDES(D, odims_p, CFL_SIZE), out_p, MD_STRIDES(D, idims1_p, CFL_SIZE), in1_p, MD_STRIDES(D, idims2_p, CFL_SIZE), in2_p);

	md_free(in1_p);
	md_free(in2_p);

	md_permute(D, order_p, odims, out, odims_p, out_p, CFL_SIZE);

	md_free(out_p);
}

static void md_zfmac_transp(int D, const long dims[D], const long ostr[D], complex float* out, const long istr1[D], const complex float* in1, const long istr2[D], const complex float* in2)
{
	assert(2 == D);

	unsigned long oflag = 0;
	unsigned long iflag1 = 0;
	unsigned long iflag2 = 0;

	for (int i = 0; i < D; i++) {

		if (0 != ostr[i])
			oflag = MD_SET(oflag, i);
		if (0 != istr1[i])
			iflag1 = MD_SET(iflag1, i);
		if (0 != istr2[i])
			iflag2 = MD_SET(iflag2, i);
	}

	perm_z3op(D, dims, (int[2]){ 1, 0 }, oflag, out, iflag1, in1, iflag2, in2, md_zfmac2, false);
}


struct simple_z3op_check {

	md_check_3op_t check_fun;
	md_z3op_t strided_kernel;
	bool on_gpu;
	bool on_cpu;
	bool in_place;
	bool reduction; // outptr and first inptr must equal
	bool thread_save; // kernel can be called in open_mp region
};

struct simple_3op_check {

	md_check_3op_t check_fun;
	md_3op_t strided_kernel;
	bool on_gpu;
	bool on_cpu;
	bool in_place;
	bool reduction; // outptr and first inptr must equal
	bool thread_save; // kernel can be called in open_mp region
};

#if 0
//not used yet
struct simple_s2op_check {

	md_check_2op_t check_fun;
	md_s2op_t strided_kernel;
	_Bool on_gpu;
	_Bool on_cpu;
	_Bool in_place;
};
#endif

/**
 * Optimized threeop wrapper. Use when inputs are constants -- copy from flpmath.c
 *
 * @param D number of dimensions
 * @param dim dimensions
 * @param ostr output strides
 * @param optr output
 * @param istr1 input 1 strides
 * @param iptr1 input 1 (constant)
 * @param istr2 input 2 strides
 * @param iptr2 input 2 (constant)
 * @param size size of data structures, e.g. complex float
 * @param too three-op multiply function
 */
static void optimized_threeop_oii(unsigned int D, const long dim[D], const long ostr[D], void* optr, const long istr1[D], const void* iptr1, const long istr2[D], const void* iptr2, size_t sizes[3], md_nary_opt_fun_t too)
{
	const long (*nstr[3])[D?D:1] = { (const long (*)[D?D:1])ostr, (const long (*)[D?D:1])istr1, (const long (*)[D?D:1])istr2 };
	void *nptr[3] = { optr, (void*)iptr1, (void*)iptr2 };

	unsigned int io = 1 + ((iptr1 == optr) ? 2 : 0) + ((iptr2 == optr) ? 4 : 0);

	optimized_nop(3, io, D, dim, nstr, nptr, sizes, too);
}


/**
 * Functions for optimizing fmac using blas
 * Checks if strides strides define a matrix,
 * i.e. one dimension is continuously in memory and followed by the other
 */
static bool is_matrix(const long dims[3], const long strs[3], int i1, int i2, long size)
{
	assert(i1 != i2);

	bool a = (   (strs[i1] == (long)size)
		  && (strs[i2] == (long)size * dims[i1]));

	bool b = (   (strs[i2] == (long)size)
		  && (strs[i1] == (long)size * dims[i2]));

	return a || b;
}


/**
 * Output: 3 if mat-mat-mul, -1, else
 *
 * if successful, the out strides have the form:
 * nostrs:  (s, 0, ndim[0]*s)
 * nistrs1: (s, ndim[0]*s, 0) or (ndim[1]*s, s, 0)
 * nistrs2: (0, s, ndim[1]*s) or (0, ndim[2]*s, s)
 *
 * Fixme: we could loose restriction for matrix lying contiguously in memory
 */
static int check_gemm(int N, long ndims[N], long nostrs[N], long nistrs1[N], long nistrs2[N], const long dims[N], const long ostrs[N], const long istrs1[N], const long istrs2[N], long size)
{
	md_singleton_dims(N, ndims);
	md_singleton_strides(N, nostrs);
	md_singleton_strides(N, nistrs1);
	md_singleton_strides(N, nistrs2);

	long tdims[N];
	long tostrs[N];
	long tistrs1[N];
	long tistrs2[N];

	md_copy_dims(N, tdims, dims);
	md_copy_strides(N, tostrs, ostrs);
	md_copy_strides(N, tistrs1, istrs1);
	md_copy_strides(N, tistrs2, istrs2);

	long (*strs[3])[N] = { &tostrs, &tistrs1, &tistrs2 };

	N = simplify_dims(3, N, tdims, strs);

	if (3 > N)
		return -1;

	/*
	 * Find zeros in strides, matmuls have strides of the form
	 * (0, x, x)
	 * (x, 0, x)
	 * (x, x, 0)
	 * or permutations
	 */
	int opos = -1;
	int ipos1 = -1;
	int ipos2 = -1;

	for (int i = 0; i < 3; i++) {

		if (0 == tostrs[i])
			opos = i;

		if (0 == tistrs1[i])
			ipos1 = i;

		if (0 == tistrs2[i])
			ipos2 = i;
	}

	// pos of zeros do not equal
	bool matrix = (   (opos != ipos1)
		       && (opos != ipos2)
                       && (ipos1 != ipos2)
                       && (3 == opos + ipos1 + ipos2));

	// Check if matrix dims are continuos in memory
	matrix = matrix && is_matrix(tdims, tostrs, (opos + 1) % 3, (opos + 2) % 3, size);
	matrix = matrix && is_matrix(tdims, tistrs1, (ipos1 + 1) % 3, (ipos1 + 2) % 3, size);
	matrix = matrix && is_matrix(tdims, tistrs2, (ipos2 + 1) % 3, (ipos2 + 2) % 3, size);

	// ipos1 is permuted to index 2:
	matrix = matrix && (tostrs[ipos1] > size);

	if (!matrix)
		return -1;

	/*
	 * Permute dims such that strides of output have the form
	 * (size, 0, x)
	 * the in strides have the form
	 * (x, x, 0)
	 * (0, x, x)
	 */
	int perm[N];

	for (int i = 3; i < N; i++)
		perm[i] = i;

	perm[0] = ipos2;
	perm[1] = opos;
	perm[2] = ipos1;

	md_permute_dims(N, perm, ndims, tdims);
	md_permute_dims(N, perm, nostrs, tostrs);
	md_permute_dims(N, perm, nistrs1, tistrs1);
	md_permute_dims(N, perm, nistrs2, tistrs2);

	return 3;
}


/**
 * Output: 2 if mat-vec-mul, -1, else
 *
 * if successful, the out strides have the form:
 * nostrs:  (s, 0)
 * nistrs1: (s, (ndim[0]+x)*s) or ((ndim[1]+x)*s, s)
 * nistrs2: (0, s)
 */
static int check_gemv(int N, long ndims[N], long nostrs[N], long nistrs1[N], long nistrs2[N], const long dims[N], const long ostrs[N], const long istrs1[N], const long istrs2[N], long size)
{
	md_singleton_dims(N, ndims);
	md_singleton_strides(N, nostrs);
	md_singleton_strides(N, nistrs1);
	md_singleton_strides(N, nistrs2);

	long tdims[N];
	long tostrs[N];
	long tistrs1[N];
	long tistrs2[N];

	md_copy_dims(N, tdims, dims);
	md_copy_strides(N, tostrs, ostrs);
	md_copy_strides(N, tistrs1, istrs1);
	md_copy_strides(N, tistrs2, istrs2);

	long (*strs[3])[N] = { &tostrs, &tistrs1, &tistrs2 };

	N = simplify_dims(3, N, tdims, strs);

	if (2 > N)
		return -1;

	int perm[N];

	for (int i = 2; i < N; i++)
		perm[i] = i;

	perm[0] = (tostrs[0] == 0) ? 1 : 0;
	perm[1] = (tostrs[0] == 0) ? 0 : 1;

	md_permute_dims(N, perm, ndims, tdims);
	md_permute_dims(N, perm, nostrs, tostrs);
	md_permute_dims(N, perm, nistrs1, tistrs1);
	md_permute_dims(N, perm, nistrs2, tistrs2);

	bool matvecmul = true;

	matvecmul = matvecmul && (0 == nostrs[0] % size) && (size <= nostrs[0]) && (0 == nostrs[1]);	//(s*x, 0)
	matvecmul = matvecmul && (0 == nistrs2[1] % size) && (size <= nistrs2[1]) && (0 == nistrs2[0]);	//(0, s*x)

	matvecmul = matvecmul && (0 == nistrs1[0] % size) && (0 == nistrs1[1] % size);
	matvecmul = matvecmul && (   ((size == nistrs1[0]) && (size * ndims[0] <= nistrs1[1]))
				  || ((size == nistrs1[1]) && (size * ndims[1] <= nistrs1[0])) );		//nistrs1: (s, (ndim[0]+x)*s) or ((ndim[1]+x)*s, s)

	if (!matvecmul)
		return -1;

	return 2;
}


/**
 * Output: 2 if symmetric rank one update, -1, else
 *
 * if successful, the out strides have the form:
 * nostrs: (s, s*(dim[0]+1))
 * the in strides have the form
 * nistrs1: (s*(1+x), 0)
 * nistrs2: (0, s*(1+x))
 */
static int check_ger(int N, long ndims[N], long nostrs[N], long nistrs1[N], long nistrs2[N], const long dims[N], const long ostrs[N], const long istrs1[N], const long istrs2[N], long size)
{
	md_singleton_dims(N, ndims);
	md_singleton_strides(N, nostrs);
	md_singleton_strides(N, nistrs1);
	md_singleton_strides(N, nistrs2);

	long tdims[N];
	long tostrs[N];
	long tistrs1[N];
	long tistrs2[N];

	md_copy_dims(N, tdims, dims);
	md_copy_strides(N, tostrs, ostrs);
	md_copy_strides(N, tistrs1, istrs1);
	md_copy_strides(N, tistrs2, istrs2);

	long (*strs[3])[N] = { &tostrs, &tistrs1, &tistrs2 };

	N = simplify_dims(3, N, tdims, strs);

	if ((2 > N) || ((size != tostrs[0]) && (size != tostrs[1])))
		return -1;

	int perm[N];

	for (int i = 2; i < N; i++)
		perm[i] = i;

	perm[0] = (tostrs[0] == size) ? 0 : 1;
	perm[1] = (tostrs[0] == size) ? 1 : 0;

	md_permute_dims(N, perm, ndims, tdims);
	md_permute_dims(N, perm, nostrs, tostrs);
	md_permute_dims(N, perm, nistrs1, tistrs1);
	md_permute_dims(N, perm, nistrs2, tistrs2);

	bool ger = true;
	ger = ger && (0 == nistrs1[1]) && (0 < nistrs1[0]) && (0 == nistrs1[0] % size);
	ger = ger && (0 == nistrs2[0]) && (0 < nistrs2[1]) && (0 == nistrs2[1] % size);
	ger = ger && (size == nostrs[0]) && (0 == nostrs[1] % size) && (nostrs[0] * ndims[0] <= nostrs[1]);

	return ger ? 2 : -1;
}


/**
 * Output: 1 if scalar-vec update, -1, else
 *
 * if successful, the out strides have the form:
 * nostrs:  (s*(1+x))
 * the in strides have the form
 * nistrs1: (s*(1+x))
 * nistrs2: (0)
 */
static int check_axpy(int N, long ndims[N], long nostrs[N], long nistrs1[N], long nistrs2[N], const long dims[N], const long ostrs[N], const long istrs1[N], const long istrs2[N], long size)
{
	md_singleton_dims(N, ndims);
	md_singleton_strides(N, nostrs);
	md_singleton_strides(N, nistrs1);
	md_singleton_strides(N, nistrs2);

	long tdims[N];
	long tostrs[N];
	long tistrs1[N];
	long tistrs2[N];

	md_copy_dims(N, tdims, dims);
	md_copy_strides(N, tostrs, ostrs);
	md_copy_strides(N, tistrs1, istrs1);
	md_copy_strides(N, tistrs2, istrs2);

	long (*strs[3])[N] = { &tostrs, &tistrs1, &tistrs2 };

	N = simplify_dims(3, N, tdims, strs);

	if ((1 > N) || (   (0 != tostrs[0] % size) || (0 >= tostrs[0])
			|| (0 != tistrs1[0] % size) || (0 >= tistrs1[0])
			|| (0 != tistrs2[0])))
		return -1;

	md_copy_dims(N, ndims, tdims);
	md_copy_strides(N, nostrs, tostrs);
	md_copy_strides(N, nistrs1, tistrs1);
	md_copy_strides(N, nistrs2, tistrs2);

	return 1;
}


/**
 * Output: 1 if dot, -1, else
 *
 * if successful, the out strides have the form:
 * nostrs: (0)
 * the in strides have the form
 * nistrs1: (s*x)
 * nistrs2: (s*x)
 */
static int check_dot(int N, long ndims[N], long nostrs[N], long nistrs1[N], long nistrs2[N], const long dims[N], const long ostrs[N], const long istrs1[N], const long istrs2[N], long size)
{
	md_singleton_dims(N, ndims);
	md_singleton_strides(N, nostrs);
	md_singleton_strides(N, nistrs1);
	md_singleton_strides(N, nistrs2);

	long tdims[N];
	long tostrs[N];
	long tistrs1[N];
	long tistrs2[N];

	md_copy_dims(N, tdims, dims);
	md_copy_strides(N, tostrs, ostrs);
	md_copy_strides(N, tistrs1, istrs1);
	md_copy_strides(N, tistrs2, istrs2);

	long (*strs[3])[N] = { &tostrs, &tistrs1, &tistrs2 };

	N = simplify_dims(3, N, tdims, strs);

	if ((1 > N) || (   (0 != tostrs[0])
			|| (0 != tistrs1[0] % size) || (0 >= tistrs1[0])
			|| (0 != tistrs2[0] % size) || (0 >= tistrs2[0]) ))
		return -1;

	md_copy_dims(N, ndims, tdims);
	md_copy_strides(N, nostrs, tostrs);
	md_copy_strides(N, nistrs1, tistrs1);
	md_copy_strides(N, nistrs2, tistrs2);

	return 1;
}

/**
 * Output: 2 if outer dot, -1, else
 *
 * if successful, the out strides have the form:
 * nostrs: (s, 0)
 * the in strides have the form
 * nistrs1: (s, s*dim[0])
 * nistrs2: (s, s*dim[0])
 */
static int check_dot_outer(int N, long ndims[N], long nostrs[N], long nistrs1[N], long nistrs2[N], const long dims[N], const long ostrs[N], const long istrs1[N], const long istrs2[N], long size)
{
	md_singleton_dims(N, ndims);
	md_singleton_strides(N, nostrs);
	md_singleton_strides(N, nistrs1);
	md_singleton_strides(N, nistrs2);

	long tdims[N];
	long tostrs[N];
	long tistrs1[N];
	long tistrs2[N];

	md_copy_dims(N, tdims, dims);
	md_copy_strides(N, tostrs, ostrs);
	md_copy_strides(N, tistrs1, istrs1);
	md_copy_strides(N, tistrs2, istrs2);

	long (*strs[3])[N] = { &tostrs, &tistrs1, &tistrs2 };

	N = simplify_dims(3, N, tdims, strs);

	if ((1 > N) || (   (size != tostrs[0])  || (0 != tostrs[1])
			|| (size != tistrs1[0]) || (size * tdims[0] != tistrs1[1])
			|| (size != tistrs2[0]) || (size * tdims[0] != tistrs2[1]) ))
		return -1;

	if (128 < tdims[0])
		return -1;
	if (128 * tdims[0] > tdims[1])
		return -1;

	md_copy_dims(N, ndims, tdims);
	md_copy_strides(N, nostrs, tostrs);
	md_copy_strides(N, nistrs1, tistrs1);
	md_copy_strides(N, nistrs2, tistrs2);

	return 2;
}

/**
 * Check if strides arise from md_calc_strides where dims are tenmul dims, i.e. dims equal or are 1.
 * First stride must be non-trivial for all args.
 * Example:
 * dims:	[4, 2, 3]
 * ostr:	[s, 4s, 8s]
 * istr1:	[s, 4s, 0]
 * istr2:	[s, 0, 2s]
 */
static int check_batched_select(int N, long ndims[N], long nostrs[N], long nistrs1[N], long nistrs2[N], const long dims[N], const long ostrs[N], const long istrs1[N], const long istrs2[N], long size)
{
	md_singleton_dims(N, ndims);
	md_singleton_strides(N, nostrs);
	md_singleton_strides(N, nistrs1);
	md_singleton_strides(N, nistrs2);

	long tdims[N];
	long tostrs[N];
	long tistrs1[N];
	long tistrs2[N];

	md_copy_dims(N, tdims, dims);
	md_copy_strides(N, tostrs, ostrs);
	md_copy_strides(N, tistrs1, istrs1);
	md_copy_strides(N, tistrs2, istrs2);

	long (*strs[3])[N] = { &tostrs, &tistrs1, &tistrs2 };

	N = simplify_dims(3, N, tdims, strs);

	md_copy_dims(N, ndims, tdims);
	md_copy_strides(N, nostrs, tostrs);
	md_copy_strides(N, nistrs1, tistrs1);
	md_copy_strides(N, nistrs2, tistrs2);

	long todims[N];
	long tidims1[N];
	long tidims2[N];

	md_select_dims(N, MD_BIT(0) | md_nontriv_strides(N, nostrs), todims, ndims);
	md_select_dims(N, MD_BIT(0) | md_nontriv_strides(N, nistrs1), tidims1, ndims);
	md_select_dims(N, MD_BIT(0) | md_nontriv_strides(N, nistrs2), tidims2, ndims);

	md_calc_strides(N, tostrs, todims, size);
	md_calc_strides(N, tistrs1, tidims1, size);
	md_calc_strides(N, tistrs2, tidims2, size);

	int i = 0;
	while ( i < (int)N
		&& (tostrs[i] == nostrs[i])
		&& (tistrs1[i] == nistrs1[i])
		&& (tistrs2[i] == nistrs2[i]))
		i++;

	if (1 >= i)
		return -1;

	return MIN(4, i);
}

/**
 * Output: 2 if diagonal-general matrix multiplication, -1, else
 *
 * if successful, the out strides have the form:
 * nostrs: (s, s*(dims[0] + x))
 * the in strides have the form
 * nistrs1: (s, s*(dims[0] + x))
 * nistrs2: (s*x, 0) or (0, s*x)
 */
static int check_dgmm(int N, long ndims[N], long nostrs[N], long nistrs1[N], long nistrs2[N], const long dims[N], const long ostrs[N], const long istrs1[N], const long istrs2[N], long size)
{
	md_singleton_dims(N, ndims);
	md_singleton_strides(N, nostrs);
	md_singleton_strides(N, nistrs1);
	md_singleton_strides(N, nistrs2);

	long tdims[N];
	long tostrs[N];
	long tistrs1[N];
	long tistrs2[N];

	md_copy_dims(N, tdims, dims);
	md_copy_strides(N, tostrs, ostrs);
	md_copy_strides(N, tistrs1, istrs1);
	md_copy_strides(N, tistrs2, istrs2);

	long (*strs[3])[N] = { &tostrs, &tistrs1, &tistrs2 };

	N = simplify_dims(3, N, tdims, strs);

	if (2 > N)
		return -1;

	int perm[N];

	for (int i = 2; i < N; i++)
		perm[i] = i;

	perm[0] = (tostrs[0] == size) ? 0 : 1;
	perm[1] = (tostrs[0] == size) ? 1 : 0;

	md_permute_dims(N, perm, ndims, tdims);
	md_permute_dims(N, perm, nostrs, tostrs);
	md_permute_dims(N, perm, nistrs1, tistrs1);
	md_permute_dims(N, perm, nistrs2, tistrs2);

	bool dgmm = true;
	dgmm = dgmm && (size == nostrs[0]) && (0 == nostrs[1] % size) && (size * ndims[0] <= nostrs[1]);
	dgmm = dgmm && (size == nistrs1[0]) && (0 == nistrs1[1] % size) && (size * ndims[0] <= nistrs1[1]);
	dgmm = dgmm && (0 == nistrs2[0] % size) && (0 == nistrs2[1] % size);
	dgmm = dgmm && (0 == nistrs2[0] * nistrs2[1]);
	dgmm = dgmm && ((0 < nistrs2[0]) || (0 < nistrs2[1]));

	if (!dgmm)
		return -1;

	return 2;
}

/**
 * Output: 2 if inner matrix can be reduced over non-continuous dimension and istrs1 == ostsrs, -1, else
 *
 * if successful, the out strides have the form:
 * nostrs: (s, 0, ...)
 * the in strides have the form
 * nistrs1: (s, 0, ...)
 * nistrs2: (s, s * dim[0], ...)
 */
static int check_reduce_outer(int N, long ndims[N], long nostrs[N], long nistrs1[N], long nistrs2[N], const long dims[N], const long ostrs[N], const long istrs1[N], const long istrs2[N], long size)
{
	md_singleton_dims(N, ndims);
	md_singleton_strides(N, nostrs);
	md_singleton_strides(N, nistrs1);
	md_singleton_strides(N, nistrs2);

	long tdims[N];
	long tostrs[N];
	long tistrs1[N];
	long tistrs2[N];

	md_copy_dims(N, tdims, dims);
	md_copy_strides(N, tostrs, ostrs);
	md_copy_strides(N, tistrs1, istrs1);
	md_copy_strides(N, tistrs2, istrs2);

	long (*strs[3])[N] = { &tostrs, &tistrs1, &tistrs2 };

	N = simplify_dims(3, N, tdims, strs);

	for (int i = 0; i < N; i++)
		if (tostrs[i] != tistrs1[i])
			return -1;

	if (2 > N)
		return -1;

	bool reduce = true;
	reduce &= ((1 == tdims[0]) || (size == tostrs[0])) && (0 == tostrs[1]);
	reduce &= ((1 == tdims[0]) || (size == tistrs1[0])) && (0 == tistrs1[1]);
	reduce &= ((1 == tdims[0]) || (size == tistrs2[0])) && (size * tdims[0] == tistrs2[1]);

	if (!reduce)
		return -1;

	md_copy_dims(N, ndims, tdims);
	md_copy_strides(N, nostrs, tostrs);
	md_copy_strides(N, nistrs1, tistrs1);
	md_copy_strides(N, nistrs2, tistrs2);

	return 2;
}

/**
 * Output: 2 or 1 if inner matrix can be reduced over contiguous dimension and istrs1 == ostsrs, -1, else
 *
 * if successful, the out strides have the form:
 * nostrs: (0) or (0, s)
 * the in strides have the form
 * nistrs1: (0) or (0, s)
 * nistrs2: (s) or (s, s * dim[0])
 */
static int check_reduce_inner(int N, long ndims[N], long nostrs[N], long nistrs1[N], long nistrs2[N], const long dims[N], const long ostrs[N], const long istrs1[N], const long istrs2[N], long size)
{
	md_singleton_dims(N, ndims);
	md_singleton_strides(N, nostrs);
	md_singleton_strides(N, nistrs1);
	md_singleton_strides(N, nistrs2);

	long tdims[N];
	long tostrs[N];
	long tistrs1[N];
	long tistrs2[N];

	md_copy_dims(N, tdims, dims);
	md_copy_strides(N, tostrs, ostrs);
	md_copy_strides(N, tistrs1, istrs1);
	md_copy_strides(N, tistrs2, istrs2);

	long (*strs[3])[N] = { &tostrs, &tistrs1, &tistrs2 };

	N = simplify_dims(3, N, tdims, strs);

	for (int i = 0; i < N; i++)
		if (tostrs[i] != tistrs1[i])
			return -1;

	if (1 > N)
		return -1;

	bool reduce = true;
	reduce &= ((0 == tostrs[0]));
	reduce &= ((0 == tistrs1[0]));
	reduce &= ((size == tistrs2[0]));

	if (!reduce)
		return -1;

	md_copy_dims(N, ndims, tdims);
	md_copy_strides(N, nostrs, tostrs);
	md_copy_strides(N, nistrs1, tistrs1);
	md_copy_strides(N, nistrs2, tistrs2);

	reduce &= (size == tostrs[1]);
	reduce &= (size == tistrs1[1]);
	reduce &= (size * tdims[0] == tistrs2[1]);

	return reduce ? 2 : 1;
}


// computes the size of an array with strides ignoring dims with 0 strides
// returns zero if block is not contiguous in memory
static size_t get_block_size(unsigned int N, const long dims[N], const long strs[N], size_t size)
{
	long tdims[N];
	long tstrs[N];
	md_select_dims(N, md_nontriv_strides(N, strs), tdims, dims);
	md_copy_strides(N, tstrs, strs);

	long (*nstr[1])[N?N:1] = { (long (*)[N?N:1])tstrs };

	int NN = optimize_dims_gpu(1, N, tdims, nstr); // sorting of dims

	if (NN != md_calc_blockdim(NN, tdims, tstrs, size))
		return 0;

	return md_calc_size(NN, tdims) * size;
}


static bool simple_z3op(int N_checks, struct simple_z3op_check strided_calls[N_checks], unsigned int N, const long dims[N], const long ostrs[N], complex float* out, const long istrs1[N], const complex float* in1, const long istrs2[N], const complex float* in2, bool conj)
{
	if (!use_strided_vecops)
		return false;

	long size = CFL_SIZE;

	long ndims[N];
	long nostrs[N];
	long nistrs1[N];
	long nistrs2[N];

	const complex float* tin1 = NULL;
	const complex float* tin2 = NULL;

	complex float* conj_in = NULL;

	long N_in = -1;

	bool on_gpu = false;
#ifdef USE_CUDA
	on_gpu = (cuda_ondevice(out));
#endif

	if (conj && (0 == get_block_size(N, dims, istrs2, size)))
		return false; //the conjugated input is not a continuos memory block

	struct simple_z3op_check strided_call;

	for (int i = 0; i < N_checks; i++) {

		strided_call = strided_calls[i];

		bool applicable = on_gpu ? strided_call.on_gpu : strided_call.on_cpu;

		#ifdef _OPENMP
		applicable &= (strided_call.thread_save || on_gpu);
		#endif

		if (!applicable)
			continue;

		N_in = strided_call.check_fun(N, ndims, nostrs, nistrs1, nistrs2, dims, ostrs, istrs1, istrs2, size);

		if ((strided_call.reduction) && (out != in1))
			N_in = -1;

		if (-1 != N_in) {

			tin1 = in1;
			tin2 = in2;

			if (conj) {

				long size_tmp = get_block_size(N, dims, istrs2, size);
				size_tmp /= size;
				conj_in = md_alloc_sameplace(1, &size_tmp, size, in2);
				md_zconj(1, &size_tmp, conj_in, in2);
				tin2 = conj_in;
			}

			break;
		}


		N_in = strided_call.check_fun(N, ndims, nostrs, nistrs1, nistrs2, dims, ostrs, istrs2, istrs1, size);

		if ((strided_call.reduction) && (out != in2))
			N_in = -1;

		if (-1 != N_in) {

			tin1 = in2;
			tin2 = in1;

			if (conj) {

				long size_tmp = get_block_size(N, dims, istrs2, size);
				size_tmp /= size;

				conj_in = md_alloc_sameplace(1, &size_tmp, size, in2);
				md_zconj(1, &size_tmp, conj_in, in2);
				tin1 = conj_in;
			}

			break;
		}
	}

	if (-1 == N_in)
		return false;

	// FIXME: blas calls are not save with large input dimensions
	if (INT_MAX / 2 < md_calc_size(N_in, ndims)) {

		md_free(conj_in);
		return false;
	}

	size_t osize = get_block_size(N_in, ndims, nostrs, size);
	size_t isize1 = get_block_size(N_in, ndims, nistrs1, size);
	size_t isize2 = get_block_size(N_in, ndims, nistrs2, size);

	if ((0 == osize) || (0 == isize1) || (0 == isize2))
	{
		md_free(conj_in);
		return false; //cross check: data for inner kernel is contiguous in memory
	}

	// clang
	long* ndims_ptr = &ndims[0];
	long* nostrs_ptr = &nostrs[0];
	long* nistrs1_ptr = &nistrs1[0];
	long* nistrs2_ptr = &nistrs2[0];


	NESTED(void, nary_inner_z3op, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (long i = 0; i < data->size; i++)
			strided_call.strided_kernel(	N_in, ndims_ptr,
					nostrs_ptr, (complex float*)(ptr[0] + i * osize),
					nistrs1_ptr, (const complex float*)(ptr[1] + i * isize1),
					nistrs2_ptr, (const complex float*)(ptr[2] + i * isize2));
	};

	optimized_threeop_oii(	N - N_in, ndims + N_in,
				nostrs + N_in, (void*)out, nistrs1 + N_in, (void*)tin1, nistrs2 + N_in, (void*)tin2,
				(size_t[3]){ osize, isize1, isize2 }, nary_inner_z3op);

	md_free(conj_in);

	return true;
}

static bool simple_3op(int N_checks, struct simple_3op_check strided_calls[N_checks], unsigned int N, const long dims[N], const long ostrs[N], float* out, const long istrs1[N], const float* in1, const long istrs2[N], const float* in2)
{
	if (!use_strided_vecops)
		return false;

	long size = FL_SIZE;

	long ndims[N];
	long nostrs[N];
	long nistrs1[N];
	long nistrs2[N];

	const float* tin1 = NULL;
	const float* tin2 = NULL;

	long N_in = -1;

	bool on_gpu = false;
#ifdef USE_CUDA
	on_gpu = (cuda_ondevice(out));
#endif

	struct simple_3op_check strided_call;

	for (int i = 0; i < N_checks; i++) {

		strided_call = strided_calls[i];

		bool applicable = on_gpu ? strided_call.on_gpu : strided_call.on_cpu;

		#ifdef _OPENMP
		applicable &= (strided_call.thread_save || on_gpu);
		#endif

		if (!applicable)
			continue;

		N_in = strided_call.check_fun(N, ndims, nostrs, nistrs1, nistrs2, dims, ostrs, istrs1, istrs2, size);

		if ((strided_call.reduction) && (out != in1))
			N_in = -1;

		if (-1 != N_in) {

			tin1 = in1;
			tin2 = in2;

			break;
		}

		N_in = strided_call.check_fun(N, ndims, nostrs, nistrs1, nistrs2, dims, ostrs, istrs2, istrs1, size);

		if ((strided_call.reduction) && (out != in2))
			N_in = -1;

		if (-1 != N_in) {

			tin1 = in2;
			tin2 = in1;

			break;
		}
	}

	if (-1 == N_in)
		return false;

	// FIXME: blas calls are not save with large input dimensions
	if (INT_MAX / 2 < md_calc_size(N_in, ndims))
		return false;

	size_t osize = get_block_size(N_in, ndims, nostrs, size);
	size_t isize1 = get_block_size(N_in, ndims, nistrs1, size);
	size_t isize2 = get_block_size(N_in, ndims, nistrs2, size);

	if ((0 == osize) || (0 == isize1) || (0 == isize2))
		return false; //cross check: data for inner kernel is contiguous in memory

	// clang
	long* ndims_ptr = &ndims[0];
	long* nostrs_ptr = &nostrs[0];
	long* nistrs1_ptr = &nistrs1[0];
	long* nistrs2_ptr = &nistrs2[0];


	NESTED(void, nary_inner_3op, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (long i = 0; i < data->size; i++)
			strided_call.strided_kernel(	N_in, ndims_ptr,
					nostrs_ptr, (float*)(ptr[0] + i * osize),
					nistrs1_ptr, (const float*)(ptr[1] + i * isize1),
					nistrs2_ptr, (const float*)(ptr[2] + i * isize2));
	};

	optimized_threeop_oii(	N - N_in, ndims + N_in,
				nostrs + N_in, (void*)out, nistrs1 + N_in, (void*)tin1, nistrs2 + N_in, (void*)tin2,
				(size_t[3]){ osize, isize1, isize2 }, nary_inner_3op);
	return true;
}

#if 0
//not used yet
static bool simple_s2op(int N_checks, struct simple_s2op_check strided_calls[N_checks], unsigned int N, const long dims[N], const long ostrs[N], float* out, const long istrs[N], const float* in, float val)
{
	if (!use_strided_vecops)
		return false;

	long size = 4;

	long ndims[N];
	long nostrs[N];
	long nistrs[N];

	long N_in = -1;
	md_s2op_t strided_kernel = NULL;

	for (int i = 0; i < N_checks; i++) {

		bool applicable = true;
		strided_kernel = strided_calls[i].strided_kernel;

	#ifdef USE_CUDA
		if (cuda_ondevice(out))
			applicable &= strided_calls[i].on_gpu;
		else
	#endif
			applicable &= strided_calls[i].on_cpu;
		if (!applicable)
			continue;

		N_in = strided_calls[i].check_fun(N, ndims, nostrs, nistrs, dims, ostrs, istrs, size);
		if (-1 != N_in)
			break;
	}

	if (-1 == N_in)
		return false;

	size_t osize = 0;
	size_t isize = 0;

	for (int i = 0; i < N_in; i++) {

		osize = MAX(osize, (size_t)(nostrs[i] * ndims[i]));
		isize = MAX(osize, (size_t)(nistrs[i] * ndims[i]));
	}

	//clang
	long* ndims_ptr = &ndims[0];
	long* nostrs_ptr = &nostrs[0];
	long* nistrs1_ptr = &nistrs[0];


	NESTED(void, nary_inner_z3op, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (long i = 0; i < data->size; i++)
			strided_kernel(	N_in, ndims_ptr,
					nostrs_ptr, (float*)(ptr[0] + i * osize),
					nistrs1_ptr, (const float*)(ptr[1] + i * isize),
					val);
	};

	optimized_twoop_oi(	N - N_in, ndims + N_in,
				nostrs + N_in, (void*)out, nistrs + N_in, (void*)in,
				(size_t[3]){ osize, isize }, nary_inner_z3op);

	return true;
}
#endif

bool simple_zfmac(int N, const long dims[N], const long ostrs[N], complex float* out, const long istrs1[N], const complex float* in1, const long istrs2[N], const complex float* in2)
{
	if (!use_strided_vecops)
		return false;

	if (simple_zconvcorr(N, dims, ostrs, out, istrs1, in1, istrs2, in2))
		return true;

	struct simple_z3op_check strided_calls[] = {
		{ check_gemm,	blas_zfmac_cgemm, true, true, false, false, blas_threadsafe },
		{ check_gemv,	blas_zfmac_cgemv, true, true, false, false, blas_threadsafe },
		{ check_ger,	blas_zfmac_cgeru, true, true, false, false, blas_threadsafe },
		{ check_axpy,	blas_zfmac_caxpy, true, true, false, false, blas_threadsafe },
		{ check_dot,	blas_zfmac_cdotu, true, true, false, false, blas_threadsafe },
		{ check_batched_select,	zfmac_gpu_batched_loop, true, false, false, false, true },
		{ check_dot_outer, md_zfmac_transp, true, false, false, false, blas_threadsafe }
	};

	return simple_z3op(	ARRAY_SIZE(strided_calls), strided_calls,
				N, dims, ostrs, out, istrs1, in1, istrs2, in2, false);
}

bool simple_zfmacc(int N, const long dims[N], const long ostrs[N], complex float* out, const long istrs1[N], const complex float* in1, const long istrs2[N], const complex float* in2)
{
	struct simple_z3op_check strided_calls[] = {
		{ check_gemm,  blas_zfmac_cgemm, true, true, false, false, blas_threadsafe },
		{ check_gemv,  blas_zfmac_cgemv, true, true, false, false, blas_threadsafe },
		{ check_ger,   blas_zfmac_cgeru, true, true, false, false, blas_threadsafe },
		{ check_axpy,  blas_zfmac_caxpy, true, true, false, false, blas_threadsafe },
		{ check_dot,   blas_zfmac_cdotu, true, true, false, false, blas_threadsafe },
		{ check_batched_select,	zfmac_gpu_batched_loop, true, false, false, false, true },
		{ check_dot_outer, md_zfmac_transp, true, false, false, false, blas_threadsafe }
	};

	return simple_z3op(	ARRAY_SIZE(strided_calls), strided_calls,
				N, dims, ostrs, out, istrs1, in1, istrs2, in2, true);
}

bool simple_fmac(int N, const long dims[N], const long ostrs[N], float* out, const long istrs1[N], const float* in1, const long istrs2[N], const float* in2)
{
	struct simple_3op_check strided_calls[] = {
		{ check_gemm,  blas_fmac_sgemm, true, true, false, false, blas_threadsafe },
		{ check_gemv,  blas_fmac_sgemv, true, true, false, false, blas_threadsafe },
		{ check_ger,   blas_fmac_sger,  true, true, false, false, blas_threadsafe },
		{ check_axpy,  blas_fmac_saxpy, true, true, false, false, blas_threadsafe },
		{ check_dot,   blas_fmac_sdot,  true, true, false, false, blas_threadsafe }
	};

	return simple_3op(	ARRAY_SIZE(strided_calls), strided_calls,
				N, dims, ostrs, out, istrs1, in1, istrs2, in2);
}

bool simple_zmul(int N, const long dims[N], const long ostrs[N], complex float* out, const long istrs1[N], const complex float* in1, const long istrs2[N], const complex float* in2)
{
	struct simple_z3op_check strided_calls[] = {
		{ check_ger,   blas_zmul_cgeru, true, true, false, false, blas_threadsafe },
		{ check_dgmm,  blas_zmul_cdgmm, true, false, true, false, blas_threadsafe },
		{ check_axpy,  blas_zmul_cscal, true, true, true, false, blas_threadsafe }
	};

	return simple_z3op(	ARRAY_SIZE(strided_calls), strided_calls,
				N, dims, ostrs, out, istrs1, in1, istrs2, in2, false);
}

bool simple_zmulc(int N, const long dims[N], const long ostrs[N], complex float* out, const long istrs1[N], const complex float* in1, const long istrs2[N], const complex float* in2)
{
	struct simple_z3op_check strided_calls[] = {
		{ check_ger,   blas_zmul_cgeru, true, true, false, false, blas_threadsafe },
		{ check_dgmm,  blas_zmul_cdgmm, true, false, true, false, blas_threadsafe },
		{ check_axpy,  blas_zmul_cscal, true, true, true, false, blas_threadsafe }
	};

	return simple_z3op(	ARRAY_SIZE(strided_calls), strided_calls,
				N, dims, ostrs, out, istrs1, in1, istrs2, in2, true);
}

bool simple_mul(int N, const long dims[N], const long ostrs[N], float* out, const long istrs1[N], const float* in1, const long istrs2[N], const float* in2)
{
	struct simple_3op_check strided_calls[] = {
		{ check_ger,   blas_mul_sger, true, true, false, false, blas_threadsafe },
		{ check_dgmm,  blas_mul_sdgmm, true, false, true, false, blas_threadsafe },
		{ check_axpy,  blas_mul_sscal, true, true, true, false, blas_threadsafe }
	};

	return simple_3op(	ARRAY_SIZE(strided_calls), strided_calls,
				N, dims, ostrs, out, istrs1, in1, istrs2, in2);
}

bool simple_zadd(int N, const long dims[N], const long ostrs[N], complex float* out, const long istrs1[N], const complex float* in1, const long istrs2[N], const complex float* in2)
{
	struct simple_z3op_check strided_calls[] = {
#ifdef NON_DETERMINISTIC
		{ check_reduce_outer,	reduce_zadd_outer_gpu, true, false, false, true, true },
		{ check_reduce_inner,	reduce_zadd_inner_gpu, true, false, false, true, true },
#endif
		{ check_reduce_outer,	reduce_zadd_gemv, true, true, false, true, blas_threadsafe },
		{ check_reduce_inner,	reduce_zadd_gemv, true, true, false, true, blas_threadsafe },
	};

	return simple_z3op(	ARRAY_SIZE(strided_calls), strided_calls,
				N, dims, ostrs, out, istrs1, in1, istrs2, in2, false);
}

bool simple_add(int N, const long dims[N], const long ostrs[N], float* out, const long istrs1[N], const float* in1, const long istrs2[N], const float* in2)
{
	struct simple_3op_check strided_calls[] = {
#ifdef NON_DETERMINISTIC
		{ check_reduce_outer,	reduce_add_outer_gpu, true, false, false, true, true },
		{ check_reduce_inner,	reduce_add_inner_gpu, true, false, false, true, true },
#endif
		{ check_reduce_outer,	reduce_add_gemv, true, true, false, true, blas_threadsafe },
		{ check_reduce_inner,	reduce_add_gemv, true, true, false, true, blas_threadsafe },
	};

	return simple_3op(	ARRAY_SIZE(strided_calls), strided_calls,
				N, dims, ostrs, out, istrs1, in1, istrs2, in2);
}

bool simple_zmax(int N, const long dims[N], const long ostrs[N], complex float* out, const long istrs1[N], const complex float* in1, const long istrs2[N], const complex float* in2)
{
	struct simple_z3op_check strided_calls[] = {
		{ check_reduce_outer,	reduce_zmax_outer_gpu, true, false, false, true, true },
		{ check_reduce_inner,	reduce_zmax_inner_gpu, true, false, false, true, true },
	};

	return simple_z3op(	ARRAY_SIZE(strided_calls), strided_calls,
				N, dims, ostrs, out, istrs1, in1, istrs2, in2, false);
}
