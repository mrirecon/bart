/* Copyright 2024. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "num/rand.h"
#include "num/vptr.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/vptr.h"

#include "vptr_iter.h"

static int vptr_get_N_float(const void* x)
{
	int N = vptr_get_N(x);
	int size = vptr_get_size(x);

	if (size == FL_SIZE)
		return N;

	if (size == CFL_SIZE)
		return 1 + N;

	assert(0);
	return 0;
}

static void vptr_get_dims_float(const void* x, int N, long dims[N], long strs[N])
{
	assert(N == vptr_get_N_float(x));

	int size = vptr_get_size(x);

	if (size == FL_SIZE) {

		vptr_get_dims(x, N, dims);
		md_calc_strides(N, strs, dims, FL_SIZE);
		return;
	}

	if (size == CFL_SIZE) {

		dims[N - 1] = 2;
		strs[N - 1] = FL_SIZE;
		vptr_get_dims(x, N - 1, dims);
		md_calc_strides(N - 1, strs, dims, CFL_SIZE);
		return;
	}

	assert(0);
}

typedef void (*md_2op_t)(int D, const long dims[D], const long ostrs[D], float* optr, const long istrs1[D], const float* iptr1);
typedef void (*md_s2op_t)(int D, const long dims[D], const long ostrs[D], float* optr, const long istrs1[D], const float* iptr1, float scalar);
typedef void (*md_3op_t)(int D, const long dims[D], const long ostrs[D], float* optr, const long istrs1[D], const float* iptr1, const long istrs2[D], const float* iptr2);

static void vptr_md_2op(md_2op_t op, long N, float* optr, const float* iptr)
{
	if (0 == N)
		return;

	float* toptr = vptr_resolve_range(optr);
	float* tiptr = vptr_resolve_range(iptr);

	assert((toptr == optr) == (tiptr == iptr));

	int D = vptr_get_N_float(toptr);
	long dims[D];
	long strs[D];

	vptr_get_dims_float(toptr, D, dims, strs);
	op(D, dims, strs, toptr, strs, tiptr);

	vptr_md_2op(op, N - md_calc_size(D, dims), optr + md_calc_size(D, dims), iptr + md_calc_size(D, dims));
}

static void vptr_md_s2op(md_s2op_t op, long N, float* optr, const float* iptr, float scalar)
{
	if (0 == N)
		return;

	float* toptr = vptr_resolve_range(optr);
	float* tiptr = vptr_resolve_range(iptr);

	assert((toptr == optr) == (tiptr == iptr));

	int D = vptr_get_N_float(toptr);
	long dims[D];
	long strs[D];

	vptr_get_dims_float(toptr, D, dims, strs);
	op(D, dims, strs, toptr, strs, tiptr, scalar);

	vptr_md_s2op(op, N - md_calc_size(D, dims), optr + md_calc_size(D, dims), iptr + md_calc_size(D, dims), scalar);
}

static void vptr_md_3op(md_3op_t op, long N, float* optr, const float* iptr1, const float* iptr2)
{
	if (0 == N)
		return;

	float* toptr = vptr_resolve_range(optr);
	float* tiptr1 = vptr_resolve_range(iptr1);
	float* tiptr2 = vptr_resolve_range(iptr2);

	assert((toptr == optr) == (tiptr1 == iptr1));
	assert((toptr == optr) == (tiptr2 == iptr2));

	int D = vptr_get_N_float(toptr);
	long dims[D];
	long strs[D];

	vptr_get_dims_float(toptr, D, dims, strs);
	op(D, dims, strs, toptr, strs, tiptr1, strs, tiptr2);

	vptr_md_3op(op, N - md_calc_size(D, dims), optr + md_calc_size(D, dims), iptr1 + md_calc_size(D, dims), iptr2 + md_calc_size(D, dims));
}



static float* vptr_float_malloc(long N)
{
	return vptr_alloc_size((size_t)N * FL_SIZE);
}

static float* vptr_float_malloc_gpu(long N)
{
	float* ret = vptr_alloc_size((size_t)N * FL_SIZE);
	vptr_set_gpu(ret, true);

	return ret;
}

static float* vptr_float_malloc_sameplace(long N, const void* ref)
{
	return (is_vptr_gpu(ref) ? vptr_float_malloc_gpu : vptr_float_malloc)(N);
}

static void vptr_float_free(float* ptr)
{
	md_free(ptr);
}

static void vptr_float_clear(long N, float* x)
{
	if (0 == N)
		return;

	if (!vptr_is_init(x)) {

		vptr_set_clear(x);
	} else {

		float* tx = vptr_resolve_range(x);

		int D = vptr_get_N(tx);
		size_t size = vptr_get_size(tx);

		long dims[D];
		vptr_get_dims(tx, D, dims);
		md_clear(D, dims, tx, size);

		vptr_float_clear(N - md_calc_size(D, dims) * (long)(size / FL_SIZE), x + md_calc_size(D, dims) * (long)(size / FL_SIZE));
	}
}

static void md_float_copy(int N, const long dims[N], const long ostrs[N], float* optr, const long istrs[N], const float* iptr)
{
	md_copy2(N, dims, ostrs, optr, istrs, iptr, FL_SIZE);
}

static void vptr_float_copy(long N, float* a, const float* x)
{
	if (is_vptr(x) && !vptr_is_init(x)) {

		vptr_float_clear(N, a);
		return;
	}

	if (is_vptr(a) && is_vptr(x)) {

		vptr_set_dims_sameplace(a, x);
		vptr_md_2op(md_float_copy, N, a, x);
	} else {

		//copying scalars to and from cpu
		assert(1 == N || 2 == N);
		assert(!is_vptr(x) || vptr_is_init(x));

		if (is_vptr(a) && !vptr_is_init(a))
			vptr_set_dims(a, 1, MD_DIMS(1), 1 == N ? FL_SIZE : CFL_SIZE, NULL);

		md_copy(1, MD_DIMS(1), a, x, 1 == N ? FL_SIZE : CFL_SIZE);
	}
}

static void vptr_swap(long N, float* a, float* x)
{
	float* tmp = vptr_float_malloc_sameplace(N, x);
	vptr_float_copy(N, tmp, a);
	vptr_float_copy(N, a, x);
	vptr_float_copy(N, x, tmp);
	vptr_float_free(tmp);
}

static double* vptr_dot2(long N, const float* x, const float* y)
{
	if (!vptr_is_init(x) || !vptr_is_init(y)) {

		double* ret = vptr_alloc_sameplace(1, MD_DIMS(1), DL_SIZE, x);
		md_clear(1, MD_DIMS(1), ret, DL_SIZE);

		return vptr_wrap_range(1, (void**)&ret, true);
	}

	int R = 0;
	long N2 = N;

	const float* x2 = x;
	const float* y2 = y;

	while (0 < N2) {

		float* tx = vptr_resolve_range(x2);
		float* ty = vptr_resolve_range(y2);

		assert((tx == x2) == (ty == y2));

		int D = vptr_get_N_float(tx);
		long dims[D];
		long strs[D];

		vptr_get_dims_float(tx, D, dims, strs);

		N2 -= md_calc_size(D, dims);
		x2 += md_calc_size(D, dims);
		y2 += md_calc_size(D, dims);

		R++;
	}

	void* retp[R];

	for(int i = 0; i < R; i++) {

		float* tx = vptr_resolve_range(x);
		float* ty = vptr_resolve_range(y);

		assert((tx == x) == (ty == y));

		int D = vptr_get_N_float(tx);
		long dims[D];
		long strs[D];

		retp[i] = vptr_alloc_sameplace(1, MD_DIMS(1), DL_SIZE, tx);
		md_clear(1, MD_DIMS(1), retp[i], DL_SIZE);
		vptr_get_dims_float(tx, D, dims, strs);

		md_fmacD2(D, dims, MD_SINGLETON_STRS(D), retp[i], strs, tx, strs, ty);

		N -= md_calc_size(D, dims);
		x += md_calc_size(D, dims);
		y += md_calc_size(D, dims);
	}

	return vptr_wrap_range(R, retp, true);
}

static double vptr_get_dot2(double* x)
{
	double ret = 0;

	int N = vptr_get_len(x) / DL_SIZE;

	for (int i = 0; i < N; i++) {

		double tmp;
		md_copy(1, MD_DIMS(1), &tmp, x + i, DL_SIZE);
		ret += tmp;
	}

	md_free(x);

	return ret;
}

static double vptr_dot(long N, const float* x, const float* y)
{
	return vptr_get_dot2(vptr_dot2(N, x, y));
}

static double vptr_norm(long N, const float* x)
{
	return sqrt(vptr_dot(N, x, x));
}

static double* vptr_norm2(long N, const float* vec)
{
	return vptr_dot2(N, vec, vec);
}

static double vptr_get_norm2(double*x)
{
	return sqrt(vptr_get_dot2(x));
}



static void vptr_smul(long N, float alpha, float* a, const float* x)
{
	if (!vptr_is_init(x)) {

		vptr_set_clear(a);
		return;
	}

	vptr_set_dims_sameplace(a, x);

	vptr_md_s2op(md_smul2, N, a, x, alpha);
}

static void vptr_sub(long N, float* a, const float* x, const float* y)
{
	if (!vptr_is_init(x)) {

		vptr_smul(N, -1., a, y);
		return;
	}

	if (!vptr_is_init(y)) {

		vptr_float_copy(N, a, x);
		return;
	}

	vptr_set_dims_sameplace(a, x);

	vptr_md_3op(md_sub2, N, a, x, y);
}

static void vptr_add(long N, float* a, const float* x, const float* y)
{
	if (!vptr_is_init(y)) {

		vptr_float_copy(N, a, x);
		return;
	}

	if (!vptr_is_init(x)) {

		vptr_float_copy(N, a, y);
		return;
	}

	vptr_md_3op(md_add2, N, a, x, y);
}

static void vptr_xpay(long N, float alpha, float* a, const float* x)
{
	if (a == x) {

		vptr_smul(N, alpha + 1, a, a);
	} else {

		vptr_smul(N, alpha, a, a);
		vptr_add(N, a, a, x);
	}
}

static void saxpy_wrap(int N, const long dims[N], const long ostrs[N], float* a, const long istrs[N], const float* x, float alpha)
{
	md_axpy2(N, dims, ostrs, a, alpha, istrs, x);
}

static void vptr_saxpy(long N, float* a, float alpha, const float* x)
{
	if (!vptr_is_init(x))
		return;

	vptr_set_dims_sameplace(a, x);

	vptr_md_s2op(saxpy_wrap, N, a, x, alpha);
}


static void vptr_axpbz(long N, float* out, const float a, const float* x, const float b, const float* z)
{
	vptr_set_dims_sameplace(out, x);

	float* tmp1 = vptr_float_malloc_sameplace(N, x);
	float* tmp2 = vptr_float_malloc_sameplace(N, z);

	vptr_smul(N, a, tmp1, x);
	vptr_smul(N, b, tmp2, z);
	vptr_add(N, out, tmp1, tmp2);

	vptr_float_free(tmp1);
	vptr_float_free(tmp2);
}


static void vptr_mul(long N, float* a, const float* x, const float* y)
{
	if (!vptr_is_init(x) || !vptr_is_init(y)) {

		vptr_set_clear(a);
		return;
	}

	vptr_set_dims_sameplace(a, x);
	vptr_md_3op(md_mul2, N, a, x, y);
}

static void vptr_fmac(long N, float* a, const float* x, const float* y)
{
	if (!vptr_is_init(x) || !vptr_is_init(y)) {

		vptr_float_clear(N, a);
		return;
	}

	vptr_set_dims_sameplace(a, x);

	vptr_md_3op(md_fmac2, N, a, x, y);
}


static void vptr_div(long N, float* a, const float* x, const float* y)
{
	if (!vptr_is_init(x) || !vptr_is_init(y)) {

		vptr_set_clear(a);
		return;
	}

	vptr_set_dims_sameplace(a, x);

	vptr_md_3op(md_div2, N, a, x, y);
}

static void vptr_sqrt(long N, float* a, const float* x)
{
	if (!vptr_is_init(x)) {

		vptr_set_clear(a);
		return;
	}

	vptr_set_dims_sameplace(a, x);

	vptr_md_2op(md_sqrt2, N, a, x);
}

static void vptr_smax(long N, float alpha, float* a, const float* x)
{
	vptr_md_s2op(md_smax2, N, a, x, alpha);
}



static void vptr_le(long N, float* a, const float* x, const float* y)
{
	vptr_set_dims_sameplace(a, x);

	vptr_md_3op(md_lessequal2, N, a, x, y);
}

static void vptr_zmul(long /*N*/, complex float* /*dst*/, const complex float* /*src1*/, const complex float* /*src2*/)
{
	error("zmul in iter not supported for vptr/MPI\n");
}

static void vptr_zsmax(long /*N*/, float /*val*/, complex float* /*dst*/, const complex float* /*src*/)
{
	error("zsmax in iter not supported for vptr/MPI\n");
}

static void vptr_rand(long N, float* dst)
{
	if (0 == N)
		return;

	float* tdst = vptr_resolve_range(dst);

	int D = vptr_get_N(tdst);
	long dims[D];
	vptr_get_dims(tdst, D, dims);

	assert(CFL_SIZE == vptr_get_size(tdst));

	md_gaussian_rand(D, dims, (complex float*)tdst);

	static bool printed = false;

	if (!printed && (N != 2 * md_calc_size(D, dims)))
		debug_printf(DP_WARN, "Random number generator on range of vptrs gives different random numbers than on continous memory!\n");

	printed = true;

	vptr_rand(N - 2 * md_calc_size(D, dims), dst + 2 * md_calc_size(D, dims));
}

// defined in iter/vec.h
struct vec_iter_s {

	float* (*allocate)(long N);
	void (*del)(float* x);
	void (*clear)(long N, float* x);
	void (*copy)(long N, float* a, const float* x);
	void (*swap)(long N, float* a, float* x);

	double (*norm)(long N, const float* x);
	double (*dot)(long N, const float* x, const float* y);

	void (*sub)(long N, float* a, const float* x, const float* y);
	void (*add)(long N, float* a, const float* x, const float* y);

	void (*smul)(long N, float alpha, float* a, const float* x);
	void (*xpay)(long N, float alpha, float* a, const float* x);
	void (*axpy)(long N, float* a, float alpha, const float* x);
	void (*axpbz)(long N, float* out, const float a, const float* x, const float b, const float* z);
	void (*fmac)(long N, float* a, const float* x, const float* y);

	void (*mul)(long N, float* a, const float* x, const float* y);
	void (*div)(long N, float* a, const float* x, const float* y);
	void (*sqrt)(long N, float* a, const float* x);

	void (*smax)(long N, float alpha, float* a, const float* x);
	void (*smin)(long N, float alpha, float* a, const float* x);
	void (*sadd)(long N, float* x, float y);
	void (*sdiv)(long N, float* a, float x, const float* y);
	void (*le)(long N, float* a, const float* x, const float* y);

	void (*zmul)(long N, complex float* dst, const complex float* src1, const complex float* src2);
	void (*zsmax)(long N, float val, complex float* dst, const complex float* src1);

	void (*rand)(long N, float* dst);

	void (*xpay_bat)(long Bi, long N, long Bo, const float* beta, float* a, const float* x);
	void (*dot_bat)(long Bi, long N, long Bo, float* dst, const float* src1, const float* src2);
	void (*axpy_bat)(long Bi, long N, long Bo, float* a, const float* alpha, const float* x);

	double* (*norm2)(long N, const float* x);
	double* (*dot2)(long N, const float* x, const float* y);
	double (*get_norm2)(double* x);
	double (*get_dot2)(double* x);
};

extern const struct vec_iter_s vptr_iter_ops;
const struct vec_iter_s vptr_iter_ops = {

	.allocate = vptr_float_malloc,
	.del = vptr_float_free,
	.clear = vptr_float_clear,
	.copy = vptr_float_copy,
	.dot = vptr_dot,
	.norm = vptr_norm,
	.axpy = vptr_saxpy,
	.xpay = vptr_xpay,
	.axpbz = vptr_axpbz,
	.smul = vptr_smul,
	.add = vptr_add,
	.sub = vptr_sub,
	.swap = vptr_swap,
	.zmul = vptr_zmul,
	.rand = vptr_rand,
	.mul = vptr_mul,
	.fmac = vptr_fmac,
	.div = vptr_div,
	.sqrt = vptr_sqrt,
	.smax = vptr_smax,
	.smin = NULL,
	.sadd = NULL,
	.sdiv = NULL,
	.le = vptr_le,
	.zsmax = vptr_zsmax,
	.xpay_bat = NULL,
	.dot_bat = NULL,
	.axpy_bat = NULL,

	.norm2 = vptr_norm2,
	.dot2 = vptr_dot2,
	.get_norm2 = vptr_get_norm2,
	.get_dot2 = vptr_get_dot2,
};

extern const struct vec_iter_s vptr_iter_ops;
const struct vec_iter_s vptr_iter_ops_gpu = {

	.allocate = vptr_float_malloc_gpu,
	.del = vptr_float_free,
	.clear = vptr_float_clear,
	.copy = vptr_float_copy,
	.dot = vptr_dot,
	.norm = vptr_norm,
	.axpy = vptr_saxpy,
	.xpay = vptr_xpay,
	.axpbz = vptr_axpbz,
	.smul = vptr_smul,
	.add = vptr_add,
	.sub = vptr_sub,
	.swap = vptr_swap,
	.zmul = vptr_zmul,
	.rand = vptr_rand,
	.mul = vptr_mul,
	.fmac = vptr_fmac,
	.div = vptr_div,
	.sqrt = vptr_sqrt,
	.smax = vptr_smax,
	.smin = NULL,
	.sadd = NULL,
	.sdiv = NULL,
	.le = vptr_le,
	.zsmax = vptr_zsmax,
	.xpay_bat = NULL,
	.dot_bat = NULL,
	.axpy_bat = NULL,

	.norm2 = vptr_norm2,
	.dot2 = vptr_dot2,
	.get_norm2 = vptr_get_norm2,
	.get_dot2 = vptr_get_dot2,
};