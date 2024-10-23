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

#include "vptr_iter.h"



static float* vptr_float_malloc(long N)
{
	return vptr_alloc_size((size_t)N * FL_SIZE);
}

static float* vptr_float_malloc_gpu(long N)
{
	float* ret = vptr_alloc_size((size_t)N * FL_SIZE);
	vptr_set_gpu(ret);

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
	if (!vptr_is_init(x))
		vptr_clear(x);
	else
		md_clear(1, MD_DIMS(N), x, FL_SIZE);
}

static void vptr_float_copy(long N, float* a, const float* x)
{
	if (is_vptr(x) && !vptr_is_init(x)) {

		vptr_float_clear(N, a);
		return;
	}

	if (is_vptr(a) && is_vptr(x))
		vptr_set_dims_sameplace(a, x);

	md_copy(1, MD_DIMS(N), a, x, FL_SIZE);
}

static void vptr_swap(long N, float* a, float* x)
{
	float* tmp = vptr_float_malloc_sameplace(N, x);
	vptr_float_copy(N, tmp, a);
	vptr_float_copy(N, a, x);
	vptr_float_copy(N, x, tmp);
	vptr_float_free(tmp);
}

static double vptr_dot(long N, const float* x, const float* y)
{
	if (!vptr_is_init(x) || !vptr_is_init(y))
		return 0.;

	return md_scalar(1, MD_DIMS(N), x, y);
}

static double vptr_norm(long N, const float* x)
{
	return sqrt(vptr_dot(N, x, x));
}



static void vptr_smul(long N, float alpha, float* a, const float* x)
{
	if (!vptr_is_init(x)) {

		vptr_clear(a);
		return;
	}

	vptr_set_dims_sameplace(a, x);

	md_smul(1, MD_DIMS(N), a, x, alpha);
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

	md_sub(1, MD_DIMS(N), a, x, y);
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

	md_add(1, MD_DIMS(N), a, x, y);
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

static void vptr_saxpy(long N, float* a, float alpha, const float* x)
{
	if (!vptr_is_init(x))
		return;

	vptr_set_dims_sameplace(a, x);

	md_axpy(1, MD_DIMS(N), a, alpha, x);
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

		vptr_clear(a);
		return;
	}

	vptr_set_dims_sameplace(a, x);
	md_mul(1, MD_DIMS(N), a, x, y);
}

static void vptr_fmac(long N, float* a, const float* x, const float* y)
{
	if (!vptr_is_init(x) || !vptr_is_init(y)) {

		vptr_float_clear(N, a);
		return;
	}

	vptr_set_dims_sameplace(a, x);

	md_fmac(1, MD_DIMS(N), a, x, y);
}


static void vptr_div(long N, float* a, const float* x, const float* y)
{
	if (!vptr_is_init(x) || !vptr_is_init(y)) {

		vptr_clear(a);
		return;
	}

	vptr_set_dims_sameplace(a, x);

	md_div(1, MD_DIMS(N), a, x, y);
}

static void vptr_sqrt(long N, float* a, const float* x)
{
	if (!vptr_is_init(x)) {

		vptr_clear(a);
		return;
	}

	vptr_set_dims_sameplace(a, x);

	md_sqrt(1, MD_DIMS(N), a, x);
}

static void vptr_smax(long N, float alpha, float* a, const float* x)
{
	md_smax(1, MD_DIMS(N), a, x, alpha);
}



static void vptr_le(long N, float* a, const float* x, const float* y)
{
	vptr_set_dims_sameplace(a, x);

	md_lessequal(1, MD_DIMS(N), a, x, y);
}

static void vptr_zmul(long N, complex float* dst, const complex float* src1, const complex float* src2)
{
	md_zmul(1, MD_DIMS(N), dst, src1, src2);
}

static void vptr_zsmax(long N, float val, complex float* dst, const complex float* src)
{
	md_zsmax(1, MD_DIMS(N), dst, src, val);
}

static void vptr_rand(long N, float* dst)
{
	if (0 >= N)
		return;

	const struct vptr_shape_s* shape = vptr_get_shape(dst);
	assert(CFL_SIZE == shape->size);

	md_gaussian_rand(shape->N, shape->dims, (complex float*)dst);

	static bool printed = false;

	if (!printed && (N != 2 * md_calc_size(shape->N, shape->dims)))
		debug_printf(DP_WARN, "Random number generator on range of vptrs gives different random numbers than on continous memory!\n");

	printed = true;

	vptr_rand(N - 2 * md_calc_size(shape->N, shape->dims), dst + 2 * md_calc_size(shape->N, shape->dims));
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

};