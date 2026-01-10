/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2015-2022. Uecker Lab. University Medical Center Göttingen
 * Copyright 2021-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Martin Uecker, Jonathan Tamir, Moritz Blumenthal
 *
 * Publications:
 *
 * Sylvester JJ.
 * Thoughts on inverse orthogonal matrices, simultaneous sign successions,
 * and tessellated pavements in two or more colours, with applications to Newton’s
 * rule, ornamental tile-work, and the theory of numbers.
 * Philosophical Magazine 1867; 34:461-475.
 */


#include <complex.h>
#include <assert.h>
#include <math.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/wavelet.h"
#include "num/conv.h"
#include "num/ops.h"
#include "num/iovec.h"
#include "num/multiplace.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/linop.h"

#include "someops.h"
#include "linops/fmac.h"


struct cdiag_s {

	linop_data_t super;

	int N;
	const long* dims;
	const long* strs;
	const long* ddims;
	const long* dstrs;

	struct multiplace_array_s* diag;
	struct multiplace_array_s* normal;
	bool rmul;
};

static DEF_TYPEID(cdiag_s);


static void cdiag_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(cdiag_s, _data);

	const complex float* diag = multiplace_read(data->diag, src);

	(data->rmul ? md_zrmul2 : md_zmul2)(data->N, data->dims, data->strs, dst, data->strs, src, data->dstrs, diag);
}

static void cdiag_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(cdiag_s, _data);

	const complex float* diag = multiplace_read(data->diag, src);

	(data->rmul ? md_zrmul2 : md_zmulc2)(data->N, data->dims, data->strs, dst, data->strs, src, data->dstrs, diag);
}

static void cdiag_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(cdiag_s, _data);

#pragma omp critical (linop_cdiag_normal)
	if (NULL == data->normal) {

		const complex float* diag = multiplace_read(data->diag, src);

		complex float* tmp = md_alloc_sameplace(data->N, data->ddims, CFL_SIZE, dst);
		(data->rmul ? md_zrmul : md_zmulc)(data->N, data->ddims, tmp, diag, diag);
		data->normal = multiplace_move_F(data->N, data->ddims, CFL_SIZE, tmp);
	}

	const complex float* normal = multiplace_read(data->normal, src);

	(data->rmul ? md_zrmul2 : md_zmul2)(data->N, data->dims, data->strs, dst, data->strs, src, data->dstrs, normal);
}

static void cdiag_free(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(cdiag_s, _data);

	multiplace_free(data->normal);
	multiplace_free(data->diag);
	xfree(data->ddims);
	xfree(data->dims);
	xfree(data->dstrs);
	xfree(data->strs);

	xfree(data);
}

static struct linop_s* linop_gdiag_create(int N, const long dims[N], unsigned long flags, const complex float* diag, bool rdiag)
{
	PTR_ALLOC(struct cdiag_s, data);
	SET_TYPEID(cdiag_s, data);

	data->rmul = rdiag;

	data->N = N;
	PTR_ALLOC(long[N], ddims);
	PTR_ALLOC(long[N], dstrs);
	PTR_ALLOC(long[N], dims2);
	PTR_ALLOC(long[N], strs);

	md_select_dims(N, flags, *ddims, dims);
	md_calc_strides(N, *dstrs, *ddims, CFL_SIZE);

	md_copy_dims(N, *dims2, dims);
	md_calc_strides(N, *strs, dims, CFL_SIZE);

	data->dims = *PTR_PASS(dims2);
	data->strs = *PTR_PASS(strs);
	data->ddims = *PTR_PASS(ddims);
	data->dstrs = *PTR_PASS(dstrs);

	data->diag = (NULL == diag) ? NULL : multiplace_move(N, data->ddims, CFL_SIZE, diag);
	data->normal = NULL;

	return linop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), cdiag_apply, cdiag_adjoint, cdiag_normal, NULL, cdiag_free);
}



/**
 * Create a operator y = D x where D is a diagonal matrix
 *
 * @param N number of dimensions
 * @param dims input and output dimensions
 * @param flags bitmask specifying the dimensions present in diag
 * @param diag diagonal matrix
 */
struct linop_s* linop_cdiag_create(int N, const long dims[N], unsigned long flags, const complex float* diag)
{
	return linop_gdiag_create(N, dims, flags, diag, false);
}


/**
 * Create a operator y = D x where D is a diagonal matrix
 *
 * @param N number of dimensions
 * @param dims input and output dimensions
 * @param flags bitmask specifying the dimensions present in diag
 * @param diag diagonal matrix
 */
struct linop_s* linop_rdiag_create(int N, const long dims[N], unsigned long flags, const complex float* diag)
{
	return linop_gdiag_create(N, dims, flags, diag, true);
}

void linop_gdiag_set_diag(const struct linop_s* lop, int N, const long ddims[N], const complex float* diag)
{
	auto _data = linop_get_data(lop);
	auto data = CAST_DOWN(cdiag_s, _data);

	assert(data->N == N);
	assert(md_check_equal_dims(N, ddims, data->ddims, ~0UL));

	multiplace_free(data->diag);
	multiplace_free(data->normal);

	data->normal = NULL;
	data->diag = multiplace_move(N, data->ddims, CFL_SIZE, diag);
}

void linop_gdiag_set_diag_F(const struct linop_s* lop, int N, const long ddims[N], const complex float* diag)
{
	auto _data = linop_get_data(lop);
	auto data = CAST_DOWN(cdiag_s, _data);

	assert(data->N == N);
	assert(md_check_equal_dims(N, ddims, data->ddims, ~0UL));

	multiplace_free(data->diag);
	multiplace_free(data->normal);

	data->normal = NULL;
	data->diag = multiplace_move_F(N, data->ddims, CFL_SIZE, diag);
}

void linop_gdiag_set_diag_ref(const struct linop_s* lop, int N, const long ddims[N], const complex float* diag)
{
	auto _data = linop_get_data(lop);
	auto data = CAST_DOWN(cdiag_s, _data);

	assert(data->N == N);
	assert(md_check_equal_dims(N, ddims, data->ddims, ~0UL));

	multiplace_free(data->diag);
	multiplace_free(data->normal);

	data->normal = NULL;
	data->diag = multiplace_move_wrapper(N, data->ddims, CFL_SIZE, diag);
}

struct scale_s {

	linop_data_t super;

	int N;
	const long* dims;
	const long* strs;
	complex float scale;
};

static DEF_TYPEID(scale_s);

static void scale_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(scale_s, _data);

	md_zsmul2(data->N, data->dims, data->strs, dst, data->strs, src, data->scale);
}

static void scale_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(scale_s, _data);

	md_zsmul2(data->N, data->dims, data->strs, dst, data->strs, src, conjf(data->scale));
}

static void scale_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(scale_s, _data);

	md_zsmul2(data->N, data->dims, data->strs, dst, data->strs, src, conjf(data->scale) * data->scale);
}

static void scale_free(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(scale_s, _data);

	xfree(data->dims);
	xfree(data->strs);
	xfree(data);
}

/**
 * Create a scaling linear operator: y = a x,
 * where a is a complex float
 *
 * @param N number of dimensions
 * @param dims dimensions
 * @param scale scaling factor a
 */

struct linop_s* linop_scale_create(int N, const long dims[N], const complex float scale)
{
	if (1 == scale)
		return linop_identity_create(N, dims);

	PTR_ALLOC(struct scale_s, data);
	SET_TYPEID(scale_s, data);

	data->scale = scale;

	data->N = N;
	PTR_ALLOC(long[N], dims2);
	PTR_ALLOC(long[N], strs);

	md_copy_dims(N, *dims2, dims);
	md_calc_strides(N, *strs, dims, CFL_SIZE);

	data->dims = *PTR_PASS(dims2);
	data->strs = *PTR_PASS(strs);

	return linop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), scale_apply, scale_adjoint, scale_normal, NULL, scale_free);
}



struct zconj_s {

	linop_data_t super;

	int N;
	const long* dims;
};

DEF_TYPEID(zconj_s);

static void zconj_fun(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(zconj_s, _data);

	md_zconj(data->N, data->dims, dst, src);
}

static void zconj_del(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(zconj_s, _data);

	xfree(data->dims);
	xfree(data);
}


struct linop_s* linop_zconj_create(int N, const long dims[N])
{
	PTR_ALLOC(struct zconj_s, data);
	SET_TYPEID(zconj_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);

	data->N = N;
	data->dims = *PTR_PASS(ndims);

	return linop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), zconj_fun, zconj_fun, NULL, NULL, zconj_del);
}



struct zreal_s {

	linop_data_t super;

	int N;
	const long* dims;
};

static DEF_TYPEID(zreal_s);

static void zreal_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(zreal_s, _data);

	md_zreal(data->N, data->dims, dst, src);
}

static void zreal_free(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(zreal_s, _data);

	xfree(data->dims);
	xfree(data);
}

/**
 * @param N number of dimensions
 * @param dims dimensions
 */

struct linop_s* linop_zreal_create(int N, const long dims[N])
{
	PTR_ALLOC(struct zreal_s, data);
	SET_TYPEID(zreal_s, data);

	data->N = N;

	PTR_ALLOC(long[N], dims2);
	md_copy_dims(N, *dims2, dims);

	data->dims = *PTR_PASS(dims2);

	return linop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), zreal_apply, zreal_apply, zreal_apply, NULL, zreal_free);
}



/**
 * Create an Identity linear operator: I x
 * @param N number of dimensions
 * @param dims dimensions of input (domain)
 */
struct linop_s* linop_identity_create(int N, const long dims[N])
{
	auto op = operator_identity_create(N, dims);
	auto result = linop_from_ops(op, op, op, NULL);
	operator_free(op);

	return result;
}

bool linop_is_identity(const struct linop_s* lop)
{
	return check_simple_copy(lop->forward);
}

struct copy_block_s {

	linop_data_t super;

	int N;

	const long* odims;
	const long* idims;

	const long* pos;
};

static DEF_TYPEID(copy_block_s);

static void copy_block_forward(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(copy_block_s, _data);

	for (int i = 0; i < data->N; i++)
		if (data->odims[i] > data->idims[i]) {

			md_clear(data->N, data->odims, dst, CFL_SIZE);
			break;
		}

	md_copy_block(data->N, data->pos, data->odims, dst, data->idims, src, CFL_SIZE);
}

static void copy_block_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(copy_block_s, _data);

	for (int i = 0; i < data->N; i++)
		if (data->idims[i] > data->odims[i]) {

			md_clear(data->N, data->idims, dst, CFL_SIZE);
			break;
		}

	md_copy_block(data->N, data->pos, data->idims, dst, data->odims, src, CFL_SIZE);
}

static void copy_block_free(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(copy_block_s, _data);

	xfree(data->odims);
	xfree(data->idims);
	xfree(data->pos);

	xfree(data);
}

struct linop_s* linop_copy_block_create(int N, const long pos[N], const long odims[N], const long idims[N])
{
	PTR_ALLOC(struct copy_block_s, data);
	SET_TYPEID(copy_block_s, data);

	data->N = N;
	data->odims = ARR_CLONE(long[N], odims);
	data->idims = ARR_CLONE(long[N], idims);
	data->pos = ARR_CLONE(long[N], pos);

	return linop_create(N, odims, N, idims, CAST_UP(PTR_PASS(data)), copy_block_forward, copy_block_adjoint, NULL, NULL, copy_block_free);
}


/**
 * Create a resize linear operator: y = M x,
 * where M either crops or expands the the input dimensions to match the output dimensions.
 * Uses centered zero-padding and centered cropping
 *
 * @param N number of dimensions
 * @param out_dims output dimensions
 * @param in_dims input dimensions
 */
struct linop_s* linop_resize_center_create(int N, const long out_dims[N], const long in_dims[N])
{
	long pos[N];
	for (int i = 0; i < N; i++)
		pos[i] = labs((out_dims[i] / 2) - (in_dims[i] / 2));


	return linop_copy_block_create(N, pos, out_dims, in_dims);
}

struct linop_s* linop_resize_create(int N, const long out_dims[N], const long in_dims[N])
{
	//FIXME: inconsistent with md_resize
	return linop_resize_center_create(N, out_dims, in_dims);
}

struct linop_s* linop_expand_create(int N, const long out_dims[N], const long in_dims[N])
{
	long pos[N];
	for (int i = 0; i < N; i++)
		pos[i] = 0;

	return linop_copy_block_create(N, pos, out_dims, in_dims);
}



struct padding_op_s {

	linop_data_t super;

	int N;
	const long* strs_out;

	const long* dims_for;
	const long* strs_for;
	long offset_out_for;
	long offset_in_for;

	const long* dims_mid;
	const long* strs_mid;
	long offset_out_mid;
	long offset_in_mid;

	const long* dims_after;
	const long* strs_after;
	long offset_out_after;
	long offset_in_after;

	const long* dims_in;
	const long* dims_out;
};

static DEF_TYPEID(padding_op_s);


static void padding_forward(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(padding_op_s, _data);

	int N = data->N;

	assert(dst != src);

	md_clear(N, data->dims_out, dst, CFL_SIZE); //needed for PADDING_SAME

	md_copy2(N, data->dims_for, data->strs_out, dst + data->offset_out_for, data->strs_for, src + data->offset_in_for, CFL_SIZE);
	md_copy2(N, data->dims_mid, data->strs_out, dst + data->offset_out_mid, data->strs_mid, src + data->offset_in_mid, CFL_SIZE);
	md_copy2(N, data->dims_after, data->strs_out, dst + data->offset_out_after, data->strs_after, src + data->offset_in_after, CFL_SIZE);
}

static void padding_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(padding_op_s, _data);

	complex float* dst_tmp = md_alloc_sameplace(data->N, data->dims_in, CFL_SIZE, dst);

	// strided copies are more efficient than strided sum (gpu)
	md_clear(data->N, data->dims_in, dst, CFL_SIZE);
	md_copy2(data->N, data->dims_mid, data->strs_mid, dst + data->offset_in_mid, data->strs_out, src + data->offset_out_mid, CFL_SIZE);

	md_clear(data->N, data->dims_in, dst_tmp, CFL_SIZE);
	md_copy2(data->N, data->dims_for, data->strs_for, dst_tmp + data->offset_in_for, data->strs_out, src + data->offset_out_for, CFL_SIZE);
	md_zadd(data->N, data->dims_in, dst, dst, dst_tmp);

	md_clear(data->N, data->dims_in, dst_tmp, CFL_SIZE);
	md_copy2(data->N, data->dims_after, data->strs_after, dst_tmp + data->offset_in_after, data->strs_out, src + data->offset_out_after, CFL_SIZE);
	md_zadd(data->N, data->dims_in, dst, dst, dst_tmp);

	md_free(dst_tmp);
}

static void padding_free(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(padding_op_s, _data);

	xfree(data->strs_out);

	xfree(data->dims_for);
	xfree(data->strs_for);
	xfree(data->dims_mid);
	xfree(data->strs_mid);
	xfree(data->dims_after);
	xfree(data->strs_after);

	xfree(data->dims_in);
	xfree(data->dims_out);

	xfree(data);
}

struct linop_s* linop_padding_create_onedim(int N, const long dims[N], enum PADDING pad_type, int pad_dim, long pad_for, long pad_after)
{
	if ((PAD_VALID == pad_type) || (PAD_SAME == pad_type) || (PAD_CAUSAL == pad_type)) {

		long pad_for_arr[N];
		long pad_after_arr[N];

		for (int i = 0; i < N; i++) {

			pad_for_arr[i] = (pad_dim == i) ? pad_for : 0;
			pad_after_arr[i] = (pad_dim == i) ? pad_after : 0;
		}

		return linop_padding_create(N, dims, pad_type, pad_for_arr, pad_after_arr);
	}

	assert(pad_dim < N);
	assert(0 <= pad_for * pad_after); // same sign or zero

	PTR_ALLOC(long[N], dims_for);
	PTR_ALLOC(long[N], dims_mid);
	PTR_ALLOC(long[N], dims_after);
	PTR_ALLOC(long[N], dims_out);

	md_copy_dims(N, *dims_for, dims);
	md_copy_dims(N, *dims_mid, dims);
	md_copy_dims(N, *dims_after, dims);
	md_copy_dims(N, *dims_out, dims);

	(*dims_for)[pad_dim] = MAX(0l, pad_for);
	(*dims_mid)[pad_dim] = MIN(dims[pad_dim], dims[pad_dim] + pad_for + pad_after);
	(*dims_after)[pad_dim] = MAX(0l, pad_after);

	(*dims_out)[pad_dim] += (pad_for + pad_after);

	PTR_ALLOC(long[N], strs_out);
	PTR_ALLOC(long[N], strs_for);
	PTR_ALLOC(long[N], strs_mid);
	PTR_ALLOC(long[N], strs_after);

	md_calc_strides(N, *strs_out, *dims_out, CFL_SIZE);

	long pos[N];
	md_singleton_strides(N, pos);

	long offset_out_for = 0;
	long offset_out_mid = 0;
	long offset_out_after = 0;
	long offset_in_for = 0;
	long offset_in_mid = 0;
	long offset_in_after = 0;


	offset_out_for = md_calc_offset(N, *strs_out, pos) / (long)CFL_SIZE;

	pos[pad_dim] += MAX(0, pad_for);
	offset_out_mid = md_calc_offset(N, *strs_out, pos) / (long)CFL_SIZE;

	pos[pad_dim] += (*dims_mid)[pad_dim];
	offset_out_after = md_calc_offset(N, *strs_out, pos) / (long)CFL_SIZE;

	md_singleton_strides(N, pos); //pos = {0, 0, ...}

	long strs_in[N];
	md_calc_strides(N, strs_in, dims, CFL_SIZE);

	if ((0 > pad_for) || (0 > pad_after)) // reduction will always be valid type
		pad_type = PAD_VALID;

	switch (pad_type) {

	case PAD_VALID:
	case PAD_SAME:
	case PAD_CAUSAL:
		assert(0); // should be covered by copy_block

	case PAD_SYMMETRIC:

		assert(0 <= pad_for);
		assert(0 <= pad_after);

		pos[pad_dim] = pad_for - 1;

		offset_in_for = md_calc_offset(N, strs_in , pos) / (long)CFL_SIZE;

		md_calc_strides(N, *strs_for, dims, CFL_SIZE);
		(*strs_for)[pad_dim] = -(*strs_for)[pad_dim];

		offset_in_mid = 0;
		md_calc_strides(N, *strs_mid, dims, CFL_SIZE);

		pos[pad_dim] = dims[pad_dim] - 1;
		offset_in_after = md_calc_offset(N, strs_in , pos) / (long)CFL_SIZE;

		md_calc_strides(N, *strs_after, dims, CFL_SIZE);
		(*strs_after)[pad_dim] = -(*strs_after)[pad_dim];

		break;

	case PAD_REFLECT:

		assert(0 <= pad_for);
		assert(0 <= pad_after);

		pos[pad_dim] = pad_for;

		offset_in_for = md_calc_offset(N, strs_in , pos) / (long)CFL_SIZE;

		md_calc_strides(N, *strs_for, dims, CFL_SIZE);
		(*strs_for)[pad_dim] = -(*strs_for)[pad_dim];

		offset_in_mid = 0;
		md_calc_strides(N, *strs_mid, dims, CFL_SIZE);

		pos[pad_dim] = dims[pad_dim] - 2;
		offset_in_after = md_calc_offset(N, strs_in , pos) / (long)CFL_SIZE;

		md_calc_strides(N, *strs_after, dims, CFL_SIZE);
		(*strs_after)[pad_dim] = -(*strs_after)[pad_dim];

		break;

	case PAD_CYCLIC:

		assert(0 <= pad_for);
		assert(0 <= pad_after);

		pos[pad_dim] = dims[pad_dim] - pad_for;

		offset_in_for = md_calc_offset(N, strs_in , pos) / (long)CFL_SIZE;
		md_calc_strides(N, *strs_for, dims, CFL_SIZE);

		offset_in_mid = 0;
		md_calc_strides(N, *strs_mid, dims, CFL_SIZE);

		pos[pad_dim] = 0;
		offset_in_after = md_calc_offset(N, strs_in , pos) / (long)CFL_SIZE;
		md_calc_strides(N, *strs_after, dims, CFL_SIZE);

		break;

	default:
		assert(0);
	}

	PTR_ALLOC(long[N], dims_in);

	md_copy_dims(N, *dims_in, dims);

	long dims_out2[N];
	md_copy_dims(N, dims_out2, *dims_out);


	PTR_ALLOC(struct padding_op_s, data);
	SET_TYPEID(padding_op_s, data);

	data->N = N;
	data->offset_in_mid = offset_in_mid;
	data->offset_in_for = offset_in_for;
	data->offset_in_after = offset_in_after;
	data->offset_out_mid = offset_out_mid;
	data->offset_out_for = offset_out_for;
	data->offset_out_after = offset_out_after;



	data->dims_in = *PTR_PASS(dims_in);
	data->dims_out = *PTR_PASS(dims_out);
	data->dims_for = *PTR_PASS(dims_for);
	data->dims_mid = *PTR_PASS(dims_mid);
	data->dims_after = *PTR_PASS(dims_after);

	data->strs_out = *PTR_PASS(strs_out);
	data->strs_for = *PTR_PASS(strs_for);
	data->strs_mid = *PTR_PASS(strs_mid);
	data->strs_after = *PTR_PASS(strs_after);

	return linop_create(N, dims_out2, N, dims, CAST_UP(PTR_PASS(data)), padding_forward, padding_adjoint, NULL, NULL, padding_free);
}

struct linop_s* linop_padding_create(int N, const long dims[N], enum PADDING pad_type, long pad_for[N], long pad_after[N])
{
	for (int i = 0; i < N; i++)
		assert(0 <= pad_for[i] * pad_after[i]); // same sign or zero

	if ((PAD_VALID == pad_type) || (PAD_SAME == pad_type) || (PAD_CAUSAL == pad_type)) {

		long pos[N];
		long odims[N];

		for (int i = 0; i < N; i++) {

			pos[i] = labs(pad_for[i]);
			odims[i] = dims[i] + pad_after[i] + pad_for[i];
		}

		return linop_copy_block_create(N, pos, odims, dims);
	}

	struct linop_s* result = linop_identity_create(N, dims);

	for (int i = 0; i < N; i++)
		if ((0 != pad_for[i]) || (0 != pad_after[i]))
			result = linop_chain_FF(result, linop_padding_create_onedim(N, linop_codomain(result)->dims, pad_type, i, pad_for[i], pad_after[i]));

	return result;
}



struct extract_op_s {

	linop_data_t super;

	int N;
	const long* pos;
	const long* in_dims;
	const long* out_dims;
};

static DEF_TYPEID(extract_op_s);

static void extract_forward(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(extract_op_s, _data);

	md_clear(data->N, data->out_dims, dst, CFL_SIZE);
	md_copy_block(data->N, data->pos, data->out_dims, dst, data->in_dims, src, CFL_SIZE);
}

static void extract_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(extract_op_s, _data);

	md_clear(data->N, data->in_dims, dst, CFL_SIZE);
	md_copy_block(data->N, data->pos, data->in_dims, dst, data->out_dims, src, CFL_SIZE);
}

static void extract_free(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(extract_op_s, _data);

	xfree(data->out_dims);
	xfree(data->in_dims);
	xfree(data->pos);

	xfree(data);
}

extern struct linop_s* linop_extract_create(int N, const long pos[N], const long out_dims[N], const long in_dims[N])
{
	PTR_ALLOC(struct extract_op_s, data);
	SET_TYPEID(extract_op_s, data);

	data->N = N;
	data->pos = *TYPE_ALLOC(long[N]);
	data->out_dims = *TYPE_ALLOC(long[N]);
	data->in_dims = *TYPE_ALLOC(long[N]);

	md_copy_dims(N, (long*)data->pos, pos);
	md_copy_dims(N, (long*)data->out_dims, out_dims);
	md_copy_dims(N, (long*)data->in_dims, in_dims);

	return linop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), extract_forward, extract_adjoint, NULL, NULL, extract_free);
}

extern struct linop_s* linop_slice_create(int N, unsigned long flags, const long pos[N], const long dims[N])
{
	long odim[N];
	md_select_dims(N, ~flags, odim, dims);

	return linop_extract_create(N, pos, odim, dims);
}

extern struct linop_s* linop_slice_one_create(int N, int idx, long pos, const long dims[N])
{
	long _pos[N];
	md_set_dims(N, _pos, 0);

	_pos[idx] = pos;

	assert(0 <= idx);
	assert(idx < N);

	return linop_slice_create(N, MD_BIT(idx), _pos, dims);
}

struct linop_s* linop_reshape_create(int A, const long out_dims[A], int B, const long in_dims[B])
{
	PTR_ALLOC(struct linop_s, c);

	c->forward = operator_reshape_create(A, out_dims, B, in_dims);
	c->adjoint = operator_reshape_create(B, in_dims, A, out_dims);
	c->normal = operator_reshape_create(B, in_dims, B, in_dims);
	c->norm_inv = NULL;

	return PTR_PASS(c);
}



struct reshape_flagged_s {

	linop_data_t super;

	int N;
	unsigned long flags;
	const long* idims;
	const long* odims;
};

static DEF_TYPEID(reshape_flagged_s);

static void reshape_forward(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(reshape_flagged_s, _data);
	assert(dst != src);

	md_reshape(d->N, d->flags, d->odims, dst, d->idims, src, CFL_SIZE);
}

static void reshape_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(reshape_flagged_s, _data);;

	md_reshape(d->N, d->flags, d->idims, dst, d->odims, src, CFL_SIZE);
}

static void reshape_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(reshape_flagged_s, _data);

	if (dst != src)
		md_copy(d->N, d->odims, dst, src, CFL_SIZE);
}

static void reshape_free(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(reshape_flagged_s, _data);

	xfree(data->odims);
	xfree(data->idims);


	xfree(data);
}



struct linop_s* linop_reshape2_create(int N, unsigned long flags, const long out_dims[N], const long in_dims[N])
{
	if (md_check_equal_dims(N, MD_STRIDES(N, out_dims, CFL_SIZE), MD_STRIDES(N, in_dims, CFL_SIZE), ~flags))
		return linop_reshape_create(N, out_dims, N, in_dims);

	PTR_ALLOC(struct reshape_flagged_s, data);
	SET_TYPEID(reshape_flagged_s, data);

	assert(md_check_equal_dims(N, out_dims, in_dims, ~flags));
	assert(md_calc_size(N, out_dims) == md_calc_size(N, in_dims));

	data->N = N;
	data->odims = ARR_CLONE(long[N], out_dims);
	data->idims = ARR_CLONE(long[N], in_dims);
	data->flags = flags;

	return linop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), reshape_forward, reshape_adjoint, reshape_normal, NULL, reshape_free);
}



struct permute_op_s {

	linop_data_t super;

	int N;
	const long* idims;
	const long* odims;
	const int* order;
	const int* order_adj;
};

static DEF_TYPEID(permute_op_s);

static void permute_forward(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(permute_op_s, _data);

	md_permute(data->N, data->order, data->odims, dst, data->idims, src, CFL_SIZE);
}

static void permute_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(permute_op_s, _data);

	md_permute(data->N, data->order_adj, data->idims, dst, data->odims, src, CFL_SIZE);
}

static void permute_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(permute_op_s, _data);

	md_copy(data->N, data->idims, dst, src, CFL_SIZE);
}

static void permute_free(const linop_data_t* _data)
{
	auto data = CAST_DOWN(permute_op_s, _data);

	xfree(data->idims);
	xfree(data->odims);
	xfree(data->order);
	xfree(data->order_adj);

	xfree(data);
}


struct linop_s* linop_permute_create(int N, const int order[N], const long idims[N])
{
	long odims[N];
	md_permute_dims(N, order, odims, idims);

	int order_adj[N];
	for (int i = 0; i < N; i++)
		order_adj[order[i]] = i;

	PTR_ALLOC(struct permute_op_s, data);
	SET_TYPEID(permute_op_s, data);

	data->N = N;

	long* tidims = *TYPE_ALLOC(long[N]);
	long* todims = *TYPE_ALLOC(long[N]);
	int* torder = *TYPE_ALLOC(int[N]);
	int* torder_adj = *TYPE_ALLOC(int[N]);

	for (int i = 0; i < N; i++) {

		torder[i] = order[i];
		torder_adj[i] = order_adj[i];
	}

	md_copy_dims(N, tidims, idims);
	md_copy_dims(N, todims, odims);

	data->idims = tidims;
	data->odims = todims;

	data->order = torder;
	data->order_adj = torder_adj;

	return linop_create(N, odims, N, idims, CAST_UP(PTR_PASS(data)), permute_forward, permute_adjoint, permute_normal, NULL, permute_free);
}

extern struct linop_s* linop_permute_create(int N, const int order[N], const long idims[N]);

struct transpose_op_s {

	linop_data_t super;

	int N;
	int a;
	int b;
	const long* dims;
};

static DEF_TYPEID(transpose_op_s);

static void transpose_forward(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(transpose_op_s, _data);

	long odims[data->N];
	md_copy_dims(data->N, odims, data->dims);
	odims[data->a] = data->dims[data->b];
	odims[data->b] = data->dims[data->a];

	md_transpose(data->N, data->a, data->b, odims, dst, data->dims, src, CFL_SIZE);
}

static void transpose_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(transpose_op_s, _data);

	long odims[data->N];
	md_copy_dims(data->N, odims, data->dims);
	odims[data->a] = data->dims[data->b];
	odims[data->b] = data->dims[data->a];

	md_transpose(data->N, data->a, data->b, data->dims, dst, odims, src, CFL_SIZE);
}

static void transpose_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(transpose_op_s, _data);

	md_copy(data->N, data->dims, dst, src, CFL_SIZE);
}

static void transpose_free(const linop_data_t* _data)
{
	auto data = CAST_DOWN(transpose_op_s, _data);

	xfree(data->dims);

	xfree(data);
}


struct linop_s* linop_transpose_create(int N, int a, int b, const long dims[N])
{
	assert((0 <= a) && (a < N));
	assert((0 <= b) && (b < N));
	assert(a != b);

	PTR_ALLOC(struct transpose_op_s, data);
	SET_TYPEID(transpose_op_s, data);

	data->N = N;
	data->a = a;
	data->b = b;

	long* idims = *TYPE_ALLOC(long[N]);
	md_copy_dims(N, idims, dims);
	data->dims = idims;

	long odims[N];
	md_copy_dims(N, odims, dims);
	odims[a] = idims[b];
	odims[b] = idims[a];

	return linop_create(N, odims, N, idims, CAST_UP(PTR_PASS(data)),
			transpose_forward, transpose_adjoint, transpose_normal, NULL, transpose_free);
}


struct linop_s* linop_shift_create(int N, const long dims[N], int shift_dim, long shift, enum PADDING pad_type)
{
	auto lop_pad = linop_padding_create_onedim(N, dims, pad_type, shift_dim, MAX(shift, 0), MAX(-shift, 0));

	long dims_exp[N];
	md_copy_dims(N, dims_exp, dims);
	dims_exp[shift_dim] += labs(shift);

	long pos[N];
	md_set_dims(N, pos, 0);
	if (0 > shift)
		pos[shift_dim] = -shift;

	auto lop_ext = linop_extract_create(N, pos, dims, dims_exp);

	return linop_chain_FF(lop_pad, lop_ext);
}





struct flip_op_s {

	linop_data_t super;

	int N;
	unsigned long flags;
	const long* dims;
};

static DEF_TYPEID(flip_op_s);

static void flip_forward(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(flip_op_s, _data);
	md_flip(data->N, data->dims, data->flags, dst, src, CFL_SIZE);
}

static void flip_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(flip_op_s, _data);

	md_copy(data->N, data->dims, dst, src, CFL_SIZE);
}

static void flip_free(const linop_data_t* _data)
{
	auto data = CAST_DOWN(flip_op_s, _data);

	xfree(data->dims);
	xfree(data);
}


struct linop_s* linop_flip_create(int N, const long dims[N], unsigned long flags)
{
	PTR_ALLOC(struct flip_op_s, data);
	SET_TYPEID(flip_op_s, data);

	data->N = N;
	data->flags = flags;

	long* ndims = *TYPE_ALLOC(long[N]);
	md_copy_dims(N, ndims, dims);
	data->dims = ndims;

	return linop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), flip_forward, flip_forward, flip_normal, NULL, flip_free);
}


struct add_strided_s {

	linop_data_t super;

	int N;

	const long* dims;

	const long* istrs;
	const long* ostrs;

	int OO;
	const long* odims;

	int II;
	const long* idims;

	long ooffset;
	long ioffset;
};

static DEF_TYPEID(add_strided_s);

static void add_strided_forward(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(add_strided_s, _data);

	md_clear(data->OO, data->odims, dst, CFL_SIZE);
	md_zadd2(data->N, data->dims, data->ostrs, dst + data->ooffset, data->ostrs, dst + data->ooffset, data->istrs, src + data->ioffset);
}

static void add_strided_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(add_strided_s, _data);

	md_clear(data->II, data->idims, dst, CFL_SIZE);
	md_zadd2(data->N, data->dims, data->istrs, dst + data->ioffset, data->istrs, dst + data->ioffset, data->ostrs, src + data->ooffset);
}


static void add_strided_free(const linop_data_t* _data)
{
	auto data = CAST_DOWN(add_strided_s, _data);

	xfree(data->dims);
	xfree(data->odims);
	xfree(data->idims);
	xfree(data->ostrs);
	xfree(data->istrs);

	xfree(data);
}


struct linop_s* linop_add_strided_create(int N, const long dims[N], const long ostrs[N], const long istrs[N],
					int OO, const long odims[OO], int II, const long idims[II])
{
	PTR_ALLOC(struct add_strided_s, data);
	SET_TYPEID(add_strided_s, data);

	data->N = N;
	data->dims = ARR_CLONE(long[N], dims);
	data->ostrs = ARR_CLONE(long[N], ostrs);
	data->istrs = ARR_CLONE(long[N], istrs);

	data->OO = OO;
	data->II = II;

	data->odims = ARR_CLONE(long[OO], odims);
	data->idims = ARR_CLONE(long[II], idims);

	data->ioffset = 0;
	data->ooffset = 0;

	return linop_create(OO, odims, II, idims, CAST_UP(PTR_PASS(data)), add_strided_forward, add_strided_adjoint, NULL, NULL, add_strided_free);
}

struct linop_s* linop_hankelization_create(int N, const long dims[N], int dim, int window_dim, int window_size)
{
	long odims[N];
	md_copy_dims(N, odims, dims);

	assert(1 == odims[window_dim]);
	assert(window_size <= odims[dim]);

	odims[window_dim] = window_size;
	odims[dim] -= window_size - 1;
	long ostrs[N];
	long istrs[N];

	md_calc_strides(N, ostrs, odims, CFL_SIZE);
	md_calc_strides(N, istrs, dims, CFL_SIZE);
	istrs[window_dim] = istrs[dim];

	return linop_add_strided_create(N, odims, ostrs, istrs, N, odims, N, dims);
}



struct operator_matrix_s {

	linop_data_t super;

	const complex float* mat;
	const complex float* mat_gram; // A^H A
#ifdef USE_CUDA
	const complex float* mat_gpu;
	const complex float* mat_gram_gpu;
#endif
	int N;

	const long* mat_dims;
	const long* out_dims;
	const long* in_dims;

	const long* grm_dims;
	const long* gin_dims;
	const long* gout_dims;
};

static DEF_TYPEID(operator_matrix_s);


static void linop_matrix_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(operator_matrix_s, _data);
	const complex float* mat = data->mat;

#ifdef USE_CUDA
	if (cuda_ondevice(src)) {

		if (NULL == data->mat_gpu)
			data->mat_gpu = md_gpu_move(data->N, data->mat_dims, data->mat, CFL_SIZE);

		mat = data->mat_gpu;
	}
#endif

	md_ztenmul(data->N, data->out_dims, dst, data->in_dims, src, data->mat_dims, mat);
}

static void linop_matrix_apply_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(operator_matrix_s, _data);
	const complex float* mat = data->mat;

#ifdef USE_CUDA
	if (cuda_ondevice(src)) {

		if (NULL == data->mat_gpu)
			data->mat_gpu = md_gpu_move(data->N, data->mat_dims, data->mat, CFL_SIZE);

		mat = data->mat_gpu;
	}
#endif

	md_ztenmulc(data->N, data->in_dims, dst, data->out_dims, src, data->mat_dims, mat);
}

static void linop_matrix_apply_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(operator_matrix_s, _data);

	if (NULL == data->mat_gram) {

		complex float* tmp = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, src);

		linop_matrix_apply(_data, tmp, src);
		linop_matrix_apply_adjoint(_data, dst, tmp);

		md_free(tmp);

	} else {

		const complex float* mat_gram = data->mat_gram;
#ifdef USE_CUDA
		if (cuda_ondevice(src)) {

			if (NULL == data->mat_gram_gpu)
				data->mat_gram_gpu = md_gpu_move(2 * data->N, data->grm_dims, data->mat_gram, CFL_SIZE);

			mat_gram = data->mat_gram_gpu;
		}
#endif
		md_ztenmul(2 * data->N, data->gout_dims, dst, data->gin_dims, src, data->grm_dims, mat_gram);
	}
}

static void linop_matrix_del(const linop_data_t* _data)
{
	auto data = CAST_DOWN(operator_matrix_s, _data);

	xfree(data->out_dims);
	xfree(data->mat_dims);
	xfree(data->in_dims);
	xfree(data->gin_dims);
	xfree(data->gout_dims);
	xfree(data->grm_dims);

	md_free(data->mat);
	md_free(data->mat_gram);
#ifdef USE_CUDA
	md_free(data->mat_gpu);
	md_free(data->mat_gram_gpu);
#endif
	xfree(data);
}


static void shadow_dims(int N, long out[2 * N], const long in[N])
{
	for (int i = 0; i < N; i++) {

		out[2 * i + 0] = in[i];
		out[2 * i + 1] = 1;
	}
}


/* O I M G
 * 1 1 1 1   - not used
 * 1 1 A !   - forbidden
 * 1 A 1 !   - forbidden
 * A 1 1 !   - forbidden
 * A A 1 1   - replicated
 * A 1 A 1   - output
 * 1 A A A/A - input
 * A A A A   - batch
 */
static struct operator_matrix_s* linop_matrix_priv2(int N, const long out_dims[N], const long in_dims[N], const long matrix_dims[N], const complex float* matrix)
{
	// to get assertions and cost estimate

	long max_dims[N];
	md_tenmul_dims(N, max_dims, out_dims, in_dims, matrix_dims);


	PTR_ALLOC(struct operator_matrix_s, data);
	SET_TYPEID(operator_matrix_s, data);

	data->N = N;

	PTR_ALLOC(long[N], out_dims1);
	md_copy_dims(N, *out_dims1, out_dims);
	data->out_dims = *PTR_PASS(out_dims1);

	PTR_ALLOC(long[N], mat_dims1);
	md_copy_dims(N, *mat_dims1, matrix_dims);
	data->mat_dims = *PTR_PASS(mat_dims1);

	PTR_ALLOC(long[N], in_dims1);
	md_copy_dims(N, *in_dims1, in_dims);
	data->in_dims = *PTR_PASS(in_dims1);


	complex float* mat = md_alloc(N, matrix_dims, CFL_SIZE);

	md_copy(N, matrix_dims, mat, matrix, CFL_SIZE);

	data->mat = mat;
	data->mat_gram = NULL;
#ifdef USE_CUDA
	data->mat_gpu = NULL;
	data->mat_gram_gpu = NULL;
#endif

#if 1
	// pre-multiply gram matrix (if there is a cost reduction)

	unsigned long out_flags = md_nontriv_dims(N, out_dims);
	unsigned long in_flags = md_nontriv_dims(N, in_dims);

	unsigned long del_flags = in_flags & ~out_flags;
	unsigned long new_flags = out_flags & ~in_flags;

	/* we double (again) for the gram matrix
	 */

	PTR_ALLOC(long[2 * N], mat_dims2);
	PTR_ALLOC(long[2 * N], in_dims2);
	PTR_ALLOC(long[2 * N], gmt_dims2);
	PTR_ALLOC(long[2 * N], gin_dims2);
	PTR_ALLOC(long[2 * N], grm_dims2);
	PTR_ALLOC(long[2 * N], gout_dims2);

	shadow_dims(N, *gmt_dims2, matrix_dims);
	shadow_dims(N, *mat_dims2, matrix_dims);
	shadow_dims(N, *in_dims2, in_dims);
	shadow_dims(N, *gout_dims2, in_dims);
	shadow_dims(N, *gin_dims2, in_dims);
	shadow_dims(N, *grm_dims2, matrix_dims);

	/* move removed input dims into shadow position
	 * for the gram matrix can have an output there
	 */
	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(del_flags, i)) {

			assert((*mat_dims2)[2 * i + 0] == (*in_dims2)[2 * i + 0]);

			(*mat_dims2)[2 * i + 1] = (*mat_dims2)[2 * i + 0];
			(*mat_dims2)[2 * i + 0] = 1;

			(*in_dims2)[2 * i + 1] = (*gin_dims2)[2 * i + 0];
			(*in_dims2)[2 * i + 0] = 1;
		}
	}

	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(new_flags, i)) {

			(*grm_dims2)[2 * i + 0] = 1;
			(*grm_dims2)[2 * i + 1] = 1;
		}

		if (MD_IS_SET(del_flags, i)) {

			(*gout_dims2)[2 * i + 1] = (*gin_dims2)[2 * i + 0];
			(*gout_dims2)[2 * i + 0] = 1;

			(*grm_dims2)[2 * i + 0] = in_dims[i];
			(*grm_dims2)[2 * i + 1] = in_dims[i];
		}
	}


	long gmx_dims[2 * N];
	md_tenmul_dims(2 * N, gmx_dims, *gout_dims2, *gin_dims2, *grm_dims2);

	long mult_mat = md_calc_size(N, max_dims);
	long mult_gram = md_calc_size(2 * N, gmx_dims);

	if (mult_gram < 2 * mult_mat) {	// FIXME: rethink

		debug_printf(DP_DEBUG2, "Gram matrix: 2x %ld vs %ld\n", mult_mat, mult_gram);

		complex float* mat_gram = md_alloc(2 * N, *grm_dims2, CFL_SIZE);

		md_ztenmulc(2 * N, *grm_dims2, mat_gram, *gmt_dims2, matrix, *mat_dims2, matrix);

		data->mat_gram = mat_gram;
	}

	PTR_FREE(gmt_dims2);
	PTR_FREE(mat_dims2);
	PTR_FREE(in_dims2);

	data->gin_dims = *PTR_PASS(gin_dims2);
	data->gout_dims = *PTR_PASS(gout_dims2);
	data->grm_dims = *PTR_PASS(grm_dims2);
#else
	data->gin_dims = NULL;
	data->gout_dims = NULL;
	data->grm_dims = NULL;
#endif

	return PTR_PASS(data);
}


static struct operator_matrix_s* linop_matrix_priv(int N, const long out_dims[N], const long in_dims[N], const long matrix_dims[N], const complex float* matrix)
{
	unsigned long out_flags = md_nontriv_dims(N, out_dims);
	unsigned long in_flags = md_nontriv_dims(N, in_dims);

	unsigned long del_flags = in_flags & ~out_flags;

	/* we double dimensions for chaining which can lead to
	 * matrices with the same input and output dimension
	 */

	long out_dims2[2 * N];
	long mat_dims2[2 * N];
	long in_dims2[2 * N];

	shadow_dims(N, out_dims2, out_dims);
	shadow_dims(N, mat_dims2, matrix_dims);
	shadow_dims(N, in_dims2, in_dims);

	/* move removed input dims into shadow position
	 * which makes chaining easier below
	 */
	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(del_flags, i)) {

			assert(1 == out_dims2[2 * i + 0]);
			assert(mat_dims2[2 * i + 0] == in_dims2[2 * i + 0]);

			mat_dims2[2 * i + 1] = mat_dims2[2 * i + 0];
			mat_dims2[2 * i + 0] = 1;

			in_dims2[2 * i + 1] = in_dims[i];
			in_dims2[2 * i + 0] = 1;
		}
	}

	return linop_matrix_priv2(2 * N, out_dims2, in_dims2, mat_dims2, matrix);
}



/**
 * Operator interface for a true matrix:
 * out = mat * in
 * in:	[x x x x 1 x x K x x]
 * mat:	[x x x x T x x K x x]
 * out:	[x x x x T x x 1 x x]
 * where the x's are arbitrary dimensions and T and K may be transposed
 *
 * @param N number of dimensions
 * @param out_dims output dimensions after applying the matrix (codomain)
 * @param in_dims input dimensions to apply the matrix (domain)
 * @param matrix_dims dimensions of the matrix
 * @param matrix matrix data
 */
struct linop_s* linop_matrix_create(int N, const long out_dims[N], const long in_dims[N], const long matrix_dims[N], const complex float* matrix)
{
	struct operator_matrix_s* data = linop_matrix_priv(N, out_dims, in_dims, matrix_dims, matrix);

	return linop_create(N, out_dims, N, in_dims, CAST_UP(data),
			linop_matrix_apply, linop_matrix_apply_adjoint,
			linop_matrix_apply_normal, NULL, linop_matrix_del);
}


/**
 * Efficiently chain two matrix linops by multiplying the actual matrices together.
 * Stores a copy of the new matrix.
 * Returns: C = B A
 *
 * @param a first matrix (applied to input)
 * @param b second matrix (applied to output of first matrix)
 */
struct linop_s* linop_matrix_chain(const struct linop_s* a, const struct linop_s* b)
{
	const auto a_data = CAST_DOWN(operator_matrix_s, linop_get_data(a));
	const auto b_data = CAST_DOWN(operator_matrix_s, linop_get_data(b));

	// check compatibility
	assert(linop_codomain(a)->N == linop_domain(b)->N);
	assert(md_check_compat(linop_codomain(a)->N, 0u, linop_codomain(a)->dims, linop_domain(b)->dims));

	int D = linop_domain(a)->N;

	unsigned long outB_flags = md_nontriv_dims(D, linop_codomain(b)->dims);
	unsigned long inB_flags = md_nontriv_dims(D, linop_domain(b)->dims);

	unsigned long delB_flags = inB_flags & ~outB_flags;

	int N = a_data->N;
	assert(N == 2 * D);

	long in_dims[N];
	md_copy_dims(N, in_dims, a_data->in_dims);

	long matA_dims[N];
	md_copy_dims(N, matA_dims, a_data->mat_dims);

	long matB_dims[N];
	md_copy_dims(N, matB_dims, b_data->mat_dims);

	long out_dims[N];
	md_copy_dims(N, out_dims, b_data->out_dims);

	for (int i = 0; i < D; i++) {

		if (MD_IS_SET(delB_flags, i)) {

			matA_dims[2 * i + 0] = a_data->mat_dims[2 * i + 1];
			matA_dims[2 * i + 1] = a_data->mat_dims[2 * i + 0];

			in_dims[2 * i + 0] = a_data->in_dims[2 * i + 1];
			in_dims[2 * i + 1] = a_data->in_dims[2 * i + 0];
		}
	}


	long matrix_dims[N];
	md_singleton_dims(N, matrix_dims);

	unsigned long iflags = md_nontriv_dims(N, in_dims);
	unsigned long oflags = md_nontriv_dims(N, out_dims);
	unsigned long flags = iflags | oflags;

	// we combine a and b and sum over dims not in input or output

	md_max_dims(N, flags, matrix_dims, matA_dims, matB_dims);

	debug_printf(DP_DEBUG1, "tensor chain: %ld x %ld -> %ld\n",
			md_calc_size(N, matA_dims), md_calc_size(N, matB_dims), md_calc_size(N, matrix_dims));


	complex float* matrix = md_alloc(N, matrix_dims, CFL_SIZE);

	debug_print_dims(DP_DEBUG2, N, matrix_dims);
	debug_print_dims(DP_DEBUG2, N, in_dims);
	debug_print_dims(DP_DEBUG2, N, matA_dims);
	debug_print_dims(DP_DEBUG2, N, matB_dims);
	debug_print_dims(DP_DEBUG2, N, out_dims);

	md_ztenmul(N, matrix_dims, matrix, matA_dims, a_data->mat, matB_dims, b_data->mat);

	// priv2 takes our doubled dimensions

	struct operator_matrix_s* data = linop_matrix_priv2(N, out_dims, in_dims, matrix_dims, matrix);

	/* although we internally use different dimensions we define the
	 * correct interface
	 */
	struct linop_s* c = linop_create(linop_codomain(b)->N, linop_codomain(b)->dims,
			linop_domain(a)->N, linop_domain(a)->dims, CAST_UP(data),
			linop_matrix_apply, linop_matrix_apply_adjoint,
			linop_matrix_apply_normal, NULL, linop_matrix_del);

	md_free(matrix);

	return c;
}





struct fft_linop_s {

	linop_data_t super;

	float nscale;

	int N;
	unsigned long flags;
	long* dims;
	long* strs;
};

static DEF_TYPEID(fft_linop_s);

static void fft_linop_apply(const linop_data_t* _data, complex float* out, const complex float* in)
{
	const auto data = CAST_DOWN(fft_linop_s, _data);

	//FIXME: copy is unnecessary now, as the plan is searched on the fly from cache.
	//CAVEAT: tests/test-nlinv-pf-vcc fails on RISC-V if copy is removed.
	if (in != out)
		md_copy2(data->N, data->dims, data->strs, out, data->strs, in, CFL_SIZE);

	fft(data->N, data->dims, data->flags, out, out);
}

static void fft_linop_adjoint(const linop_data_t* _data, complex float* out, const complex float* in)
{
	const auto data = CAST_DOWN(fft_linop_s, _data);

	//FIXME: copy is unnecessary now, as the plan is searched on the fly from cache.
	//CAVEAT: tests/test-nlinv-pf-vcc fails on RISC-V if copy is removed.
	if (in != out)
		md_copy2(data->N, data->dims, data->strs, out, data->strs, in, CFL_SIZE);

	ifft(data->N, data->dims, data->flags, out, out);
}

static void fft_linop_free(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(fft_linop_s, _data);

	xfree(data->dims);
	xfree(data->strs);

	xfree(data);
}

static void fft_linop_normal(const linop_data_t* _data, complex float* out, const complex float* in)
{
	const auto data = CAST_DOWN(fft_linop_s, _data);

	md_zsmul(data->N, data->dims, out, in, data->nscale);
}

/**
 * Uncentered forward Fourier transform linear operator
 *
 * @param N number of dimensions
 * @param dims dimensions of input
 * @param flags bitmask of the dimensions to apply the Fourier transform
 */
struct linop_s* linop_fft_create(int N, const long dims[N], unsigned long flags)
{

	PTR_ALLOC(struct fft_linop_s, data);
	SET_TYPEID(fft_linop_s, data);

	data->N = N;
	data->flags = flags;

	data->dims = *TYPE_ALLOC(long[N]);
	md_copy_dims(N, data->dims, dims);

	data->strs = *TYPE_ALLOC(long[N]);
	md_calc_strides(N, data->strs, data->dims, CFL_SIZE);

	long fft_dims[N];
	md_select_dims(N, flags, fft_dims, dims);
	data->nscale = (float)md_calc_size(N, fft_dims);

	return linop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), fft_linop_apply, fft_linop_adjoint, fft_linop_normal, NULL, fft_linop_free);
}

/**
 * Uncentered backward Fourier transform linear operator
 *
 * @param N number of dimensions
 * @param dims dimensions of input
 * @param flags bitmask of the dimensions to apply the Fourier transform
 */
struct linop_s* linop_ifft_create(int N, const long dims[N], unsigned long flags)
{
	struct linop_s* lop_fft = linop_fft_create(N, dims, flags);
	struct linop_s* lop_ifft = (struct linop_s*)linop_get_adjoint(lop_fft);

	linop_free(lop_fft);
	return lop_ifft;
}


static struct linop_s* linop_fft_create_priv(	int N, const long dims[N], unsigned long flags,
						bool forward, unsigned long center_flags, unsigned long unitary_flags,
						unsigned long pre_flags, const complex float* pre_diag,
						unsigned long post_flags, const complex float* post_diag)
{
	flags = flags & md_nontriv_dims(N, dims);
	center_flags = center_flags & flags;
	unitary_flags = unitary_flags & flags;

	struct linop_s* lop = (forward ? linop_fft_create : linop_ifft_create)(N, dims, flags);

	if (0 != (center_flags | unitary_flags)) {

		long fft_mod_dims[N];
		md_select_dims(N, center_flags, fft_mod_dims, dims);

		complex float* fftmod_a = md_alloc(N, fft_mod_dims, CFL_SIZE);

		long fft_scale_dims[N];
		md_select_dims(N, unitary_flags, fft_scale_dims, dims);
		md_zfill(N, fft_mod_dims, fftmod_a, 1. / sqrtf(sqrtf(md_calc_size(N, fft_scale_dims))));

		(forward ? fftmod : ifftmod)(N, fft_mod_dims, center_flags & flags, fftmod_a, fftmod_a);

		struct linop_s* lop_fftmod = linop_cdiag_create(N, dims, center_flags, fftmod_a);
		md_free(fftmod_a);

		lop = linop_chain_FF(lop, linop_clone(lop_fftmod));
		lop = linop_chain_FF(lop_fftmod, lop);
	}

	if (NULL != post_diag) {

		lop = linop_chain_FF(lop, linop_cdiag_create(N, dims, post_flags, post_diag));
	}

	if (NULL != pre_diag) {

		lop = linop_chain_FF(linop_cdiag_create(N, dims, pre_flags, pre_diag), lop);
	}

	return lop;
}


/**
 * Centered forward Fourier transform linear operator
 *
 * @param N number of dimensions
 * @param dims dimensions of input
 * @param flags bitmask of the dimensions to apply the Fourier transform
 */
struct linop_s* linop_fftc_create(int N, const long dims[N], unsigned long flags)
{
	return linop_fft_create_priv(N, dims, flags, true, flags, flags, 0, NULL, 0, NULL);
}


/**
 * Centered inverse Fourier transform linear operator
 *
 * @param N number of dimensions
 * @param dims dimensions of input
 * @param flags bitmask of the dimensions to apply the Fourier transform
 */
struct linop_s* linop_ifftc_create(int N, const long dims[N], unsigned long flags)
{
	return linop_fft_create_priv(N, dims, flags, false, flags, flags, 0, NULL, 0, NULL);
}

/**
 * Centered forward Fourier transform linear operator chained with cdiag operator from both sides
 *
 * @param N number of dimensions
 * @param dims dimensions of input
 * @param flags bitmask of the dimensions to apply the Fourier transform
 * @param center_flags bitmask for centered fft
 * @param unitary_flags bitmask for unitary scaling
 * @param pre_flags bitmask of the dimensions of the pre-chained diag operator
 * @param pre_diag diagonal of the pre-chained diag operator
 * @param post_flags bitmask of the dimensions of the post-chained diag operator
 * @param post_diag diagonal of the post-chained diag operator
 */
struct linop_s* linop_fft_generic_create(int N, const long dims[N], unsigned long flags, unsigned long center_flags, unsigned long unitary_flags,
					   unsigned long pre_flag, const complex float* pre_diag, unsigned long post_flag, const complex float* post_diag)
{
	return linop_fft_create_priv(N, dims, flags, true, center_flags, unitary_flags, pre_flag, pre_diag, post_flag, post_diag);
}


/**
 * Centered inverse Fourier transform linear operator chained with cdiag operator from both sides
 *
 * @param N number of dimensions
 * @param dims dimensions of input
 * @param flags bitmask of the dimensions to apply the Fourier transform
 * @param center_flags bitmask for centered fft
 * @param unitary_flags bitmask for unitary scaling
 * @param pre_flags bitmask of the dimensions of the pre-chained diag operator
 * @param pre_diag diagonal of the pre-chained diag operator
 * @param post_flags bitmask of the dimensions of the post-chained diag operator
 * @param post_diag diagonal of the post-chained diag operator
 */
struct linop_s* linop_ifft_generic_create(int N, const long dims[N], unsigned long flags, unsigned long center_flags, unsigned long unitary_flags,
					    unsigned long pre_flag, const complex float* pre_diag, unsigned long post_flag, const complex float* post_diag)
{
	return linop_fft_create_priv(N, dims, flags, false, center_flags, unitary_flags, pre_flag, pre_diag, post_flag, post_diag);
}


struct linop_cdf97_s {

	linop_data_t super;

	int N;
	const long* dims;
	unsigned long flags;
};

static DEF_TYPEID(linop_cdf97_s);

static void linop_cdf97_apply(const linop_data_t* _data, complex float* out, const complex float* in)
{
	const auto data = CAST_DOWN(linop_cdf97_s, _data);

	md_copy(data->N, data->dims, out, in, CFL_SIZE);
	md_cdf97z(data->N, data->dims, data->flags, out);
}

static void linop_cdf97_adjoint(const linop_data_t* _data, complex float* out, const complex float* in)
{
	const auto data = CAST_DOWN(linop_cdf97_s, _data);

	md_copy(data->N, data->dims, out, in, CFL_SIZE);
	md_icdf97z(data->N, data->dims, data->flags, out);
}

static void linop_cdf97_normal(const linop_data_t* _data, complex float* out, const complex float* in)
{
	const auto data = CAST_DOWN(linop_cdf97_s, _data);

	md_copy(data->N, data->dims, out, in, CFL_SIZE);
}

static void linop_cdf97_free(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(linop_cdf97_s, _data);

	xfree(data->dims);

	xfree(data);
}



/**
 * Wavelet CFD9/7 transform operator
 *
 * @param N number of dimensions
 * @param dims dimensions of input
 * @param flags bitmask of the dimensions to apply the Fourier transform
 */
struct linop_s* linop_cdf97_create(int N, const long dims[N], unsigned long flags)
{
	PTR_ALLOC(struct linop_cdf97_s, data);
	SET_TYPEID(linop_cdf97_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);

	data->N = N;
	data->dims = *ndims;
	data->flags = flags;

	return linop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), linop_cdf97_apply, linop_cdf97_adjoint, linop_cdf97_normal, NULL, linop_cdf97_free);
}



struct conv_data_s {

	linop_data_t super;

	struct conv_plan* plan;
};

static DEF_TYPEID(conv_data_s);

static void linop_conv_forward(const linop_data_t* _data, complex float* out, const complex float* in)
{
	auto data = CAST_DOWN(conv_data_s, _data);

	conv_exec(data->plan, out, in);
}

static void linop_conv_adjoint(const linop_data_t* _data, complex float* out, const complex float* in)
{
	auto data = CAST_DOWN(conv_data_s, _data);

	conv_adjoint(data->plan, out, in);
}

static void linop_conv_free(const linop_data_t* _data)
{
	auto data = CAST_DOWN(conv_data_s, _data);

	conv_free(data->plan);

	xfree(data);
}


/**
 * Convolution operator
 *
 * @param N number of dimensions
 * @param flags bitmask of the dimensions to apply convolution
 * @param ctype
 * @param cmode
 * @param odims output dimensions
 * @param idims input dimensions
 * @param kdims kernel dimensions
 * @param krn convolution kernel
 */
struct linop_s* linop_conv_create(int N, unsigned long flags, enum conv_type ctype, enum conv_mode cmode, const long odims[N],
                const long idims[N], const long kdims[N], const complex float* krn)
{
	PTR_ALLOC(struct conv_data_s, data);
	SET_TYPEID(conv_data_s, data);

	data->plan = conv_plan(N, flags, ctype, cmode, odims, idims, kdims, krn);

	return linop_create(N, odims, N, idims, CAST_UP(PTR_PASS(data)), linop_conv_forward, linop_conv_adjoint, NULL, NULL, linop_conv_free);
}


struct linop_s* linop_conv_gaussian_create(int N, enum conv_type ctype, const long dims[N], const float sigma[N])
{
	unsigned long flags = 0;

	long kdims[N];
	md_singleton_dims(N, kdims);

	for (int i = 0; i < N; i++) {

		if (0. == sigma[i])
			continue;

		flags |= MD_BIT(i);

		kdims[i] = MIN(8. * sigma[i] + 1, dims[i]);
	}

	complex float* krn = md_alloc(N, kdims, CFL_SIZE);
	md_zfill(N, kdims, krn, 1.);

	for (int i = 0; i < N; i++) {

		if (0. == sigma[i])
			continue;

		complex float filter[kdims[i]];

		float tot = 0;

		for (int j = 0; j < kdims[i]; j++) {

			float x = (j - (kdims[i] / 2)) / sigma[i];
			filter[j] = expf(-0.5 * x * x);
			tot += expf(-0.5 * x * x);
		}

		for (int j = 0; j < kdims[i]; j++)
			filter[j] /= tot;

		long fdims[N];
		md_select_dims(N, MD_BIT(i), fdims, kdims);

		md_zmul2(N, kdims, MD_STRIDES(N, kdims, CFL_SIZE), krn, MD_STRIDES(N, kdims, CFL_SIZE), krn, MD_STRIDES(N, fdims, CFL_SIZE), filter);
	}

	auto ret = linop_conv_create(N, flags, ctype, CONV_SYMMETRIC, dims, dims, kdims, krn);

	md_free(krn);

	return ret;
}


/**
 * This function creates a Hadamard operator for multiplying input data with
 * a negative normalized Hadamard matrix.
 * It uses a temporary dimension N for multiplying the input data with the
 * Hadamard matrix and then reshapes the output to the original dimensions.
 *
 * @param N             Number of dimensions
 * @param in_dims       Input dimensions
 * @param hadamard_dim  Dimension to apply Hadamard transform (must be power of 2)
 */
struct linop_s* linop_hadamard_create(int N, const long in_dims[N], int hadamard_dim)
{
	int size = in_dims[hadamard_dim];

	// size has to be a power of two

	debug_printf(DP_DEBUG2, "Hadamard size: %d\n", size);

	assert((size > 1) && ((size & (size - 1)) == 0));

	long in2_dims[N + 1];
	md_copy_dims(N, in2_dims, in_dims);
	in2_dims[N] = 1;

	long matr_dims[N + 1];
	md_select_dims(N + 1, MD_BIT(hadamard_dim), matr_dims, in2_dims);
	matr_dims[N] = size;

	complex float* matrix = md_alloc(N + 1, matr_dims, CFL_SIZE);

	md_zfill(N + 1, matr_dims, matrix, 0.0f);

	// sylvester's construction

	matrix[0] = 1.0f;

	for (int n = 1; n < size; n *= 2) {

		for (int i = 0; i < n; i++) {

			for (int j = 0; j < n; j++) {

				matrix[(i + n) * size + j] = matrix[i * size + j];
				matrix[i * size + (j + n)] = matrix[i * size + j];
				matrix[(i + n) * size + (j + n)] = -matrix[i * size + j];
			}
		}
	}

	// normalize matrix

	md_zsmul(N + 1, matr_dims, matrix, matrix, -1.0f / sqrtf((float)size));

	long out_dims[N + 1];
	md_select_dims(N + 1, ~MD_BIT(hadamard_dim), out_dims, in2_dims);
	out_dims[N] = size;

	auto lop_reshape = linop_reshape_create(N + 1, in2_dims, N, in_dims);
	auto lop_hadamard = (struct linop_s *)linop_fmac_dims_create(N + 1, out_dims, in2_dims, matr_dims, matrix);
	lop_hadamard = linop_chain_FF(lop_reshape, lop_hadamard);

	// transpose so output dimensions are input dimensions

	auto lop_transpose = linop_transpose_create(N + 1, N, hadamard_dim, linop_codomain(lop_hadamard)->dims);

	lop_hadamard = linop_chain_FF(lop_hadamard, lop_transpose);
	lop_hadamard = linop_reshape_out_F(lop_hadamard, N, in_dims);

	md_free(matrix);

	return lop_hadamard;
}
