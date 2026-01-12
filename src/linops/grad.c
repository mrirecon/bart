/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2016-2019. Martin Uecker.
 * Copyright 2024-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#include <complex.h>
#include <assert.h>
#include <strings.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "linops/linop.h"

#include "misc/misc.h"

#include "grad.h"


typedef void (*md_zfdiff_core_t)(int D, const long dims[D], int d, bool adj, const long ostr[D], complex float* out, const long istr[D], const complex float* in);

static void md_zfdiff_core2(int D, const long dims[D], int d, bool dir, bool adj, const long ostr[D], complex float* out, const long istr[D], const complex float* in)
{
	long pos[D];
	md_set_dims(D, pos, 0);

	if (adj)
		pos[d] = dir ? 1 : -1;
	else
		pos[d] = dir ? -1 : 1;

	md_circ_shift2(D, dims, pos, ostr, out, istr, in, CFL_SIZE);

	if (dir)
		md_zsub2(D, dims, ostr, out, ostr, out, istr, in);
	else
		md_zsub2(D, dims, ostr, out, istr, in, ostr, out);
}

static void md_zfdiff_f_core2(int D, const long dims[D], int d, bool adj, const long ostr[D], complex float* out, const long istr[D], const complex float* in)
{
	md_zfdiff_core2(D, dims, d, true, adj, ostr, out, istr, in);
}

static void md_zfdiff_b_core2(int D, const long dims[D], int d, bool adj, const long ostr[D], complex float* out, const long istr[D], const complex float* in)
{
	md_zfdiff_core2(D, dims, d, true, adj, ostr, out, istr, in);
}


static void md_zfdiff_z_core2(int D, const long dims[D], int d, bool adj, const long ostr[D], complex float* out, const long istr[D], const complex float* in)
{
	long pos[D];
	md_set_dims(D, pos, 0);

	pos[d] = -1;
	md_circ_shift2(D, dims, pos, ostr, out, istr, in, CFL_SIZE);

	complex float* tmp = md_alloc_sameplace(D, dims, CFL_SIZE, out);

	pos[d] = 1;
	md_circ_shift2(D, dims, pos, MD_STRIDES(D, dims, CFL_SIZE), tmp, istr, in, CFL_SIZE);

	md_zsub2(D, dims, ostr, out, ostr, out, MD_STRIDES(D, dims, CFL_SIZE), tmp);

	md_free(tmp);

	md_zsmul2(D, dims, ostr, out, ostr, out, adj ? -0.5 : 0.5);
}



static void grad_op(md_zfdiff_core_t grad, int D, const long dims[D], int d, unsigned long flags, complex float* out, const complex float* in)
{
	int N = bitcount(flags);

	assert(N == dims[d]);
	assert(!MD_IS_SET(flags, d));

	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	long dims1[D];
	md_select_dims(D, ~MD_BIT(d), dims1, dims);

	long strs1[D];
	md_calc_strides(D, strs1, dims1, CFL_SIZE);

	unsigned long flags2 = flags;

	for (int i = 0; i < N; i++) {

		int lsb = ffsl((long)flags2) - 1;
		flags2 = MD_CLEAR(flags2, lsb);

		grad(D, dims1, lsb, false, strs, (void*)out + i * strs[d], strs1, in);
	}

	assert(0 == flags2);
}


static void grad_adjoint(md_zfdiff_core_t grad, int D, const long dims[D], int d, unsigned long flags, complex float* out, const complex float* in)
{
	int N = bitcount(flags);

	assert(N == dims[d]);
	assert(!MD_IS_SET(flags, d));

	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	long dims1[D];
	md_select_dims(D, ~MD_BIT(d), dims1, dims);

	long strs1[D];
	md_calc_strides(D, strs1, dims1, CFL_SIZE);

	unsigned long flags2 = flags;

	complex float* tmp = md_alloc_sameplace(D, dims1, CFL_SIZE, out);

	md_clear(D, dims1, out, CFL_SIZE);
	md_clear(D, dims1, tmp, CFL_SIZE);

	for (int i = 0; i < N; i++) {

		int lsb = ffsl((long)flags2) - 1;
		flags2 = MD_CLEAR(flags2, lsb);

		grad(D, dims1, lsb, true, strs1, tmp, strs, (const void*)in + i * strs[d]);
		md_zadd(D, dims1, out, out, tmp);
	}

	md_free(tmp);

	assert(0 == flags2);
}




struct grad_s {

	linop_data_t super;

	md_zfdiff_core_t grad;

	int N;
	int d;
	long* dims;
	unsigned long flags;
};

static DEF_TYPEID(grad_s);

static void grad_op_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(grad_s, _data);

	grad_op(data->grad, data->N, data->dims, data->d, data->flags, dst, src);
}
	
static void grad_op_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(grad_s, _data);

	grad_adjoint(data->grad, data->N, data->dims, data->d, data->flags, dst, src);
}


static void grad_op_free(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(grad_s, _data);

	xfree(data->dims);
	xfree(data);
}

static struct linop_s* linop_grad_internal_create(md_zfdiff_core_t grad, long N, const long dims[N], int d, unsigned long flags)
{
	PTR_ALLOC(struct grad_s, data);
	SET_TYPEID(grad_s, data);

	data->grad = grad;

	int NO = N;

	if (N == d) {

		// as a special case, id d is one after the last dimensions,
		// we extend the output dimensions by one.

		NO++;

	} else {

		assert(1 == dims[d]);
	}

	long dims2[NO];
	md_copy_dims(N, dims2, dims);

	assert(!MD_IS_SET(flags, d));

	dims2[d] = bitcount(flags);

	data->N = NO;
	data->d = d;
	data->flags = flags;

	data->dims = *TYPE_ALLOC(long[N + 1]);

	md_copy_dims(NO, data->dims, dims2);

	return linop_create(NO, dims2, N, dims, CAST_UP(PTR_PASS(data)), grad_op_apply, grad_op_adjoint, NULL, NULL, grad_op_free);
}

struct linop_s* linop_grad_forward_create(long N, const long dims[N], int d, unsigned long flags)
{
	return linop_grad_internal_create(md_zfdiff_f_core2, N, dims, d, flags);
}

struct linop_s* linop_grad_backward_create(long N, const long dims[N], int d, unsigned long flags)
{
	return linop_grad_internal_create(md_zfdiff_b_core2, N, dims, d, flags);
}

struct linop_s* linop_grad_zentral_create(long N, const long dims[N], int d, unsigned long flags)
{
	return linop_grad_internal_create(md_zfdiff_z_core2, N, dims, d, flags);
}

struct linop_s* linop_grad_create(long N, const long dims[N], int d, unsigned long flags)
{
	return linop_grad_backward_create(N, dims, d, flags);
}

