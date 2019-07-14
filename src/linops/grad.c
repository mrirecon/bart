/* Copyright 2014. The Regents of the University of California.
 * Copyright 2016-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */
 
#include <complex.h>
#include <assert.h>
#include <strings.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "linops/linop.h"

#include "misc/misc.h"

#include "grad.h"



static void grad_dims(unsigned int D, long dims2[D], int d, unsigned int flags, const long dims[D])
{
	md_copy_dims(D, dims2, dims);

	assert(1 == dims[d]);
	assert(!MD_IS_SET(flags, d));

	dims2[d] = bitcount(flags);
}


static void grad_op(unsigned int D, const long dims[D], int d, unsigned int flags, complex float* out, const complex float* in)
{
	unsigned int N = bitcount(flags);

	assert(N == dims[d]);
	assert(!MD_IS_SET(flags, d));

	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	long dims1[D];
	md_select_dims(D, ~MD_BIT(d), dims1, dims);

	long strs1[D];
	md_calc_strides(D, strs1, dims1, CFL_SIZE);

	unsigned int flags2 = flags;

	for (unsigned int i = 0; i < N; i++) {

		unsigned int lsb = ffs(flags2) - 1;
		flags2 = MD_CLEAR(flags2, lsb);

		md_zfdiff2(D, dims1, lsb, strs, (void*)out + i * strs[d], strs1, in);
	}

	assert(0 == flags2);
}


static void grad_adjoint(unsigned int D, const long dims[D], int d, unsigned int flags, complex float* out, const complex float* in)
{
	unsigned int N = bitcount(flags);

	assert(N == dims[d]);
	assert(!MD_IS_SET(flags, d));

	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	long dims1[D];
	md_select_dims(D, ~MD_BIT(d), dims1, dims);

	long strs1[D];
	md_calc_strides(D, strs1, dims1, CFL_SIZE);

	unsigned int flags2 = flags;

	complex float* tmp = md_alloc_sameplace(D, dims1, CFL_SIZE, out);

	md_clear(D, dims1, out, CFL_SIZE);
	md_clear(D, dims1, tmp, CFL_SIZE);

	for (unsigned int i = 0; i < N; i++) {

		unsigned int lsb = ffs(flags2) - 1;
		flags2 = MD_CLEAR(flags2, lsb);

		md_zfdiff_backwards2(D, dims1, lsb, strs1, tmp, strs, (void*)in + i * strs[d]);
		md_zadd(D, dims1, out, out, tmp);
	}

	md_free(tmp);

	assert(0 == flags2);
}




struct grad_s {

	INTERFACE(linop_data_t);

	int N;
	int d;
	long* dims;
	unsigned long flags;
};

static DEF_TYPEID(grad_s);

static void grad_op_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(grad_s, _data);

	grad_op(data->N, data->dims, data->d, data->flags, dst, src);
}
	
static void grad_op_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(grad_s, _data);

	grad_adjoint(data->N, data->dims, data->d, data->flags, dst, src);
}

static void grad_op_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(grad_s, _data);

	complex float* tmp = md_alloc_sameplace(data->N, data->dims, CFL_SIZE, dst);

	// this could be implemented more efficiently
	grad_op(data->N, data->dims, data->d, data->flags, tmp, src);
	grad_adjoint(data->N, data->dims, data->d, data->flags, dst, tmp);

	md_free(tmp);
}

static void grad_op_free(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(grad_s, _data);

	xfree(data->dims);
	xfree(data);
}

struct linop_s* linop_grad_create(long N, const long dims[N], int d, unsigned int flags)
{
	PTR_ALLOC(struct grad_s, data);
	SET_TYPEID(grad_s, data);

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
	dims2[d] = 1;

	grad_dims(NO, dims2, d, flags, dims2);

	data->N = NO;
	data->d = d;
	data->flags = flags;

	data->dims = *TYPE_ALLOC(long[N + 1]);

	md_copy_dims(NO, data->dims, dims2);

	return linop_create(NO, dims2, N, dims, CAST_UP(PTR_PASS(data)), grad_op_apply, grad_op_adjoint, grad_op_normal, NULL, grad_op_free);
}

