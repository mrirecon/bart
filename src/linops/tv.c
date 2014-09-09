/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014 Martin Uecker <uecker@eecs.berkeley.edu>
 */
 
#include <complex.h>
#include <assert.h>
#include <strings.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "linops/linop.h"

#include "misc/misc.h"

#include "tv.h"




static unsigned int bitcount(unsigned int flags)
{
	unsigned int N = 0;

	for (; flags > 0; N++)
		flags &= (flags - 1);
		
	return N;
}


void tv_op(unsigned int D, const long dims[D], unsigned int flags, complex float* out, const complex float* in)
{
	unsigned int N = bitcount(flags);

	assert(N == dims[D - 1]);	// we use the highest dim to store our different partial derivatives

	unsigned int flags2 = flags;

	for (unsigned int i = 0; i < N; i++) {

		unsigned int lsb = ffs(flags2) - 1;
		flags2 ^= (1 << lsb);

		md_zfdiff(D - 1, dims, lsb, out + i * md_calc_size(D - 1, dims), in);
	}

	assert(0 == flags2);
}


void tv_adjoint(unsigned int D, const long dims[D], unsigned int flags, complex float* out, const complex float* in)
{
	unsigned int N = bitcount(flags);

	assert(N == dims[D - 1]);	// we use the highest dim to store our different partial derivatives

	unsigned int flags2 = flags;

	complex float* tmp = md_alloc_sameplace(D - 1, dims, CFL_SIZE, out);

	md_clear(D - 1, dims, out, CFL_SIZE);
	md_clear(D - 1, dims, tmp, CFL_SIZE);

	for (unsigned int i = 0; i < N; i++) {

		unsigned int lsb = ffs(flags2) - 1;
		flags2 ^= (1 << lsb);

		md_zfdiff_backwards(D - 1, dims, lsb, tmp, in + i * md_calc_size(D - 1, dims));
	
		md_zadd(D - 1, dims, out, out, tmp);
	}

	md_free(tmp);

	assert(0 == flags2);
}



void tv(unsigned int D, const long dims[D], unsigned int flags, complex float* out, const complex float* in)
{
	unsigned int N = bitcount(flags);

	long dims2[D + 1];
	md_copy_dims(D, dims2, dims);
	dims2[D] = N;

	complex float* tmp = md_alloc_sameplace(D + 1, dims2, CFL_SIZE, out);

	tv_op(D + 1, dims2, flags, tmp, in);

	// rss should be moved elsewhere
	md_rss(D + 1, dims2, flags, out, tmp);
	md_free(tmp);
}




struct tv_s {

	unsigned int N;
	long* dims;
	unsigned long flags;
};

static void tv_op_apply(const void* _data, complex float* dst, const complex float* src)
{
	const struct tv_s* data = _data;

	tv_op(data->N, data->dims, data->flags, dst, src);
}
	
static void tv_op_adjoint(const void* _data, complex float* dst, const complex float* src)
{
	const struct tv_s* data = _data;

	tv_adjoint(data->N, data->dims, data->flags, dst, src);
}

static void tv_op_normal(const void* _data, complex float* dst, const complex float* src)
{
	const struct tv_s* data = _data;

	complex float* tmp = md_alloc_sameplace(data->N, data->dims, CFL_SIZE, dst);

	// this could be implemented more efficiently
	tv_op(data->N, data->dims, data->flags, tmp, src);
	tv_adjoint(data->N, data->dims, data->flags, dst, tmp);

	md_free(tmp);
}


static void tv_op_free(const void* _data)
{
	const struct tv_s* data = _data;
	free(data->dims);
	free((void*)data);
}


struct linop_s* tv_init(long N, const long dims[N], unsigned int flags)
{
	struct tv_s* data = xmalloc(sizeof(struct tv_s));
	
	data->N = N + 1;
	data->dims = xmalloc((N + 1) * sizeof(long));
	data->flags = flags;

	md_copy_dims(N, data->dims, dims);
	data->dims[N] = bitcount(flags);
	
	return linop_create(N + 1, data->dims, N, dims, data, tv_op_apply, tv_op_adjoint, tv_op_normal, NULL, tv_op_free);
}

