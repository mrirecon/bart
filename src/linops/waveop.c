/* Copyright 2015. The Regents of the University of California.
 * Copyright 2016-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <assert.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "linops/linop.h"

#include "misc/misc.h"

#include "wavelet3/wavelet.h"

#include "waveop.h"

struct wavelet_s {

	INTERFACE(linop_data_t);

	unsigned int N;
	unsigned int flags;
	const long* dims;
	const long* istr;
	const long* minsize;
};

DEF_TYPEID(wavelet_s);

static void wavelet_forward(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct wavelet_s* data = CAST_DOWN(wavelet_s, _data);

	long shifts[data->N];
	for (unsigned int i = 0; i < data->N; i++)
		shifts[i] = 0;

	fwt(data->N, data->flags, shifts, data->dims, dst, data->istr, src, data->minsize, 4, wavelet3_dau2);
}

static void wavelet_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct wavelet_s* data = CAST_DOWN(wavelet_s, _data);

	long shifts[data->N];
	for (unsigned int i = 0; i < data->N; i++)
		shifts[i] = 0;

	iwt(data->N, data->flags, shifts, data->dims, data->istr, dst, src, data->minsize, 4, wavelet3_dau2);
}

static void wavelet_del(const linop_data_t* _data)
{
	const struct wavelet_s* data = CAST_DOWN(wavelet_s, _data);

	xfree(data->dims);
	xfree(data->istr);
	xfree(data->minsize);

	xfree(data);
}

struct linop_s* linop_wavelet3_create(unsigned int N, unsigned int flags, const long dims[N], const long istr[N], const long minsize[N])
{
	PTR_ALLOC(struct wavelet_s, data);
	SET_TYPEID(wavelet_s, data);

	data->N = N;
	data->flags = flags;

	long (*ndims)[N] = TYPE_ALLOC(long[N]);
	md_copy_dims(N, *ndims, dims);
	data->dims = *ndims;

	long (*nistr)[N] = TYPE_ALLOC(long[N]);
	md_copy_strides(N, *nistr, istr);
	data->istr = *nistr;

	long (*nminsize)[N] = TYPE_ALLOC(long[N]);
	md_copy_dims(N, *nminsize, minsize);
	data->minsize = *nminsize;

	long odims[N];
	md_singleton_dims(N, odims);
//	md_select_dims(N, ~flags, odims, dims);

	assert(1 == odims[0]);

	odims[0] = wavelet_coeffs(N, flags, dims, minsize, 4);

	long ostr[N];
	md_calc_strides(N, ostr, odims, CFL_SIZE);

	return linop_create2(N, odims, ostr, N, dims, istr, CAST_UP(PTR_PASS(data)), wavelet_forward, wavelet_adjoint, NULL, NULL, wavelet_del);
}



