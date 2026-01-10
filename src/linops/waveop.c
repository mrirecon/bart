/* Copyright 2015. The Regents of the University of California.
 * Copyright 2016-2018. Martin Uecker.
 * Copyright 2017. University of Oxford.
 * Copyright 2024-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2017 Martin Uecker
 * 2017 Sofia Dimoudi
 */

#include <assert.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"

#include "linops/linop.h"

#include "misc/misc.h"

#include "wavelet/wavelet.h"

#include "waveop.h"

struct wavelet_s {

	linop_data_t super;

	int N;
	unsigned long flags;
	const long* idims;
	const long* istr;
	const long* odims;
	const long* ostr;
	const long* minsize;
	long* shifts;
	bool randshift;
	struct bart_rand_state *rand_state;
	int flen;
	const void* filter;
};

static DEF_TYPEID(wavelet_s);

static void wavelet_forward(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(wavelet_s, _data);

	if (data->randshift) {

		for (int i = 0; i < data->N; i++) {

			if (MD_IS_SET(data->flags, i)) {

				int levels = wavelet_num_levels(data->N, MD_BIT(i), data->idims, data->minsize, data->flen);
				data->shifts[i] = rand_range_state(data->rand_state, (1 << levels) + 1u); // +1, as we want to include the limit

				assert(data->shifts[i] < data->idims[i]);
			}
		}
	}

	fwt2(data->N, data->flags, data->shifts, data->odims, data->ostr, dst, data->idims, data->istr, src, data->minsize, data->flen, data->filter);
}

static void wavelet_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(wavelet_s, _data);

	iwt2(data->N, data->flags, data->shifts, data->idims, data->istr, dst, data->odims, data->ostr, src, data->minsize, data->flen, data->filter);
}

static void wavelet_del(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(wavelet_s, _data);

	xfree(data->odims);
	xfree(data->ostr);
	xfree(data->idims);
	xfree(data->istr);
	xfree(data->minsize);
	xfree(data->shifts);
	xfree(data->rand_state);

	xfree(data);
}

struct linop_s* linop_wavelet_create(int N, unsigned long flags, const long dims[N], const long istr[N], enum wtype wtype, const long minsize[N], bool randshift)
{
	PTR_ALLOC(struct wavelet_s, data);
	SET_TYPEID(wavelet_s, data);

	data->N = N;
	data->flags = flags;
	data->randshift = randshift;
	data->rand_state = rand_state_create(1);
	data->flen = 0;
	data->filter = NULL;

	switch (wtype) {

	case WAVELET_HAAR:
		data->flen = ARRAY_SIZE(wavelet_haar[0][0]);
		data->filter = &wavelet_haar;
		break;

	case WAVELET_DAU2:
		data->flen = ARRAY_SIZE(wavelet_dau2[0][0]);
		data->filter = &wavelet_dau2;
		break;

	case WAVELET_CDF44:
		data->flen = ARRAY_SIZE(wavelet_cdf44[0][0]);
		data->filter = &wavelet_cdf44;
		break;
	}

	long (*idims)[N] = TYPE_ALLOC(long[N]);
	md_copy_dims(N, *idims, dims);
	data->idims = *idims;

	long (*nistr)[N] = TYPE_ALLOC(long[N]);
	md_copy_strides(N, *nistr, istr);
	data->istr = *nistr;

	long (*nminsize)[N] = TYPE_ALLOC(long[N]);
	md_copy_dims(N, *nminsize, minsize);
	data->minsize = *nminsize;

	long (*odims)[N] = TYPE_ALLOC(long[N]);
	wavelet_coeffs2(N, flags, *odims, dims, minsize, data->flen);
	data->odims = *odims;

	long (*ostr)[N] = TYPE_ALLOC(long[N]);
	md_calc_strides(N, *ostr, *odims, CFL_SIZE);
	data->ostr = *ostr;

	long (*shifts)[N] = TYPE_ALLOC(long[N]);
	for (int i = 0; i < data->N; i++)
		(*shifts)[i] = 0;

	data->shifts = *shifts;

	return linop_create2(N, *odims, *ostr, N, dims, istr, CAST_UP(PTR_PASS(data)), wavelet_forward, wavelet_adjoint, NULL, NULL, wavelet_del);
}



