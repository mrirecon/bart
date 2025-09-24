/* Copyright 2014. The Regents of the University of California.
 * Copyright 2016-2022. Martin Uecker
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013 Frank Ong <uecker@eecs.berkeley.edu>
 * 2013-2022 Martin Uecker <uecker@tugraz.at>
 */

#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <stdlib.h>

#include "misc/misc.h"
#include "misc/types.h"

#include "num/multind.h"
#include "num/ops.h"
#include "num/ops_p.h"
#include "num/rand.h"

#include "wavelet/wavelet.h"

#include "wavthresh.h"


struct wavelet_thresh_s {

	operator_data_t super;

	int N;
	const long* dims;
	const long* minsize;
	unsigned long flags;
	unsigned long jflags;
	float lambda;
	bool randshift;
	struct bart_rand_state* rand_state;

	int flen;
	const void* filter;
};

static DEF_TYPEID(wavelet_thresh_s);


static void wavelet_thresh_apply(const operator_data_t* _data, float mu, complex float* out, const complex float* in)
{
	const auto data = CAST_DOWN(wavelet_thresh_s, _data);

	long shift[data->N];
	for (int i = 0; i < data->N; i++)
		shift[i] = 0;

	if (data->randshift) {

		for (int i = 0; i < data->N; i++) {

			if (MD_IS_SET(data->flags, i)) {

				int levels = wavelet_num_levels(data->N, MD_BIT(i), data->dims, data->minsize, data->flen);
				shift[i] = rand_range_state(data->rand_state, (1 << levels) + 1u); // +1, as we want to include the limit

				assert(shift[i] < data->dims[i]);
			}
		}
	}

	wavelet_thresh(data->N, data->lambda * mu, data->flags, data->jflags, shift, data->dims,
		out, in, data->minsize, data->flen, data->filter);
}



void wavthresh_rand_state_set(const struct operator_p_s* op, unsigned long long x)
{
	auto data = CAST_DOWN(wavelet_thresh_s, operator_p_get_data(op));
	rand_state_update(data->rand_state, x);
}


static void wavelet_thresh_del(const operator_data_t* _data)
{
	const auto data = CAST_DOWN(wavelet_thresh_s, _data);
	xfree(data->dims);
	xfree(data->minsize);
	xfree(data->rand_state);
	xfree(data);
}


/**
 * Proximal operator for l1-norm with Wavelet transform: f(x) = lambda || W x ||_1
 *
 * @param N number of dimensions
 * @param dims dimensions of x
 * @param flags bitmask for Wavelet transform
 * @param jflags bitmask for joint thresholding
 * Qparam wtype wavelet type
 * @param minsize minimum size of coarse Wavelet scale
 * @param lambda threshold parameter
 * @param randshift random shifting
 */
const struct operator_p_s* prox_wavelet_thresh_create(int N, const long dims[N], unsigned long flags, unsigned long jflags,
				enum wtype wtype, const long minsize[N], float lambda, bool randshift)
{
	PTR_ALLOC(struct wavelet_thresh_s, data);
	SET_TYPEID(wavelet_thresh_s, data);

	data->N = N;

	long (*ndims)[N] = TYPE_ALLOC(long[N]);
	md_copy_dims(N, (*ndims), dims);
	data->dims = *ndims;

	long (*nminsize)[N] = TYPE_ALLOC(long[N]);
	md_copy_dims(N, (*nminsize), minsize);
	data->minsize = *nminsize;

	data->flags = flags;
	data->jflags = jflags;
	data->lambda = lambda;
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


	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), wavelet_thresh_apply, wavelet_thresh_del);
}

