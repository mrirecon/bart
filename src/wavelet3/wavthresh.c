/* Copyright 2014. The Regents of the University of California.
 * Copyright 2016-2017. Martin Uecker
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013 Frank Ong <uecker@eecs.berkeley.edu>
 * 2013-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#define _GNU_SOURCE
#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <stdlib.h>

#include "misc/misc.h"
#include "misc/types.h"

#include "num/multind.h"
#include "num/ops.h"

#include "wavelet3/wavelet.h"

#include "wavthresh.h"


struct wavelet3_thresh_s {

	INTERFACE(operator_data_t);

	unsigned int N;
	const long* dims;
	const long* minsize;
	unsigned int flags;
	float lambda;
	bool randshift;
	int rand_state;
};

DEF_TYPEID(wavelet3_thresh_s);


static int rand_lim(unsigned int* state, int limit)
{
        int divisor = RAND_MAX / (limit + 1);
        int retval;

        do {
                retval = rand_r(state) / divisor;

        } while (retval > limit);

        return retval;
}


static void wavelet3_thresh_apply(const operator_data_t* _data, float mu, complex float* out, const complex float* in)
{
	const struct wavelet3_thresh_s* data = CAST_DOWN(wavelet3_thresh_s, _data);

	long shift[data->N];
	for (unsigned int i = 0; i < data->N; i++)
		shift[i] = 0;

	if (data->randshift) {

		for (unsigned int i = 0; i < data->N; i++) {

			if (MD_IS_SET(data->flags, i)) {

				int levels = wavelet_num_levels(data->N, MD_BIT(i), data->dims, data->minsize, 4);
				shift[i] = rand_lim((unsigned int*)&data->rand_state, 1 << levels);

				assert(shift[i] < data->dims[i]);
			}
		}
	}

	wavelet3_thresh(data->N, data->lambda * mu, data->flags, shift, data->dims,
		out, in, data->minsize, 4, wavelet3_dau2);
}

static void wavelet3_thresh_del(const operator_data_t* _data)
{
	const struct wavelet3_thresh_s* data = CAST_DOWN(wavelet3_thresh_s, _data);
	xfree(data->dims);
	xfree(data->minsize);
	xfree(data);
}


/**
 * Proximal operator for l1-norm with Wavelet transform: f(x) = lambda || W x ||_1
 *
 * @param N number of dimensions
 * @param dims dimensions of x
 * @param flags bitmask for Wavelet transform
 * @param minsize minimium size of coarse Wavelet scale
 * @param lambda threshold parameter
 * @param randshift random shifting
 */
const struct operator_p_s* prox_wavelet3_thresh_create(unsigned int N, const long dims[N], unsigned int flags, const long minsize[N], float lambda, bool randshift)
{
	PTR_ALLOC(struct wavelet3_thresh_s, data);
	SET_TYPEID(wavelet3_thresh_s, data);

	data->N = N;

	long (*ndims)[N] = TYPE_ALLOC(long[N]);
	md_copy_dims(N, (*ndims), dims);
	data->dims = *ndims;

	long (*nminsize)[N] = TYPE_ALLOC(long[N]);
	md_copy_dims(N, (*nminsize), minsize);
	data->minsize = *nminsize;

	data->flags = flags;
	data->lambda = lambda;
	data->randshift = randshift;
	data->rand_state = 1;

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), wavelet3_thresh_apply, wavelet3_thresh_del);
}



