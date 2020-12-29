/* Copyright 2014. The Regents of the University of California.
 * Copyright 2017, 2021. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2021 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <assert.h>
#include <stdbool.h>

#include "num/fft.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/misc.h"

#include "conv.h"




struct conv_plan {

	enum conv_mode cmode;
	enum conv_type ctype;

	int N;
	unsigned int flags;

	const struct operator_s* fft1;
	const struct operator_s* ifft1;
	const struct operator_s* fft2;
	const struct operator_s* ifft2;

	long* idims;
	long* odims;

	long* dims;
	long* dims1;
	long* dims2;
	long* str1;
	long* str2;

	long* kdims;
	long* kstr;

	complex float* kernel;
};





struct conv_plan* conv_plan(int N, unsigned int flags, enum conv_type ctype, enum conv_mode cmode, const long odims[N],  
		const long idims1[N], const long idims2[N], const complex float* src2)
{
	assert(   (!((CONV_VALID == ctype) || (CONV_EXTENDED == ctype)))
	       || (CONV_CAUSAL == cmode));

	PTR_ALLOC(struct conv_plan, plan);

	plan->N = N;
	plan->flags = flags;
	plan->cmode = cmode;
	plan->ctype = ctype;

	plan->dims  = *TYPE_ALLOC(long[N]);
	plan->dims1 = *TYPE_ALLOC(long[N]);
	plan->dims2 = *TYPE_ALLOC(long[N]);
	plan->kdims = *TYPE_ALLOC(long[N]);

	plan->idims = *TYPE_ALLOC(long[N]);
	plan->odims = *TYPE_ALLOC(long[N]);

	complex float U = 1.;

	long shift[N];

        for (int i = 0; i < N; i++) {

		plan->idims[i] = idims1[i];
		plan->odims[i] = odims[i];
	
		if (MD_IS_SET(flags, i)) {

			switch (ctype) {

			case CONV_CYCLIC:
		
				assert(odims[i] == idims1[i]);
				plan->dims1[i] = idims1[i];
				plan->dims2[i] = odims[i];
				shift[i] = 0;
				break;

			case CONV_TRUNCATED:

				assert(odims[i] == idims1[i]);
				plan->dims1[i] = idims1[i] + idims2[i] - 1;
				plan->dims2[i] = odims[i] + idims2[i] - 1;
				shift[i] = 0;
				break;

			case CONV_VALID:

				plan->dims1[i] = odims[i] + idims2[i] - 1;
				plan->dims2[i] = odims[i] + idims2[i] - 1;
				assert(idims1[i] == plan->dims1[i]);
				shift[i] = (1 - idims2[i]);
				break;

			case CONV_EXTENDED:

				plan->dims1[i] = idims1[i] + idims2[i] - 1;
				plan->dims2[i] = idims1[i] + idims2[i] - 1;
				assert(odims[i] == plan->dims2[i]);
				shift[i] = 0;
			}

                	plan->kdims[i] = (1 == idims2[i]) ? 1 : plan->dims1[i];

			U *= (float)plan->dims1[i];

		} else {

			// O I K:
			// X X X
			// X X 1
			// X 1 X
			// X 1 1 (inefficient)
			// 1 X X

			assert((1 == idims1[i]) || (idims1[i] == odims[i]) || (idims1[i] == idims2[i]));
			assert((1 == idims2[i]) || (idims2[i] == odims[i]) || (idims2[i] == idims1[i]));
			assert((1 == odims[i]) || (idims2[i] == odims[i]) || (idims1[i] == odims[i]));

			plan->dims1[i] = idims1[i];
			plan->dims2[i] = odims[i];
			plan->kdims[i] = idims2[i];

			shift[i] = 0;
		}

		plan->dims[i] = MAX(plan->dims1[i], plan->dims2[i]);
	}

	plan->str1 = *TYPE_ALLOC(long[N]);
	plan->str2 = *TYPE_ALLOC(long[N]);
	plan->kstr = *TYPE_ALLOC(long[N]);

	md_calc_strides(N, plan->str1, plan->dims1, CFL_SIZE);
	md_calc_strides(N, plan->str2, plan->dims2, CFL_SIZE);
	md_calc_strides(N, plan->kstr, plan->kdims, CFL_SIZE);

	switch (cmode) {

	case CONV_SYMMETRIC:

		for (int i = 0; i < N; i++)
			shift[i] -= MD_IS_SET(flags, i) ? ((idims2[i] - 1) / 2) : 0;

		break;

	case CONV_CAUSAL:

		break;

	case CONV_ANTICAUSAL:

		for (int i = 0; i < N; i++)
			shift[i] -= MD_IS_SET(flags, i) ? (idims2[i] - 1) : 0;

		break;

	default:

		assert(0);
	}

	for (int i = 0; i < N; i++)
		shift[i] = (plan->kdims[i] + shift[i]) % plan->kdims[i];

	plan->kernel = md_alloc_sameplace(N, plan->kdims, CFL_SIZE, src2);

	md_resize(N, plan->kdims, plan->kernel, idims2, src2, CFL_SIZE);
	md_circ_shift(N, plan->kdims, shift, plan->kernel, plan->kernel, CFL_SIZE);
	ifft(N, plan->kdims, flags, plan->kernel, plan->kernel);
        md_zsmul(N, plan->kdims, plan->kernel, plan->kernel, 1. / U);

	plan->fft1 = fft_create(N, plan->dims1, plan->flags, NULL, NULL, false);
	plan->ifft1 = fft_create(N, plan->dims1, plan->flags, NULL, NULL, true);
	plan->fft2 = fft_create(N, plan->dims2, plan->flags, NULL, NULL, false);
	plan->ifft2 = fft_create(N, plan->dims2, plan->flags, NULL, NULL, true);

	return PTR_PASS(plan);
}




void conv_free(struct conv_plan* plan)
{
	fft_free(plan->fft1);
	fft_free(plan->ifft1);
	fft_free(plan->fft2);
	fft_free(plan->ifft2);

	md_free(plan->kernel);

	xfree(plan->dims);
	xfree(plan->dims1);
	xfree(plan->dims2);
	xfree(plan->kdims);
	xfree(plan->str1);
	xfree(plan->str2);
	xfree(plan->kstr);
	xfree(plan->idims);
	xfree(plan->odims);

	xfree(plan);
}



static void conv_cyclic(struct conv_plan* plan, complex float* dst, const complex float* src)
{
	// FIXME: optimize tmp away when possible
	complex float* tmp = md_alloc_sameplace(plan->N, plan->dims1, CFL_SIZE, plan->kernel);

	fft_exec(plan->ifft1, tmp, src);

	md_clear(plan->N, plan->dims2, dst, CFL_SIZE);
        md_zfmac2(plan->N, plan->dims, plan->str2, dst, plan->str1, tmp, plan->kstr, plan->kernel);

	fft_exec(plan->fft2, dst, dst);

	md_free(tmp);
}

static void conv_cyclicH(struct conv_plan* plan, complex float* dst, const complex float* src)
{
	complex float* tmp = md_alloc_sameplace(plan->N, plan->dims2, CFL_SIZE, plan->kernel);

	fft_exec(plan->ifft2, tmp, src);

	md_clear(plan->N, plan->dims1, dst, CFL_SIZE);
        md_zfmacc2(plan->N, plan->dims, plan->str1, dst, plan->str2, tmp, plan->kstr, plan->kernel);

	fft_exec(plan->fft1, dst, dst);

	md_free(tmp);
}


void conv_exec(struct conv_plan* plan, complex float* dst, const complex float* src)
{
	complex float* tmp = md_alloc_sameplace(plan->N, plan->dims, CFL_SIZE, plan->kernel);

	md_resize(plan->N, plan->dims1, tmp, plan->idims, src, CFL_SIZE);

	conv_cyclic(plan, tmp, tmp);

	md_resize(plan->N, plan->odims, dst, plan->dims2, tmp, CFL_SIZE);

	md_free(tmp);
}


void conv_adjoint(struct conv_plan* plan, complex float* dst, const complex float* src)
{
	complex float* tmp = md_alloc_sameplace(plan->N, plan->dims, CFL_SIZE, plan->kernel);

	md_resize(plan->N, plan->dims2, tmp, plan->odims, src, CFL_SIZE);

	conv_cyclicH(plan, tmp, tmp);

	md_resize(plan->N, plan->idims, dst, plan->dims1, tmp, CFL_SIZE);

	md_free(tmp);
}




void conv(int N, unsigned int flags, enum conv_type ctype, enum conv_mode cmode, const long odims[N], complex float* dst, 
		const long idims1[N], const complex float* src1, const long idims2[N], const complex float* src2)
{
	struct conv_plan* plan = conv_plan(N, flags, ctype, cmode, odims, idims1, idims2, src2);
	conv_exec(plan, dst, src1);
	conv_free(plan);
}




void convH(int N, unsigned int flags, enum conv_type ctype, enum conv_mode cmode, const long odims[N], complex float* dst, 
		const long idims1[N], const complex float* src1, const long idims2[N], const complex float* src2)
{
	struct conv_plan* plan = conv_plan(N, flags, ctype, cmode, idims1, odims, idims2, src2); // idims1 <-> odims
	conv_adjoint(plan, dst, src1);
	conv_free(plan);
}




