/* Copyright 2014. The Regents of the University of California.
 * Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012, 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
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

	struct fft_plan_s* fft_plan;

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

        for (int i = 0; i < N; i++) {

		plan->idims[i] = idims1[i];
		plan->odims[i] = odims[i];
	
		if (MD_IS_SET(flags, i)) {

			assert(idims2[i] <= idims1[i]);

			if (cmode == CONV_SYMMETRIC) {

				assert((0 == idims1[i] % 2) && (1 == idims2[i] % 2));

				if (1 == (idims1[i] / 2) % 2)
					U *= -1.i;
			}

			switch (ctype) {
			case CONV_CYCLIC:
		
				assert(odims[i] == idims1[i]);
				plan->dims1[i] = idims1[i];
				plan->dims2[i] = odims[i];
				break;

			case CONV_TRUNCATED:

				assert(odims[i] == idims1[i]);
				plan->dims1[i] = idims1[i] + idims2[i] - 1;
				plan->dims2[i] = odims[i] + idims2[i] - 1;
				break;

			case CONV_VALID:

				plan->dims1[i] = odims[i] + idims2[i] - 1;
				plan->dims2[i] = odims[i] + idims2[i] - 1;
				assert(idims1[i] == plan->dims1[i]);
				break;

			case CONV_EXTENDED:

				plan->dims1[i] = idims1[i] + idims2[i] - 1;
				plan->dims2[i] = idims1[i] + idims2[i] - 1;
				assert(odims[i] == plan->dims2[i]);
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
		}

		plan->dims[i] = MAX(plan->dims1[i], plan->dims2[i]);
	}

	plan->str1 = *TYPE_ALLOC(long[N]);
	plan->str2 = *TYPE_ALLOC(long[N]);
	plan->kstr = *TYPE_ALLOC(long[N]);

	md_calc_strides(N, plan->str1, plan->dims1, CFL_SIZE);
	md_calc_strides(N, plan->str2, plan->dims2, CFL_SIZE);
	md_calc_strides(N, plan->kstr, plan->kdims, CFL_SIZE);

        plan->kernel = md_alloc_sameplace(N, plan->kdims, CFL_SIZE, src2);

	switch (cmode) {

	case CONV_SYMMETRIC:

	        md_resize_center(N, plan->kdims, plan->kernel, idims2, src2, CFL_SIZE);
       		ifft(N, plan->kdims, flags, plan->kernel, plan->kernel);
	        fftmod(N, plan->kdims, flags, plan->kernel, plan->kernel);
		break;

	case CONV_CAUSAL:

	        md_resize(N, plan->kdims, plan->kernel, idims2, src2, CFL_SIZE);
       		ifft(N, plan->kdims, flags, plan->kernel, plan->kernel);
		break;

	case CONV_ANTICAUSAL:

	        md_resize(N, plan->kdims, plan->kernel, idims2, src2, CFL_SIZE);
       		fft(N, plan->kdims, flags, plan->kernel, plan->kernel);
		break;

	default:
		assert(0);
	}

	
        md_zsmul(N, plan->kdims, plan->kernel, plan->kernel, 1. / U);

//	plan->fftplan = fft_plan(N, plan->dims, plan->flags);

	return PTR_PASS(plan);
}




void conv_free(struct conv_plan* plan)
{
	md_free(plan->kernel);

	// fft_free_plan

	free(plan->dims);
	free(plan->dims1);
	free(plan->dims2);
	free(plan->kdims);
	free(plan->str1);
	free(plan->str2);
	free(plan->kstr);
	free(plan->idims);
	free(plan->odims);

	free(plan);
}



static void conv_cyclic(struct conv_plan* plan, complex float* dst, const complex float* src1)
{
	// FIXME: optimize tmp away when possible
	complex float* tmp = md_alloc_sameplace(plan->N, plan->dims1, CFL_SIZE, plan->kernel);
        ifft(plan->N, plan->dims1, plan->flags, tmp, src1);
	md_clear(plan->N, plan->dims2, dst, CFL_SIZE);
        md_zfmac2(plan->N, plan->dims, plan->str2, dst, plan->str1, tmp, plan->kstr, plan->kernel);
  //      md_zmul2(plan->N, plan->dims, plan->str2, dst, plan->str1, tmp, plan->kstr, plan->kernel);
        fft(plan->N, plan->dims2, plan->flags, dst, dst);
	md_free(tmp);
}

static void conv_cyclicH(struct conv_plan* plan, complex float* dst, const complex float* src1)
{
	complex float* tmp = md_alloc_sameplace(plan->N, plan->dims1, CFL_SIZE, plan->kernel);
        ifft(plan->N, plan->dims2, plan->flags, tmp, src1);
	md_clear(plan->N, plan->dims1, dst, CFL_SIZE);
        md_zfmacc2(plan->N, plan->dims, plan->str1, dst, plan->str2, tmp, plan->kstr, plan->kernel);
        //md_zmulc2(plan->N, plan->dims1, plan->str1, dst, plan->str2, tmp, plan->kstr, plan->kernel);
        fft(plan->N, plan->dims1, plan->flags, dst, dst);
	md_free(tmp);
}



void conv_exec(struct conv_plan* plan, complex float* dst, const complex float* src1)
{
	bool crop = (CONV_SYMMETRIC == plan->cmode);
	bool pre = (CONV_TRUNCATED == plan->ctype) || (CONV_EXTENDED == plan->ctype);
	bool post = (CONV_TRUNCATED == plan->ctype) || (CONV_VALID == plan->ctype);

	complex float* tmp = NULL;

	if (pre || post) {

		tmp = md_alloc_sameplace(plan->N, plan->dims1, CFL_SIZE, plan->kernel);
	}

	if (pre)
		(crop ? md_resize_center : md_resize)(plan->N, plan->dims1, tmp, plan->idims, src1, CFL_SIZE);

	conv_cyclic(plan, post ? tmp : dst, pre ? tmp : src1);

	if (post)
		(crop ? md_resize_center : md_resize)(plan->N, plan->odims, dst, plan->dims2, tmp, CFL_SIZE);

	if (pre || post)
		md_free(tmp);
}



void conv_adjoint(struct conv_plan* plan, complex float* dst, const complex float* src1)
{
	bool crop = (CONV_SYMMETRIC == plan->cmode);
	bool post = (CONV_TRUNCATED == plan->ctype) || (CONV_EXTENDED == plan->ctype);
	bool pre = (CONV_TRUNCATED == plan->ctype) || (CONV_VALID == plan->ctype);

	complex float* tmp = NULL;

	if (pre || post) {

		tmp = md_alloc_sameplace(plan->N, plan->dims1, CFL_SIZE, plan->kernel);
	}

	if (pre)
		(crop ? md_resize_center : md_resize)(plan->N, plan->dims2, tmp, plan->odims, src1, CFL_SIZE);

	conv_cyclicH(plan, post ? tmp : dst, pre ? tmp : src1);

	if (post)
		(crop ? md_resize_center : md_resize)(plan->N, plan->idims, dst, plan->dims1, tmp, CFL_SIZE);

	if (pre || post)
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




