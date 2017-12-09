/* Copyright 2013, 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2013, 2015 Martin Uecker <uecker@eecs.berkeley.edu>
 *
 *
 * Internal interface to the CUFFT library used in fft.c.
 */

#include <stdbool.h>
#include <complex.h>
#include <assert.h>

#include "misc/misc.h"
#include "num/multind.h"

#include "fft-cuda.h"

#ifdef USE_CUDA
#include <cufft.h>
#include "num/gpuops.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


struct fft_cuda_plan_s {

	cufftHandle cufft;

	bool backwards;

	long batch;
	long idist;
	long odist;
};

struct iovec {

	long n; 
	long is; 
	long os; 
};





struct fft_cuda_plan_s* fft_cuda_plan(unsigned int D, const long dimensions[D], unsigned long flags, const long ostrides[D], const long istrides[D], bool backwards)
{
	PTR_ALLOC(struct fft_cuda_plan_s, plan);
	unsigned int N = D;

	plan->batch = 1;
	plan->odist = 0;
	plan->idist = 0;
	plan->backwards = backwards;

	struct iovec dims[N];
	struct iovec hmdims[N];

	assert(0 != flags);
	
	// the cufft interface is strange, but we do our best...
	unsigned int k = 0;
	unsigned int l = 0;

	for (unsigned int i = 0; i < N; i++) {

		if (1 == dimensions[i])
			continue;

		if (MD_IS_SET(flags, i)) {

			dims[k].n = dimensions[i];
			dims[k].is = istrides[i] / CFL_SIZE; 
			dims[k].os = ostrides[i] / CFL_SIZE;
			k++;

		} else  {

			hmdims[l].n = dimensions[i];
			hmdims[l].is = istrides[i] / CFL_SIZE;
			hmdims[l].os = ostrides[i] / CFL_SIZE;
			l++;
		}
	}

	assert(k > 0);

	int cudims[k];
	int cuiemb[k];
	int cuoemb[k];

	long batchdims[l];
	long batchistr[l];
	long batchostr[l];

	int lis = dims[0].is;
	int los = dims[0].os;

	if (k > 3)
		goto errout;

	for (unsigned int i = 0; i < k; i++) {

		assert(dims[i].is == lis);
		assert(dims[i].os == los);

		cudims[k - 1 - i] = dims[i].n;
		cuiemb[k - 1 - i] = dims[i].n;
		cuoemb[k - 1 - i] = dims[i].n;
	
		lis = dims[i].n * dims[i].is;
		los = dims[i].n * dims[i].os;
	}

	for (unsigned int i = 0; i < l; i++) {

		batchdims[i] = hmdims[i].n;

		batchistr[i] = hmdims[i].is;
		batchostr[i] = hmdims[i].os;
	}

	int istride = dims[0].is;
	int ostride = dims[0].os;
	int idist = lis;
	int odist = los;
	int cubs = 1;


	// check that batch dimensions can be collapsed to one

	unsigned int bi = md_calc_blockdim(l, batchdims, batchistr, hmdims[0].is);
	unsigned int bo = md_calc_blockdim(l, batchdims, batchostr, hmdims[0].os);

	if (bi != bo)
		goto errout;

	if (bi > 0) {

		idist = hmdims[0].is;
		odist = hmdims[0].os;
		cubs = md_calc_size(bi, batchdims);
	}

	if (l != bi) {

		// check that batch dimensions can be collapsed to one

		if (l - bi != md_calc_blockdim(l - bi, batchdims + bi, batchistr + bi, hmdims[bi].is))
			goto errout;

		if (l - bo != md_calc_blockdim(l - bo, batchdims + bo, batchostr + bo, hmdims[bo].os))
			goto errout;

		plan->idist = hmdims[bi].is;
		plan->odist = hmdims[bo].os;
		plan->batch = md_calc_size(l - bi, batchdims + bi);
	}

	assert(k <= 3);

	int err;

	#pragma omp critical
	err = cufftPlanMany(&plan->cufft, k,
				cudims, cuiemb, istride, idist,
				        cuoemb, ostride, odist, CUFFT_C2C, cubs);

	if (CUFFT_SUCCESS != err)
		goto errout;

	return PTR_PASS(plan);

errout:
	PTR_FREE(plan);
	return NULL;
}



void fft_cuda_free_plan(struct fft_cuda_plan_s* cuplan)
{
	cufftDestroy(cuplan->cufft);
	xfree(cuplan);
}


void fft_cuda_exec(struct fft_cuda_plan_s* cuplan, complex float* dst, const complex float* src)
{
	assert(cuda_ondevice(src));
	assert(cuda_ondevice(dst));
	assert(NULL != cuplan);

	int err;

	for (int i = 0; i < cuplan->batch; i++) {

		if (CUFFT_SUCCESS != (err = cufftExecC2C(cuplan->cufft,
							(cufftComplex*)src + i * cuplan->idist,
							(cufftComplex*)dst + i * cuplan->odist,
							(!cuplan->backwards) ? CUFFT_FORWARD : CUFFT_INVERSE)))
			error("CUFFT: %d\n", err);
	}
}


#endif
