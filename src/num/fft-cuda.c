/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2013 Martin Uecker <uecker@eecs.berkeley.edu>
 *
 *
 * Internal interfact to the CUFFT library used in fft.c.
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
	struct fft_cuda_plan_s* plan = xmalloc(sizeof(struct fft_cuda_plan_s));
	unsigned int N = D;

#if 0

	long squeeze_dimensions[D];
	unsigned int j = 0;
	int rank = 0;
	int idist = 1;
	int batch = 1;
	for (unsigned int i = 0; i < N; i++) {
		assert( istrides[i] == ostrides[i] );

		if (1 == dimensions[i])
		{
			if (flags & (1 << i))
				flags--;
		} else
		{
			// get squeezed dimension
			squeeze_dimensions[j] = dimension[i];
			j++;
			
			// get fft rank, batch
			if (flags & (1 << i))
			{
				rank++;
			} else
			{
				batch *= dimension[i];
			}
		}
	}
	unsigned int squeeze_D = j;

	assert( (flags==1) || (flags==3) || (flags==7));
	assert( rank <=3 );


	int n[rank];
	for( unsigned int i = 0; i < rank; i++)
		n[i] = squeeze_dimensions[rank-1-i];

	int* inembed = n+1;
	

#else
	struct iovec dims[N];
	struct iovec hmdims[N];

	plan->backwards = backwards;
	plan->batch = 1;
	plan->odist = 0;
	plan->idist = 0;

	
	// the cufft interface is strange, but we do our best...
	unsigned int k = 0;
	unsigned int l = 0;

	for (unsigned int i = 0; i < N; i++) {

		if (1 == dimensions[i])
			continue;

		if (flags & (1 << i)) {

			dims[k].n = dimensions[i];
			dims[k].is = istrides[i] / sizeof(complex float); 
			dims[k].os = ostrides[i] / sizeof(complex float);
			k++;

		} else  {

			hmdims[l].n = dimensions[i];
			hmdims[l].is = istrides[i] / sizeof(complex float);
			hmdims[l].os = ostrides[i] / sizeof(complex float);
			l++;
		}
	}

	assert(k <= 3);
	int cudims[k];
//	int cuiemb[k];
//	int cuoemb[k];

	long batchdims[l];
	long batchistr[l];
	long batchostr[l];

	int lis = 1;
	int los = 1;

	for (unsigned int i = 0; i < k; i++) {

		assert(0 == dims[i].is % lis);
		assert(0 == dims[i].os % los);

		cudims[i] = dims[i].n;

		//cuiemb[i] = dims[i].is;// / lis;
		//cuoemb[i] = dims[i].os;// / los;
	
		lis = dims[i].is;
		los = dims[i].os;

	}

	for (unsigned int i = 0; i < l; i++) {

		batchdims[i] = hmdims[i].n;

		batchistr[i] = hmdims[i].is;
		batchostr[i] = hmdims[i].os;
	}


	unsigned int batchsize = md_calc_size(l, batchdims);

	assert(l == md_calc_blockdim(l, batchdims, batchistr, hmdims[0].is));
	assert(l == md_calc_blockdim(l, batchdims, batchostr, hmdims[0].os));

//	printf("CUFFT %d %d %d x %d\n", k, cudims[0], cudims[1], batchsize);
#if 1
	assert(2 == k);

	plan->batch = batchsize;
	plan->idist = hmdims[0].is;
	plan->odist = hmdims[0].os;

	#pragma omp critical
	if (CUFFT_SUCCESS != cufftPlan2d(&plan->cufft, cudims[1], cudims[0], CUFFT_C2C))
		abort();

#else
//PI cufftPlanMany(cufftHandle *plan,
//                                   int rank,
//                                   int *n,
//                                   int *inembed, int istride, int idist,
//                                   int *onembed, int ostride, int odist,
//                                   cufftType type,
//                                   int batch);

	#pragma omp critical
	if (CUFFT_SUCCESS != cufftPlanMany(&plan->cufft, k, cudims, cuiemb, cuiemb[0], hmdims[0].is, cuoemb, cuoemb[0], hmdims[0].os, CUFFT_C2C, batchsize))
		abort();
#endif
#endif

	return plan;
}



void fft_cuda_free_plan(struct fft_cuda_plan_s* cuplan)
{
	cufftDestroy(cuplan->cufft);
	free(cuplan);
}


void fft_cuda_exec(struct fft_cuda_plan_s* cuplan, complex float* dst, const complex float* src)
{
	assert(cuda_ondevice(src));
	assert(cuda_ondevice(dst));
	assert(NULL != cuplan);

	int err;

	for (int i = 0; i < cuplan->batch; i++)
		if (CUFFT_SUCCESS != (err = cufftExecC2C(cuplan->cufft, (cufftComplex*)src + i * cuplan->idist, (cufftComplex*)dst + i * cuplan->odist, (!cuplan->backwards) ? CUFFT_FORWARD : CUFFT_INVERSE))) {
			fprintf(stderr, "CUFFT: %d\n", err);
			abort();
		}
}


#endif
