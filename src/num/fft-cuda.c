/* Copyright 2013, 2015. The Regents of the University of California.
 * Copyright 2019. Uecker Lab, University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2019 Martin Uecker
 * Christian Holme
 *
 *
 * Internal interface to the CUFFT library used in fft.c.
 */

#include <stdbool.h>
#include <complex.h>
#include <assert.h>
#include <limits.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "num/multind.h"

#include "fft-cuda.h"

#ifdef USE_CUDA
#include <cufft.h>
#include "num/gpuops.h"

#ifndef CFL_SIZE
#define CFL_SIZE (long)sizeof(complex float)
#endif

struct fft_cuda_plan_s {

	cufftHandle cufft;
	bool cufft_initialized;
	size_t workspace_size;

	struct fft_cuda_plan_s* chain;

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

static char* cufft_error_string(enum cufftResult_t err);
static void cufft_error(const char* file, int line, enum cufftResult_t code)
{
	const char* err_str = cufft_error_string(code);
	error("cuFFT Error: %s in %s:%d \n", err_str, file, line);
}

#define CUFFT_ERROR(x)	({ CUDA_ASYNC_ERROR_NOTE("before cuFFT call"); enum cufftResult_t errval = (x); if (CUFFT_SUCCESS != errval) cufft_error(__FILE__, __LINE__, errval); CUDA_ASYNC_ERROR_NOTE("after cuFFT call");})


// detect if flags has blocks of 1's separated by 0's
static bool noncontiguous_flags(int D, unsigned long flags)
{
	bool o = false;
	bool z = false;

	for (int i = 0; i < D; i++) {

		bool curr_bit = MD_IS_SET(flags, i);

		if (curr_bit) // found a block of ones
			o = true;

		if (o && !curr_bit) // found the end of a block of ones
			z = true;

		if (o && z && curr_bit) // found a second block of ones
			return true;
	}

	return false;
}


static struct fft_cuda_plan_s* fft_cuda_plan0(int D, const long dimensions[D], unsigned long flags, const long ostrides[D], const long istrides[D], bool backwards)
{
	// TODO: This is not optimal, as it will often create separate fft's where they
	// are not needed. And since we compute blocks, we could also recurse
	// into both blocks...

	if (noncontiguous_flags(D, flags))
		return NULL;

	PTR_ALLOC(struct fft_cuda_plan_s, plan);
	int N = D;

	plan->batch = 1;
	plan->odist = 0;
	plan->idist = 0;
	plan->backwards = backwards;
	plan->chain = NULL;
	plan->cufft_initialized = false;
	plan->workspace_size = 0;

	struct iovec dims[N];
	struct iovec hmdims[N];

	assert(0 != flags);
	
	// the cufft interface is strange, but we do our best...
	int k = 0;
	int l = 0;

	for (int i = 0; i < N; i++) {

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
	int idist;
	int odist;
	int cubs = 1;
	int bi;
	int bo;

	int istride = dims[0].is;
	int ostride = dims[0].os;

	if (k > 3)
		goto errout;

	for (int i = 0; i < k; i++) {

		// assert(dims[i].is == lis);
		// assert(dims[i].os == los);

		cudims[k - 1 - i] = dims[i].n;
		cuiemb[k - 1 - i] = dims[i].n;
		cuoemb[k - 1 - i] = dims[i].n;
	
		lis = dims[i].n * dims[i].is;
		los = dims[i].n * dims[i].os;
	}

	for (int i = 0; i < l; i++) {

		batchdims[i] = hmdims[i].n;

		batchistr[i] = hmdims[i].is;
		batchostr[i] = hmdims[i].os;
	}

	idist = lis;
	odist = los;


	// check that batch dimensions can be collapsed to one

	bi = md_calc_blockdim(l, batchdims, batchistr, (size_t)hmdims[0].is);
	bo = md_calc_blockdim(l, batchdims, batchostr, (size_t)hmdims[0].os);

	if (bi != bo)
		goto errout;

	if (bi > 0) {

		idist = hmdims[0].is;
		odist = hmdims[0].os;
		cubs = md_calc_size(bi, batchdims);
	}

	if (l != bi) {

		// check that batch dimensions can be collapsed to one

		if (l - bi != md_calc_blockdim(l - bi, batchdims + bi, batchistr + bi, (size_t)hmdims[bi].is))
			goto errout;

		if (l - bo != md_calc_blockdim(l - bo, batchdims + bo, batchostr + bo, (size_t)hmdims[bo].os))
			goto errout;

		plan->idist = hmdims[bi].is;
		plan->odist = hmdims[bo].os;
		plan->batch = md_calc_size(l - bi, batchdims + bi);
	}

	assert(k <= 3);

	CUFFT_ERROR(cufftCreate(&plan->cufft));
	CUFFT_ERROR(cufftSetAutoAllocation(plan->cufft, 0));
	CUFFT_ERROR(cufftMakePlanMany(plan->cufft, k,
				cudims, cuiemb, istride, idist,
				cuoemb, ostride, odist, CUFFT_C2C, cubs, &(plan->workspace_size)));
	CUFFT_ERROR(cufftSetStream(plan->cufft, cuda_get_stream()));

	plan->cufft_initialized = true;
	

	return PTR_PASS(plan);

errout:
	PTR_FREE(plan);
	return NULL;
}


static unsigned long find_msb(unsigned long flags)
{
	for (int i = 1; i < CHAR_BIT * (int)sizeof(flags); i *= 2)
		flags |= flags >> i;

	return (flags + 1) / 2;
}


struct fft_cuda_plan_s* fft_cuda_plan(int D, const long dimensions[D], unsigned long flags, const long ostrides[D], const long istrides[D], bool backwards)
{
	assert(0u != flags);
	assert(0u == (flags & ~md_nontriv_dims(D, dimensions)));

	struct fft_cuda_plan_s* plan = fft_cuda_plan0(D, dimensions, flags, ostrides, istrides, backwards);

	if (NULL != plan)
		return plan;

	unsigned long msb = find_msb(flags);

	if (flags & msb) {

		struct fft_cuda_plan_s* plan = fft_cuda_plan0(D, dimensions, msb, ostrides, istrides, backwards);

		if (NULL == plan)
			return NULL;

		plan->chain = fft_cuda_plan(D, dimensions, flags & ~msb, ostrides, ostrides, backwards);

		if (NULL == plan->chain) {

			fft_cuda_free_plan(plan);
			return NULL;
		}

		return plan;
	}

	return NULL;
}


void fft_cuda_free_plan(struct fft_cuda_plan_s* cuplan)
{
	if (NULL != cuplan->chain)
		fft_cuda_free_plan(cuplan->chain);

	if (cuplan->cufft_initialized) {

		CUFFT_ERROR(cufftDestroy(cuplan->cufft));
		cuplan->cufft_initialized = false;
	}

	xfree(cuplan);
}

void fft_cuda_exec(struct fft_cuda_plan_s* cuplan, complex float* dst, const complex float* src)
{
	assert(cuda_ondevice(src));
	assert(cuda_ondevice(dst));
	assert(NULL != cuplan);

	
	assert(cuplan->cufft_initialized);
	size_t workspace_size = cuplan->workspace_size;
	cufftHandle cufft = cuplan->cufft;

	void* workspace = md_alloc_gpu(1, MAKE_ARRAY(1l), workspace_size);
	CUDA_ERROR_PTR(dst, src, workspace);
	CUFFT_ERROR(cufftSetWorkArea(cufft, workspace));
	for (int i = 0; i < cuplan->batch; i++)
		CUFFT_ERROR(cufftExecC2C(cufft,
					 (cufftComplex*)src + i * cuplan->idist,
					 (cufftComplex*)dst + i * cuplan->odist,
					 (!cuplan->backwards) ? CUFFT_FORWARD : CUFFT_INVERSE));
	
	md_free(workspace);

	if (NULL != cuplan->chain)
		fft_cuda_exec(cuplan->chain, dst, dst);
}


static char* cufft_error_string(enum cufftResult_t err)
{
	switch (err) {
		case CUFFT_SUCCESS: 			return "CUFFT_SUCCESS"; break;
		case CUFFT_INVALID_PLAN: 		return "CUFFT_INVALID_PLAN"; break;
		case CUFFT_ALLOC_FAILED: 		return "CUFFT_ALLOC_FAILED"; break;
		case CUFFT_INVALID_TYPE: 		return "CUFFT_INVALID_TYPE"; break;
		case CUFFT_INVALID_VALUE: 		return "CUFFT_INVALID_VALUE"; break;
		case CUFFT_INTERNAL_ERROR: 		return "CUFFT_INTERNAL_ERROR"; break;
		case CUFFT_EXEC_FAILED: 		return "CUFFT_EXEC_FAILED"; break;
		case CUFFT_SETUP_FAILED: 		return "CUFFT_SETUP_FAILED"; break;
		case CUFFT_INVALID_SIZE: 		return "CUFFT_INVALID_SIZE"; break;
		case CUFFT_UNALIGNED_DATA: 		return "CUFFT_UNALIGNED_DATA"; break;
		case CUFFT_INCOMPLETE_PARAMETER_LIST: 	return "CUFFT_INCOMPLETE_PARAMETER_LIST"; break;
		case CUFFT_INVALID_DEVICE: 		return "CUFFT_INVALID_DEVICE"; break;
		case CUFFT_PARSE_ERROR: 		return "CUFFT_PARSE_ERROR"; break;
		case CUFFT_NO_WORKSPACE: 		return "CUFFT_NO_WORKSPACE"; break;
		case CUFFT_NOT_IMPLEMENTED: 		return "CUFFT_NOT_IMPLEMENTED"; break;
		case CUFFT_LICENSE_ERROR: 		return "CUFFT_LICENSE_ERROR"; break;
		case CUFFT_NOT_SUPPORTED: 		return "CUFFT_NOT_SUPPORTED"; break;
		default: return "Not a valid error string!\n"; break;
	}
}

#endif // USE_CUDA

