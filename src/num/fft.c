/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2011, 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2014 Frank Ong <frankong@berkeley.edu>
 *
 * 
 * FFT. It uses FFTW or CUFFT internally.
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <math.h>

#include <fftw3.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "fft.h"
#undef fft_plan_s

#ifdef USE_CUDA
#include "num/gpuops.h"
#include "fft-cuda.h"
#endif


void fftscale2(unsigned int N, const long dimensions[N], unsigned long flags, const long ostrides[N], complex float* dst, const long istrides[N], const complex float* src)
{
	long fft_dims[N];
	md_select_dims(N, flags, fft_dims, dimensions);
	float scale = 1. / sqrtf((float)md_calc_size(N, fft_dims));
	md_zsmul2(N, dimensions, ostrides, dst, istrides, src, scale);
}

void fftscale(unsigned int N, const long dims[N], unsigned long flags, complex float* dst, const complex float* src)
{
	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);
	fftscale2(N, dims, flags, strs, dst, strs, src);
}


static void fftmod2_r(unsigned int N, const long dims[N], unsigned long flags, const long ostrs[N], complex float* dst, const long istrs[N], const complex float* src, bool evenodd)
{
	if (0 == flags) {

		md_zsmul2(N, dims, ostrs, dst, istrs, src, evenodd ? -1 : 1.);
		return;
	}

	/* This will create 2^X calls to md_zsmul depending on the number X
	 * of bits set in flags. One could optimize this to two calls for
	 * certain cases by combining dimensions. */

	/* this will also currently be slow on the GPU because we do not
	 * support strides there on the lowest level */

	unsigned int i = N - 1;
	while (!MD_IS_SET(flags, i))
		i--;

#if 1
	// If there is only one dimensions left and it is the innermost
	// which is contiguous optimize using md_zfftmod2

	if ((0u == MD_CLEAR(flags, i)) && (0 == i)
		&& (0 == dims[0] % 2) && (CFL_SIZE == ostrs[0]) && (CFL_SIZE == istrs[0])) {

		md_zfftmod2(N, dims, ostrs, dst, istrs, src, evenodd);
		return;
	}
#endif

	long tdims[N];
	md_copy_dims(N, tdims, dims);

	long tostrs[N];
	md_copy_dims(N, tostrs, ostrs);
	tostrs[i] *= 2;

	long tistrs[N];
	md_copy_dims(N, tistrs, istrs);
	tistrs[i] *= 2;

//#pragma omp parallel sections
	{
//#pragma omp section
		{
			long tdims1[N];
			md_copy_dims(N, tdims1, tdims );
			tdims1[i] = (dims[i] + 1) / 2;
			fftmod2_r(N, tdims1, MD_CLEAR(flags, i), tostrs, (void*)dst + 0 * ostrs[i], tistrs, (void*)src + 0 * istrs[i], !evenodd);
		}

//#pragma omp section
		{
			long tdims2[N];
			md_copy_dims(N, tdims2, tdims );
			tdims2[i] = (dims[i] + 0) / 2;
			fftmod2_r(N, tdims2, MD_CLEAR(flags, i), tostrs, (void*)dst + 1 * ostrs[i], tistrs, (void*)src + 1 * istrs[i], evenodd);
		}
	}
}

void fftmod2(unsigned int N, const long dims[N], unsigned long flags, const long ostrs[N], complex float* dst, const long istrs[N], const complex float* src)
{
	fftmod2_r(N, dims, flags, ostrs, dst, istrs, src, true);
}


/*
	fftmodk. The current fftmod produces negative output if N/2 is odd.
	The correct usage should be fftmod, fft and then fftmodk, which compensates the
	additional negative factor when N/2 is odd.
*/
void fftmodk2(unsigned int N, const long dims[N], unsigned long flags, const long ostrs[N], complex float* dst, const long istrs[N], const complex float* src)
{
	long dimsum = 0;
	for(unsigned int i = 0; i < N; i++)
		if (MD_IS_SET(flags, i))
			dimsum += dims[i] / 2;

	bool evenodd = ((dimsum % 2) == 0);
		
	fftmod2_r(N, dims, flags, ostrs, dst, istrs, src, evenodd);
}

void fftmod(unsigned int N, const long dimensions[N], unsigned long flags, complex float* dst, const complex float* src)
{
	long strs[N];
	md_calc_strides(N, strs, dimensions, CFL_SIZE);
	fftmod2(N, dimensions, flags, strs, dst, strs, src);
}

void fftmodk(unsigned int N, const long dimensions[N], unsigned long flags, complex float* dst, const complex float* src)
{
	long strs[N];
	md_calc_strides(N, strs, dimensions, CFL_SIZE);
	fftmodk2(N, dimensions, flags, strs, dst, strs, src);
}




void fftshift2(unsigned int N, const long dims[N], unsigned long flags, const long ostrs[N], complex float* dst, const long istrs[N], const complex float* src)
{
	long pos[N];
	md_set_dims(N, pos, 0);
	for (unsigned int i = 0; i < N; i++)
		if (MD_IS_SET(flags, i))
			pos[i] = dims[i] / 2;

	md_circ_shift2(N, dims, pos, ostrs, dst, istrs, src, CFL_SIZE);
}

void fftshift(unsigned int N, const long dimensions[N], unsigned long flags, complex float* dst, const complex float* src)
{
	long strs[N];
	md_calc_strides(N, strs, dimensions, CFL_SIZE);
	fftshift2(N, dimensions, flags, strs, dst, strs, src);
}



struct fft_plan_s {

	fftwf_plan fftw;
	
#ifdef  USE_CUDA
	struct fft_cuda_plan_s* cuplan;
#endif
};



static fftwf_plan fft_fftwf_plan(unsigned int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src, bool backwards)
{
	unsigned int N = D;
	fftwf_iodim dims[N];
	fftwf_iodim hmdims[N];
	unsigned int k = 0;
	unsigned int l = 0;

	//FFTW seems to be fine with this
	//assert(0 != flags); 

	for (unsigned int i = 0; i < N; i++) {

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

	fftwf_plan fftwf;

	#pragma omp critical
	fftwf = fftwf_plan_guru_dft(k, dims, l, hmdims, (complex float*)src, dst, backwards ? 1 : (-1), FFTW_ESTIMATE);

	return fftwf;
}


static void fft_apply(const void* _plan, unsigned int N, void* args[N])
{
	complex float* dst = args[0];
	const complex float* src = args[1];
	const struct fft_plan_s* plan = _plan;

	assert(2 == N);

#ifdef  USE_CUDA
	if (cuda_ondevice(src)) {

		assert(NULL != plan->cuplan);
		fft_cuda_exec(plan->cuplan, dst, src);

	} else 
#endif
	fftwf_execute_dft(plan->fftw, (complex float*)src, dst);
}


static void fft_free_plan(const void* _data)
{
	const struct fft_plan_s* plan = _data;

	fftwf_destroy_plan(plan->fftw);
#ifdef	USE_CUDA
	if (NULL != plan->cuplan)
		fft_cuda_free_plan(plan->cuplan);
#endif
	free((void*)plan);
}

const struct operator_s* fft_create2(unsigned int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src, bool backwards)
{
	struct fft_plan_s* plan = xmalloc(sizeof(struct fft_plan_s));

	plan->fftw = fft_fftwf_plan(D, dimensions, flags, ostrides, dst, istrides, src, backwards);

#ifdef  USE_CUDA
	plan->cuplan = NULL;

	if (cuda_ondevice(src))
		plan->cuplan = fft_cuda_plan(D, dimensions, flags, ostrides, istrides, backwards);
#endif

	return operator_create2(D, dimensions, ostrides, D, dimensions, istrides, plan, fft_apply, fft_free_plan);
}

const struct operator_s* fft_create(unsigned int D, const long dimensions[D], unsigned long flags, complex float* dst, const complex float* src, bool backwards)
{
	long strides[D];
	md_calc_strides(D, strides, dimensions, CFL_SIZE);
	return fft_create2(D, dimensions, flags, strides, dst, strides, src, backwards);
}




void fft_exec(const struct operator_s* o, complex float* dst, const complex float* src)
{
	operator_apply_unchecked(o, dst, src);
}




void fft_free(const struct operator_s* o)
{
	operator_free(o);
}


void fft2(unsigned int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	const struct operator_s* plan = fft_create2(D, dimensions, flags, ostrides, dst, istrides, src, false);
	fft_exec(plan, dst, src);
	fft_free(plan);
}

void ifft2(unsigned int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	const struct operator_s* plan = fft_create2(D, dimensions, flags, ostrides, dst, istrides, src, true);
	fft_exec(plan, dst, src);
	fft_free(plan);
}

void fft(unsigned int D, const long dimensions[D], unsigned long flags, complex float* dst, const complex float* src)
{
	const struct operator_s* plan = fft_create(D, dimensions, flags, dst, src, false);
	fft_exec(plan, dst, src);
	fft_free(plan);
}

void ifft(unsigned int D, const long dimensions[D], unsigned long flags, complex float* dst, const complex float* src)
{
	const struct operator_s* plan = fft_create(D, dimensions, flags, dst, src, true);
	fft_exec(plan, dst, src);
	fft_free(plan);
}

void fftc(unsigned int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src)
{
	fftmod(D, dimensions, flags, dst, src);
	fft(D, dimensions, flags, dst, dst);
	fftmodk(D, dimensions, flags, dst, dst);
}

void ifftc(unsigned int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src)
{
	fftmodk(D, dimensions, flags, dst, src);
	ifft(D, dimensions, flags, dst, dst);
	fftmod(D, dimensions, flags, dst, dst);
}

void fftc2(unsigned int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	fftmod2(D, dimensions, flags, ostrides, dst, istrides, src);
	fft2(D, dimensions, flags, ostrides, dst, ostrides, dst);
	fftmodk2(D, dimensions, flags, ostrides, dst, ostrides, dst);
}

void ifftc2(unsigned int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	fftmodk2(D, dimensions, flags, ostrides, dst, istrides, src);
	ifft2(D, dimensions, flags, ostrides, dst, ostrides, dst);
	fftmod2(D, dimensions, flags, ostrides, dst, ostrides, dst);
}

void fftu(unsigned int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src)
{
	fft(D, dimensions, flags, dst, src);
	fftscale(D, dimensions, flags, dst, dst);
}

void ifftu(unsigned int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src)
{
	ifft(D, dimensions, flags, dst, src);
	fftscale(D, dimensions, flags, dst, dst);
}

void fftu2(unsigned int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	fft2(D, dimensions, flags, ostrides, dst, istrides, src);
	fftscale2(D, dimensions, flags, ostrides, dst, ostrides, dst);
}

void ifftu2(unsigned int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	ifft2(D, dimensions, flags, ostrides, dst, istrides, src);
	fftscale2(D, dimensions, flags, ostrides, dst, ostrides, dst);
}

void fftuc(unsigned int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src)
{
	fftc(D, dimensions, flags, dst, src);
	fftscale(D, dimensions, flags, dst, dst);
}

void ifftuc(unsigned int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src)
{
	ifftc(D, dimensions, flags, dst, src);
	fftscale(D, dimensions, flags, dst, dst);
}

void fftuc2(unsigned int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	fftc2(D, dimensions, flags, ostrides, dst, istrides, src);
	fftscale2(D, dimensions, flags, ostrides, dst, ostrides, dst);
}

void ifftuc2(unsigned int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	ifftc2(D, dimensions, flags, ostrides, dst, istrides, src);
	fftscale2(D, dimensions, flags, ostrides, dst, ostrides, dst);
}


bool fft_threads_init = false;

void fft_set_num_threads(unsigned int n)
{
	#pragma omp critical
	if (!fft_threads_init) {

		fft_threads_init = true;
		fftwf_init_threads();
	}

	#pragma omp critical
        fftwf_plan_with_nthreads(n);
}



