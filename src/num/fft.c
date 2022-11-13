/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2016-2022. Martin Uecker.
 * Copyright 2018. Massachusetts Institute of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2011-2022 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014 Frank Ong <frankong@berkeley.edu>
 * 2018 Siddharth Iyer <ssi@mit.edu>
 *
 * 
 * FFT. It uses FFTW or CUFFT internally.
 *
 *
 * Gauss, Carl F. 1805. "Nachlass: Theoria Interpolationis Methodo Nova
 * Tractata." Werke 3, pp. 265-327, Königliche Gesellschaft der
 * Wissenschaften, Göttingen, 1866
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
#include "num/gpukrnls.h"
#include "fft-cuda.h"
#define LAZY_CUDA
#endif


void fftscale2(int N, const long dimensions[N], unsigned long flags, const long ostrides[N], complex float* dst, const long istrides[N], const complex float* src)
{
	long fft_dims[N];
	md_select_dims(N, flags, fft_dims, dimensions);

	float scale = 1. / sqrtf((float)md_calc_size(N, fft_dims));

	md_zsmul2(N, dimensions, ostrides, dst, istrides, src, scale);
}

void fftscale(int N, const long dims[N], unsigned long flags, complex float* dst, const complex float* src)
{
	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	fftscale2(N, dims, flags, strs, dst, strs, src);
}


static double fftmod_phase(long length, int j)
{
	long center1 = length / 2;
	double shift = (double)center1 / (double)length;
	return ((double)j - (double)center1 / 2.) * shift;
}

static complex double fftmod_phase2(long n, int j, bool inv, double phase)
{
	phase += fftmod_phase(n, j);
	double rem = phase - floor(phase);
	double sgn = inv ? -1. : 1.;
#if 1
	if (rem == 0.)
		return 1.;

	if (rem == 0.5)
		return -1.;

	if (rem == 0.25)
		return 1.i * sgn;

	if (rem == 0.75)
		return -1.i * sgn;
#endif
	return cexp(M_PI * 2.i * sgn * rem);
}

static void zfftmod_3d_4(const long dims[3], complex float* dst, const complex float* src, bool inv, double phase)
{
	double rem = phase - floor(phase);
	double sgn = inv ? -1. : 1.;

	complex double scale_1 = cexp(M_PI * 2.i * sgn * rem);

	if ((1 != dims[0]) && (0 != dims[0] % 8))
		scale_1 *= -1;
	
	if ((1 != dims[1]) && (0 != dims[1] % 8))
		scale_1 *= -1;
	
	if ((1 != dims[2]) && (0 != dims[2] % 8))
		scale_1 *= -1;


	#pragma omp parallel for collapse(3)
	for (long z = 0; z < dims[2]; z++)
		for (long y = 0; y < dims[1]; y++)
			for (long x = 0; x < dims[0]; x++) {

				complex double scale = scale_1;

				if (1 == x % 2)
					scale = -scale;
				
				if (1 == y % 2)
					scale = -scale;

				if (1 == z % 2)
					scale = -scale;

				long idx = x + dims[0] * y + dims[0] * dims[1] * z;
				
				dst[idx] = scale * src[idx];		
			}
}

static void zfftmod_3d(const long dims[3], complex float* dst, const complex float* src, bool inv, double phase)
{
	if (   ((dims[0] == 1) || (dims[0] % 4 == 0))
	    && ((dims[1] == 1) || (dims[1] % 4 == 0))
	    && ((dims[2] == 1) || (dims[2] % 4 == 0)))
		{
			zfftmod_3d_4(dims, dst, src, inv, phase);
			return;
		}

	#pragma omp parallel for collapse(3)
	for (long z = 0; z < dims[2]; z++)
		for (long y = 0; y < dims[1]; y++)
			for (long x = 0; x < dims[0]; x++) {

				long pos[3] = { x, y, z };
				long idx = x + dims[0] * y + dims[0] * dims[1] * z;
				
				double phase0 = phase;
				for (int i = 2; i > 0; i--)
					phase0 += fftmod_phase(dims[i], pos[i]);
				
				complex double scale = fftmod_phase2(dims[0], pos[0], inv, phase0);

				dst[idx] = scale * src[idx];		
			}
}

static void fftmod2_r(int N, const long dims[N], unsigned long flags, const long ostrs[N], complex float* dst, const long istrs[N], const complex float* src, bool inv, double phase)
{
	flags &= md_nontriv_dims(N, dims);

	if (0 == flags) {

		md_zsmul2(N, dims, ostrs, dst, istrs, src, cexp(M_PI * 2.i * (inv ? -phase : phase)));
		return;
	}

	if (   ((3 == flags) && md_check_equal_dims(2, ostrs, istrs, 3) && md_check_equal_dims(2, ostrs, MD_STRIDES(2, dims, CFL_SIZE), 3))
	    || ((7 == flags) && md_check_equal_dims(3, ostrs, istrs, 7) && md_check_equal_dims(3, ostrs, MD_STRIDES(3, dims, CFL_SIZE), 7)) ){

		long tdims[3] = { dims[0], dims[1], (7 == flags) ? dims[2] : 1 };
		long* tptr = &(tdims[0]);
 
		NESTED(void, nary_zfftmod, (void* ptr[]))
		{
		#ifdef USE_CUDA
			if (cuda_ondevice(dst))
				cuda_zfftmod_3d(tptr, ptr[0], ptr[1], inv, phase);
			else
	 	#endif
				zfftmod_3d(tptr, ptr[0], ptr[1], inv, phase);
		};

		const long* strs[2] = { ostrs + (3 == flags ? 2 : 3), istrs + (3 == flags ? 2 : 3) };
		void* ptr[2] = { (void*)dst, (void*)src };

		md_nary(2, N - (3 == flags ? 2 : 3), dims + (3 == flags ? 2 : 3), strs, ptr, nary_zfftmod);

		return;
	}

	/* this will also currently be slow on the GPU because we do not
	 * support strides there on the lowest level */

	int i = N - 1;

	while (!MD_IS_SET(flags, i))
		i--;

#if 1
	// If there is only one dimensions left and it is the innermost
	// which is contiguous optimize using md_zfftmod2

	if ((0u == MD_CLEAR(flags, i)) && (1 == md_calc_size(i, dims))
		&& (CFL_SIZE == ostrs[i]) && (CFL_SIZE == istrs[i])) {

		md_zfftmod2(N - i, dims + i, ostrs + i, dst, istrs + i, src, inv, phase);
		return;
	}
#endif

	long tdims[N];
	md_select_dims(N, ~MD_BIT(i), tdims, dims);

	struct cuda_threads_s* gpu_stat = gpu_threads_create(dst);

#ifdef USE_CUDA
	// FIXME: New threads initialize the 0 GPU by default
	// As long as gpu_threads_enter is not implemented other active devices other than the 0th device will fail
	// As a workaround use CUDA_VISIBLE_DEVICES environment variable to select a GPU and hide other GPUS by the driver 
	assert((0 == cuda_get_device_internal_unchecked()) || !cuda_ondevice(dst));
#endif

	#pragma omp parallel for
	for (int j = 0; j < dims[i]; j++) {

		gpu_threads_enter(gpu_stat);

		fftmod2_r(N, tdims, MD_CLEAR(flags, i),
			ostrs, (void*)dst + j * ostrs[i], istrs, (void*)src + j * istrs[i],
			inv, phase + fftmod_phase(dims[i], j));
		
		gpu_threads_leave(gpu_stat);
	}
	
	gpu_threads_free(gpu_stat);
}


static unsigned long clear_singletons(int N, const long dims[N], unsigned long flags)
{
       return (0 == N) ? flags : clear_singletons(N - 1, dims, (1 == dims[N - 1]) ? MD_CLEAR(flags, N - 1) : flags);
}


void fftmod2(int N, const long dims[N], unsigned long flags, const long ostrs[N], complex float* dst, const long istrs[N], const complex float* src)
{
	fftmod2_r(N, dims, clear_singletons(N, dims, flags), ostrs, dst, istrs, src, false, 0.);
}


/*
 *	The correct usage is fftmod before and after fft and
 *      ifftmod before and after ifft (this is different from
 *	how fftshift/ifftshift has to be used)
 */
void ifftmod2(int N, const long dims[N], unsigned long flags, const long ostrs[N], complex float* dst, const long istrs[N], const complex float* src)
{
	fftmod2_r(N, dims, clear_singletons(N, dims, flags), ostrs, dst, istrs, src, true, 0.);
}

void fftmod(int N, const long dimensions[N], unsigned long flags, complex float* dst, const complex float* src)
{
	long strs[N];
	md_calc_strides(N, strs, dimensions, CFL_SIZE);
	fftmod2(N, dimensions, flags, strs, dst, strs, src);
}

void ifftmod(int N, const long dimensions[N], unsigned long flags, complex float* dst, const complex float* src)
{
	long strs[N];
	md_calc_strides(N, strs, dimensions, CFL_SIZE);
	ifftmod2(N, dimensions, flags, strs, dst, strs, src);
}






void ifftshift2(int N, const long dims[N], unsigned long flags, const long ostrs[N], complex float* dst, const long istrs[N], const complex float* src)
{
	long pos[N];
	md_set_dims(N, pos, 0);

	for (int i = 0; i < N; i++)
		if (MD_IS_SET(flags, i))
			pos[i] = dims[i] - dims[i] / 2;

	md_circ_shift2(N, dims, pos, ostrs, dst, istrs, src, CFL_SIZE);
}

void ifftshift(int N, const long dimensions[N], unsigned long flags, complex float* dst, const complex float* src)
{
	long strs[N];
	md_calc_strides(N, strs, dimensions, CFL_SIZE);
	ifftshift2(N, dimensions, flags, strs, dst, strs, src);
}

void fftshift2(int N, const long dims[N], unsigned long flags, const long ostrs[N], complex float* dst, const long istrs[N], const complex float* src)
{
	long pos[N];
	md_set_dims(N, pos, 0);

	for (int i = 0; i < N; i++)
		if (MD_IS_SET(flags, i))
			pos[i] = dims[i] / 2;

	md_circ_shift2(N, dims, pos, ostrs, dst, istrs, src, CFL_SIZE);
}

void fftshift(int N, const long dimensions[N], unsigned long flags, complex float* dst, const complex float* src)
{
	long strs[N];
	md_calc_strides(N, strs, dimensions, CFL_SIZE);
	fftshift2(N, dimensions, flags, strs, dst, strs, src);
}

struct fft_plan_s {

	INTERFACE(operator_data_t);

	fftwf_plan fftw;

	int D;
	unsigned long flags;
	bool backwards;
	const long* dims;
	const long* istrs;
	const long* ostrs;

#ifdef  USE_CUDA
	struct fft_cuda_plan_s* cuplan;
#endif
};

static DEF_TYPEID(fft_plan_s);

bool use_fftw_wisdom = false;

static char* fftw_wisdom_name(int N, bool backwards, unsigned long flags, const long dims[N])
{
	if (!use_fftw_wisdom)
		return NULL;

	char* tbpath = getenv("TOOLBOX_PATH");

	if (NULL == tbpath) {

		debug_printf(DP_WARN, "FFTW wisdom only works with TOOLBOX_PATH set!\n");
		return NULL;
	}

	// Space for path and null terminator.
	int space = snprintf(NULL, 0, "%s/save/fftw/N_%d_BACKWARD_%d_FLAGS_%lu_DIMS", tbpath, N, backwards, flags);

	// Space for dimensions.
	for (int idx = 0; idx < N; idx ++)
		space += snprintf(NULL, 0, "_%lu", dims[idx]);

	// Space for extension.
	space += snprintf(NULL, 0, ".fftw");
	// Space for null terminator.
	space += 1;

	int len = space;
	char* loc = calloc(space, sizeof(char));

	if (NULL == loc)
		error("memory out");

	int ret = snprintf(loc, len, "%s/save/fftw/N_%d_BACKWARD_%d_FLAGS_%lu_DIMS", tbpath, N, backwards, flags);

	assert(ret < len);
	len -= ret;

	for (int idx = 0; idx < N; idx++) {

		char tmp[64];
		ret = sprintf(tmp, "_%lu", dims[idx]);
		assert(ret < 64);
		len -= ret;
		strcat(loc, tmp);
	}

	strcat(loc, ".fftw");
	len -= 5;
	assert(1 == len);
	assert('\0' == loc[space - 1]);

	return loc;
}


static fftwf_plan fft_fftwf_plan(int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src, bool backwards, bool measure)
{
	fftwf_plan fftwf;

	int N = D;
	fftwf_iodim64 dims[N];
	fftwf_iodim64 hmdims[N];
	int k = 0;
	int l = 0;

	char* wisdom = fftw_wisdom_name(D, backwards, flags, dimensions);

	if (NULL != wisdom)
		fftwf_import_wisdom_from_filename(wisdom);


	//FFTW seems to be fine with this
	//assert(0 != flags); 

	for (int i = 0; i < N; i++) {

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

	#pragma omp critical
	fftwf = fftwf_plan_guru64_dft(k, dims, l, hmdims, (complex float*)src, dst,
				backwards ? 1 : (-1), measure ? FFTW_MEASURE : FFTW_ESTIMATE);


	if (NULL != wisdom) {

		fftwf_export_wisdom_to_filename(wisdom);
		xfree(wisdom);
	}

	return fftwf;
}


static void fft_apply(const operator_data_t* _plan, unsigned int N, void* args[N])
{
	complex float* dst = args[0];
	const complex float* src = args[1];
	const auto plan = CAST_DOWN(fft_plan_s, _plan);

	assert(2 == N);

	if (0u == plan->flags) {

		md_copy2(plan->D, plan->dims, plan->ostrs, dst, plan->istrs, src, CFL_SIZE);
		return;
	}

#ifdef  USE_CUDA
	if (cuda_ondevice(src)) {
#ifdef	LAZY_CUDA
		if (NULL == plan->cuplan)
			((struct fft_plan_s*)plan)->cuplan = fft_cuda_plan(plan->D, plan->dims, plan->flags, plan->ostrs, plan->istrs, plan->backwards);
#endif
		if (NULL == plan->cuplan)
			error("Failed to plan a GPU FFT (too large?)\n");

		fft_cuda_exec(plan->cuplan, dst, src);

	} else 
#endif
	{
		assert(NULL != plan->fftw);
		fftwf_execute_dft(plan->fftw, (complex float*)src, dst);
	}
}


static void fft_free_plan(const operator_data_t* _data)
{
	const auto plan = CAST_DOWN(fft_plan_s, _data);

	if (NULL != plan->fftw)
		fftwf_destroy_plan(plan->fftw);

#ifdef	USE_CUDA
	if (NULL != plan->cuplan)
		fft_cuda_free_plan(plan->cuplan);
#endif
	xfree(plan->dims);
	xfree(plan->istrs);
	xfree(plan->ostrs);

	xfree(plan);
}


const struct operator_s* fft_measure_create(int D, const long dimensions[D], unsigned long flags, bool inplace, bool backwards)
{
	flags &= md_nontriv_dims(D, dimensions);

	PTR_ALLOC(struct fft_plan_s, plan);
	SET_TYPEID(fft_plan_s, plan);

	complex float* src = md_alloc(D, dimensions, CFL_SIZE);
	complex float* dst = inplace ? src : md_alloc(D, dimensions, CFL_SIZE);

	long strides[D];
	md_calc_strides(D, strides, dimensions, CFL_SIZE);

	plan->fftw = NULL;

	if (0u != flags)
		plan->fftw = fft_fftwf_plan(D, dimensions, flags, strides, dst, strides, src, backwards, true);

	md_free(src);

	if (!inplace)
		md_free(dst);

#ifdef  USE_CUDA
	plan->cuplan = NULL;
#ifndef LAZY_CUDA
	if (cuda_ondevice(src) && (0u != flags)
		plan->cuplan = fft_cuda_plan(D, dimensions, flags, strides, strides, backwards);
#endif
#endif
	plan->D = D;
	plan->flags = flags;
	plan->backwards = backwards;

	PTR_ALLOC(long[D], dims);
	md_copy_dims(D, *dims, dimensions);
	plan->dims = *PTR_PASS(dims);

	PTR_ALLOC(long[D], istrs);
	md_copy_strides(D, *istrs, strides);
	plan->istrs = *PTR_PASS(istrs);

	PTR_ALLOC(long[D], ostrs);
	md_copy_strides(D, *ostrs, strides);
	plan->ostrs = *PTR_PASS(ostrs);

	return operator_create2(D, dimensions, strides, D, dimensions, strides, CAST_UP(PTR_PASS(plan)), fft_apply, fft_free_plan);
}


const struct operator_s* fft_create2(int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src, bool backwards)
{
	flags &= md_nontriv_dims(D, dimensions);

	PTR_ALLOC(struct fft_plan_s, plan);
	SET_TYPEID(fft_plan_s, plan);

	plan->fftw = NULL;

	if (0u != flags)
		plan->fftw = fft_fftwf_plan(D, dimensions, flags, ostrides, dst, istrides, src, backwards, false);

#ifdef  USE_CUDA
	plan->cuplan = NULL;
#ifndef LAZY_CUDA
	if (cuda_ondevice(src) && (0u != flags)
		plan->cuplan = fft_cuda_plan(D, dimensions, flags, ostrides, istrides, backwards);
#endif
#endif
	plan->D = D;
	plan->flags = flags;
	plan->backwards = backwards;

	PTR_ALLOC(long[D], dims);
	md_copy_dims(D, *dims, dimensions);
	plan->dims = *PTR_PASS(dims);

	PTR_ALLOC(long[D], istrs);
	md_copy_strides(D, *istrs, istrides);
	plan->istrs = *PTR_PASS(istrs);

	PTR_ALLOC(long[D], ostrs);
	md_copy_strides(D, *ostrs, ostrides);
	plan->ostrs = *PTR_PASS(ostrs);

	return operator_create2(D, dimensions, ostrides, D, dimensions, istrides, CAST_UP(PTR_PASS(plan)), fft_apply, fft_free_plan);
}

const struct operator_s* fft_create(int D, const long dimensions[D], unsigned long flags, complex float* dst, const complex float* src, bool backwards)
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


void fft2(int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	const struct operator_s* plan = fft_create2(D, dimensions, flags, ostrides, dst, istrides, src, false);
	fft_exec(plan, dst, src);
	fft_free(plan);
}

void ifft2(int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	const struct operator_s* plan = fft_create2(D, dimensions, flags, ostrides, dst, istrides, src, true);
	fft_exec(plan, dst, src);
	fft_free(plan);
}

void fft(int D, const long dimensions[D], unsigned long flags, complex float* dst, const complex float* src)
{
	const struct operator_s* plan = fft_create(D, dimensions, flags, dst, src, false);
	fft_exec(plan, dst, src);
	fft_free(plan);
}

void ifft(int D, const long dimensions[D], unsigned long flags, complex float* dst, const complex float* src)
{
	const struct operator_s* plan = fft_create(D, dimensions, flags, dst, src, true);
	fft_exec(plan, dst, src);
	fft_free(plan);
}

void fftc(int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src)
{
	fftmod(D, dimensions, flags, dst, src);
	fft(D, dimensions, flags, dst, dst);
	fftmod(D, dimensions, flags, dst, dst);
}

void ifftc(int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src)
{
	ifftmod(D, dimensions, flags, dst, src);
	ifft(D, dimensions, flags, dst, dst);
	ifftmod(D, dimensions, flags, dst, dst);
}

void fftc2(int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	fftmod2(D, dimensions, flags, ostrides, dst, istrides, src);
	fft2(D, dimensions, flags, ostrides, dst, ostrides, dst);
	fftmod2(D, dimensions, flags, ostrides, dst, ostrides, dst);
}

void ifftc2(int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	ifftmod2(D, dimensions, flags, ostrides, dst, istrides, src);
	ifft2(D, dimensions, flags, ostrides, dst, ostrides, dst);
	ifftmod2(D, dimensions, flags, ostrides, dst, ostrides, dst);
}

void fftu(int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src)
{
	fft(D, dimensions, flags, dst, src);
	fftscale(D, dimensions, flags, dst, dst);
}

void ifftu(int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src)
{
	ifft(D, dimensions, flags, dst, src);
	fftscale(D, dimensions, flags, dst, dst);
}

void fftu2(int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	fft2(D, dimensions, flags, ostrides, dst, istrides, src);
	fftscale2(D, dimensions, flags, ostrides, dst, ostrides, dst);
}

void ifftu2(int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	ifft2(D, dimensions, flags, ostrides, dst, istrides, src);
	fftscale2(D, dimensions, flags, ostrides, dst, ostrides, dst);
}

void fftuc(int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src)
{
	fftc(D, dimensions, flags, dst, src);
	fftscale(D, dimensions, flags, dst, dst);
}

void ifftuc(int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src)
{
	ifftc(D, dimensions, flags, dst, src);
	fftscale(D, dimensions, flags, dst, dst);
}

void fftuc2(int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	fftc2(D, dimensions, flags, ostrides, dst, istrides, src);
	fftscale2(D, dimensions, flags, ostrides, dst, ostrides, dst);
}

void ifftuc2(int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	ifftc2(D, dimensions, flags, ostrides, dst, istrides, src);
	fftscale2(D, dimensions, flags, ostrides, dst, ostrides, dst);
}


bool fft_threads_init = false;

void fft_set_num_threads(int n)
{
#ifdef FFTWTHREADS
	#pragma omp critical
	if (!fft_threads_init) {

		fft_threads_init = true;
		fftwf_init_threads();
	}

	#pragma omp critical
        fftwf_plan_with_nthreads(n);
#else
	UNUSED(n);
#endif
}



