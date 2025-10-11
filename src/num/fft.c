/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2016-2022. Martin Uecker.
 * Copyright 2018. Massachusetts Institute of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2011-2022 Martin Uecker
 * 2014 Frank Ong
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

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops.h"
#include "num/fft_plan.h"

#include "num/mpi_ops.h"
#include "num/vptr.h"

#include "misc/misc.h"

#include "fft.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#include "num/gpukrnls.h"
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

	if (is_vptr(dst) || is_vptr(src)) {

		unsigned long mpi_flags = vptr_block_loop_flags(N, dims, ostrs, dst, CFL_SIZE) | vptr_block_loop_flags(N, dims, istrs, src, CFL_SIZE);

		long ldims[N];
		long bdims[N];

		md_select_dims(N, ~mpi_flags, bdims, dims);
		md_select_dims(N, mpi_flags, ldims, dims);

		long* bdimsp = &bdims[0];
		const long * istrsp = &istrs[0];
		const long * ostrsp = &ostrs[0];

		NESTED(void, nary_mpi_fftmod, (void* ptr[]))
		{
			fftmod2_r(N, bdimsp, flags, ostrsp, ptr[0], istrsp, ptr[1], inv, phase);
		};

		md_nary(2, N, ldims, (const long*[2]){ ostrs, istrs }, (void*[2]){ dst, (void*)src }, nary_mpi_fftmod);

		return;
	}

	if (0 == flags) {

		md_zsmul2(N, dims, ostrs, dst, istrs, src, cexp(M_PI * 2.i * (inv ? -phase : phase)));
		return;
	}

	dst = vptr_resolve(dst);
	src = vptr_resolve(src);

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

#pragma omp parallel for
	for (int j = 0; j < dims[i]; j++) {

		fftmod2_r(N, tdims, MD_CLEAR(flags, i),
			ostrs, (void*)dst + j * ostrs[i], istrs, (void*)src + j * istrs[i],
			inv, phase + fftmod_phase(dims[i], j));

	}
}


static unsigned long clear_singletons(int N, const long dims[N], unsigned long flags)
{
       return (0 == N) ? flags : clear_singletons(N - 1, dims, (1 == dims[N - 1]) ? MD_CLEAR(flags, N - 1) : flags);
}


void fftmod2(int N, const long dims[N], unsigned long flags, const long ostrs[N], complex float* dst, const long istrs[N], const complex float* src)
{
	long bdims[N];
	md_select_dims(N, ~flags, bdims, dims);

	if (4 < md_calc_size(N, bdims)) {

		long fdims[N];
		md_select_dims(N, flags, fdims, dims);
		complex float* tmp = md_alloc_sameplace(N, fdims, CFL_SIZE, dst);
		md_zfill(N, fdims, tmp, 1.);
		fftmod(N, fdims, flags, tmp, tmp);
		md_zmul2(N, dims, ostrs, dst, istrs, src, MD_STRIDES(N, fdims, CFL_SIZE), tmp);
		md_free(tmp);
		return;
	}

	fftmod2_r(N, dims, clear_singletons(N, dims, flags), ostrs, dst, istrs, src, false, 0.);
}


/*
 *	The correct usage is fftmod before and after fft and
 *      ifftmod before and after ifft (this is different from
 *	how fftshift/ifftshift has to be used)
 */
void ifftmod2(int N, const long dims[N], unsigned long flags, const long ostrs[N], complex float* dst, const long istrs[N], const complex float* src)
{
	long bdims[N];
	md_select_dims(N, ~flags, bdims, dims);

	if (4 < md_calc_size(N, bdims)) {

		long fdims[N];
		md_select_dims(N, flags, fdims, dims);
		complex float* tmp = md_alloc_sameplace(N, fdims, CFL_SIZE, dst);
		md_zfill(N, fdims, tmp, 1.);
		ifftmod(N, fdims, flags, tmp, tmp);
		md_zmul2(N, dims, ostrs, dst, istrs, src, MD_STRIDES(N, fdims, CFL_SIZE), tmp);
		md_free(tmp);
		return;
	}

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

void fft2(int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	if (is_vptr(dst) || is_vptr(src)) {

		unsigned long mpi_flags =  vptr_block_loop_flags(D, dimensions, ostrides, dst, CFL_SIZE)
					 | vptr_block_loop_flags(D, dimensions, istrides, src, CFL_SIZE);

		assert(0 == (mpi_flags & flags));

		long ldims[D];
		long fdims[D];
		md_select_dims(D, ~flags, ldims, dimensions);
		md_select_dims(D,  flags, fdims, dimensions);

		const long* fdims_p = fdims;
		const long* ostrs_p = ostrides;
		const long* istrs_p = istrides;

		NESTED(void, nary_mpi_fft, (void* ptr[]))
		{
			fft2(D, fdims_p, flags, ostrs_p, ptr[0], istrs_p, ptr[1]);
		};

		md_nary(2, D, ldims, (const long*[2]){ ostrides, istrides }, (void*[2]){ dst, (void*)src }, nary_mpi_fft);

		return;
	}

	const struct operator_s* plan = fft_create2(D, dimensions, flags, ostrides, dst, istrides, src, false);
	operator_apply_unchecked(plan, dst, src);
	operator_free(plan);
}

void ifft2(int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	if (is_vptr(dst) || is_vptr(src)) {

		unsigned long mpi_flags =  vptr_block_loop_flags(D, dimensions, ostrides, dst, CFL_SIZE)
					 | vptr_block_loop_flags(D, dimensions, istrides, src, CFL_SIZE);

		assert(0 == (mpi_flags & flags));

		long ldims[D];
		long fdims[D];
		md_select_dims(D, ~flags, ldims, dimensions);
		md_select_dims(D,  flags, fdims, dimensions);

		const long* fdims_p = fdims;
		const long* ostrs_p = ostrides;
		const long* istrs_p = istrides;

		NESTED(void, nary_mpi_fft, (void* ptr[]))
		{
			ifft2(D, fdims_p, flags, ostrs_p, ptr[0], istrs_p, ptr[1]);
		};

		md_nary(2, D, ldims, (const long*[2]){ ostrides, istrides }, (void*[2]){ dst, (void*)src }, nary_mpi_fft);

		return;
	}

	const struct operator_s* plan = fft_create2(D, dimensions, flags, ostrides, dst, istrides, src, true);
	operator_apply_unchecked(plan, dst, src);
	operator_free(plan);
}

void fft(int D, const long dimensions[D], unsigned long flags, complex float* dst, const complex float* src)
{
	fft2(D, dimensions, flags, MD_STRIDES(D, dimensions, CFL_SIZE), dst, MD_STRIDES(D, dimensions, CFL_SIZE), src);
}

void ifft(int D, const long dimensions[D], unsigned long flags, complex float* dst, const complex float* src)
{
	ifft2(D, dimensions, flags, MD_STRIDES(D, dimensions, CFL_SIZE), dst, MD_STRIDES(D, dimensions, CFL_SIZE), src);
}

void fftc(int D, const long dimensions[D], unsigned long flags, complex float* dst, const complex float* src)
{
	fftmod(D, dimensions, flags, dst, src);
	fft(D, dimensions, flags, dst, dst);
	fftmod(D, dimensions, flags, dst, dst);
}

void ifftc(int D, const long dimensions[D], unsigned long flags, complex float* dst, const complex float* src)
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

void fftu(int D, const long dimensions[D], unsigned long flags, complex float* dst, const complex float* src)
{
	fft(D, dimensions, flags, dst, src);
	fftscale(D, dimensions, flags, dst, dst);
}

void ifftu(int D, const long dimensions[D], unsigned long flags, complex float* dst, const complex float* src)
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

void fftuc(int D, const long dimensions[D], unsigned long flags, complex float* dst, const complex float* src)
{
	fftc(D, dimensions, flags, dst, src);
	fftscale(D, dimensions, flags, dst, dst);
}

void ifftuc(int D, const long dimensions[D], unsigned long flags, complex float* dst, const complex float* src)
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



