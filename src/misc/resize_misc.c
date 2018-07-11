/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2013-2014 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <complex.h>
#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/filter.h"

#include "resize.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

// FIXME: implement inverse, adjoint, etc..

static void fft_xzeropad2(unsigned int N, const long dims[N], unsigned int d, unsigned int x, const long ostr[N], complex float* dst, const long istr[N], const complex float* src)
{
	assert(d < N);

	long tdims[N + 1];
	md_copy_dims(N, tdims, dims);
	tdims[N] = x;

	long tostr[N + 1];
	md_copy_strides(N, tostr, ostr);
	tostr[d] = x * ostr[d];
	tostr[N] = ostr[d];

	long pdims[N + 1];
	md_select_dims(N + 1, MD_BIT(d) | MD_BIT(N), pdims, tdims);

	long pstr[N + 1];
	md_calc_strides(N + 1, pstr, pdims, CFL_SIZE);

	complex float* shift = md_alloc_sameplace(N + 1, pdims, CFL_SIZE, src);

	float pos[N];
	for (unsigned int i = 0; i < N; i++)
		pos[i] = 0.;

	for (unsigned int i = 0; i < x; i++) {

		pos[d] = -(1. / (float)x) * i;
		linear_phase(N, pdims, pos, (void*)shift + i * pstr[N]);
	}

	long tistr[N + 1];
	md_copy_strides(N, tistr, istr);
	tistr[N] = 0;

	md_zmul2(N + 1, tdims, tostr, dst, tistr, src, pstr, shift);

	md_free(shift);

	fftc2(N + 1, tdims, MD_BIT(d), tostr, dst, tostr, dst);
}

static void fft_xzeropad(unsigned int N, const long dims[N], unsigned int d, unsigned int x, complex float* dst, const complex float* src)
{
	long odims[N];
	long ostrs[N];
	long istrs[N];

	md_copy_dims(N, odims, dims);
	odims[d] = x * dims[d];

	md_calc_strides(N, ostrs, odims, CFL_SIZE);
	md_calc_strides(N, istrs, dims, CFL_SIZE);

	fft_xzeropad2(N, dims, d, x, ostrs, dst, istrs, src);
}


static void fft_zeropad_simple(unsigned int N, unsigned int flags, const long odims[N], complex float* dst, const long idims[N], const complex float* src)
{
	md_resize_center(N, odims, dst, idims, src, CFL_SIZE);
	fftc(N, odims, flags, dst, dst);
}

#if 0
static void fft_zeropad_simpleH(unsigned int N, unsigned int flags, const long odims[N], complex float* dst, const long idims[N], const complex float* src)
{
	complex float* tmp = md_alloc_sameplace(N, idims, CFL_SIZE, src);
	ifftc(N, idims, flags, tmp, src);
	md_resize_center(N, odims, dst, idims, tmp, CFL_SIZE);
	md_free(tmp);
}
#endif

static void fft_zeropad_r(unsigned int N, const long odims[N], complex float* dst, const long idims[N], const complex float* src)
{
	int i = N - 1;

	while (odims[i] == idims[i]) {

		if (0 == i) {

			if (dst != src)
				md_copy(N, odims, dst, src, CFL_SIZE);

			return;
		}

		i--;
	}

	//printf("%d %ld %ld\n", i, odims[i], idims[i]);
	assert(odims[i] > idims[i]);

	long tdims[N];
	md_copy_dims(N, tdims, idims);
	tdims[i] = odims[i];

	complex float* tmp = md_alloc_sameplace(N, tdims, CFL_SIZE, src);

#if 1
	if (0 == tdims[i] % idims[i]) {

		fft_xzeropad(N, idims, i, tdims[i] / idims[i], tmp, src);

	} else {
#else
	{
#endif

		fft_zeropad_simple(N, MD_BIT(i), tdims, tmp, idims, src);
	}

	fft_zeropad_r(N, odims, dst, tdims, tmp);

	md_free(tmp);
}


/*
 * perform zero-padded FFT
 *
 */
void fft_zeropad(unsigned int N, unsigned int flags, const long odims[N], complex float* dst, const long idims[N], const complex float* src)
{
	unsigned int lflags = 0;

	for (unsigned int i = 0; i < N; i++)
		if (odims[i] > idims[i])
			lflags = MD_SET(lflags, i);

	assert(flags == lflags);

	unsigned int sflags = 0;

	for (unsigned int i = 0; i < N; i++)
		if (odims[i] < idims[i])
			sflags = MD_SET(sflags, i);

	assert(0 == sflags);

	fft_zeropad_r(N, odims, dst, idims, src);
}



static void fft_zeropadH_r(unsigned int N, const long odims[N], complex float* dst, const long idims[N], const complex float* src)
{
	int i = N - 1;

	while (odims[i] == idims[i]) {

		if (0 == i) {

			if (dst != src)
				md_copy(N, odims, dst, src, CFL_SIZE);

			return;
		}

		i--;
	}

	assert (idims[i] > odims[i]);

	long tdims[N];
	md_copy_dims(N, tdims, odims);
	tdims[i] = idims[i];

	complex float* tmp = md_alloc_sameplace(N, tdims, CFL_SIZE, src);

	fft_zeropadH_r(N, tdims, tmp, idims, src);
	ifftc(N, tdims, MD_BIT(i), tmp, tmp);
	md_resize_center(N, odims, dst, tdims, tmp, CFL_SIZE);

	md_free(tmp);
}


/*
 * perform zero-padded FFT
 *
 */
void fft_zeropadH(unsigned int N, unsigned int flags, const long odims[N], complex float* dst, const long idims[N], const complex float* src)
{
	unsigned int lflags = 0;

	for (unsigned int i = 0; i < N; i++)
		if (odims[i] > idims[i])
			lflags = MD_SET(lflags, i);

	assert(0 == lflags);

	unsigned int sflags = 0;

	for (unsigned int i = 0; i < N; i++)
		if (odims[i] < idims[i])
			sflags = MD_SET(sflags, i);

	assert(flags == sflags);

	fft_zeropadH_r(N, odims, dst, idims, src);
}



/* scale using zero-padding in the Fourier domain
 *
 */

void sinc_resize(unsigned int D, const long out_dims[D], complex float* out, const long in_dims[D], const complex float* in)
{
	complex float* tmp = md_alloc_sameplace(D, in_dims, CFL_SIZE, in);

	unsigned int flags = 0;

	for (unsigned int i = 0; i < D; i++)
		if (out_dims[i] != in_dims[i])
			flags = MD_SET(flags, i);

	fftmod(D, in_dims, flags, tmp, in);
	fft(D, in_dims, flags, tmp, tmp);
	fftmod(D, in_dims, flags, tmp, tmp);
	// NOTE: the inner fftmod/ifftmod should cancel for N % 4 == 0
	// and could be replaced by a sign change for N % 4 == 1

	// md_resize_center can size up or down
	md_resize_center(D, out_dims, out, in_dims, tmp, CFL_SIZE);

	md_free(tmp);

	ifftmod(D, out_dims, flags, out, out);	// see above
	ifft(D, out_dims, flags, out, out);
	ifftmod(D, out_dims, flags, out, out);
}







/* scale using zero-padding in the Fourier domain - scale each dimensions in sequence (faster)
 *
 */
void sinc_zeropad(unsigned int D, const long out_dims[D], complex float* out, const long in_dims[D], const complex float* in)
{
	unsigned int i = D - 1;

	while (out_dims[i] == in_dims[i]) {

		if (0 == i) {

			if (out != in)
				md_copy(D, out_dims, out, in, CFL_SIZE);

			return;
		}

		i--;
	}

	assert(out_dims[i] > in_dims[i]);

	long tmp_dims[D];
	for (unsigned int l = 0; l < D; l++)
		tmp_dims[l] = in_dims[l];

	tmp_dims[i] = out_dims[i];

	//printf("Resizing...%d: %ld->%ld\n", i, in_dims[i], tmp_dims[i]);

	sinc_resize(D, tmp_dims, out, in_dims, in);
	sinc_zeropad(D, out_dims, out, tmp_dims, out);
}




