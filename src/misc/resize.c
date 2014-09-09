/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <complex.h>
#include <assert.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/fft.h"

#include "resize.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
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
	md_select_dims(N, ~0u, tdims, idims);
	tdims[i] = odims[i];

	complex float* tmp = md_alloc_sameplace(N, tdims, CFL_SIZE, src);

	md_resizec(N, tdims, tmp, idims, src, CFL_SIZE);

	fftc(N, tdims, (1u << i), tmp, tmp);

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
			lflags |= (1 << i);

	assert(flags == lflags);

	unsigned int sflags = 0;

	for (unsigned int i = 0; i < N; i++)
		if (odims[i] < idims[i])
			sflags |= (1 << i);

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
	ifftc(N, tdims, (1u << i), tmp, tmp);
	md_resizec(N, odims, dst, tdims, tmp, CFL_SIZE);

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
			lflags |= (1 << i);

	assert(0 == lflags);

	unsigned int sflags = 0;

	for (unsigned int i = 0; i < N; i++)
		if (odims[i] < idims[i])
			sflags |= (1 << i);

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
			flags |= (1u << i);

	fftmod(D, in_dims, flags, tmp, in);
	fft(D, in_dims, flags, tmp, tmp);

	// Use md_resizec crop or zero pad, depending on whether we are sizing down or up
	md_resizec(D, out_dims, out, in_dims, tmp, CFL_SIZE);

	md_free(tmp);

	ifft(D, out_dims, flags, out, out);
	fftmod(D, out_dims, flags, out, out);
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




