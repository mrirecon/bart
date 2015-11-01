/* Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 *	2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#define _GNU_SOURCE
#include <stdbool.h>
#include <assert.h>
#include <complex.h>
#include <strings.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/misc.h"

#include "mdfft.h"





static void rot45z2(unsigned int D, unsigned int M,
	const long dim[D], const long ostr[D], complex float* optr,
	const long istr[D], const complex float* iptr)
{
	assert(M < D);
	assert(2 == dim[M]);
	assert(optr != iptr);

	long dims2[D];
	md_copy_dims(D, dims2, dim);
	dims2[M] = 1;

	long ostr2[D];
	md_copy_strides(D, ostr2, ostr);
	ostr2[M] *= 2;

	long istr2[D];
	md_copy_strides(D, istr2, istr);
	istr2[M] *= 2;

	md_zadd2(D, dims2, ostr2,         optr           , istr2, iptr, istr2, ((void*)iptr) + istr[M]);
	md_zsub2(D, dims2, ostr2, ((void*)optr) + ostr[M], istr2, iptr, istr2, ((void*)iptr) + istr[M]);
}



static unsigned int find_bit(unsigned long N)
{
	return ffsl(N) - 1;
}

static unsigned int next_powerof2(unsigned int x)
{
	x--;

	for (unsigned int i = 0, n = 1; i < 6; i++, n *= 2)
		x = (x >> n) | x;

	return x + 1;
}


static void compute_chirp(unsigned int L, bool dir, unsigned int M, complex float krn[M])
{
	krn[0] = 1.;

	for (unsigned int i = 1; i < M; i++)
		krn[i] = 0.;

	for (unsigned int i = 1; i < M; i++) {

		if (i < L) {

			complex float v = cexpf((dir ? -1. : 1.) * M_PI * 1.i * (float)(i * i) / (float)L);
			krn[i] = v;
			krn[M - i] = v;
		}
	}
}

static void bluestein(unsigned int N, const long dims[N],
	unsigned long flags, unsigned long dirs,
	const long ostrs[N], complex float* dst,
	const long istrs[N], const complex float* in)
{
	unsigned int D = find_bit(flags);
	unsigned int M = next_powerof2(2 * dims[D] - 1);

	assert(M >= 2 * dims[D] - 1);
	assert(0 == MD_CLEAR(flags, D));

	/* Bluestein
	 *
         * Transform DFT into convolution according to
	 *
	 *	\ksi_N^{nk} = \ksi_N^{-(k-n)^2/2 + n^2/2 + k^2/2}
	 *
	 * ... and use fft of different size to implement it.
	 */

	long kdims[N];
	md_singleton_dims(N, kdims);
	kdims[D] = M;

	long kstrs[N];
	md_calc_strides(N, kstrs, kdims, CFL_SIZE);

	complex float* xkrn = md_alloc(N, kdims, CFL_SIZE);
	compute_chirp(dims[D], MD_IS_SET(dirs, D), M, xkrn);

	complex float* krn = md_alloc_sameplace(N, kdims, CFL_SIZE, dst);
	md_copy(N, kdims, krn, xkrn, CFL_SIZE);
	md_free(xkrn);

	complex float* fkrn = md_alloc_sameplace(N, kdims, CFL_SIZE, dst);
	md_fft(N, kdims, MD_BIT(D), MD_FFT_FORWARD, fkrn, krn);

	long bdims[N];
	md_copy_dims(N, bdims, dims);
	bdims[D] = M;

	long bstrs[N];
	md_calc_strides(N, bstrs, bdims, CFL_SIZE);

	complex float* btmp = md_alloc_sameplace(N, bdims, CFL_SIZE, dst);
	md_clear(N, bdims, btmp, CFL_SIZE);

	md_zmulc2(N, dims, bstrs, btmp, istrs, in, kstrs, krn);
	md_fft2(N, bdims, MD_BIT(D), MD_FFT_INVERSE, bstrs, btmp, bstrs, btmp);
	md_zmul2(N, bdims, bstrs, btmp, bstrs, btmp, kstrs, fkrn);
	md_fft2(N, bdims, MD_BIT(D), MD_FFT_FORWARD, bstrs, btmp, bstrs, btmp);

	md_zsmul(1, MD_DIMS(M), krn, krn, 1. / (float)M);
	md_zmulc2(N, dims, ostrs, dst, bstrs, btmp, kstrs, krn);

	md_free(fkrn);
	md_free(krn);
	md_free(btmp);
}








static void compute_twiddle(unsigned int n, unsigned int m, complex float t[n][m])
{
	for (unsigned int i = 0; i < n; i++)
		for (unsigned int j = 0; j < m; j++)
			t[i][j] = cexpf(-2.i * M_PI * (float)(i * j) / (float)(n * m));
}

static void cooley_tukey(unsigned int N, const long dims[N],
		unsigned int D, unsigned int a, unsigned int b,
		unsigned long flags, unsigned long dirs,
		const long ostr[N], complex float* dst,
		const long istr[N], const complex float* in)
{
	/* Cooley-Tukey
	 *
	 * With N = A * B, \ksi_N^N = 1, split into smaller FFTs:
	 *
	 *   \ksi_N^{(B * i + j)(l + A * k)}
	 * = \ksi_N^{B * i * l + A * j * k + j * l}
	 * = \ksi_N^{B * i * l} \ksi_N^{A * j * k} \ksi_N^{j * l}
	 * = \ksi_A^{i * l} \ksi_N^{j * l} \ksi_B^{j * k}
	 */

	long xdims[N + 1];
	md_copy_dims(N, xdims, dims);
	xdims[D] = a;
	xdims[N] = b;

	long astr[N + 1];
	md_copy_strides(N, astr, istr);
	astr[D] = istr[D] * 1;
	astr[N] = istr[D] * a;

	long bstr[N + 1];
	md_copy_strides(N, bstr, ostr);
	bstr[D] = ostr[D] * b;
	bstr[N] = ostr[D] * 1;

	unsigned long flags1 = 0;
	unsigned long flags2 = MD_CLEAR(flags, D);

	long tdims[N + 1];
	long tstrs[N + 1];
	md_select_dims(N + 1, MD_BIT(D) | MD_BIT(N), tdims, xdims);
	md_calc_strides(N + 1, tstrs, tdims, CFL_SIZE);

	complex float (*xtw)[b][a] = xmalloc(a * b * CFL_SIZE);
	compute_twiddle(b, a, *xtw);

	complex float* tw = md_alloc_sameplace(N + 1, tdims, CFL_SIZE, dst);
	md_copy(N + 1, tdims, tw, &(*xtw)[0][0], CFL_SIZE);
	free(xtw);

	md_fft2(N + 1, xdims, MD_SET(flags1, N), dirs, bstr, dst, astr, in);
	(MD_IS_SET(dirs, D) ?  md_zmulc2 : md_zmul2)(N + 1, xdims, bstr, dst, bstr, dst, tstrs, tw);
	md_fft2(N + 1, xdims, MD_SET(flags2, D), dirs, bstr, dst, bstr, dst);

	md_free(tw);
}


static bool check_strides(unsigned int N, const long ostr[N], const long istr[N])
{
	bool ret = true;

	for (unsigned int i = 0; i < N; i++)
		ret = ret & (ostr[i] == istr[i]);

	return ret;
}

static unsigned int find_factor(unsigned int N)
{
	for (unsigned int i = 2; i < N; i++)
		if (0 == N % i)
			return i;
	return N;
}

void md_fft2(unsigned int N, const long dims[N],
		unsigned long flags, unsigned long dirs,
		const long ostr[N], complex float* dst,
		const long istr[N], const complex float* in)
{
	if (0 == flags) {

		if (dst == in) {

			if (!check_strides(N, ostr, istr)) {

				// detect and use inplace transpose?

				long strs[N];
				md_calc_strides(N, strs, dims, CFL_SIZE);

				complex float* tmp = md_alloc_sameplace(N, dims, CFL_SIZE, dst);
				md_copy2(N, dims, strs, tmp, istr, in, CFL_SIZE);
				md_copy2(N, dims, ostr, dst, strs, tmp, CFL_SIZE);
				md_free(tmp);
			}

			return;
		}

		md_copy2(N, dims, ostr, dst, istr, in, CFL_SIZE);
		return;
	}

	unsigned int D = find_bit(flags);

	if (1 == dims[D]) {

		md_fft2(N, dims, MD_CLEAR(flags, D), dirs, ostr, dst, istr, in);
		return;
	}

	if (2 == dims[D]) {

		if (dst == in) {

			long strs[N];
			md_calc_strides(N, strs, dims, CFL_SIZE);

			complex float* tmp = md_alloc_sameplace(N, dims, CFL_SIZE, dst);
			md_fft2(N, dims, MD_CLEAR(flags, D), dirs, strs, tmp, istr, in);
			rot45z2(N, D, dims, ostr, dst, strs, tmp);
			md_free(tmp);

		} else {

			// the nufft may do the transpose
			rot45z2(N, D, dims, ostr, dst, istr, in);
			md_fft2(N, dims, MD_CLEAR(flags, D), dirs, ostr, dst, ostr, dst);
		}

		return;
	}

	unsigned int a = find_factor(dims[D]);
	unsigned int b = dims[D] / a;

	if (1 == b) { // prime

		bluestein(N, dims, MD_BIT(D), dirs, ostr, dst, istr, in);
		md_fft2(N, dims, MD_CLEAR(flags, D), dirs, ostr, dst, ostr, dst);

	} else {

		cooley_tukey(N, dims, D, a, b, flags, dirs, ostr, dst, istr, in);
	}
}





void md_fft(unsigned int N, const long dims[N],
		unsigned long flags, unsigned long dirs,
		complex float* dst, const complex float* in)
{
	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);
	md_fft2(N, dims, flags, dirs, strs, dst, strs, in);
}


