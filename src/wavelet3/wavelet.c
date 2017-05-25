/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013 Frank Ong <uecker@eecs.berkeley.edu>
 * 2013-2014 Martin Uecker <uecker@eecs.berkeley.edu>
 */

/*
 * md_*-based multi-dimensional wavelet implementation
 *
 * - 3 levels (1d, md, md-hierarchical)
 * - all higher-level code should work for GPU as well
 *
 * Bugs:
 *
 * - GPU version is not optimized
 * - memory use could possible be reduced
 * 
 * Missing:
 *
 * - different boundary conditions
 *   (symmetric, periodic, zero)
 */

#include <complex.h>
#include <assert.h>
#include <limits.h>
#include <stdbool.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/ops.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#include "wavelet3/wl3-cuda.h"
#endif

#include "wavelet.h"



// layer 1 - 1-dimensional wavelet transform

static unsigned int bandsize(unsigned int imsize, unsigned int flen)
{
	return (imsize + flen - 1) / 2;
}

static complex float* access(const long str[3], complex float* x, long i, long j, long k)
{
	return (void*)x + str[2] * i + str[1] * j + str[0] * k;
}

static const complex float* caccess(const long str[3], const complex float* x, long i, long j, long k)
{
	return (const void*)x + str[2] * i + str[1] * j + str[0] * k;
}


static int coord(int l, int x, int flen, int k)
{
	int n = 2 * l + 1 - (flen - 1) + k;

	if (n < 0)
		n = -n - 1;

	if (n >= x)
		n = x - 1 - (n - x);

	return n;
}


static void wavelet_down3(const long dims[3], const long out_str[3], complex float* out, const long in_str[3], const complex float* in, unsigned int flen, const float filter[flen])
{
#pragma omp parallel for collapse(3)
	for (unsigned int i = 0; i < dims[2]; i++)
		for (unsigned int j = 0; j < bandsize(dims[1], flen); j++)
			for (unsigned int k = 0; k < dims[0]; k++) {

				*access(out_str, out, i, j, k) = 0.;

				for (unsigned int l = 0; l < flen; l++) {

						int n = coord(j, dims[1], flen, l);
	
						*access(out_str, out, i, j, k) += 
							*(caccess(in_str, in, i, n, k)) * filter[flen - l - 1];
				}
			}
}

static void wavelet_up3(const long dims[3], const long out_str[3], complex float* out, const long in_str[3],  const complex float* in, unsigned int flen, const float filter[flen])
{
//	md_clear2(3, dims, out_str, out, CFL_SIZE);

#pragma omp parallel for collapse(3)
	for (unsigned int i = 0; i < dims[2]; i++)
		for (unsigned int j = 0; j < dims[1]; j++)
			for (unsigned int k = 0; k < dims[0]; k++) {

		//		*access(out_str, out, i, j, k) = 0.;

				for (unsigned int l = ((j + flen / 2 - 0) - (flen - 1)) % 2; l < flen; l += 2) {

					int n = ((j + flen / 2 - 0) - (flen - 1) + l) / 2;

					if ((0 <= n) && ((unsigned int)n < bandsize(dims[1], flen)))
						*access(out_str, out, i, j, k) += 
							*caccess(in_str, in, i, n, k) * filter[flen - l - 1];
				}
			}
}


void fwt1(unsigned int N, unsigned int d, const long dims[N], const long ostr[N], complex float* low, complex float* hgh, const long istr[N], const complex float* in, const long flen, const float filter[2][2][flen])
{
	debug_printf(DP_DEBUG4, "fwt1: %d/%d\n", d, N);
	debug_print_dims(DP_DEBUG4, N, dims);

	long odims[N];
	md_copy_dims(N, odims, dims);
	odims[d] = bandsize(dims[d], flen);

	debug_print_dims(DP_DEBUG4, N, odims);

	long o = d + 1;
	long u = N - o;

	// 0 1 2 3 4 5 6|7
	// --d-- * --u--|N
	// ---o---

	assert(d == md_calc_blockdim(d, dims + 0, istr + 0, CFL_SIZE));
	assert(u == md_calc_blockdim(u, dims + o, istr + o, CFL_SIZE * md_calc_size(o, dims)));

	assert(d == md_calc_blockdim(d, odims + 0, ostr + 0, CFL_SIZE));
	assert(u == md_calc_blockdim(u, odims + o, ostr + o, CFL_SIZE * md_calc_size(o, odims)));

	// merge dims

	long wdims[3] = { md_calc_size(d, dims), dims[d], md_calc_size(u, dims + o) };
	long wistr[3] = { CFL_SIZE, istr[d], CFL_SIZE * md_calc_size(o, dims) };
	long wostr[3] = { CFL_SIZE, ostr[d], CFL_SIZE * md_calc_size(o, odims) };

#ifdef  USE_CUDA
	if (cuda_ondevice(in)) {

		assert(cuda_ondevice(low));
		assert(cuda_ondevice(hgh));

		float* flow = md_gpu_move(1, MD_DIMS(flen), filter[0][0], FL_SIZE);
		float* fhgh = md_gpu_move(1, MD_DIMS(flen), filter[0][1], FL_SIZE);

		wl3_cuda_down3(wdims, wostr, low, wistr, in, flen, flow);
		wl3_cuda_down3(wdims, wostr, hgh, wistr, in, flen, fhgh);

		md_free(flow);
		md_free(fhgh);
		return;
	}
#endif

	// no clear needed
	wavelet_down3(wdims, wostr, low, wistr, in, flen, filter[0][0]);
	wavelet_down3(wdims, wostr, hgh, wistr, in, flen, filter[0][1]);
}


void iwt1(unsigned int N, unsigned int d, const long dims[N], const long ostr[N], complex float* out, const long istr[N], const complex float* low, const complex float* hgh, const long flen, const float filter[2][2][flen])
{
	debug_printf(DP_DEBUG4, "ifwt1: %d/%d\n", d, N);
	debug_print_dims(DP_DEBUG4, N, dims);

	long idims[N];
	md_copy_dims(N, idims, dims);
	idims[d] = bandsize(dims[d], flen);

	debug_print_dims(DP_DEBUG4, N, idims);

	long o = d + 1;
	long u = N - o;

	// 0 1 2 3 4 5 6|7
	// --d-- * --u--|N
	// ---o---

	assert(d == md_calc_blockdim(d, dims + 0, ostr + 0, CFL_SIZE));
	assert(u == md_calc_blockdim(u, dims + o, ostr + o, CFL_SIZE * md_calc_size(o, dims)));
	assert(d == md_calc_blockdim(d, idims + 0, istr + 0, CFL_SIZE));
	assert(u == md_calc_blockdim(u, idims + o, istr + o, CFL_SIZE * md_calc_size(o, idims)));

	long wdims[3] = { md_calc_size(d, dims), dims[d], md_calc_size(u, dims + o) };
	long wistr[3] = { CFL_SIZE, istr[d], CFL_SIZE * md_calc_size(o, idims) };
	long wostr[3] = { CFL_SIZE, ostr[d], CFL_SIZE * md_calc_size(o, dims) };

	md_clear(3, wdims, out, CFL_SIZE);	// we cannot clear because we merge outputs

#ifdef  USE_CUDA
	if (cuda_ondevice(out)) {

		assert(cuda_ondevice(low));
		assert(cuda_ondevice(hgh));

		float* flow = md_gpu_move(1, MD_DIMS(flen), filter[1][0], FL_SIZE);
		float* fhgh = md_gpu_move(1, MD_DIMS(flen), filter[1][1], FL_SIZE);

		wl3_cuda_up3(wdims, wostr, out, wistr, low, flen, flow);
		wl3_cuda_up3(wdims, wostr, out, wistr, hgh, flen, fhgh);

		md_free(flow);
		md_free(fhgh);
		return;
	}
#endif

	wavelet_up3(wdims, wostr, out, wistr, low, flen, filter[1][0]);
	wavelet_up3(wdims, wostr, out, wistr, hgh, flen, filter[1][1]);
}


// layer 2 - multi-dimensional wavelet transform

static void wavelet_dims_r(unsigned int N, unsigned int n, unsigned int flags, long odims[2 * N], const long dims[N], const long flen)
{
	if (MD_IS_SET(flags, n)) {

		odims[0 + n] = bandsize(dims[n], flen);
		odims[N + n] = 2;
	} 

	if (n > 0)
		wavelet_dims_r(N, n - 1, flags, odims, dims, flen);
}

void wavelet_dims(unsigned int N, unsigned int flags, long odims[2 * N], const long dims[N], const long flen)
{
	md_copy_dims(N, odims, dims);
	md_singleton_dims(N, odims + N);

	wavelet_dims_r(N, N - 1, flags, odims, dims, flen);
}


void fwtN(unsigned int N, unsigned int flags, const long shifts[N], const long dims[N], const long ostr[2 * N], complex float* out, const long istr[N], const complex float* in, const long flen, const float filter[2][2][flen])
{
	long odims[2 * N];
	wavelet_dims(N, flags, odims, dims, flen);

	assert(md_calc_size(2 * N, odims) >= md_calc_size(N, dims));

	// FIXME one of these is unnecessary if we use the output

	complex float* tmpA = md_alloc_sameplace(2 * N, odims, CFL_SIZE, out);
	complex float* tmpB = md_alloc_sameplace(2 * N, odims, CFL_SIZE, out);

	long tidims[2 * N];
	md_copy_dims(N, tidims, dims);
	md_singleton_dims(N, tidims + N);
	
	long tistrs[2 * N];
	md_calc_strides(2 * N, tistrs, tidims, CFL_SIZE);

	long todims[2 * N];
	md_copy_dims(2 * N, todims, tidims);

	long tostrs[2 * N];

	// maybe we should push the randshift into lower levels

	//md_copy2(N, dims, tistrs, tmpA, istr, in, CFL_SIZE);
	md_circ_shift2(N, dims, shifts, tistrs, tmpA, istr, in, CFL_SIZE);

	for (unsigned int i = 0; i < N; i++) {

		if (MD_IS_SET(flags, i)) {

			todims[0 + i] = odims[0 + i];
			todims[N + i] = odims[N + i];

			md_calc_strides(2 * N, tostrs, todims, CFL_SIZE);
		
			fwt1(2 * N, i, tidims, tostrs, tmpB, (void*)tmpB + tostrs[N + i], tistrs, tmpA, flen, filter);

			md_copy_dims(2 * N, tidims, todims);
			md_copy_dims(2 * N, tistrs, tostrs);

			complex float* swap = tmpA;
			tmpA = tmpB;
			tmpB = swap;
		}
	}

	md_copy2(2 * N, todims, ostr, out, tostrs, tmpA, CFL_SIZE);

	md_free(tmpA);
	md_free(tmpB);
}


void iwtN(unsigned int N, unsigned int flags, const long shifts[N], const long dims[N], const long ostr[N], complex float* out, const long istr[2 * N], const complex float* in, const long flen, const float filter[2][2][flen])
{
	long idims[2 * N];
	wavelet_dims(N, flags, idims, dims, flen);

	assert(md_calc_size(2 * N, idims) >= md_calc_size(N, dims));

	complex float* tmpA = md_alloc_sameplace(2 * N, idims, CFL_SIZE, out);
	complex float* tmpB = md_alloc_sameplace(2 * N, idims, CFL_SIZE, out);

	long tidims[2 * N];
	md_copy_dims(2 * N, tidims, idims);
	
	long tistrs[2 * N];
	md_calc_strides(2 * N, tistrs, tidims, CFL_SIZE);

	long todims[2 * N];
	md_copy_dims(2 * N, todims, tidims);

	long tostrs[2 * N];

	long ishifts[N];
	for (unsigned int i = 0; i < N; i++)
		ishifts[i] = -shifts[i];

	md_copy2(2 * N, tidims, tistrs, tmpA, istr, in, CFL_SIZE);

	for (int i = N - 1; i >= 0; i--) {	// run backwards to maintain contigous blocks

		if (MD_IS_SET(flags, i)) {

			todims[0 + i] = dims[0 + i];
			todims[N + i] = 1;

			md_calc_strides(2 * N, tostrs, todims, CFL_SIZE);
		
			iwt1(2 * N, i, todims, tostrs, tmpB, tistrs, tmpA, (void*)tmpA + tistrs[N + i], flen, filter);

			md_copy_dims(2 * N, tidims, todims);
			md_copy_dims(2 * N, tistrs, tostrs);

			complex float* swap = tmpA;
			tmpA = tmpB;
			tmpB = swap;
		}
	}

	//md_copy2(N, dims, ostr, out, tostrs, tmpA, CFL_SIZE);
	md_circ_shift2(N, dims, ishifts, ostr, out, tostrs, tmpA, CFL_SIZE);

	md_free(tmpA);
	md_free(tmpB);
}

// layer 3 - hierarchical multi-dimensional wavelet transform

static long wavelet_filter_flags(unsigned int N, long flags, const long dims[N], const long min[N])
{
	for (unsigned int i = 0; i < N; i++)
		if (dims[i] < min[i])	// CHECK
			flags = MD_CLEAR(flags, i);

	return flags;
}

long wavelet_num_levels(unsigned int N, unsigned int flags, const long dims[N], const long min[N], const long flen)
{
	if (0 == flags)
		return 1;

	long wdims[2 * N];
	wavelet_dims(N, flags, wdims, dims, flen);

	return 1 + wavelet_num_levels(N, wavelet_filter_flags(N, flags, wdims, min), wdims, min, flen);
}

static long wavelet_coeffs_r(unsigned int levels, unsigned int N, unsigned int flags, const long dims[N], const long min[N], const long flen)
{
	long wdims[2 * N];
	wavelet_dims(N, flags, wdims, dims, flen);

	long coeffs = md_calc_size(N, wdims);
	long bands = md_calc_size(N, wdims + N);

	assert((0 == flags) == (0 == levels));

	if (0 == flags)
		return bands * coeffs;
	
	return coeffs * (bands - 1) + wavelet_coeffs_r(levels - 1, N, wavelet_filter_flags(N, flags, wdims, min), wdims, min, flen);
}

long wavelet_coeffs(unsigned int N, unsigned int flags, const long dims[N], const long min[N], const long flen)
{
	unsigned int levels = wavelet_num_levels(N, flags, dims, min, flen);

	assert(levels > 0);

	return wavelet_coeffs_r(levels - 1, N, flags, dims, min, flen);	
}



void fwt(unsigned int N, unsigned int flags, const long shifts[N], const long dims[N], complex float* out, const long istr[N], const complex float* in, const long minsize[N], long flen, const float filter[2][2][flen])
{
	if (0 == flags) {

		if (out != in)
			md_copy2(N, dims, istr, out, istr, in, CFL_SIZE);

		return;
	}

	unsigned long coeffs = wavelet_coeffs(N, flags, dims, minsize, flen);

	long wdims[2 * N];
	wavelet_dims(N, flags, wdims, dims, flen);

	long ostr[2 * N];
	md_calc_strides(2 * N, ostr, wdims, CFL_SIZE);

	long offset = coeffs - md_calc_size(2 * N, wdims);

	debug_printf(DP_DEBUG4, "%d %ld %ld\n", flags, coeffs, offset);

	long shifts0[N];
	for (unsigned int i = 0; i < N; i++)
		shifts0[i] = 0;

	fwtN(N, flags, shifts, dims, ostr, out + offset, istr, in, flen, filter);
	fwt(N, wavelet_filter_flags(N, flags, wdims, minsize), shifts0, wdims, out, ostr, out + offset, minsize, flen, filter);
}


void iwt(unsigned int N, unsigned int flags, const long shifts[N], const long dims[N], const long ostr[N], complex float* out, const complex float* in, const long minsize[N], const long flen, const float filter[2][2][flen])
{
	if (0 == flags) {

		if (out != in)
			md_copy2(N, dims, ostr, out, ostr, in, CFL_SIZE);

		return;
	}

	unsigned long coeffs = wavelet_coeffs(N, flags, dims, minsize, flen);

	long wdims[2 * N];
	wavelet_dims(N, flags, wdims, dims, flen);

	long istr[2 * N];
	md_calc_strides(2 * N, istr, wdims, CFL_SIZE);

	long offset = coeffs - md_calc_size(2 * N, wdims);

	debug_printf(DP_DEBUG4, "%d %ld %ld\n", flags, coeffs, offset);

	complex float* tmp = md_alloc_sameplace(2 * N, wdims, CFL_SIZE, out);

	md_copy(2 * N, wdims, tmp, in + offset, CFL_SIZE);

	long shifts0[N];
	for (unsigned int i = 0; i < N; i++)
		shifts0[i] = 0;

	// fix me we need temp storage
	iwt(N, wavelet_filter_flags(N, flags, wdims, minsize), shifts0, wdims, istr, tmp, in, minsize, flen, filter);
	iwtN(N, flags, shifts, dims, ostr, out, istr, tmp, flen, filter);

	md_free(tmp);
}


void wavelet3_thresh(unsigned int N, float lambda, unsigned int flags, const long shifts[N], const long dims[N], complex float* out, const complex float* in, const long minsize[N], long flen, const float filter[2][2][flen])
{
	unsigned long coeffs = wavelet_coeffs(N, flags, dims, minsize, flen);

	long istr[N];
	md_calc_strides(N, istr, dims, CFL_SIZE);

	complex float* tmp = md_alloc_sameplace(1, MD_DIMS(coeffs), CFL_SIZE, out);

	fwt(N, flags, shifts, dims, tmp, istr, in, minsize, flen, filter);
	md_zsoftthresh(1, MD_DIMS(coeffs), lambda, 0u, tmp, tmp);
	iwt(N, flags, shifts, dims, istr, out, tmp, minsize, flen, filter);

	md_free(tmp);
}





const float wavelet3_haar[2][2][2] = {
	{ { +0.7071067811865475, +0.7071067811865475 },
	  { -0.7071067811865475, +0.7071067811865475 }, },
	{ { +0.7071067811865475, +0.7071067811865475 },
	  { +0.7071067811865475, -0.7071067811865475 }, },
};

const float wavelet3_dau2[2][2][4] = {
	{ { -0.1294095225512603, +0.2241438680420134, +0.8365163037378077, +0.4829629131445341 },
	  { -0.4829629131445341, +0.8365163037378077, -0.2241438680420134, -0.1294095225512603 }, },
	{ { +0.4829629131445341, +0.8365163037378077, +0.2241438680420134, -0.1294095225512603 },
	  { -0.1294095225512603, -0.2241438680420134, +0.8365163037378077, -0.4829629131445341 }, },
};

const float wavelet3_cdf44[2][2][10] = {
	{ { +0.00000000000000000, +0.03782845550726404 , -0.023849465019556843, -0.11062440441843718 , +0.37740285561283066, 
	    +0.85269867900889385, +0.37740285561283066 , -0.11062440441843718 , -0.023849465019556843, +0.03782845550726404 },
	  { +0.00000000000000000, -0.064538882628697058, +0.040689417609164058, +0.41809227322161724 , -0.7884856164055829, 
	    +0.41809227322161724, +0.040689417609164058, -0.064538882628697058, +0.00000000000000000 , +0.00000000000000000 }, },
	{ { +0.00000000000000000, -0.064538882628697058, -0.040689417609164058, +0.41809227322161724 , +0.7884856164055829, 
	    +0.41809227322161724, -0.040689417609164058, -0.064538882628697058, +0.000000000000000000, +0.00000000000000000 },
	  { +0.00000000000000000, -0.03782845550726404 , -0.023849465019556843, +0.11062440441843718 , +0.37740285561283066,
            -0.85269867900889385, +0.37740285561283066 , +0.11062440441843718 , -0.023849465019556843, -0.03782845550726404 }, },
};



