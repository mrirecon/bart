/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012, 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 *
 * 
 * Implementation of CDF97 wavelets.
 *
 * Ingrid Daubechies and Wil Sweldens, Factoring wavelet transforms into
 * lifting steps. Journal of Fourier Analysis and Applications 1998,
 * Volume 4, Issue 3, pp 247-269
 *
 */

#include <stdbool.h>
#include <complex.h>
#include <assert.h>

#include "num/multind.h"
//#include "num/parallel.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#ifdef BERKELEY_SVN
#include "num/wlcuda.h"
#endif
#endif

#include "wavelet.h"

const float a[4] = { -1.586134342, -0.05298011854, 0.8829110762, 0.4435068522 };
const float scale = 1.149604398;



/**
 * This is designed to work for every n.
 * For odd n, we split n = a + b with 
 * a = b + 1 where a is the number of 
 * coarse coefficients. This splitting is
 * implicit by putting the first
 * value into the coarse coefficients.
 */
static void predict(int n, float a, int str, float* x)
{
	for (int i = 1; i < n - 1; i += 2) 
		x[i * str] += a * (x[(i - 1) * str] + x[(i + 1) * str]);

	if (0 == n % 2) 
		x[(n - 1) * str] += a * (x[(n - 2) * str] + x[0]);	// periodic
	//	x[(n - 1) * str] += 2. * a * x[(n - 2) * str]; // non-periodic
}

static void update(int n, float a, int str, float* x)
{
	for (int i = 2; i < n - 1; i += 2) 
		x[i * str] += a * (x[(i - 1) * str] + x[(i + 1) * str]);

	if (0 == n % 2) 	// +-+-+- 
		x[0] += a * (x[(n - 1) * str] + x[1 * str]); 	// periodic
	//	x[0] += 2. * a * x[1 * str]; // non-periodic
	else {			// +-+-+
		x[0] += 2. * a * x[1 * str];
		x[(n - 1) * str] += 2. * a * x[(n - 2) * str];
	}
}


static void cdf97(int n, int str, float* x)
{
	predict(n, a[0], str, x);
	update(n, a[1], str, x);
	predict(n, a[2], str, x);
	update(n, a[3], str, x);

	for (int i = 0; i < n; i++)
		x[i * str] *= (0 == i % 2) ? scale : (1. / scale);
}

static void icdf97(int n, int str, float* x)
{
	for (int i = 0; i < n; i++)
		x[i * str] *= (0 == i % 2) ? (1. / scale) : scale;

	update(n, -a[3], str, x);
	predict(n, -a[2], str, x);
	update(n, -a[1], str, x);
	predict(n, -a[0], str, x);
}



static long num_coeff(long n)
{
	return n / 2;
}

static long num_scale(long n)
{
	return n - num_coeff(n);
}




static void resort(int n, int str, float* src)
{
	float tmp[n];

	for (int i = 0; i < num_scale(n); i++)
		tmp[i] = src[(i * 2 + 0) * str];

	for (int i = 0; i < num_coeff(n); i++)
		tmp[num_scale(n) + i] = src[(i * 2 + 1) * str];

	for (int i = 0; i < n; i++)
		src[i * str] = tmp[i];
}

static void iresort(int n, int str, float* src)
{
	float tmp[n];

	for (int i = 0; i < num_scale(n); i++)
		tmp[i * 2 + 0] = src[i * str];

	for (int i = 0; i < num_coeff(n); i++)
		tmp[i * 2 + 1] = src[(num_scale(n) + i) * str];

	for (int i = 0; i < n; i++)
		src[i * str] = tmp[i];
}



static void cdf97_line(void* _data, long n, long str, void* ptr)
{
	assert(NULL == _data);
	cdf97(n, str / 4, ptr);
	resort(n, str / 4, ptr);
}

static void icdf97_line(void* _data, long n, long str, void* ptr)
{
	assert(NULL == _data);
	iresort(n, str / 4, ptr);
	icdf97(n, str / 4, ptr);
}

static void cdf97_line_nosort(void* _data, long n, long str, void* ptr)
{
	assert(NULL == _data);

#ifdef USE_CUDA
	if (cuda_ondevice(ptr))
#ifdef BERKELEY_SVN
		cuda_cdf97(n, str / 4, ptr);
#else
		assert(0);
#endif
	else
#endif
	cdf97(n, str / 4, ptr);
}

static void icdf97_line_nosort(void* _data, long n, long str, void* ptr)
{
	assert(NULL == _data);

#ifdef USE_CUDA
	if (cuda_ondevice(ptr))
#ifdef BERKELEY_SVN
		cuda_icdf97(n, str / 4, ptr);	
#else
		assert(0);
#endif
	else
#endif
	icdf97(n, str / 4, ptr);
}





void md_wavtrafo2(int D, const long dims[D], unsigned int flags, const long strs[D], void* ptr, md_trafo_fun_t fun, bool inv, bool nosort)
{
	if (0 == flags)
		return;

	bool rec = true;
	for (int i = 0; i < D; i++) {

		if (1 == dims[i])
			flags = MD_CLEAR(flags, i);

		if (MD_IS_SET(flags, i))
			rec &= (dims[i] > 32);
	}

	if (!inv)
		md_septrafo2(D, dims, flags, strs, ptr, fun, NULL);
		//md_parallel_septrafo2(D, dims, flags, strs, ptr, fun, NULL);

	if (rec) {

		long dims2[D];
		md_select_dims(D, ~0, dims2, dims);
		
		for (int i = 0; i < D; i++)
			if (MD_IS_SET(flags, i))
				dims2[i] = num_scale(dims[i]);	

		long strs2[D];
		md_copy_strides(D, strs2, strs);

		for (int i = 0; i < D; i++)
			if (nosort && (MD_IS_SET(flags, i)))
				strs2[i] *= 2;

		md_wavtrafo2(D, dims2, flags, strs2, ptr, fun, inv, nosort);
	}

	if (inv)
		md_septrafo2(D, dims, flags, strs, ptr, fun, NULL);
		//md_parallel_septrafo2(D, dims, flags, strs, ptr, fun, NULL);
}


void md_wavtrafo(int D, const long dims[D], unsigned int flags, void* ptr, size_t size, md_trafo_fun_t fun, bool inv, bool nosort)
{
	long strs[D];
	md_calc_strides(D, strs, dims, size);
	md_wavtrafo2(D, dims, flags, strs, ptr, fun, inv, nosort);
}


void md_wavtrafoz2(int D, const long dims[D], unsigned int flags, const long strs[D], complex float* x, md_trafo_fun_t fun, bool inv, bool nosort)
{
	long dims2[D + 1];
	dims2[0] = 2; // complex float
	md_copy_dims(D, dims2 + 1, dims);

	long strs2[D + 1];
	strs2[0] = sizeof(float);
	md_copy_strides(D, strs2 + 1, strs);

	md_wavtrafo2(D + 1, dims2, flags << 1, strs2, (void*)x, fun, inv, nosort);
}

void md_wavtrafoz(int D, const long dims[D], unsigned int flags, complex float* ptr, md_trafo_fun_t fun, bool inv, bool nosort)
{
	long strs[D];
	md_calc_strides(D, strs, dims, sizeof(complex float));
	md_wavtrafoz2(D, dims, flags, strs, ptr, fun, inv, nosort);
}


void md_cdf97z(int D, const long dims[D], unsigned int flags, complex float* data)
{
	md_wavtrafoz(D, dims, flags, data, cdf97_line_nosort, false, true);
}

void md_icdf97z(int D, const long dims[D], unsigned int flags, complex float* data)
{
	md_wavtrafoz(D, dims, flags, data, icdf97_line_nosort, true, true);
}

void md_cdf97z2(int D, const long dims[D], unsigned int flags, const long strs[D], complex float* data)
{
	md_wavtrafoz2(D, dims, flags, strs, data, cdf97_line_nosort, false, true);
}

void md_icdf97z2(int D, const long dims[D], unsigned int flags, const long strs[D], complex float* data)
{
	md_wavtrafoz2(D, dims, flags, strs, data, icdf97_line_nosort, true, true);
}


// FIXME: slow
void md_resortz(int D, const long dims[D], unsigned int flags, complex float* data)
{
	md_wavtrafoz(D, dims, flags, data, icdf97_line_nosort, true, true);
	md_wavtrafoz(D, dims, flags, data, cdf97_line, false, false);
}

void md_iresortz(int D, const long dims[D], unsigned int flags, complex float* data)
{
	md_wavtrafoz(D, dims, flags, data, icdf97_line, true, false);
	md_wavtrafoz(D, dims, flags, data, cdf97_line_nosort, false, true);
}


#if 0

const float d4[6] = { -sqrt(3.), sqrt(3.) / 4., -(2. - sqrt(3.)) / 4., 1., sqrt(2. + sqrt(3.)), sqrt(2. - sqrt(3.)) };

static void d4update(int n, float x[n])
{
	
}



void deb4

#endif
