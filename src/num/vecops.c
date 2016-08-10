/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2011-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014 Frank Ong <frankong@berkeley.edu>
 * 2014-2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 *
 *
 * This file defines basic operations on vectors of floats/complex floats
 * for operations on the CPU which are are used by higher level code
 * (mainly num/flpmath.c and num/italgos.c) to implement more complex
 * operations. The functions are exported by pointers stored in the
 * global variable cpu_ops of type struct vec_ops. Identical functions
 * are implemented for the GPU in gpukrnls.c.
 *
 */

#include <assert.h>
#include <math.h>
#include <complex.h>
#include <stdbool.h>

#include "misc/misc.h"

#include "vecops.h"



/**
 * Allocate memory for array of floats.
 * Note: be sure to pass 2*N if allocating for complex float
 *
 * @param N number of elements
 */
static float* allocate(long N)
{
	assert(N >= 0);
	return xmalloc((size_t)N * sizeof(float));
}

static void del(float* vec)
{
	free(vec);
}

static void copy(long N, float* dst, const float* src)
{
	for (long i = 0; i < N; i++)
		dst[i] = src[i];
}

static void float2double(long N, double* dst, const float* src)
{
	for (long i = 0; i < N; i++)
		dst[i] = src[i];
}

static void double2float(long N, float* dst, const double* src)
{
	for (long i = 0; i < N; i++)
		dst[i] = src[i];
}

/*
 * Set vector to all-zeros
 *
 * @param N vector length
 * @param vec vector
 */
static void clear(long N, float* vec)
{
	for (long i = 0; i < N; i++)
		vec[i] = 0.;
}

static double dot(long N, const float* vec1, const float* vec2)
{
	double res = 0.;

	for (long i = 0; i < N; i++)
		res += vec1[i] * vec2[i];
	//res = fma((double)vec1[i], (double)vec2[i], res);

	return res;
}

/**
 * Compute l2 norm of vec
 *
 * @param N vector length
 * @param vec vector
 */
static double norm(long N, const float* vec)
{
	double res = 0.;

	for (long i = 0; i < N; i++)
		res += vec[i] * vec[i];
	//res = fma((double)vec[i], (double)vec[i], res);

	return sqrt(res);
}


/**
 * Compute l1 norm of vec
 *
 * @param N vector length
 * @param vec vector
 */
static double asum(long N, const float* vec)
{
	double res = 0.;

	for (long i = 0; i < N; i++)
		res += fabsf(vec[i]);

	return res;
}


/**
 * Compute l1 norm of complex vec
 *
 * @param N vector length
 * @param vec vector
 */
static double zl1norm(long N, const complex float* vec)
{
	double res = 0.;

	for (long i = 0; i < N; i++)
		res += cabsf(vec[i]);

	return res;
}

static void axpy(long N, float* dst, float alpha, const float* src)
{
	if (0. != alpha)
	for (long i = 0; i < N; i++)
		dst[i] += alpha * src[i];
//		dst[i] = fmaf(alpha, src[i], dst[i]);
}

static void xpay(long N, float beta, float* dst, const float* src)
{
	for (long i = 0; i < N; i++)
		dst[i] = dst[i] * beta + src[i];
//		dst[i] = fmaf(beta, dst[i], src[i]);
}

static void smul(long N, float alpha, float* dst, const float* src)
{
	for (long i = 0; i < N; i++)
		dst[i] = alpha * src[i];
	//dst[i] = fmaf(alpha, src[i], 0.f);
}

static void add(long N, float* dst, const float* src1, const float* src2)
{
#if 1
	if (dst == src1) {

		for (long i = 0; i < N; i++)
			dst[i] += src2[i];
	} else
#endif
	for (long i = 0; i < N; i++)
		dst[i] = src1[i] + src2[i];
}

static void sub(long N, float* dst, const float* src1, const float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] = src1[i] - src2[i];
}

static void mul(long N, float* dst, const float* src1, const float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] = src1[i] * src2[i];
}

static void vec_div(long N, float* dst, const float* src1, const float* src2)
{
	for (long i = 0; i < N; i++)
		//dst[i] = src1[i] / src2[i];
		dst[i] = (src2[i] == 0) ? 0.f : src1[i] / src2[i];
}

static void fmac(long N, float* dst, const float* src1, const float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] += src1[i] * src2[i];
	//dst[i] = fmaf(src1[i], src2[i], dst[i]);
		
}

static void fmac2(long N, double* dst, const float* src1, const float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] += src1[i] * src2[i];
}


static void zmul(long N, complex float* dst, const complex float* src1, const complex float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] = src1[i] * src2[i];
}

static void zdiv(long N, complex float* dst, const complex float* src1, const complex float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] = (src2[i] == 0) ? 0.f : src1[i] / src2[i];
}

static void zpow(long N, complex float* dst, const complex float* src1, const complex float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] = cpowf(src1[i], src2[i]);
}

static void zfmac(long N, complex float* dst, const complex float* src1, const complex float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] += src1[i] * src2[i];
}

static void zfmac2(long N, complex double* dst, const complex float* src1, const complex float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] += src1[i] * src2[i];
}

static void zmulc(long N, complex float* dst, const complex float* src1, const complex float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] = src1[i] * conjf(src2[i]);
}

static void zfmacc(long N, complex float* dst, const complex float* src1, const complex float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] += src1[i] * conjf(src2[i]);
}

static void zfmacc2(long N, complex double* dst, const complex float* src1, const complex float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] += src1[i] * conjf(src2[i]);
}

static void zconj(long N, complex float* dst, const complex float* src)
{
	for (long i = 0; i < N; i++)
		dst[i] = conjf(src[i]);
}

static void zcmp(long N, complex float* dst, const complex float* src1, const complex float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] = (src1[i] == src2[i]) ? 1. : 0.;
}

static void zdiv_reg(long N, complex float* dst, const complex float* src1, const complex float* src2, complex float lambda)
{
	for (long i = 0; i < N; i++)
		dst[i] = (src2[i] == 0) ? 0.f : src1[i] / (lambda + src2[i]);
}

static void zphsr(long N, complex float* dst, const complex float* src)
{
	for (long i = 0; i < N; i++) {

		float s = cabsf(src[i]);

		/* Note: the comparison (0 == src[i]) is not enough with `--fast-math`
		 * with gcc 4.4.3 (but seems to work for 4.7.3, different computer)
		 * Test:
		 * complex float a = FLT_MIN;
		 * complex float c = a / cabsf(a);
		 * assert(!(isnan(creal(c)) || isnan(cimag(c))));
		 */

		dst[i] = (0. == s) ? 1. : (src[i] / s);
	}
}

static void zexpj(long N, complex float* dst, const complex float* src)
{
	for (long i = 0; i < N; i++)
		dst[i] = cexpf(1.I * src[i]);
}

static void zarg(long N, complex float* dst, const complex float* src)
{
	for (long i = 0; i < N; i++)
		dst[i] = cargf(src[i]);
}


static void max(long N, float* dst, const float* src1, const float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] = MAX(src1[i], src2[i]);
}


static void min(long N, float* dst, const float* src1, const float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] = MIN(src1[i], src2[i]);
}


static void vec_pow(long N, float* dst, const float* src1, const float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] = powf(src1[i], src2[i]);
}


static void vec_sqrt(long N, float* dst, const float* src)
{
	for (long i = 0; i < N; i++)
		dst[i] = sqrtf(src[i]);
}


static void vec_le(long N, float* dst, const float* src1, const float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] = (src1[i] <= src2[i]);
}

static void vec_ge(long N, float* dst, const float* src1, const float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] = (src1[i] >= src2[i]);
}

/**
 * Step (1) of soft thesholding, y = ST(x, lambda).
 * Only computes the residual, resid = MAX( (abs(x) - lambda)/abs(x)), 0 )
 *
 * @param N number of elements
 * @param lambda threshold parameter
 * @param d pointer to destination, resid
 * @param x pointer to input
 */
static void zsoftthresh_half(long N, float lambda, complex float* d, const complex float* x)
{
	for (long i = 0; i < N; i++) {

		float norm = cabsf(x[i]);
		float red = norm - lambda;
		d[i] = (red > 0.) ? (red / norm) : 0.;
	}
}


static void zsoftthresh(long N, float lambda, complex float* d, const complex float* x)
{
	for (long i = 0; i < N; i++) {

		float norm = cabsf(x[i]);
		float red = norm - lambda;
		d[i] = (red > 0.) ? (red / norm) * x[i]: 0.;
	}
}



static void softthresh_half(long N, float lambda, float* d, const float* x)
{
	for (long i = 0; i < N; i++) {

		float norm = fabsf(x[i]);
		float red = norm - lambda;
		d[i] = (red > 0.) ? (red / norm) : 0.;
	}
}



static void softthresh(long N, float lambda, float* d, const float* x)
{
	for (long i = 0; i < N; i++) {

		float norm = fabsf(x[i]);
		float red = norm - lambda;
		d[i] = (red > 0.) ? (red / norm) * x[i] : 0.;
	}
}


static void swap(long N, float* a, float* b)
{
	for (long i = 0; i < N; i++) {

		float tmp = a[i];
		a[i] = b[i];
		b[i] = tmp;
	}
}


// identical copy in num/fft.c
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

static void zfftmod(long N, complex float* dst, const complex float* src, unsigned int n, bool inv, double phase)
{
	for (long i = 0; i < N; i++)
		for (unsigned int j = 0; j < n; j++)
			dst[i * n + j] = src[i * n + j] * fftmod_phase2(n, j, inv, phase);
}



/*
 * If you add functions here, please also add to gpuops.c/gpukrnls.cu
 */
const struct vec_ops cpu_ops = {

	.float2double = float2double,
	.double2float = double2float,
	.dot = dot,
	.asum = asum,
	.zl1norm = zl1norm,

	.add = add,
	.sub = sub,
	.mul = mul,
	.div = vec_div,
	.fmac = fmac,
	.fmac2 = fmac2,

	.axpy = axpy,

	.pow = vec_pow,
	.sqrt = vec_sqrt,

	.le = vec_le,
	.ge = vec_ge,

	.zmul = zmul,
	.zdiv = zdiv,
	.zfmac = zfmac,
	.zfmac2 = zfmac2,
	.zmulc = zmulc,
	.zfmacc = zfmacc,
	.zfmacc2 = zfmacc2,

	.zpow = zpow,
	.zphsr = zphsr,
	.zconj = zconj,
	.zexpj = zexpj,
	.zarg = zarg,

	.zcmp = zcmp,
	.zdiv_reg = zdiv_reg,
	.zfftmod = zfftmod,

	.max = max,
	.min = min,

	.zsoftthresh = zsoftthresh,
	.zsoftthresh_half = zsoftthresh_half,
	.softthresh = softthresh,
	.softthresh_half = softthresh_half,
};



// defined in iter/vec.h
struct vec_iter_s {

	float* (*allocate)(long N);
	void (*del)(float* x);
	void (*clear)(long N, float* x);
	void (*copy)(long N, float* a, const float* x);
	void (*swap)(long N, float* a, float* x);

	double (*norm)(long N, const float* x);
	double (*dot)(long N, const float* x, const float* y);

	void (*sub)(long N, float* a, const float* x, const float* y);
	void (*add)(long N, float* a, const float* x, const float* y);

	void (*smul)(long N, float alpha, float* a, const float* x);
	void (*xpay)(long N, float alpha, float* a, const float* x);
	void (*axpy)(long N, float* a, float alpha, const float* x);
};


extern const struct vec_iter_s cpu_iter_ops;
const struct vec_iter_s cpu_iter_ops = {

	.allocate = allocate,
	.del = del,
	.clear = clear,
	.copy = copy,
	.dot = dot,
	.norm = norm,
	.axpy = axpy,
	.xpay = xpay,
	.smul = smul,
	.add = add,
	.sub = sub,
	.swap = swap,
};


