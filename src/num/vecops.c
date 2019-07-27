/* Copyright 2013-2018. The Regents of the University of California.
 * Copyright 2016-2019. Martin Uecker.
 * Copyright 2017. University of Oxford.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2011-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014 Frank Ong <frankong@berkeley.edu>
 * 2014-2018 Jon Tamir <jtamir@eecs.berkeley.edu>
 * 2017 Sofia Dimoudi <sofia.dimoudi@cardiov.ox.ac.uk>
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
#include "misc/debug.h"

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
	xfree(vec);
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



// we should probably replace asum and zl1norm
static void zsum(long N, complex float* vec)
{
	complex float res = 0.;

	for (long i = 0; i < N; i++)
		res += vec[i];

	vec[0] = res;
}



static void axpbz(long N, float* dst, const float a1, const float* src1, const float a2, const float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] = a1 * src1[i] + a2 * src2[i];
}

static void axpy(long N, float* dst, float alpha, const float* src)
{
	axpbz(N, dst, 1., dst, alpha, src);
	//dst[i] = fmaf(alpha, src[i], dst[i]);
}

static void xpay(long N, float beta, float* dst, const float* src)
{
	axpbz(N, dst, beta, dst, 1., src);
	//dst[i] = fmaf(beta, dst[i], src[i]);
}


static void smul(long N, float alpha, float* dst, const float* src)
{
	axpbz(N, dst, 0., src, alpha, src);
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

static void zsmul(long N, complex float val, complex float* dst, const complex float* src1)
{
	for (long i = 0; i < N; i++)
		dst[i] = src1[i] * val;
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

static void zexp(long N, complex float* dst, const complex float* src)
{
	for (long i = 0; i < N; i++)
		dst[i] = cexpf(src[i]);
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

static void zabs(long N, complex float* dst, const complex float* src)
{
	for (long i = 0; i < N; i++)
		dst[i] = cabsf(src[i]);
}


static void zmax(long N, complex float* dst, const complex float* src1, const complex float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] = (crealf(src1[i]) > crealf(src2[i])) ? src1[i] : src2[i];
}


static void max(long N, float* dst, const float* src1, const float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] = MAX(src1[i], src2[i]);
}

static void smax(long N, float val, float* dst, const float* src1)
{
	for (long i = 0; i < N; i++)
		dst[i] = MAX(src1[i], val);
}


static void min(long N, float* dst, const float* src1, const float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] = MIN(src1[i], src2[i]);
}


static void zsmax(long N, float val, complex float* dst, const complex float* src)
{
	for (long i = 0; i < N; i++)
		dst[i] = MAX(crealf(src[i]), val);
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


static void vec_zle(long N, complex float* dst, const complex float* src1, const complex float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] = (crealf(src1[i]) <= crealf(src2[i]));
}


static void vec_le(long N, float* dst, const float* src1, const float* src2)
{
	for (long i = 0; i < N; i++)
		dst[i] = (src1[i] <= src2[i]);
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

/**
 * Return the absolute value of the kth largest array element
 * To be used for hard thresholding
 *
 * @param N number of elements
 * @param k the sorted element index to pick
 * @param ar the input complex array
 *
 * @returns the absolute value of the kth largest array element.
 *
 */

static float klargest_complex_partsort( unsigned int N,  unsigned int k, const complex float* ar)
{
	assert(k <= N);
	
	complex float* tmp =  (complex float*)xmalloc(N * sizeof(complex float));
	copy(2 * N, (float*)tmp, (float*)ar);
	
	float thr = quickselect_complex(tmp, N, k);

	xfree(tmp);

	return thr;
}

/**
 * Hard thesholding, y = HT(x, thr).
 * computes the thresholded vector, y = x * (abs(x) >= t(kmax))
 * 
 * @param N number of elements
 * @param k threshold parameter, index of kth largest element of sorted x 
 * @param d pointer to destination, y
 * @param x pointer to input
 */

static void zhardthresh(long N,  unsigned int k, complex float* d, const complex float* x)
{
	float thr = klargest_complex_partsort(N, k, x);
   
	for (long i = 0; i < N; i++) {

		float norm = cabsf(x[i]);
		d[i] = (norm > thr) ? x[i] : 0.;
	}
}

/**
 * Hard thesholding mask, m = HS(x, thr).
 * computes the non-zero complex support vector, m = 1.0 * (abs(x) >= t(kmax))
 * This mask should be applied by complex multiplication.
 *
 * @param N number of elements
 * @param k threshold parameter, index of kth largest element of sorted x
 * @param d pointer to destination
 * @param x pointer to input
 */

static void zhardthresh_mask(long N,  unsigned int k, complex float* d, const complex float* x)
{
	float thr = klargest_complex_partsort(N, k, x);

	for (long i = 0; i < N; i++) {

		float norm = cabsf(x[i]);
		d[i] = (norm > thr) ? 1. : 0.;
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
#if 1
	if (0 == n % 2) {

		complex float ph = fftmod_phase2(n, 0, inv, phase);

		for (long i = 0; i < N; i++)
			for (unsigned int j = 0; j < n; j++)
				dst[i * n + j] = src[i * n + j] * ((0 == j % 2) ? ph : -ph);

		return;
	}
#endif

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
	.zsum = zsum,
	.zl1norm = zl1norm,

	.add = add,
	.sub = sub,
	.mul = mul,
	.div = vec_div,
	.fmac = fmac,
	.fmac2 = fmac2,

	.smul = smul,

	.axpy = axpy,

	.pow = vec_pow,
	.sqrt = vec_sqrt,

	.zle = vec_zle,
	.le = vec_le,

	.zmul = zmul,
	.zdiv = zdiv,
	.zfmac = zfmac,
	.zfmac2 = zfmac2,
	.zmulc = zmulc,
	.zfmacc = zfmacc,
	.zfmacc2 = zfmacc2,

	.zsmax = zsmax,
	.zsmul = zsmul,

	.zpow = zpow,
	.zphsr = zphsr,
	.zconj = zconj,
	.zexpj = zexpj,
	.zexp = zexp,
	.zarg = zarg,
	.zabs = zabs,

	.zcmp = zcmp,
	.zdiv_reg = zdiv_reg,
	.zfftmod = zfftmod,

	.zmax = zmax,

	.smax = smax,
	.max = max,
	.min = min,

	.zsoftthresh = zsoftthresh,
	.zsoftthresh_half = zsoftthresh_half,
	.softthresh = softthresh,
	.softthresh_half = softthresh_half,
	.zhardthresh = zhardthresh,
	.zhardthresh_mask = zhardthresh_mask,
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
	void (*axpbz)(long N, float* out, const float a, const float* x, const float b, const float* z);
	void (*zmul)(long N, complex float* dst, const complex float* src1, const complex float* src2);
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
	.axpbz = axpbz,
	.smul = smul,
	.add = add,
	.sub = sub,
	.swap = swap,
	.zmul = zmul,
};


