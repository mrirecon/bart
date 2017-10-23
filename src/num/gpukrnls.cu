/* Copyright 2013-2017. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-03-24 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2015,2017 Jon Tamir <jtamir@eecs.berkeley.edu>
 *
 * 
 * This file defines basic operations on vectors of floats/complex floats
 * for operations on the GPU. See the CPU version (vecops.c) for more
 * information.
 */

#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>

#include "num/gpukrnls.h"

#if 1
// see Dara's src/calib/calibcu.cu for how to get
// runtime info


// limited by hardware to 1024 on most devices
// should be a multiple of 32 (warp size)
#define BLOCKSIZE 1024

static int blocksize(int N)
{
	return BLOCKSIZE;
}

static int gridsize(int N)
{
	return (N + BLOCKSIZE - 1) / BLOCKSIZE;
}
#else
// http://stackoverflow.com/questions/5810447/cuda-block-and-grid-size-efficiencies

#define WARPSIZE 32
#define MAXBLOCKS (16 * 8)
// 16 multi processor times 8 blocks

#define MIN(x, y) ((x < y) ? (x) : (y))
#define MAX(x, y) ((x > y) ? (x) : (y))

static int blocksize(int N)
{
	int warps_total = (N + WARPSIZE - 1) / WARPSIZE;
	int warps_block = MAX(1, MIN(4, warps_total));
	
	return WARPSIZE * warps_block;
}

static int gridsize(int N)
{
	int warps_total = (N + WARPSIZE - 1) / WARPSIZE;
	int warps_block = MAX(1, MIN(4, warps_total));
	
	return MIN(MAXBLOCKS, MAX(1, warps_total / warps_block));
}
#endif

__global__ void kern_float2double(int N, double* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = src[i];
}

extern "C" void cuda_float2double(long N, double* dst, const float* src)
{
	kern_float2double<<<gridsize(N), blocksize(N)>>>(N, dst, src);
}

__global__ void kern_double2float(int N, float* dst, const double* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = src[i];
}

extern "C" void cuda_double2float(long N, float* dst, const double* src)
{
	kern_double2float<<<gridsize(N), blocksize(N)>>>(N, dst, src);
}

__global__ void kern_xpay(int N, float beta, float* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = dst[i] * beta + src[i];
}




extern "C" void cuda_xpay(long N, float beta, float* dst, const float* src)
{
	kern_xpay<<<gridsize(N), blocksize(N)>>>(N, beta, dst, src);
}


__global__ void kern_axpbz(int N, float* dst, const float a1, const float* src1, const float a2, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = a1 * src1[i] + a2 * src2[i];
}


extern "C" void cuda_axpbz(long N, float* dst, const float a1, const float* src1, const float a2, const float* src2)
{
	kern_axpbz<<<gridsize(N), blocksize(N)>>>(N, dst, a1, src1, a2, src2);
}


__global__ void kern_smul(int N, float alpha, float* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = alpha * src[i];
}

extern "C" void cuda_smul(long N, float alpha, float* dst, const float* src)
{
	kern_smul<<<gridsize(N), blocksize(N)>>>(N, alpha, dst, src);
}


typedef void (*cuda_3op_f)(int N, float* dst, const float* src1, const float* src2);

extern "C" void cuda_3op(cuda_3op_f krn, int N, float* dst, const float* src1, const float* src2)
{
	krn<<<gridsize(N), blocksize(N)>>>(N, dst, src1, src2);
}

__global__ void kern_add(int N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = src1[i] + src2[i];
}

extern "C" void cuda_add(long N, float* dst, const float* src1, const float* src2)
{
	cuda_3op(kern_add, N, dst, src1, src2);
}

__global__ void kern_sub(int N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = src1[i] - src2[i];
}

extern "C" void cuda_sub(long N, float* dst, const float* src1, const float* src2)
{
	cuda_3op(kern_sub, N, dst, src1, src2);
}


__global__ void kern_mul(int N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = src1[i] * src2[i];
}

extern "C" void cuda_mul(long N, float* dst, const float* src1, const float* src2)
{
	cuda_3op(kern_mul, N, dst, src1, src2);
}

__global__ void kern_div(int N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = src1[i] / src2[i];
}

extern "C" void cuda_div(long N, float* dst, const float* src1, const float* src2)
{
	cuda_3op(kern_div, N, dst, src1, src2);
}


__global__ void kern_fmac(int N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] += src1[i] * src2[i];
}

extern "C" void cuda_fmac(long N, float* dst, const float* src1, const float* src2)
{
	cuda_3op(kern_fmac, N, dst, src1, src2);
}


__global__ void kern_fmac2(int N, double* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] += src1[i] * src2[i];
}

extern "C" void cuda_fmac2(long N, double* dst, const float* src1, const float* src2)
{
	kern_fmac2<<<gridsize(N), blocksize(N)>>>(N, dst, src1, src2);
}


__global__ void kern_zmul(int N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = cuCmulf(src1[i], src2[i]);
}

extern "C" void cuda_zmul(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zmul<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}

__global__ void kern_zdiv(int N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = cuCdivf(src1[i], src2[i]);
}

extern "C" void cuda_zdiv(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zdiv<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}


__global__ void kern_zfmac(int N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = cuCaddf(dst[i], cuCmulf(src1[i], src2[i]));
}


extern "C" void cuda_zfmac(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zfmac<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}


__global__ void kern_zfmac2(int N, cuDoubleComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = cuCadd(dst[i], cuComplexFloatToDouble(cuCmulf(src1[i], src2[i])));
}

extern "C" void cuda_zfmac2(long N, _Complex double* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zfmac2<<<gridsize(N), blocksize(N)>>>(N, (cuDoubleComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}


__global__ void kern_zmulc(int N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = cuCmulf(src1[i], cuConjf(src2[i]));
}

extern "C" void cuda_zmulc(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zmulc<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}

__global__ void kern_zfmacc(int N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = cuCaddf(dst[i], cuCmulf(src1[i], cuConjf(src2[i])));
}


extern "C" void cuda_zfmacc(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zfmacc<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}


__global__ void kern_zfmacc2(int N, cuDoubleComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = cuCadd(dst[i], cuComplexFloatToDouble(cuCmulf(src1[i], cuConjf(src2[i]))));
}


extern "C" void cuda_zfmacc2(long N, _Complex double* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zfmacc2<<<gridsize(N), blocksize(N)>>>(N, (cuDoubleComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}


__global__ void kern_pow(int N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = powf(src1[i], src2[i]);
}

extern "C" void cuda_pow(long N, float* dst, const float* src1, const float* src2)
{
	cuda_3op(kern_pow, N, dst, src1, src2);
}

__device__ cuDoubleComplex zexpD(cuDoubleComplex x)
{
	double sc = exp(cuCreal(x));
	double si;
	double co;
	sincos(cuCimag(x), &si, &co);
	return make_cuDoubleComplex(sc * co, sc * si);
}

__device__ cuFloatComplex zexp(cuFloatComplex x)
{
	float sc = expf(cuCrealf(x));
	float si;
	float co;
	sincosf(cuCimagf(x), &si, &co);
	return make_cuFloatComplex(sc * co, sc * si);
}

__device__ float zarg(cuFloatComplex x)
{
	return atan2(cuCimagf(x), cuCrealf(x));
}

__device__ cuFloatComplex zlog(cuFloatComplex x)
{
	return make_cuFloatComplex(log(cuCabsf(x)), zarg(x));
}


// x^y = e^{y ln(x)} = e^{y 
__device__ cuFloatComplex zpow(cuFloatComplex x, cuFloatComplex y)
{
	return zexp(cuCmulf(y, zlog(x)));
}

__global__ void kern_zpow(int N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = zpow(src1[i], src2[i]);
}

extern "C" void cuda_zpow(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zpow<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}


__global__ void kern_sqrt(int N, float* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = sqrtf(fabs(src[i]));
}

extern "C" void cuda_sqrt(long N, float* dst, const float* src)
{
	kern_sqrt<<<gridsize(N), blocksize(N)>>>(N, dst, src);
}


__global__ void kern_zconj(int N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = cuConjf(src[i]);
}

extern "C" void cuda_zconj(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zconj<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}


__global__ void kern_zcmp(int N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex(((cuCrealf(src1[i]) == cuCrealf(src2[i])) && (cuCimagf(src1[i]) == cuCimagf(src2[i]))) ? 1. : 0, 0.);
}

extern "C" void cuda_zcmp(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zcmp<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}

__global__ void kern_zdiv_reg(int N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2, cuFloatComplex lambda)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = cuCdivf(src1[i], cuCaddf(src2[i], lambda));
}

extern "C" void cuda_zdiv_reg(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2, _Complex float lambda)
{
	kern_zdiv_reg<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2, make_cuFloatComplex(__real(lambda), __imag(lambda)));
}


__global__ void kern_zphsr(int N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {

		float abs = cuCabsf(src[i]); // moved out, otherwise it triggers a compiler error in nvcc
		dst[i] = (0. == abs) ? make_cuFloatComplex(1., 0.) : (cuCdivf(src[i], make_cuFloatComplex(abs, 0.)));
	}
}

extern "C" void cuda_zphsr(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zphsr<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}



__global__ void kern_zexpj(int N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {

		float abs = cuCabsf(src[i]); // moved out, otherwise it triggers a compiler error in nvcc
		dst[i] = zexp(make_cuFloatComplex(0., abs));
	}
}

extern "C" void cuda_zexpj(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zexpj<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}



__global__ void kern_zarg(int N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex(zarg(src[i]), 0.);
}

extern "C" void cuda_zarg(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zarg<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}


/**
 * (GPU) Step (1) of soft thesholding, y = ST(x, lambda).
 * Only computes the residual, resid = MAX( (abs(x) - lambda)/abs(x)), 0 )
 *
 * @param N number of elements
 * @param lambda threshold parameter
 * @param d pointer to destination, resid
 * @param x pointer to input
 */
__global__ void kern_zsoftthresh_half(int N, float lambda, cuFloatComplex* d, const cuFloatComplex* x)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {

		float norm = cuCabsf(x[i]);
		float red = norm - lambda;
		//d[i] = (red > 0.) ? (cuCmulf(make_cuFloatComplex(red / norm, 0.), x[i])) : make_cuFloatComplex(0., 0.);
		d[i] = (red > 0.) ? make_cuFloatComplex(red / norm, 0.) : make_cuFloatComplex(0., 0.);
	}
}

extern "C" void cuda_zsoftthresh_half(long N, float lambda, _Complex float* d, const _Complex float* x)
{
	kern_zsoftthresh_half<<<gridsize(N), blocksize(N)>>>(N, lambda, (cuFloatComplex*)d, (const cuFloatComplex*)x);
}


__global__ void kern_zsoftthresh(int N, float lambda, cuFloatComplex* d, const cuFloatComplex* x)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {

		float norm = cuCabsf(x[i]);
		float red = norm - lambda;
		d[i] = (red > 0.) ? (cuCmulf(make_cuFloatComplex(red / norm, 0.), x[i])) : make_cuFloatComplex(0., 0.);
	}
}


extern "C" void cuda_zsoftthresh(long N, float lambda, _Complex float* d, const _Complex float* x)
{
	kern_zsoftthresh<<<gridsize(N), blocksize(N)>>>(N, lambda, (cuFloatComplex*)d, (const cuFloatComplex*)x);
}


__global__ void kern_softthresh_half(int N, float lambda, float* d, const float* x)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {

		float norm = fabsf(x[i]);
		float red = norm - lambda;

		d[i] = (red > 0.) ? (red / norm) : 0.;
	}
}

extern "C" void cuda_softthresh_half(long N, float lambda, float* d, const float* x)
{
	kern_softthresh_half<<<gridsize(N), blocksize(N)>>>(N, lambda, d, x);
}


__global__ void kern_softthresh(int N, float lambda, float* d, const float* x)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {

		float norm = fabsf(x[i]);
		float red = norm - lambda;

		d[i] = (red > 0.) ? (red / norm * x[i]) : 0.;
	}
}

extern "C" void cuda_softthresh(long N, float lambda, float* d, const float* x)
{
	kern_softthresh<<<gridsize(N), blocksize(N)>>>(N, lambda, d, x);
}


__global__ void kern_zreal(int N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex(cuCrealf(src[i]), 0.);
}

extern "C" void cuda_zreal(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zreal<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}


__global__ void kern_le(int N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = (src1[i] <= src2[i]);
}

extern "C" void cuda_le(long N, float* dst, const float* src1, const float* src2)
{
	kern_le<<<gridsize(N), blocksize(N)>>>(N, dst, src1, src2);
}

__global__ void kern_ge(int N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = (src1[i] >= src2[i]);
}

extern "C" void cuda_ge(long N, float* dst, const float* src1, const float* src2)
{
	kern_ge<<<gridsize(N), blocksize(N)>>>(N, dst, src1, src2);
}

__device__ cuFloatComplex cuDouble2Float(cuDoubleComplex x)
{
	return make_cuFloatComplex(cuCreal(x), cuCimag(x));
}

__device__ cuDoubleComplex cuFloat2Double(cuFloatComplex x)
{
	return make_cuDoubleComplex(cuCrealf(x), cuCimagf(x));
}

// identical copy in num/fft.c
__device__ double fftmod_phase(long length, int j)
{
	long center1 = length / 2;
	double shift = (double)center1 / (double)length;
	return ((double)j - (double)center1 / 2.) * shift;
}

__device__ cuDoubleComplex fftmod_phase2(long n, int j, bool inv, double phase)
{
	phase += fftmod_phase(n, j);
	double rem = phase - floor(phase);
	double sgn = inv ? -1. : 1.;
#if 1
	if (rem == 0.)
		return make_cuDoubleComplex(1., 0.);

	if (rem == 0.5)
		return make_cuDoubleComplex(-1., 0.);

	if (rem == 0.25)
		return make_cuDoubleComplex(0., sgn);

	if (rem == 0.75)
		return make_cuDoubleComplex(0., -sgn);
#endif
	return zexpD(make_cuDoubleComplex(0., M_PI * 2. * sgn * rem));
}

__global__ void kern_zfftmod(int N, cuFloatComplex* dst, const cuFloatComplex* src, unsigned int n, _Bool inv, double phase)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		for (int j = 0; j < n; j++)
			dst[i * n + j] = cuDouble2Float(cuCmul(fftmod_phase2(n, j, inv, phase),
						 cuFloat2Double(src[i * n + j])));
}

extern "C" void cuda_zfftmod(long N, _Complex float* dst, const _Complex float* src, unsigned int n, _Bool inv, double phase)
{
	kern_zfftmod<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src, n, inv, phase);
}


#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


__global__ void kern_max(int N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = MAX(src1[i], src2[i]);
}


extern "C" void cuda_max(long N, float* dst, const float* src1, const float* src2)
{
	kern_max<<<gridsize(N), blocksize(N)>>>(N, dst, src1, src2);
}


__global__ void kern_min(int N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = MIN(src1[i], src2[i]);
}


extern "C" void cuda_min(long N, float* dst, const float* src1, const float* src2)
{
	kern_min<<<gridsize(N), blocksize(N)>>>(N, dst, src1, src2);
}


