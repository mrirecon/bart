/* Copyright 2013-2018. The Regents of the University of California.
 * Copyright 2017-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2015-2018 Jon Tamir <jtamir@eecs.berkeley.edu>
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

static long gridsize(long N)
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

static long gridsize(long N)
{
	int warps_total = (N + WARPSIZE - 1) / WARPSIZE;
	int warps_block = MAX(1, MIN(4, warps_total));

	return MIN(MAXBLOCKS, MAX(1, warps_total / warps_block));
}
#endif

__global__ void kern_float2double(long N, double* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = src[i];
}

extern "C" void cuda_float2double(long N, double* dst, const float* src)
{
	kern_float2double<<<gridsize(N), blocksize(N)>>>(N, dst, src);
}

__global__ void kern_double2float(long N, float* dst, const double* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = src[i];
}

extern "C" void cuda_double2float(long N, float* dst, const double* src)
{
	kern_double2float<<<gridsize(N), blocksize(N)>>>(N, dst, src);
}

__global__ void kern_xpay(long N, float beta, float* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = dst[i] * beta + src[i];
}

extern "C" void cuda_xpay(long N, float beta, float* dst, const float* src)
{
	kern_xpay<<<gridsize(N), blocksize(N)>>>(N, beta, dst, src);
}


__global__ void kern_axpbz(long N, float* dst, const float a1, const float* src1, const float a2, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = a1 * src1[i] + a2 * src2[i];
}


extern "C" void cuda_axpbz(long N, float* dst, const float a1, const float* src1, const float a2, const float* src2)
{
	kern_axpbz<<<gridsize(N), blocksize(N)>>>(N, dst, a1, src1, a2, src2);
}


__global__ void kern_smul(long N, float alpha, float* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = alpha * src[i];
}

extern "C" void cuda_smul(long N, float alpha, float* dst, const float* src)
{
	kern_smul<<<gridsize(N), blocksize(N)>>>(N, alpha, dst, src);
}


typedef void (*cuda_3op_f)(long N, float* dst, const float* src1, const float* src2);

extern "C" void cuda_3op(cuda_3op_f krn, int N, float* dst, const float* src1, const float* src2)
{
	krn<<<gridsize(N), blocksize(N)>>>(N, dst, src1, src2);
}

__global__ void kern_add(long N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = src1[i] + src2[i];
}

extern "C" void cuda_add(long N, float* dst, const float* src1, const float* src2)
{
	cuda_3op(kern_add, N, dst, src1, src2);
}

__global__ void kern_sadd(long N, float val, float* dst, const float* src1)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = src1[i] + val;
}

extern "C" void cuda_sadd(long N, float val, float* dst, const float* src1)
{
	kern_sadd<<<gridsize(N), blocksize(N)>>>(N, val, dst, src1);
}

__global__ void kern_zsadd(long N, cuFloatComplex val, cuFloatComplex* dst, const cuFloatComplex* src1)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = cuCaddf(src1[i], val);
}

extern "C" void cuda_zsadd(long N, _Complex float val, _Complex float* dst, const _Complex float* src1)
{
	kern_zsadd<<<gridsize(N), blocksize(N)>>>(N, make_cuFloatComplex(__real(val), __imag(val)), (cuFloatComplex*)dst, (const cuFloatComplex*)src1);
}

__global__ void kern_sub(long N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = src1[i] - src2[i];
}

extern "C" void cuda_sub(long N, float* dst, const float* src1, const float* src2)
{
	cuda_3op(kern_sub, N, dst, src1, src2);
}


__global__ void kern_mul(long N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = src1[i] * src2[i];
}

extern "C" void cuda_mul(long N, float* dst, const float* src1, const float* src2)
{
	cuda_3op(kern_mul, N, dst, src1, src2);
}

__global__ void kern_div(long N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = src1[i] / src2[i];
}

extern "C" void cuda_div(long N, float* dst, const float* src1, const float* src2)
{
	cuda_3op(kern_div, N, dst, src1, src2);
}


__global__ void kern_fmac(long N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] += src1[i] * src2[i];
}

extern "C" void cuda_fmac(long N, float* dst, const float* src1, const float* src2)
{
	cuda_3op(kern_fmac, N, dst, src1, src2);
}


__global__ void kern_fmac2(long N, double* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] += src1[i] * src2[i];
}

extern "C" void cuda_fmac2(long N, double* dst, const float* src1, const float* src2)
{
	kern_fmac2<<<gridsize(N), blocksize(N)>>>(N, dst, src1, src2);
}

__global__ void kern_zsmul(long N, cuFloatComplex val, cuFloatComplex* dst, const cuFloatComplex* src1)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = cuCmulf(src1[i], val);
}

extern "C" void cuda_zsmul(long N, _Complex float alpha, _Complex float* dst, const _Complex float* src1)
{
	kern_zsmul<<<gridsize(N), blocksize(N)>>>(N, make_cuFloatComplex(__real(alpha), __imag(alpha)), (cuFloatComplex*)dst, (const cuFloatComplex*)src1);
}


__global__ void kern_zmul(long N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = cuCmulf(src1[i], src2[i]);
}

extern "C" void cuda_zmul(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zmul<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}

__global__ void kern_zdiv(long N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride) {

		float abs = cuCabsf(src2[i]);

		dst[i] = (0. == abs) ? make_cuFloatComplex(0., 0.) : cuCdivf(src1[i], src2[i]);
	}
}

extern "C" void cuda_zdiv(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zdiv<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}


__global__ void kern_zfmac(long N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = cuCaddf(dst[i], cuCmulf(src1[i], src2[i]));
}


extern "C" void cuda_zfmac(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zfmac<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}


__global__ void kern_zfmac2(long N, cuDoubleComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = cuCadd(dst[i], cuComplexFloatToDouble(cuCmulf(src1[i], src2[i])));
}

extern "C" void cuda_zfmac2(long N, _Complex double* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zfmac2<<<gridsize(N), blocksize(N)>>>(N, (cuDoubleComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}


__global__ void kern_zmulc(long N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = cuCmulf(src1[i], cuConjf(src2[i]));
}

extern "C" void cuda_zmulc(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zmulc<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}

__global__ void kern_zfmacc(long N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = cuCaddf(dst[i], cuCmulf(src1[i], cuConjf(src2[i])));
}


extern "C" void cuda_zfmacc(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zfmacc<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}


__global__ void kern_zfmacc2(long N, cuDoubleComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = cuCadd(dst[i], cuComplexFloatToDouble(cuCmulf(src1[i], cuConjf(src2[i]))));
}


extern "C" void cuda_zfmacc2(long N, _Complex double* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zfmacc2<<<gridsize(N), blocksize(N)>>>(N, (cuDoubleComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}


__global__ void kern_pow(long N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
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

__device__ cuFloatComplex zsin(cuFloatComplex x)
{
	float si;
	float co;
	float sih;
	float coh;
	sincosf(cuCrealf(x), &si, &co);
	sih = sinhf(cuCimagf(x));
	coh = coshf(cuCimagf(x));
	return make_cuFloatComplex(si * coh , co * sih);
}

__device__ cuFloatComplex zcos(cuFloatComplex x)
{
	float si;
	float co;
	float sih;
	float coh;
	sincosf(cuCrealf(x), &si, &co);
	sih = sinhf(cuCimagf(x));
	coh = coshf(cuCimagf(x));
	return make_cuFloatComplex(co * coh , -si * sih);
}

__device__ cuFloatComplex zsinh(cuFloatComplex x)
{
	float si_i;
	float co_i;
	float sih_r;
	float coh_r;
	sincosf(cuCimagf(x), &si_i, &co_i);
	sih_r = sinhf(cuCrealf(x));
	coh_r = coshf(cuCrealf(x));
	return make_cuFloatComplex(sih_r * co_i , coh_r * si_i);
}

__device__ cuFloatComplex zcosh(cuFloatComplex x)
{
	float si_i;
	float co_i;
	float sih_r;
	float coh_r;
	sincosf(cuCimagf(x), &si_i, &co_i);
	sih_r = sinhf(cuCrealf(x));
	coh_r = coshf(cuCrealf(x));
	return make_cuFloatComplex(coh_r * co_i , sih_r * si_i);
}

__device__ float zarg(cuFloatComplex x)
{
	return atan2(cuCimagf(x), cuCrealf(x));
}

__device__ float zabs(cuFloatComplex x)
{
	return cuCabsf(x);
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

__global__ void kern_zpow(long N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = zpow(src1[i], src2[i]);
}

extern "C" void cuda_zpow(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zpow<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}


__global__ void kern_sqrt(long N, float* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = sqrtf(fabs(src[i]));
}

extern "C" void cuda_sqrt(long N, float* dst, const float* src)
{
	kern_sqrt<<<gridsize(N), blocksize(N)>>>(N, dst, src);
}


__global__ void kern_zconj(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = cuConjf(src[i]);
}

extern "C" void cuda_zconj(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zconj<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}


__global__ void kern_zcmp(long N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex(((cuCrealf(src1[i]) == cuCrealf(src2[i])) && (cuCimagf(src1[i]) == cuCimagf(src2[i]))) ? 1. : 0, 0.);
}

extern "C" void cuda_zcmp(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zcmp<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}

__global__ void kern_zdiv_reg(long N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2, cuFloatComplex lambda)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = cuCdivf(src1[i], cuCaddf(src2[i], lambda));
}

extern "C" void cuda_zdiv_reg(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2, _Complex float lambda)
{
	kern_zdiv_reg<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2, make_cuFloatComplex(__real(lambda), __imag(lambda)));
}


__global__ void kern_zphsr(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride) {

		float abs = cuCabsf(src[i]); // moved out, otherwise it triggers a compiler error in nvcc
		dst[i] = (0. == abs) ? make_cuFloatComplex(1., 0.) : (cuCdivf(src[i], make_cuFloatComplex(abs, 0.)));
	}
}

extern "C" void cuda_zphsr(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zphsr<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}


__global__ void kern_zexp(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = zexp(src[i]);
}

extern "C" void cuda_zexp(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zexp<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}


__global__ void kern_zexpj(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride) {

		float re = cuCrealf(src[i]); // moved out, otherwise it triggers a compiler error in nvcc
		float im = cuCimagf(src[i]); // moved out, otherwise it triggers a compiler error in nvcc
		dst[i] = zexp(make_cuFloatComplex(-im, re));
	}
}

extern "C" void cuda_zexpj(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zexpj<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}

__global__ void kern_zlog(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride){

		float abs = cuCabsf(src[i]);
		dst[i] = (0. == abs) ? make_cuFloatComplex(0., 0.) : zlog(src[i]);
	}
}

extern "C" void cuda_zlog(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zlog<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}

__global__ void kern_zarg(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex(zarg(src[i]), 0.);
}

extern "C" void cuda_zarg(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zarg<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}

__global__ void kern_zsin(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = zsin(src[i]);
}

extern "C" void cuda_zsin(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zsin<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}

__global__ void kern_zcos(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = zcos(src[i]);
}

extern "C" void cuda_zcos(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zcos<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}

__global__ void kern_zsinh(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = zsinh(src[i]);
}

extern "C" void cuda_zsinh(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zsinh<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}

__global__ void kern_zcosh(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = zcosh(src[i]);
}

extern "C" void cuda_zcosh(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zcosh<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}


__global__ void kern_zabs(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex(zabs(src[i]), 0.);
}

extern "C" void cuda_zabs(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zabs<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}

__global__ void kern_exp(long N, float* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = expf(src[i]);
}

extern "C" void cuda_exp(long N, float* dst, const float* src)
{
	kern_exp<<<gridsize(N), blocksize(N)>>>(N, dst, src);
}

__global__ void kern_log(long N, float* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = (0. == src[i]) ? 0. : logf(src[i]);
}

extern "C" void cuda_log(long N, float* dst, const float* src)
{
	kern_log<<<gridsize(N), blocksize(N)>>>(N, dst, src);
}


__global__ void kern_zatanr(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex(atan(cuCrealf(src[i])), 0.);
}

extern "C" void cuda_zatanr(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zatanr<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}

__global__ void kern_zacos(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex(acosf(cuCrealf(src[i])), 0.);
}

extern "C" void cuda_zacos(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zacos<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
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
__global__ void kern_zsoftthresh_half(long N, float lambda, cuFloatComplex* d, const cuFloatComplex* x)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride) {

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


__global__ void kern_zsoftthresh(long N, float lambda, cuFloatComplex* d, const cuFloatComplex* x)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride) {

		float norm = cuCabsf(x[i]);
		float red = norm - lambda;
		d[i] = (red > 0.) ? (cuCmulf(make_cuFloatComplex(red / norm, 0.), x[i])) : make_cuFloatComplex(0., 0.);
	}
}


extern "C" void cuda_zsoftthresh(long N, float lambda, _Complex float* d, const _Complex float* x)
{
	kern_zsoftthresh<<<gridsize(N), blocksize(N)>>>(N, lambda, (cuFloatComplex*)d, (const cuFloatComplex*)x);
}


__global__ void kern_softthresh_half(long N, float lambda, float* d, const float* x)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride) {

		float norm = fabsf(x[i]);
		float red = norm - lambda;

		d[i] = (red > 0.) ? (red / norm) : 0.;
	}
}

extern "C" void cuda_softthresh_half(long N, float lambda, float* d, const float* x)
{
	kern_softthresh_half<<<gridsize(N), blocksize(N)>>>(N, lambda, d, x);
}


__global__ void kern_softthresh(long N, float lambda, float* d, const float* x)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride) {

		float norm = fabsf(x[i]);
		float red = norm - lambda;

		d[i] = (red > 0.) ? (red / norm * x[i]) : 0.;
	}
}

extern "C" void cuda_softthresh(long N, float lambda, float* d, const float* x)
{
	kern_softthresh<<<gridsize(N), blocksize(N)>>>(N, lambda, d, x);
}


__global__ void kern_zreal(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex(cuCrealf(src[i]), 0.);
}

extern "C" void cuda_zreal(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zreal<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}


__global__ void kern_zle(long N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex((cuCrealf(src1[i]) <= cuCrealf(src2[i])), 0.);
}

extern "C" void cuda_zle(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zle<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}


__global__ void kern_le(long N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = (src1[i] <= src2[i]);
}

extern "C" void cuda_le(long N, float* dst, const float* src1, const float* src2)
{
	kern_le<<<gridsize(N), blocksize(N)>>>(N, dst, src1, src2);
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

__global__ void kern_zfftmod(long N, cuFloatComplex* dst, const cuFloatComplex* src, unsigned int n, _Bool inv, double phase)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
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


__global__ void kern_zmax(long N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride) {

		dst[i].x = MAX(src1[i].x, src2[i].x);
		dst[i].y = 0.0;
	}
}


extern "C" void cuda_zmax(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zmax<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}



__global__ void kern_smax(long N, float val, float* dst, const float* src1)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = MAX(src1[i], val);
}


extern "C" void cuda_smax(long N, float val, float* dst, const float* src1)
{
	kern_smax<<<gridsize(N), blocksize(N)>>>(N, val, dst, src1);
}


__global__ void kern_max(long N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = MAX(src1[i], src2[i]);
}


extern "C" void cuda_max(long N, float* dst, const float* src1, const float* src2)
{
	kern_max<<<gridsize(N), blocksize(N)>>>(N, dst, src1, src2);
}


__global__ void kern_min(long N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = MIN(src1[i], src2[i]);
}


extern "C" void cuda_min(long N, float* dst, const float* src1, const float* src2)
{
	kern_min<<<gridsize(N), blocksize(N)>>>(N, dst, src1, src2);
}

__global__ void kern_zsmax(long N, float val, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride) {

		dst[i].x = MAX(src[i].x, val);
		dst[i].y = 0.0;
	}
}

extern "C" void cuda_zsmax(long N, float alpha, _Complex float* dst, const _Complex float* src)
{
	kern_zsmax<<<gridsize(N), blocksize(N)>>>(N, alpha, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}

__global__ void kern_reduce_zsum(long N, cuFloatComplex* dst)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	cuFloatComplex sum = make_cuFloatComplex(0., 0.);

	for (long i = start; i < N; i += stride)
		sum = cuCaddf(sum, dst[i]);

	if (start < N)
		dst[start] = sum;
}

extern "C" void cuda_zsum(long N, _Complex float* dst)
{
	int B = blocksize(N);

	while (N > 1) {

		kern_reduce_zsum<<<1, B>>>(N, (cuFloatComplex*)dst);
		N = MIN(B, N);
		B /= 32;
	}
}


__global__ void kern_pdf_gauss(long N, float mu, float sig, float* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = expf(- (src[i] - mu) * (src[i] - mu) / (2 * sig * sig)) / (sqrtf(2 * M_PI) * sig);
}

extern "C" void cuda_pdf_gauss(long N, float mu, float sig, float* dst, const float* src)
{
	kern_pdf_gauss<<<gridsize(N), blocksize(N)>>>(N, mu, sig, dst, src);
}


__global__ void kern_real(int N, float* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = cuCrealf(src[i]);
}

extern "C" void cuda_real(long N, float* dst, const _Complex float* src)
{
	kern_real<<<gridsize(N), blocksize(N)>>>(N, dst, (cuFloatComplex*)src);
}

__global__ void kern_imag(int N, float* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = cuCimagf(src[i]);
}

extern "C" void cuda_imag(long N, float* dst, const _Complex float* src)
{
	kern_imag<<<gridsize(N), blocksize(N)>>>(N, dst, (cuFloatComplex*)src);
}

__global__ void kern_zcmpl_real(int N, cuFloatComplex* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex(src[i], 0);
}

extern "C" void cuda_zcmpl_real(long N, _Complex float* dst, const float* src)
{
	kern_zcmpl_real<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, src);
}

__global__ void kern_zcmpl_imag(int N, cuFloatComplex* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex(0., src[i]);
}

extern "C" void cuda_zcmpl_imag(long N, _Complex float* dst, const float* src)
{
	kern_zcmpl_imag<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, src);
}

__global__ void kern_zcmpl(int N, cuFloatComplex* dst, const float* real_src, const float* imag_src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex(real_src[i], imag_src[i]);
}

extern "C" void cuda_zcmpl(long N, _Complex float* dst, const float* real_src, const float* imag_src)
{
	kern_zcmpl<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, real_src, imag_src);
}

__global__ void kern_zfill(int N, cuFloatComplex val, cuFloatComplex* dst)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = val;
}

extern "C" void cuda_zfill(long N, _Complex float val, _Complex float* dst)
{
	kern_zfill<<<gridsize(N), blocksize(N)>>>(N, make_cuFloatComplex(__real(val), __imag(val)), (cuFloatComplex*)dst);
}
