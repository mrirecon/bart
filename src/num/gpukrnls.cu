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
#include "num/gpuops.h"
#include "num/multind.h"
#include "num/gpu_misc.h"

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

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static long gridsize(long N)
{
	// to ensure that "start" does not overflow we need to restrict gridsize!
	return MIN((N + BLOCKSIZE - 1) / BLOCKSIZE, 65536 - 1);
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
	kern_float2double<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, src);
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
	kern_double2float<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, src);
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
	kern_xpay<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, beta, dst, src);
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
	kern_axpbz<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, a1, src1, a2, src2);
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
	kern_smul<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, alpha, dst, src);
}


typedef void (*cuda_3op_f)(long N, float* dst, const float* src1, const float* src2);

extern "C" void cuda_3op(cuda_3op_f krn, int N, float* dst, const float* src1, const float* src2)
{
	krn<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, src1, src2);
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
	kern_sadd<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, val, dst, src1);
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
	kern_zsadd<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, make_cuFloatComplex(__real(val), __imag(val)), (cuFloatComplex*)dst, (const cuFloatComplex*)src1);
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
	kern_fmac2<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, src1, src2);
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
	kern_zsmul<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, make_cuFloatComplex(__real(alpha), __imag(alpha)), (cuFloatComplex*)dst, (const cuFloatComplex*)src1);
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
	kern_zmul<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
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
	kern_zdiv<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
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
	kern_zfmac<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
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
	kern_zfmac2<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuDoubleComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
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
	kern_zmulc<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
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
	kern_zfmacc<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
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
	kern_zfmacc2<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuDoubleComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}


#define MAX_DIMS 3
struct stride_desc {

	long dims[MAX_DIMS];
	long ostrs[MAX_DIMS];
	long istrs1[MAX_DIMS];
	long istrs2[MAX_DIMS];
};

__global__ void kern_zfmac_strides(stride_desc strides, long N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride) {

		for (long z = 0; z < strides.dims[2]; z++) {

			for (long y = 0; y < strides.dims[1]; y++) {

				for (long x = 0; x < strides.dims[0]; x++) {

					long o_offset = x * strides.ostrs[0] + y * strides.ostrs[1] + z * strides.ostrs[2];
					long i1_offset = x * strides.istrs1[0] + y * strides.istrs1[1] + z * strides.istrs1[2];
					long i2_offset = x * strides.istrs2[0] + y * strides.istrs2[1] + z * strides.istrs2[2];

					dst[i + o_offset] = cuCaddf(dst[i + o_offset], cuCmulf(src1[i + i1_offset], src2[i + i2_offset]));
				}
			}
		}
	}
}

//this version needs to start less kernels
extern "C" void cuda_zfmac_strided(long N, long dims[3], unsigned long oflags, unsigned long iflags1, unsigned long iflags2, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	struct stride_desc s;

	md_copy_dims(3, s.dims, dims);

	long odims[3];
	long idims1[3];
	long idims2[3];

	md_select_dims(3, oflags, odims, dims);
	md_select_dims(3, iflags1, idims1, dims);
	md_select_dims(3, iflags2, idims2, dims);

	md_calc_strides(3, s.ostrs, odims, N);
	md_calc_strides(3, s.istrs1, idims1, N);
	md_calc_strides(3, s.istrs2, idims2, N);

	kern_zfmac_strides<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(s, N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}


__global__ void kern_zfmacc_strides(stride_desc strides, long N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride) {

		for (long z = 0; z < strides.dims[2]; z++) {

			for (long y = 0; y < strides.dims[1]; y++) {

				for (long x = 0; x < strides.dims[0]; x++) {

					long o_offset = x * strides.ostrs[0] + y * strides.ostrs[1] + z * strides.ostrs[2];
					long i1_offset = x * strides.istrs1[0] + y * strides.istrs1[1] + z * strides.istrs1[2];
					long i2_offset = x * strides.istrs2[0] + y * strides.istrs2[1] + z * strides.istrs2[2];

					dst[i + o_offset] = cuCaddf(dst[i + o_offset], cuCmulf(src1[i + i1_offset],  cuConjf(src2[i + i2_offset])));
				}
			}
		}
	}
}


extern "C" void cuda_zfmacc_strided(long N, long dims[3], unsigned long oflags, unsigned long iflags1, unsigned long iflags2, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	struct stride_desc s;

	md_copy_dims(3, s.dims, dims);

	long odims[3];
	long idims1[3];
	long idims2[3];

	md_select_dims(3, oflags, odims, dims);
	md_select_dims(3, iflags1, idims1, dims);
	md_select_dims(3, iflags2, idims2, dims);

	md_calc_strides(3, s.ostrs, odims, N);
	md_calc_strides(3, s.istrs1, idims1, N);
	md_calc_strides(3, s.istrs2, idims2, N);

	kern_zfmacc_strides<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(s, N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
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

static __device__ __host__ cuDoubleComplex zexpD(cuDoubleComplex x)
{
	double sc = exp(cuCreal(x));
	double si;
	double co;
	sincos(cuCimag(x), &si, &co);
	return make_cuDoubleComplex(sc * co, sc * si);
}

static __device__ cuFloatComplex zexp(cuFloatComplex x)
{
	float sc = expf(cuCrealf(x));
	float si;
	float co;
	sincosf(cuCimagf(x), &si, &co);
	return make_cuFloatComplex(sc * co, sc * si);
}

static __device__ cuFloatComplex zsin(cuFloatComplex x)
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

static __device__ cuFloatComplex zcos(cuFloatComplex x)
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

static __device__ cuFloatComplex zsinh(cuFloatComplex x)
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

static __device__ cuFloatComplex zcosh(cuFloatComplex x)
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

static __device__ float zarg(cuFloatComplex x)
{
	return atan2(cuCimagf(x), cuCrealf(x));
}

static __device__ float zabs(cuFloatComplex x)
{
	return cuCabsf(x);
}

static __device__ cuFloatComplex zlog(cuFloatComplex x)
{
	return make_cuFloatComplex(log(cuCabsf(x)), zarg(x));
}


// x^y = e^{y ln(x)} = e^{y
static __device__ cuFloatComplex zpow(cuFloatComplex x, cuFloatComplex y)
{
	if ((0 == y.x) && (0 == y.y))
		return make_cuFloatComplex(1., 0.);

	if (((0 == x.x) && (0 == x.y)) && (0. < y.x))
		return make_cuFloatComplex(0., 0.);

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
	kern_zpow<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
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
	kern_sqrt<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, src);
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
	kern_zconj<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
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
	kern_zcmp<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
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
	kern_zdiv_reg<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2, make_cuFloatComplex(__real(lambda), __imag(lambda)));
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
	kern_zphsr<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
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
	kern_zexp<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
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
	kern_zexpj<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
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
	kern_zlog<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
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
	kern_zarg<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
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
	kern_zsin<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
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
	kern_zcos<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
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
	kern_zsinh<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
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
	kern_zcosh<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
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
	kern_zabs<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
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
	kern_exp<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, src);
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
	kern_log<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, src);
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
	kern_zatanr<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
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
	kern_zacos<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
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
	kern_zsoftthresh_half<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, lambda, (cuFloatComplex*)d, (const cuFloatComplex*)x);
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
	kern_zsoftthresh<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, lambda, (cuFloatComplex*)d, (const cuFloatComplex*)x);
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
	kern_softthresh_half<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, lambda, d, x);
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
	kern_softthresh<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, lambda, d, x);
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
	kern_zreal<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
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
	kern_zle<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
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
	kern_le<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, src1, src2);
}

static __device__ cuFloatComplex cuDouble2Float(cuDoubleComplex x)
{
	return make_cuFloatComplex(cuCreal(x), cuCimag(x));
}

static __device__ cuDoubleComplex cuFloat2Double(cuFloatComplex x)
{
	return make_cuDoubleComplex(cuCrealf(x), cuCimagf(x));
}

// identical copy in num/fft.c
static __device__ double fftmod_phase(long length, int j)
{
	long center1 = length / 2;
	double shift = (double)center1 / (double)length;
	return ((double)j - (double)center1 / 2.) * shift;
}

static __device__ cuDoubleComplex fftmod_phase2(long n, int j, bool inv, double phase)
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
	kern_zfftmod<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src, n, inv, phase);
}


__global__ void kern_fftmod_3d_4(long X, long Y, long Z, cuFloatComplex* dst, const cuFloatComplex* src, bool inv, cuDoubleComplex scale_1)
{
	int startX = threadIdx.x + blockDim.x * blockIdx.x;
	int strideX = blockDim.x * gridDim.x;

	int startY = threadIdx.y + blockDim.y * blockIdx.y;
	int strideY = blockDim.y * gridDim.y;

	int startZ = threadIdx.z + blockDim.z * blockIdx.z;
	int strideZ = blockDim.z * gridDim.z;

	for (long z = startZ; z < Z; z += strideZ)
		for (long y = startY; y < Y; y += strideY)
			for (long x = startX; x < X; x +=strideX) {

				long idx = x + X * (y + Y * z);
				
				cuDoubleComplex scale = scale_1;

				if (1 == x % 2) {

					scale.x = -scale.x;
					scale.y = -scale.y;
				}
				
				if (1 == y % 2) {

					scale.x = -scale.x;
					scale.y = -scale.y;
				}

				if (1 == z % 2) {

					scale.x = -scale.x;
					scale.y = -scale.y;
				}

				
				dst[idx] = cuDouble2Float(cuCmul(scale, cuFloat2Double(src[idx])));				
			}
}

__global__ void kern_fftmod_3d(long X, long Y, long Z, cuFloatComplex* dst, const cuFloatComplex* src, bool inv, double phase)
{
	int startX = threadIdx.x + blockDim.x * blockIdx.x;
	int strideX = blockDim.x * gridDim.x;

	int startY = threadIdx.y + blockDim.y * blockIdx.y;
	int strideY = blockDim.y * gridDim.y;

	int startZ = threadIdx.z + blockDim.z * blockIdx.z;
	int strideZ = blockDim.z * gridDim.z;

	long dims[3] = { X, Y, Z };

	for (long z = startZ; z < Z; z += strideZ)
		for (long y = startY; y < Y; y += strideY)
			for (long x = startX; x < X; x +=strideX) {

				long pos[3] = { x, y, z };
				long idx = x + X * (y + Y * z);
				
				double phase0 = phase;
				for (int i = 2; i > 0; i--)
					phase0 += fftmod_phase(dims[i], pos[i]);

				dst[idx] = cuDouble2Float(cuCmul(fftmod_phase2(dims[0], x, inv, phase0), cuFloat2Double(src[idx])));				
			}
}

extern "C" void cuda_zfftmod_3d(const long dims[3], _Complex float* dst, const _Complex float* src, _Bool inv, double phase)
{
	if (   ((dims[0] == 1) || (dims[0] % 4 == 0))
	    && ((dims[1] == 1) || (dims[1] % 4 == 0))
	    && ((dims[2] == 1) || (dims[2] % 4 == 0)))
		{
			double rem = phase - floor(phase);
			double sgn = inv ? -1. : 1.;

			cuDoubleComplex scale = zexpD(make_cuDoubleComplex(0, M_PI * 2. * sgn * rem));

			if ((1 != dims[0]) && (0 != dims[0] % 8)) {

				scale.x *= -1;
				scale.y *= -1;
			}
				
			if ((1 != dims[1]) && (0 != dims[1] % 8)) {

				scale.x *= -1;
				scale.y *= -1;
			}
	
			if ((1 != dims[2]) && (0 != dims[2] % 8)) {

				scale.x *= -1;
				scale.y *= -1;
			}


			kern_fftmod_3d_4<<<getGridSize3(dims, (const void*)kern_fftmod_3d), getBlockSize3(dims, (const void*)kern_fftmod_3d), 0, cuda_get_stream()>>>(dims[0], dims[1], dims[2], (cuFloatComplex*)dst, (const cuFloatComplex*)src, inv, scale);
			return;
		}

	kern_fftmod_3d<<<getGridSize3(dims, (const void*)kern_fftmod_3d), getBlockSize3(dims, (const void*)kern_fftmod_3d), 0, cuda_get_stream()>>>(dims[0], dims[1], dims[2], (cuFloatComplex*)dst, (const cuFloatComplex*)src, inv, phase);
}




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
	kern_zmax<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
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
	kern_smax<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, val, dst, src1);
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
	kern_max<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, src1, src2);
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
	kern_min<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, src1, src2);
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
	kern_zsmax<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, alpha, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
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

		kern_reduce_zsum<<<1, B, 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst);
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
	kern_pdf_gauss<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, mu, sig, dst, src);
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
	kern_real<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, (cuFloatComplex*)src);
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
	kern_imag<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, (cuFloatComplex*)src);
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
	kern_zcmpl_real<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, src);
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
	kern_zcmpl_imag<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, src);
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
	kern_zcmpl<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, real_src, imag_src);
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
	kern_zfill<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, make_cuFloatComplex(__real(val), __imag(val)), (cuFloatComplex*)dst);
}


__global__ static void kern_compress(long N, uint32_t* dst, const float* src)
{
	long idx;
	long idx_init;
	unsigned int stride;
	unsigned int thread;

	thread = threadIdx.x;
	idx_init = blockDim.x * blockIdx.x; //if idx would contain thread id, the loop might diverge ->deadlock with syncthreads

	stride = blockDim.x * gridDim.x;

	for (idx = idx_init; idx < N; idx += stride) {
		
		long i = idx + thread;

		extern __shared__ float tmp_float[];

		tmp_float[thread] = (i < N) ? src[i] : 0;

		__syncthreads();

		if ((0 == i % 32) && (i - 32 < N)) {

			uint32_t result = 0;
			for (int j = 0; j < 32; j++)
				if (0. != tmp_float[thread + j])
					result = MD_SET(result, j);
			
			dst[i / 32] = result;
		}
	}
}

__global__ static void kern_decompress(long N, float* dst, const uint32_t* src)
{
	long idx;
	long idx_init;
	unsigned int stride;
	unsigned int thread;

	thread = threadIdx.x;
	idx_init = blockDim.x * blockIdx.x; //if idx would contain thread id, the loop might diverge ->deadlock with syncthreads

	stride = blockDim.x * gridDim.x;

	for (idx = idx_init; idx < N; idx += stride) {
		
		long i = idx + thread;

		extern __shared__ uint32_t tmp_uint32[];

		if ((0 == i % 32) && (i - 32 < N))
			tmp_uint32[thread / 32] = src[i / 32];
		
		__syncthreads();

		if (i < N)
			dst[i] = MD_IS_SET(tmp_uint32[thread / 32], thread % 32) ? 1. : 0.;		
	}
}

extern "C" void cuda_compress(long N, uint32_t* dst, const float* src)
{
	kern_compress<<<gridsize(N), blocksize(N), blocksize(N) * sizeof(float), cuda_get_stream()>>>(N, dst, src);
}

extern "C" void cuda_decompress(long N, float* dst, const uint32_t* src)
{
	kern_decompress<<<gridsize(N), blocksize(N), blocksize(N), cuda_get_stream()>>>(N, dst, src);
}