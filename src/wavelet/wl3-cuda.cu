/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <complex.h>
#include <assert.h>
#include <stdbool.h>

#include <cuda.h>
#include <cuComplex.h>

#include "misc/misc.h"

#include "wl3-cuda.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(_Complex float)
#endif


__device__ long Wdot(dim3 a, dim3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ dim3 Wpmuladd(dim3 a, dim3 b, dim3 c)
{
	dim3 r(a.x * b.x + c.x, a.y * b.y + c.y, a.z * b.z + c.z);
	return r;
}

__device__ dim3 Wpmul(dim3 a, dim3 b)
{
	dim3 r(a.x * b.x, a.y * b.y, a.z * b.z);
	return r;
}

__host__ __device__ int bandsize(unsigned int imsize, unsigned int flen)
{
	return (imsize + flen - 1) / 2;
}

__host__ __device__ int coord(int l, int x, int flen, int k)
{
	int n = 2 * l + 1 - (flen - 1) + k;

	if (n < 0)
		n = -n - 1;

	if (n >= x)
		n = x - 1 - (n - x);

	return n;
}

__global__ void kern_down3(dim3 dims, dim3 ostr, cuFloatComplex* out, dim3 istr, const cuFloatComplex* in, unsigned int flen, const float* filter)
{
	dim3 ind = Wpmuladd(blockIdx, blockDim, threadIdx);
	
	if ((ind.x >= dims.x) || (ind.y >= bandsize(dims.y, flen)) || (ind.z >= dims.z))
		return;

	cuFloatComplex y = make_cuFloatComplex(0., 0.);

	for (unsigned int l = 0; l < flen; l++) {

		int n = coord(ind.y, dims.y, flen, l);
		dim3 ac = ind;
		ac.y = n;

		y.x += in[Wdot(ac, istr)].x * filter[flen - l - 1];
		y.y += in[Wdot(ac, istr)].y * filter[flen - l - 1];
	}

	out[Wdot(ind, ostr)] = y;
}

__global__ void kern_up3(dim3 dims, dim3 ostr, cuFloatComplex* out, dim3 istr, const cuFloatComplex* in, unsigned int flen, const float* filter)
{
	dim3 ind = Wpmuladd(blockIdx, blockDim, threadIdx);

	if ((ind.x >= dims.x) || (ind.y >= dims.y) || (ind.z >= dims.z))
		return;

//	cuFloatComplex y = make_cuFloatComplex(0., 0.);
	cuFloatComplex y = out[Wdot(ind, ostr)];

	for (unsigned int l = ((ind.y + flen / 2 - 0) - (flen - 1)) % 2; l < flen; l += 2) {

		int n = ((ind.y + flen / 2 - 0) - (flen - 1) + l) / 2;

		dim3 ac = ind;
		ac.y = n;

		if ((0 <= n) && ((unsigned int)n < bandsize(dims.y, flen))) {

			y.x += in[Wdot(ac, istr)].x * filter[flen - l - 1];
			y.y += in[Wdot(ac, istr)].y * filter[flen - l - 1];
		}
	}

	out[Wdot(ind, ostr)] = y;
}

// extern "C" size_t cuda_shared_mem;

extern "C" void wl3_cuda_down3(const long dims[3], const long out_str[3], _Complex float* out, const long in_str[3], const _Complex float* in, unsigned int flen, const float filter[__VLA(flen)])
{
	dim3 dims3(dims[0], dims[1], dims[2]);
	dim3 ostrs(out_str[0] / CFL_SIZE, out_str[1] / CFL_SIZE, out_str[2] / CFL_SIZE);
	dim3 istrs(in_str[0] / CFL_SIZE, in_str[1] / CFL_SIZE, in_str[2] / CFL_SIZE);

	long d1 = bandsize(dims[1], flen);

	int T = 8;
	dim3 th(T, T, T);
	dim3 bl((dims[0] + T - 1) / T, (d1 + T - 1) / T, (dims[2] + T - 1) / T);

	kern_down3<<< bl, th >>>(dims3, ostrs, (cuFloatComplex*)out, istrs, (const cuFloatComplex*)in, flen, filter);
}

extern "C" void wl3_cuda_up3(const long dims[3], const long out_str[3], _Complex float* out, const long in_str[3],  const _Complex float* in, unsigned int flen, const float filter[__VLA(flen)])
{
	dim3 dims3(dims[0], dims[1], dims[2]);
	dim3 ostrs(out_str[0] / CFL_SIZE, out_str[1] / CFL_SIZE, out_str[2] / CFL_SIZE);
	dim3 istrs(in_str[0] / CFL_SIZE, in_str[1] / CFL_SIZE, in_str[2] / CFL_SIZE);

	int T = 8;
	dim3 th(T, T, T);
	dim3 bl((dims[0] + T - 1) / T, (dims[1] + T - 1) / T, (dims[2] + T - 1) / T);

	kern_up3<<< bl, th >>>(dims3, ostrs, (cuFloatComplex*)out, istrs, (const cuFloatComplex*)in, flen, filter);
}



