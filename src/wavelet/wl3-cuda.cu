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

#include "num/gpuops.h"

#include "wl3-cuda.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(_Complex float)
#endif

struct ldim3 {

	unsigned long x;
	unsigned long y;
	unsigned long z;
};


__device__ long Wdot(ldim3 a, ldim3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ ldim3 Wpmuladd(dim3 a, dim3 b, dim3 c)
{
	struct ldim3 r = { a.x * b.x + c.x, a.y * b.y + c.y, a.z * b.z + c.z};
	return r;
}

__device__ ldim3 Wpmul(ldim3 a, ldim3 b)
{
	struct ldim3 r = { a.x * b.x, a.y * b.y, a.z * b.z };
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

__global__ void kern_down3(ldim3 dims, ldim3 ostr, cuFloatComplex* out, ldim3 istr, const cuFloatComplex* in, unsigned int flen, const float* filter)
{
	ldim3 ind = Wpmuladd(blockIdx, blockDim, threadIdx);

	if ((ind.x >= dims.x) || (ind.y >= bandsize(dims.y, flen)))
		return;

	for( ; ind.z < dims.z; ind.z += (blockDim.z * gridDim.z)) {
	
		cuFloatComplex y = make_cuFloatComplex(0., 0.);

		for (unsigned int l = 0; l < flen; l++) {

			int n = coord(ind.y, dims.y, flen, l);
			ldim3 ac = ind;
			ac.y = n;

			y.x += in[Wdot(ac, istr)].x * filter[flen - l - 1];
			y.y += in[Wdot(ac, istr)].y * filter[flen - l - 1];
		}

		out[Wdot(ind, ostr)] = y;
	}
}

__global__ void kern_up3(ldim3 dims, ldim3 ostr, cuFloatComplex* out, ldim3 istr, const cuFloatComplex* in, unsigned int flen, const float* filter)
{
	ldim3 ind = Wpmuladd(blockIdx, blockDim, threadIdx);

	if ((ind.x >= dims.x) || (ind.y >= dims.y))
		return;

	for( ; ind.z < dims.z; ind.z += (blockDim.z * gridDim.z)) {

		cuFloatComplex y = out[Wdot(ind, ostr)];

		int odd = (ind.y + 1) % 2;

		for (unsigned int l = odd; l < flen; l += 2) {

			int j = (ind.y + l - 1) / 2;

			ldim3 ac = ind;
			ac.y = j;

			if ((0 <= j) && ((unsigned int)j < bandsize(dims.y, flen))) {

				y.x += in[Wdot(ac, istr)].x * filter[flen - l - 1];
				y.y += in[Wdot(ac, istr)].y * filter[flen - l - 1];
			}
		}

		out[Wdot(ind, ostr)] = y;
	}
}

// extern "C" size_t cuda_shared_mem;

extern "C" void wl3_cuda_down3(const long dims[3], const long out_str[3], _Complex float* out, const long in_str[3], const _Complex float* in, unsigned int flen, const float filter[__VLA(flen)])
{
	struct ldim3 dims3 = { (unsigned long)dims[0], (unsigned long)dims[1], (unsigned long)dims[2] };
	struct ldim3 ostrs = { out_str[0] / CFL_SIZE, out_str[1] / CFL_SIZE, out_str[2] / CFL_SIZE };
	struct ldim3 istrs = { in_str[0] / CFL_SIZE, in_str[1] / CFL_SIZE, in_str[2] / CFL_SIZE };

	long d1 = bandsize(dims[1], flen);

	int T = 8;
	dim3 th(T, T, T);
	dim3 bl((dims[0] + T - 1) / T, (d1 + T - 1) / T, MIN(65535, (dims[2] + T - 1) / T));

	kern_down3<<< bl, th, 0, cuda_get_stream() >>>(dims3, ostrs, (cuFloatComplex*)out, istrs, (const cuFloatComplex*)in, flen, filter);

	CUDA_KERNEL_ERROR;
}

extern "C" void wl3_cuda_up3(const long dims[3], const long out_str[3], _Complex float* out, const long in_str[3],  const _Complex float* in, unsigned int flen, const float filter[__VLA(flen)])
{
	ldim3 dims3 = { (unsigned long)dims[0], (unsigned long)dims[1], (unsigned long)dims[2] };
	ldim3 ostrs = { out_str[0] / CFL_SIZE, out_str[1] / CFL_SIZE, out_str[2] / CFL_SIZE };
	ldim3 istrs = { in_str[0] / CFL_SIZE, in_str[1] / CFL_SIZE, in_str[2] / CFL_SIZE };

	int T = 8;
	dim3 th(T, T, T);
	dim3 bl((dims[0] + T - 1) / T, (dims[1] + T - 1) / T, MIN(65535, (dims[2] + T - 1) / T));

	kern_up3<<< bl, th, 0, cuda_get_stream() >>>(dims3, ostrs, (cuFloatComplex*)out, istrs, (const cuFloatComplex*)in, flen, filter);

	CUDA_KERNEL_ERROR;
}



