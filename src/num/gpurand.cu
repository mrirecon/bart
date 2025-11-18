/* Copyright 2023. Institute of Biomedical Imaging. Graz University of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Christian Holme, Moritz Blumenthal
 */


#include <stdbool.h>
#include <assert.h>


#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>

#include <stdint.h>

#include "num/gpuops.h"
#include "num/gpukrnls_misc.h"

struct philox_state {
	// for philox 4x32: 64bits of state, and 2 64bit counters:
	uint64_t state;
	uint64_t ctr1;
	uint64_t ctr2;
};


__device__ static void philox_4x32(const uint64_t state, const uint64_t ctr1, const uint64_t ctr2, uint64_t out[2]);


// shift by 11 to get a 53-bit integer, as double has 53 bits of precision in the mantissa
__device__ static inline double ull2double(uint64_t x)
{
	return (double)(x >> 11) * 0x1.0p-53;
}

__device__ static uint64_t rand64_state(struct philox_state* state)
{

	uint64_t out[2];
	philox_4x32(state->state, state->ctr1, state->ctr2, out);
	state->ctr1++;

	return out[0];
}

__device__ static double uniform_rand_state(struct philox_state* state)
{
	return ull2double(rand64_state(state));
}



__device__ static cuDoubleComplex gaussian_stable_rand(struct philox_state state)
{
	double u1, u2, s;
	uint64_t out[2];

	philox_4x32(state.state, state.ctr1, state.ctr2, out);

	struct philox_state gauss_state = { .state = out[0], .ctr1 = 0, .ctr2 = 0 };

	do {
		philox_4x32(gauss_state.state, gauss_state.ctr1, gauss_state.ctr2, out);
		gauss_state.ctr1++;

		u1 = 2. * ull2double(out[0]) - 1.;
		u2 = 2. * ull2double(out[1]) - 1.;
		s = u1 * u1 + u2 * u2;

	} while (s > 1.);

	double re = sqrt(-2. * log(s) / s) * u1;
	double im = sqrt(-2. * log(s) / s) * u2;

	return make_cuDoubleComplex(re, im);
}


__global__ void kern_gaussian_rand(long N, cuFloatComplex* dst, struct philox_state state, uint64_t offset)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	uint64_t ctr1 = state.ctr1;

	for (long i = start; i < N; i += stride) {

		state.ctr1 = ctr1;
		state.ctr2 = (uint64_t) i + offset;
		dst[i] = cuComplexDoubleToFloat(gaussian_stable_rand(state));
	}
}

extern "C" void cuda_gaussian_rand(long N, _Complex float* dst,  uint64_t state, uint64_t ctr1, uint64_t offset)
{
	struct philox_state ph_state = {.state = state, .ctr1 = ctr1, .ctr2 = 0};
	kern_gaussian_rand<<<getGridSize(N, (const void*) kern_gaussian_rand), getBlockSize(N, (const void*) kern_gaussian_rand), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, ph_state, offset);
	CUDA_KERNEL_ERROR;
}



__global__ void kern_uniform_rand(long N, cuFloatComplex* dst, struct philox_state state, uint64_t offset)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	uint64_t ctr1 = state.ctr1;

	for (long i = start; i < N; i += stride) {

		state.ctr1 = ctr1;
		state.ctr2 = (uint64_t) i + offset;
		dst[i] = make_cuFloatComplex(uniform_rand_state(&state), 0.f);
	}
}

extern "C" void cuda_uniform_rand(long N, _Complex float* dst,  uint64_t state, uint64_t ctr1, uint64_t offset)
{
	struct philox_state ph_state = {.state = state, .ctr1 = ctr1, .ctr2 = 0};
	kern_uniform_rand<<<getGridSize(N, (const void*) kern_uniform_rand), getBlockSize(N, (const void*) kern_uniform_rand), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, ph_state, offset);
	CUDA_KERNEL_ERROR;
}


__global__ void kern_rand_one(long N, cuFloatComplex* dst, double p, struct philox_state state, uint64_t offset)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	uint64_t ctr1 = state.ctr1;

	for (long i = start; i < N; i += stride) {

		state.ctr1 = ctr1;
		state.ctr2 = (uint64_t) i + offset;
		dst[i] = make_cuFloatComplex(uniform_rand_state(&state) < p, 0.f);
	}

}

extern "C" void cuda_rand_one(long N, _Complex float* dst, double p, uint64_t state, uint64_t ctr1, uint64_t offset)
{
	struct philox_state ph_state = {.state = state, .ctr1 = ctr1, .ctr2 = 0};
	kern_rand_one<<<getGridSize(N, (const void*) kern_rand_one), getBlockSize(N, (const void*) kern_rand_one), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst,  p, ph_state, offset);
	CUDA_KERNEL_ERROR;
}



#include "num/philox.inc"
