/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <cstdint>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>

#include "misc/misc.h"
#include "num/gpuops.h"
#include "num/gpu_conv.h"
#include "num/multind.h"

// limited by hardware to 1024 on most devices
// should be a multiple of 32 (warp size)
#define BLOCKSIZE 1024


static void getBlockSize3_internal(int block[3], const long dims[3], const void* func)
{
	cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, func);
	int threads = attr.maxThreadsPerBlock;

	block[0] = 1;
	block[1] = 1;
	block[2] = 1;

	while ((threads >= 2) && (block[0] < dims[0])) {

		block[0] *= 2;
		threads /= 2;
	}

	while ((threads >= 2) && (block[1] < dims[1])) {

		block[1] *= 2;
		threads /= 2;
	}

	while ((threads >= 2) && (block[2] < dims[2])) {

		block[2] *= 2;
		threads /= 2;
	}
}

static dim3 getBlockSize3(const long dims[3], const void* func)
{
	int block[3];

	getBlockSize3_internal(block, dims, func);

	return dim3(block[0], block[1], block[2]);
}

static long gridsize_int(long N, int blocksize)
{
	return (N + blocksize - 1) / blocksize;
}

static dim3 getGridSize3(const long dims[3], const void* func)
{
	int block[3];

	getBlockSize3_internal(block, dims, func);

	return dim3(gridsize_int(dims[0], block[0]), gridsize_int(dims[1], block[1]), gridsize_int(dims[2], block[2]));
}



static dim3 blocksize(int N, const void* func)
{
	const long dims[3] = { N, 1, 1};
	return getBlockSize3(dims, func);
}

static dim3 gridsize(long N, const void* func)
{
	const long dims[3] = { N, 1, 1};
	return getGridSize3(dims, func);
}


template <int DIMS, typename T> 
struct im2col_descriptor {

	T NC;		// number channels
	T istrs_NC;		// 1
	T ostrs_NC;		// 1

	T odims[DIMS];	// dimensions of the convolution (not including channel)
	T kdims[DIMS];
	T idims[DIMS];

	T istrs_odims[DIMS];	// input strides of im2col (in elements)
	T istrs_kdims[DIMS];
	T ostrs_kdims[DIMS];	// output strides of im2col (in elements)
	T ostrs_odims[DIMS];

	T N_in_elements;		// channels * in-dims
	T N_out_elements;		// channels * out-dims * krn-dims
	T N_out_elements_o_only;	// channels * out-dims
	T N_out_elements_k_only;	// channels * krn-dims

	bool triv_strides_dilation;	// trivial dilation and strides
};

template <int DIMS, typename T> 
static struct im2col_descriptor<DIMS, T>get_im2col_descriptor(const long odims[5], const long idims[5], const long kdims[5], const long dilation[5], const long strides[5])
{
	struct im2col_descriptor<DIMS, T>config;

	config.NC = idims[1];
	config.istrs_NC = 1;
	config.ostrs_NC = 1;

	config.N_in_elements = idims[1];
	config.N_out_elements = idims[1];
	config.N_out_elements_o_only = idims[1];
	config.N_out_elements_k_only = idims[1];

	config.triv_strides_dilation = true;

	long istrs[5];
	md_calc_strides(5, istrs, idims, 1);

	for (int i = 0; i < DIMS; i++) {

		config.odims[i] = 1;
		config.kdims[i] = 1;
		config.idims[i] = 1;
		config.istrs_odims[i] = 0;
		config.istrs_kdims[i] = 0;
		config.ostrs_odims[i] = 0;
		config.ostrs_kdims[i] = 0;
	}


	for (int i = 2, j = 0; i < 5; i++) {

		if (!((1 < odims[i]) || (1 < kdims[i])))
		 	continue;
		
		assert(j < DIMS);

		config.odims[j] = odims[i];
		config.kdims[j] = kdims[i];
		config.idims[j] = idims[i];

		config.istrs_odims[j] = istrs[i] * (NULL == strides ? 1 : strides[i]);
		config.istrs_kdims[j] = istrs[i] * (NULL == dilation ? 1 : dilation[i]);

		config.N_in_elements *= idims[i];
		config.N_out_elements_o_only *= odims[i];
		config.N_out_elements_k_only *= kdims[i];
		config.N_out_elements *= odims[i] * kdims[i];

		config.triv_strides_dilation &=
				(   (config.istrs_odims[j] == istrs[i])
				 && (config.istrs_kdims[j] == istrs[i]));

		j++;
	}

	config.ostrs_odims[0] = config.N_out_elements_k_only;
	config.ostrs_kdims[0] = config.NC;

	for (int i = 1; i < DIMS; i++) {

		config.ostrs_odims[i] = config.ostrs_odims[i - 1] * config.odims[i - 1];
		config.ostrs_kdims[i] = config.ostrs_kdims[i - 1] * config.kdims[i - 1];
	}

	return config;
}

// loop over out-dims and krn-dims and copy elements from input (copies one element per thread)
template <int DIMS, typename T, bool transp> 
__global__ static void kern_im2col_valid(struct im2col_descriptor<DIMS, T> config, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (T i = start; i < config.N_out_elements; i += stride) {

		T i_cur = i;
		T i_new = i;
		T in_index = 0;

		if (1 < config.NC) {

			i_new = i_cur / config.NC;
			in_index = (i_cur - config.NC * i_new) * config.istrs_NC;
			i_cur = i_new;
		}

		for (int j = 0; j < DIMS; j++) {

			i_new = i_cur / config.kdims[j];
			in_index += config.istrs_kdims[j] * (i_cur - config.kdims[j] * i_new);
			i_cur = i_new;
		}

		for (int j = 0; j < DIMS - 1; j++) {

			i_new = i_cur / config.odims[j];
			in_index += config.istrs_odims[j] * (i_cur - config.odims[j] * i_new);
			i_cur = i_new;
		}

		in_index += i_cur * config.istrs_odims[DIMS - 1];

		if (transp) {

			atomicAdd(&(dst[in_index].x), src[i].x);
			atomicAdd(&(dst[in_index].y), src[i].y);
		} else {

			dst[i] = src[in_index];
		}
	}
}

// loop over in-dims and copy elements from input to all corresponding output position
template <int DIMS, typename T, bool transp> 
__global__ static void kern_im2col_valid_no_dil_str(struct im2col_descriptor<DIMS, T> config, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (T i = start; i < config.N_in_elements; i += stride) {

		T i_cur = i;
		T i_new = i_cur / config.NC;
		T c = i_cur - i_new * config.NC;

		T idx_i[3] = { 0, 0, 0 };
		T idx_k[3] = { 0, 0, 0 };

		for (int j = 0; j < DIMS - 1; j++) {

			i_cur = i_new;
			i_new = i_cur / config.idims[j];
			idx_i[j] = i_cur - i_new * config.idims[j];
		}

		idx_i[DIMS - 1] = i_new;

		cuFloatComplex tmp = transp ? dst[i] : src[i];

		T kdims[3];
		for (int j = 0; j < 3; j++)
			kdims[j] = (j < DIMS) ? config.kdims[j] : 1;

		for (idx_k[2] = 0; idx_k[2] < kdims[2]; idx_k[2]++)
		for (idx_k[1] = 0; idx_k[1] < kdims[1]; idx_k[1]++)
		for (idx_k[0] = 0; idx_k[0] < kdims[0]; idx_k[0]++) {

			bool copy = true;

			T o_stride = config.N_out_elements_k_only;
			T k_stride = config.NC;
			T index = c;

			for (int j = 0; j < DIMS; j++) {

				copy = copy && (idx_k[j] <= idx_i[j]) && (idx_i[j] < idx_k[j] + config.odims[j]);

				index += (idx_i[j] - idx_k[j]) * o_stride + idx_k[j] * k_stride;
				
				o_stride *= config.odims[j];
				k_stride *= config.kdims[j];
			}

			if (copy) {
			
				if (transp)
					tmp = cuCaddf(tmp, src[index]);
				else
					dst[index] = tmp;
			}
		}

		if (transp)
			dst[i] = tmp;
	}
}

template <int DIMS, typename T, bool transp> 
static void cuda_im2col_int(_Complex float* dst, const _Complex float* src, const long odims[5], const long idims[5], const long kdims[5], const long dilation[5], const long strides[5])
{
	struct im2col_descriptor<DIMS, T> config = get_im2col_descriptor<DIMS, T>(odims, idims, kdims, dilation, strides);

	bool func1 = true;
#ifdef NON_DETERMINISTIC
	bool func2 = true;
#else
	bool func2 = !transp;
#endif

	func1 = func1 && config.triv_strides_dilation;
	func1 = func1 && (!func2 || (1 < config.NC));


	if (func1) {
	
		const void* func = (const void*)kern_im2col_valid_no_dil_str<DIMS, T, transp>;
		kern_im2col_valid_no_dil_str<DIMS, T, transp><<<gridsize(config.N_in_elements, func), blocksize(config.N_in_elements, func), 0, cuda_get_stream() >>>(config, (cuFloatComplex*) dst, (cuFloatComplex*) src);
		return;
	}
	
	if (func2) {

		const void* func = (const void*)kern_im2col_valid<DIMS, T, transp>;
		kern_im2col_valid<DIMS, T, transp><<<gridsize(config.N_in_elements, func), blocksize(config.N_in_elements, func), 0, cuda_get_stream() >>>(config, (cuFloatComplex*) dst, (cuFloatComplex*) src);
		return;
	}

	assert(0);
}

template <bool transp> 
static void cuda_im2col_int2(_Complex float* dst, const _Complex float* src, const long odims[5], const long idims[5], const long kdims[5], const long dilation[5], const long strides[5])
{
	long Nout = idims[1] * md_calc_size(3, kdims + 2) * md_calc_size(3, odims + 2);
	int DIMS = bitcount(md_nontriv_dims(3, kdims + 2) | md_nontriv_dims(3, odims + 2));

	for (int i = 0 ; i < 3; i++)
		if (1 == odims[DIMS + 1] * kdims[DIMS + 1] * idims[DIMS + 1] * (NULL != dilation ? dilation[DIMS + 1] : 1) * (NULL != strides ? strides[DIMS + 1] : 1))
			DIMS --;
	
	DIMS = 3;

	switch (DIMS) {

	case 1:
		if (INT32_MAX / 2 > Nout)
			cuda_im2col_int<1, uint32_t, transp>(dst, src, odims, idims, kdims, dilation, strides);
		else
			cuda_im2col_int<1, uint64_t, transp>(dst, src, odims, idims, kdims, dilation, strides);
		break;
	
	case 2:
		if (INT32_MAX / 2 > Nout)
			cuda_im2col_int<2, uint32_t, transp>(dst, src, odims, idims, kdims, dilation, strides);
		else
			cuda_im2col_int<2, uint64_t, transp>(dst, src, odims, idims, kdims, dilation, strides);
		break;
	
	case 3:
		if (INT32_MAX / 2 > Nout)
			cuda_im2col_int<3, uint32_t, transp>(dst, src, odims, idims, kdims, dilation, strides);
		else
			cuda_im2col_int<3, uint64_t, transp>(dst, src, odims, idims, kdims, dilation, strides);
		break;
	default:
		assert(0);
	}
}

/* *
 * Optimized kernel for copying im2col (complex float only)
 *
 * @args dst
 * @args src
 * @args odims		[OC,  1, OX, OY, OZ]
 * @args idims		[ 1, IC, IX, IY, IZ]
 * @args kdims		[OC, IC, KX, KY, KZ]
 * @args dilation	[ 1,  1, DX, DY, DZ] or NULL
 * @args strides	[ 1,  1, SX, SY, SZ] or NULL
 *
 * Copy:
 * dims:	[IC, KX, KY, KZ, OX, OY, OZ]
 * ostrs:	trivial strides of dims
 * istrs:	[ISC, ISX * DX, ISY * DY, ISZ * DZ, ISX * SX, ISY * SY, ISZ * SZ]
 * where IS* are trivial strides of idims
 * */
extern "C" void cuda_im2col(_Complex float* dst, const _Complex float* src, const long odims[5], const long idims[5], const long kdims[5], const long dilation[5], const long strides[5])
{
	cuda_im2col_int2<false>(dst, src, odims, idims, kdims, dilation, strides);
}

/* *
 * Transposed/adjoint of cuda im2col
 *
 * @args dst
 * @args src
 * @args odims		[OC,  1, OX, OY, OZ]
 * @args idims		[ 1, IC, IX, IY, IZ]
 * @args kdims		[OC, IC, KX, KY, KZ]
 * @args dilation	[ 1,  1, DX, DY, DZ] or NULL
 * @args strides	[ 1,  1, SX, SY, SZ] or NULL
 *
 * zadd with strides:
 * dims:	[IC, KX, KY, KZ, OX, OY, OZ]
 * ostrs:	[ISC, ISX * DX, ISY * DY, ISZ * DZ, ISX * SX, ISY * SY, ISZ * SZ]
 * istrs:	trivial strides of dims
 * where IS* are trivial strides of idims
 * */
extern "C" void cuda_im2col_transp(_Complex float* dst, const _Complex float* src, const long odims[5], const long idims[5], const long kdims[5], const long dilation[5], const long strides[5])
{
	cuda_im2col_int2<true>(dst, src, odims, idims, kdims, dilation, strides);
}

