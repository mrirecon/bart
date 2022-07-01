/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <stdio.h>
#ifdef USE_CUDA
#ifdef USE_CUDNN

#include <complex.h>
#include <stdbool.h>
#include <cudnn.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/gpuops.h"
#include "num/optimize.h"


#include "misc/debug.h"
#include "misc/misc.h"

#include "cudnn_wrapper.h"

static void cudnn_error(const char* file, int line, cudnnStatus_t code)
{
	const char *err_str = cudnnGetErrorString(code);
	error("cDNN error: %s in %s:%d \n", err_str, file, line);
}

#define CUDNN_ERROR(x)	({ CUDA_ASYNC_ERROR_NOTE("before cuDNN call"); cudnnStatus_t errval = (x); if (CUDNN_STATUS_SUCCESS  != errval) cudnn_error(__FILE__, __LINE__, errval); CUDA_ASYNC_ERROR_NOTE("after cuDNN call"); })
#define CUDNN_CALL(x)	({ CUDA_ASYNC_ERROR_NOTE("before cuDNN call"); cudnn_set_gpulock(); cudnnStatus_t errval = (x); if (CUDNN_STATUS_SUCCESS  != errval) cudnn_error(__FILE__, __LINE__, errval); cudnn_unset_gpulock(); CUDA_ASYNC_ERROR_NOTE("after cuBLAS call"); })


static cudnnHandle_t handle[MAX_CUDA_DEVICES];
static int num_devices_initialized = 0;


void cudnn_init(void)
{
	if (0 != num_devices_initialized)
		error("Cannot reinitialize cuDNN, deinit it first!");

	int old_device = cuda_get_device();

	for (int device = 0; device < cuda_num_devices(); ++device) {

		cuda_set_device(device);
		CUDNN_ERROR(cudnnCreate(handle + device));
	}

	cuda_set_device(old_device);

	num_devices_initialized = cuda_num_devices();	
}


void cudnn_deinit(void)
{
	if (cuda_num_devices() != num_devices_initialized)
		error("Cannot deinitialize cuDNN, number of devices has changed from initialization!");

	for (int device = 0; device < cuda_num_devices(); ++device)
		CUDNN_ERROR(cudnnDestroy(handle[device]));

	num_devices_initialized = 0;	
}

#ifdef _OPENMP
#include <omp.h>
static bool gpulock_init = false;
static omp_lock_t gpulock[MAX_CUDA_DEVICES];
static void cudnn_set_gpulock(void)
{
	#pragma omp critical (init_cudnn_gpulock)
	if (!gpulock_init) {

		for (int i = 0; i < MAX_CUDA_DEVICES; i++)
			omp_init_lock(&gpulock[i]);

		gpulock_init = true;		
	}

	omp_set_lock(&gpulock[cuda_get_device()]);
}

static void cudnn_unset_gpulock(void)
{
	omp_unset_lock(&gpulock[cuda_get_device()]);
}
#else
static void cudnn_set_gpulock(void)
{
	return;
}

static void cudnn_unset_gpulock(void)
{
	return;
}
#endif

static cudnnHandle_t get_handle(void)
{
	if (cuda_num_devices() != num_devices_initialized)
		error("cuDNN not initialized correctly!");

	cudnnHandle_t result = handle[cuda_get_device()];
	CUDNN_ERROR(cudnnSetStream(result, cuda_get_stream()));

	return result;
}

#if 0
static void destroy_handle(void)
{
	CUDNN_ERROR(cudnnDestroy(handle));
	handle_created = false;
}
#endif

#if 0
static void cudnn_print_tensor_descriptor(const cudnnTensorDescriptor_t tensorDesc) {

	int nbDims = CUDNN_DIM_MAX;

	cudnnDataType_t dataType;
 	int dimA[nbDims];
 	int strideA[nbDims];

	CUDNN_ERROR(cudnnGetTensorNdDescriptor(tensorDesc, nbDims, &dataType, &nbDims, dimA, strideA));

	printf("Tensor Dims:\n");
	print_int(nbDims, dimA);
	printf("Tensor Strides:\n");
	print_int(nbDims, strideA);
}

static void cudnn_print_filter_descriptor(const cudnnFilterDescriptor_t filterDesc) {

	int nbDims = CUDNN_DIM_MAX;

	cudnnDataType_t dataType;
	cudnnTensorFormat_t format;
 	int dimA[nbDims];

	CUDNN_ERROR(cudnnGetFilterNdDescriptor(filterDesc, nbDims, &dataType, &format, &nbDims, dimA));

	printf("Filter Dims:\n");
	print_int(nbDims, dimA);
}

static void cudnn_print_convolution_descriptor(const cudnnConvolutionDescriptor_t convDesc) {

	int nbDims = CUDNN_DIM_MAX-2;

	cudnnDataType_t dataType;
	cudnnConvolutionMode_t mode;

 	int padA[nbDims];
	int filterStrideA[nbDims];
	int dilationA[nbDims];

	CUDNN_ERROR(cudnnGetConvolutionNdDescriptor(convDesc, nbDims, &nbDims, padA, filterStrideA, dilationA, &mode, &dataType));

	printf("Padding:\n");
	print_int(nbDims, padA);

	printf("Strides:\n");
	print_int(nbDims, filterStrideA);

	printf("Dilation:\n");
	print_int(nbDims, dilationA);
}
#endif

static cudnnTensorDescriptor_t bart_to_cudnn_float_tensor_descriptor(unsigned int D, const long dims[D], const long str[D])
{
	int nbDims = MAX(D, 3u);
	int dimA[nbDims];
	int strideA[nbDims];

	for (int i = 0; i < nbDims; i++) {

		dimA[i] = 1;
		strideA[i] = 1;
	}

	for (int i = 0; i < (int)D; i++) {

		dimA[D - 1 - i] = dims[i];
		strideA[D - 1 - i] = str[i] ? str[i] / FL_SIZE : 1;
	}

	cudnnTensorDescriptor_t result;
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&result));
	CUDNN_ERROR(cudnnSetTensorNdDescriptor(result, CUDNN_DATA_FLOAT, nbDims, dimA, strideA));

	return result;
}

static void cudnn_smul2(unsigned int D, const long dims[D], const long ostr[D], float* optr, const long istr[D], const float* iptr, float val)
{
	cudnnTensorDescriptor_t odesc = bart_to_cudnn_float_tensor_descriptor(D, dims, ostr);
	cudnnTensorDescriptor_t idesc = bart_to_cudnn_float_tensor_descriptor(D, dims, istr);

	float alpha = val;
	float beta = 0;

	CUDNN_CALL(cudnnTransformTensor(get_handle(), &alpha, idesc, iptr, &beta, odesc, optr));

	CUDNN_ERROR(cudnnDestroyTensorDescriptor(odesc));
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(idesc));
}

#define MAX_DIMS 16
struct conv_desc_s {

	unsigned int N;

	long odims[MAX_DIMS];
	long idims[MAX_DIMS];
	long kdims[MAX_DIMS];

	long ostrs[MAX_DIMS];
	long istrs[MAX_DIMS];
	long kstrs[MAX_DIMS];

	long strides[MAX_DIMS];
	long dilations[MAX_DIMS];

	bool conv;

	unsigned long conv_flags;
	unsigned long batch_flags;
	unsigned long channel_in_flags;
	unsigned long channel_out_flags;
	unsigned long group_flags;
};

static int flag_to_index(unsigned long flag)
{
	if (1 != bitcount(flag))
		return -1;

	for (unsigned int i = 0; i < 8 * sizeof(flag); i++)
		if (MD_IS_SET(flag, i))
			return i;
	return -1;
}

static bool check_cudnn_convcorr(struct conv_desc_s bart_conv_desc)
{
	if (3 < bitcount(bart_conv_desc.conv_flags))
		return false;
	if (1 < bitcount(bart_conv_desc.channel_in_flags))
		return false;
	if (1 < bitcount(bart_conv_desc.channel_out_flags))
		return false;
	if (0 < bitcount(bart_conv_desc.group_flags))
		return false;
	if (1 < bitcount(bart_conv_desc.batch_flags))
		return false;
	if (bart_conv_desc.conv)  // should work, not tested
		return false;

	return true;
}

static struct conv_desc_s create_conv_desc(	int N,
						const long odims[N], const long ostrs[N],
						const long idims[N], const long istrs[N],
						const long kdims[N], const long kstrs[N],
						const long dilations[N],
						const long strides[N],
						unsigned long conv_flags,
						bool conv
						)
{
	struct conv_desc_s result;

	unsigned long non_singleton_flags = md_nontriv_dims(N, odims) | md_nontriv_dims(N, idims) | md_nontriv_dims(N, kdims);

	assert(MAX_DIMS >= N);

	result.N = N;
	result.conv_flags = conv_flags & non_singleton_flags;
	result.conv = conv;

	md_singleton_dims(MAX_DIMS, result.odims);
	md_singleton_dims(MAX_DIMS, result.idims);
	md_singleton_dims(MAX_DIMS, result.kdims);

	md_singleton_dims(MAX_DIMS, result.dilations);
	md_singleton_dims(MAX_DIMS, result.strides);

	md_singleton_strides(MAX_DIMS, result.ostrs);
	md_singleton_strides(MAX_DIMS, result.istrs);
	md_singleton_strides(MAX_DIMS, result.kstrs);

	md_copy_dims(N, result.odims, odims);
	md_copy_dims(N, result.idims, idims);
	md_copy_dims(N, result.kdims, kdims);

	if (NULL != dilations)
		md_copy_dims(N, result.dilations, dilations);
	if (NULL != strides)
		md_copy_dims(N, result.strides, strides);

	md_copy_strides(N, result.ostrs, ostrs);
	md_copy_strides(N, result.istrs, istrs);
	md_copy_strides(N, result.kstrs, kstrs);

	result.batch_flags = 0;
	result.channel_in_flags = 0;
	result.channel_out_flags = 0;
	result.group_flags = 0;

	for (int i = 0; i < N; i++) {

		if ((odims[i] == idims[i]) && (1 == kdims[i]) && (1 != odims[i]))
			result.batch_flags = MD_SET(result.batch_flags, i);

		if ((kdims[i] == idims[i]) && (1 == odims[i]) && (1 != kdims[i]))
			result.channel_in_flags = MD_SET(result.channel_in_flags, i);

		if ((kdims[i] == odims[i]) && (1 == idims[i]) && (1 != kdims[i]))
			result.channel_out_flags = MD_SET(result.channel_out_flags, i);

		if ((kdims[i] == idims[i]) && (kdims[i] == odims[i]) && (1 != idims[i]))
			result.group_flags = MD_SET(result.group_flags, i);

	}

	result.batch_flags &= ~conv_flags;
	result.channel_in_flags &= ~conv_flags;
	result.channel_out_flags &= ~conv_flags;
	result.group_flags &= ~conv_flags;

	result.batch_flags &= non_singleton_flags;
	result.channel_in_flags &= non_singleton_flags;
	result.channel_out_flags &= non_singleton_flags;
	result.group_flags &= non_singleton_flags;

	return result;
}

struct cudnn_filter_s {

	cudnnTensorFormat_t format;
	cudnnFilterDescriptor_t filter_desc;

	bool transform_needed;
	cudnnTensorDescriptor_t transformed_filter_tensor_desc;
	cudnnTensorDescriptor_t input_filter_tensor_desc;

	size_t size_transformed;
};

static struct cudnn_filter_s get_filter_descriptor(struct conv_desc_s conv_desc, cudnnTensorFormat_t format)
{
	struct cudnn_filter_s result;
	result.format = format;

	result.size_transformed = md_calc_size(conv_desc.N, conv_desc.kdims) * FL_SIZE;

	assert(1 >= bitcount(conv_desc.channel_in_flags));
	assert(1 >= bitcount(conv_desc.channel_out_flags));
	assert(1 <= bitcount(conv_desc.conv_flags));

	int in_channel_index = -1;
	int out_channel_index = -1;

	for (unsigned int i = 0; i < conv_desc.N; i++) {

		if (MD_IS_SET(conv_desc.channel_in_flags, i))
			in_channel_index = i;
		if (MD_IS_SET(conv_desc.channel_out_flags, i))
			out_channel_index = i;
	}

	int nbDims = bitcount(conv_desc.conv_flags) + 2;
	int filterDimA[MAX(4, nbDims)];
	int filterStrA[MAX(4, nbDims)];

	filterDimA[0] = (-1 == out_channel_index) ? 1 : conv_desc.kdims[out_channel_index];
	filterStrA[0] = (-1 == out_channel_index) ? 1 : conv_desc.kstrs[out_channel_index] / FL_SIZE;

	filterDimA[1] = (-1 == in_channel_index) ? 1 : conv_desc.kdims[in_channel_index];
	filterStrA[1] = (-1 == in_channel_index) ? 1 : conv_desc.kstrs[in_channel_index] / FL_SIZE;

	for (int i = 0, ir = nbDims - 1; ir >= 2; i++) {

		if (!MD_IS_SET(conv_desc.conv_flags, i))
			continue;

		filterDimA[ir] = conv_desc.kdims[i];
		filterStrA[ir] = conv_desc.kstrs[i] / FL_SIZE;

		ir--;
	}

	for (int i = 0; i < nbDims; i++) {

		if (0 == filterStrA[i]) {

			assert(1 == filterDimA[i]);
			filterStrA[i] = 1;
		}
	}

	if (3 == nbDims) {

		filterDimA[3] = 1;
		filterStrA[3] = 1;
	}

	CUDNN_ERROR(cudnnCreateFilterDescriptor(&result.filter_desc));
	CUDNN_ERROR(cudnnSetFilterNdDescriptor(result.filter_desc, CUDNN_DATA_FLOAT, format, MAX(4, nbDims), filterDimA));

	int filterStrT[MAX(4, nbDims)];
	filterStrT[MAX(4, nbDims) - 1] = (format == CUDNN_TENSOR_NCHW) ? 1 : filterDimA[1];

	for (int i = MAX(4, nbDims) - 2; i >=2; i--)
		filterStrT[i] = filterStrT[i + 1] * filterDimA[i + 1];

	filterStrT[1] = (format == CUDNN_TENSOR_NCHW) ? filterDimA[2] * filterStrT[2] : 1;
	filterStrT[0] = (format == CUDNN_TENSOR_NCHW) ? filterDimA[1] * filterStrT[1] : filterDimA[2] * filterStrT[2];


	CUDNN_ERROR(cudnnCreateTensorDescriptor(&result.transformed_filter_tensor_desc));
	CUDNN_ERROR(cudnnSetTensorNdDescriptor(result.transformed_filter_tensor_desc, CUDNN_DATA_FLOAT, MAX(4, nbDims), filterDimA, filterStrT));

	CUDNN_ERROR(cudnnCreateTensorDescriptor(&result.input_filter_tensor_desc));
	CUDNN_ERROR(cudnnSetTensorNdDescriptor(result.input_filter_tensor_desc, CUDNN_DATA_FLOAT, MAX(4, nbDims), filterDimA, filterStrA));

	result.transform_needed = true;

	for (int i = 0; i < MAX(4, nbDims); i++)
		if ((1 != filterDimA[i]) && (filterStrT[i] != filterStrA[i]))
			result.transform_needed = true;

	return result;
}

static void free_filter_descriptor(struct cudnn_filter_s desc)
{
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(desc.transformed_filter_tensor_desc));
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(desc.input_filter_tensor_desc));
	CUDNN_ERROR(cudnnDestroyFilterDescriptor(desc.filter_desc));
}

struct cudnn_tensor_s {

	cudnnTensorFormat_t format;

	bool transform_needed;
	cudnnTensorDescriptor_t transformed_tensor_desc;
	cudnnTensorDescriptor_t input_tensor_desc;

	size_t size_transformed;
};

static struct cudnn_tensor_s get_tensor_descriptor(struct conv_desc_s conv_desc, bool output, cudnnTensorFormat_t format)
{
	struct cudnn_tensor_s result;
	result.format = format;

	unsigned long channel_flags = output ? conv_desc.channel_out_flags : conv_desc.channel_in_flags;
	long* dims = output ? conv_desc.odims : conv_desc.idims;
	long* strs = output ? conv_desc.ostrs : conv_desc.istrs;

	result.size_transformed = md_calc_size(conv_desc.N, dims) * FL_SIZE;

	assert(1 >= bitcount(channel_flags));
	assert(1 <= bitcount(conv_desc.conv_flags));
	assert(1 >= bitcount(conv_desc.batch_flags));

	int channel_index = -1;
	int batch_index = -1;

	for (unsigned int i = 0; i < conv_desc.N; i++) {

		if (MD_IS_SET(channel_flags, i))
			channel_index = i;
		if (MD_IS_SET(conv_desc.batch_flags, i))
			batch_index = i;
	}

	int nbDims = bitcount(conv_desc.conv_flags) + 2;
	int dimA[MAX(4, nbDims)];
	int strA[MAX(4, nbDims)];

	dimA[0] = (-1 == batch_index) ? 1 : dims[batch_index];
	strA[0] = (-1 == batch_index) ? 1 : strs[batch_index] / FL_SIZE;

	dimA[1] = (-1 == channel_index) ? 1 : dims[channel_index];
	strA[1] = (-1 == channel_index) ? 1 : strs[channel_index] / FL_SIZE;

	for (int i = 0, ir = nbDims - 1; ir >= 2; i++) {

		if (!MD_IS_SET(conv_desc.conv_flags, i))
			continue;

		dimA[ir] = dims[i];
		strA[ir] = strs[i] / FL_SIZE;

		ir--;
	}

	for (int i = 0; i < nbDims; i++) {

		if (0 == strA[i]) {

			assert(1 == dimA[i]);
			strA[i] = 1;
		}
	}

	if (3 == nbDims) {

		dimA[3] = 1;
		strA[3] = 1;
	}

	int strT[MAX(4, nbDims)];
	strT[MAX(4, nbDims) - 1] = (format == CUDNN_TENSOR_NCHW) ? 1 : dimA[1];

	for (int i = MAX(4, nbDims) - 2; i >=2; i--)
		strT[i] = strT[i + 1] * dimA[i + 1];

	strT[1] = (format == CUDNN_TENSOR_NCHW) ? dimA[2] * strT[2] : 1;
	strT[0] = (format == CUDNN_TENSOR_NCHW) ? dimA[1] * strT[1] : dimA[2] * strT[2];

	//for (int i = 0; i < MAX(4, nbDims); i++)
	//	if(1 == dimA[i])
	//		strT[i] = 1;

	CUDNN_ERROR(cudnnCreateTensorDescriptor(&result.input_tensor_desc));
	CUDNN_ERROR(cudnnSetTensorNdDescriptor(result.input_tensor_desc, CUDNN_DATA_FLOAT, MAX(4, nbDims), dimA, strA));

	CUDNN_ERROR(cudnnCreateTensorDescriptor(&result.transformed_tensor_desc));
	CUDNN_ERROR(cudnnSetTensorNdDescriptor(result.transformed_tensor_desc, CUDNN_DATA_FLOAT, MAX(4, nbDims), dimA, strT));

	result.transform_needed = true;

	for (int i = MAX(4, nbDims) - 1; i >=0; i--)
		if ((strT[i] != strA[i]) &&( 1 != dimA[i]))
			result.transform_needed = false;

	return result;
}

static void free_tensor_descriptor(struct cudnn_tensor_s desc)
{
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(desc.transformed_tensor_desc));
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(desc.input_tensor_desc));
}

static cudnnConvolutionDescriptor_t get_conv_descriptor(struct conv_desc_s conv_desc)
{
	cudnnConvolutionDescriptor_t result;
	CUDNN_ERROR(cudnnCreateConvolutionDescriptor(&result));

	int nbDims = bitcount(conv_desc.conv_flags) + 2;

	int padA[MAX(4, nbDims)];
	int filterStrideA[MAX(4, nbDims)];
	int dilationA[MAX(4, nbDims)];

	for (int i = 0; i < MAX(4, nbDims); i++) {

		padA[i] = 0;
		filterStrideA[i] = 1;
		dilationA[i] = 1;
	}

	for (int i = 0, ir = nbDims - 1; ir >= 2; i++) {

		if (!MD_IS_SET(conv_desc.conv_flags, i))
			continue;

		filterStrideA[ir] = conv_desc.strides[i];
		dilationA[ir] = conv_desc.dilations[i];

		ir--;
	}

	CUDNN_ERROR(cudnnSetConvolutionNdDescriptor(result, MAX(4, nbDims) - 2, padA + 2, filterStrideA+ 2, dilationA + 2, conv_desc.conv ? CUDNN_CONVOLUTION : CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	return result;
}




static bool cudnn_convcorr_fwd_int(	float alpha,
					float beta,
 					const cudnnConvolutionDescriptor_t conv_desc,
					const cudnnTensorDescriptor_t in_desc,
					const float* in,
 					const cudnnFilterDescriptor_t krn_desc,
 					const float* krn,
 					const cudnnTensorDescriptor_t out_desc,
					float* out)
{
	int N_algos;
	CUDNN_CALL(cudnnGetConvolutionForwardAlgorithmMaxCount(get_handle(), &N_algos));
	cudnnConvolutionFwdAlgoPerf_t algos[N_algos];
	CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(get_handle(), in_desc, krn_desc, conv_desc, out_desc, N_algos, &N_algos, algos));

	size_t in_size;
	size_t out_size;
	CUDNN_ERROR(cudnnGetTensorSizeInBytes(in_desc, &in_size));
	CUDNN_ERROR(cudnnGetTensorSizeInBytes(out_desc, &out_size));

	cudnnConvolutionFwdAlgoPerf_t* algo = NULL;
	for (int i = 0; i < N_algos; i++) {

		bool applicable= 8 * (in_size + out_size) > algos[i].memory;
		applicable = applicable && (algos[i].status == CUDNN_STATUS_SUCCESS);
		#ifndef NON_DETERMINISTIC
		applicable = applicable && algos[i].determinism == CUDNN_DETERMINISTIC;
		#endif

		if (applicable){
			algo = algos + i;
			break;
		}

		if (i == N_algos - 1)
			return false;
	}

	size_t ws_size = algo->memory;
	void* workspace = (0 < ws_size) ? md_alloc_gpu(1, MD_DIMS(1), ws_size) : NULL;

	cudnn_set_gpulock();
	cudnnStatus_t status = cudnnConvolutionForward(get_handle(), &alpha, in_desc, in, krn_desc, krn, conv_desc, algo->algo, workspace, ws_size, &beta, out_desc, out);
	cudnn_unset_gpulock();
	md_free(workspace);

	if (CUDNN_STATUS_NOT_SUPPORTED == status)
		return false;

	CUDNN_ERROR(status);
	return true;
}

static bool cudnn_convcorr_bwd_krn_int(	float alpha,
					float beta,
 					const cudnnConvolutionDescriptor_t conv_desc,
					const cudnnTensorDescriptor_t in_desc,
					const float* in,
 					const cudnnFilterDescriptor_t krn_desc,
 					float* krn,
 					const cudnnTensorDescriptor_t out_desc,
					const float* out)
{
	int N_algos;
	CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(get_handle(), &N_algos));
	cudnnConvolutionBwdFilterAlgoPerf_t algos[N_algos];
	CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm_v7(get_handle(), in_desc, out_desc, conv_desc, krn_desc, N_algos, &N_algos, algos));

	size_t in_size;
	size_t out_size;
	CUDNN_ERROR(cudnnGetTensorSizeInBytes(in_desc, &in_size));
	CUDNN_ERROR(cudnnGetTensorSizeInBytes(out_desc, &out_size));

	cudnnConvolutionBwdFilterAlgoPerf_t* algo = NULL;
	for (int i = 0; i < N_algos; i++) {

		bool applicable= 8 * (in_size + out_size) > algos[i].memory;
		applicable = applicable && (algos[i].status == CUDNN_STATUS_SUCCESS);
		#ifndef NON_DETERMINISTIC
		applicable = applicable && algos[i].determinism == CUDNN_DETERMINISTIC;
		#endif

		if (applicable){
			algo = algos + i;
			break;
		}

		if (i == N_algos - 1)
			return false;
	}

	size_t ws_size = algo->memory;
	void* workspace = (0 < ws_size) ? md_alloc_gpu(1, MD_DIMS(1), ws_size) : NULL;

	cudnn_set_gpulock();
	cudnnStatus_t status = cudnnConvolutionBackwardFilter(get_handle(), &alpha, in_desc, in, out_desc, out, conv_desc, algo->algo, workspace, ws_size, &beta, krn_desc, krn);
	cudnn_unset_gpulock();
	md_free(workspace);

	if (CUDNN_STATUS_NOT_SUPPORTED == status)
		return false;

	CUDNN_ERROR(status);
	return true;
}

static bool cudnn_convcorr_bwd_in_int(	float alpha,
					float beta,
 					const cudnnConvolutionDescriptor_t conv_desc,
					const cudnnTensorDescriptor_t in_desc,
					float* in,
 					const cudnnFilterDescriptor_t krn_desc,
 					const float* krn,
 					const cudnnTensorDescriptor_t out_desc,
					const float* out)
{
	int N_algos;
	CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(get_handle(), &N_algos));
	cudnnConvolutionBwdDataAlgoPerf_t algos[N_algos];
	CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm_v7(get_handle(), krn_desc, out_desc, conv_desc, in_desc, N_algos, &N_algos, algos));

	size_t in_size;
	size_t out_size;
	CUDNN_ERROR(cudnnGetTensorSizeInBytes(in_desc, &in_size));
	CUDNN_ERROR(cudnnGetTensorSizeInBytes(out_desc, &out_size));

	cudnnConvolutionBwdDataAlgoPerf_t* algo = NULL;
	for (int i = 0; i < N_algos; i++) {

		bool applicable= 8 * (in_size + out_size) > algos[i].memory;
		applicable = applicable && (algos[i].status == CUDNN_STATUS_SUCCESS);
		#ifndef NON_DETERMINISTIC
		applicable = applicable && algos[i].determinism == CUDNN_DETERMINISTIC;
		#endif

		if (applicable){
			algo = algos + i;
			break;
		}

		if (i == N_algos - 1)
			return false;
	}

	size_t ws_size = algo->memory;
	void* workspace = (0 < ws_size) ? md_alloc_gpu(1, MD_DIMS(1), ws_size) : NULL;

	cudnn_set_gpulock();
	cudnnStatus_t status = cudnnConvolutionBackwardData(get_handle(), &alpha, krn_desc, krn, out_desc, out, conv_desc, algo->algo, workspace, ws_size, &beta, in_desc, in);
	cudnn_unset_gpulock();
	md_free(workspace);

	if (CUDNN_STATUS_NOT_SUPPORTED == status)
		return false;

	CUDNN_ERROR(status);
	return true;
}

static void cudnn_tensor_transform(float alpha, float beta, cudnnTensorDescriptor_t dst_desc, float* dst, cudnnTensorDescriptor_t src_desc, const float* src)
{
	CUDNN_CALL(cudnnTransformTensor(get_handle(), &alpha, src_desc, src, &beta, dst_desc, dst));
}

static void cudnn_tensor_transform_split_complex(float alpha, float beta, cudnnTensorDescriptor_t real_desc, float* real, float* imag, cudnnTensorDescriptor_t comp_desc, const complex float* comp)
{
	int nbDims = 8;
	cudnnDataType_t dataType;
	int dimA_comp[nbDims];
	int strA_comp[nbDims];

	CUDNN_ERROR(cudnnGetTensorNdDescriptor(comp_desc, nbDims, &dataType, &nbDims, dimA_comp, strA_comp));

	bool decomp_bart = true;

	long dims[nbDims];
	long strs[nbDims];

	for (int i = 0; i < nbDims; i++) {

		dims[i] = dimA_comp[i];
		strs[i] = strA_comp[i] * FL_SIZE;

		decomp_bart = decomp_bart && ((1 == dimA_comp[i]) || (0 == strA_comp[i] % 2));

		strA_comp[i] = (1 == dimA_comp[i]) ? 1 : strA_comp[i] / 2;
	}

	long (*tstrs[1])[nbDims] = {(long (*)[nbDims])strs};
	decomp_bart = decomp_bart && (1 == optimize_dims_gpu(1, nbDims, dims, tstrs));

	if (decomp_bart) {

		float* real_tmp = md_alloc_gpu(1, dims, FL_SIZE);
		float* imag_tmp = md_alloc_gpu(1, dims, FL_SIZE);

		md_real(1, dims, real_tmp, comp);
		md_imag(1, dims, imag_tmp, comp);

		cudnnTensorDescriptor_t tDesc;
		CUDNN_ERROR(cudnnCreateTensorDescriptor(&tDesc));
		CUDNN_ERROR(cudnnSetTensorNdDescriptor(tDesc, dataType, nbDims, dimA_comp, strA_comp));

		CUDNN_CALL(cudnnTransformTensor(get_handle(), &alpha, tDesc, real_tmp, &beta, real_desc, real));
		CUDNN_CALL(cudnnTransformTensor(get_handle(), &alpha, tDesc, imag_tmp, &beta, real_desc, imag));

		CUDNN_ERROR(cudnnDestroyTensorDescriptor(tDesc));

		md_free(real_tmp);
		md_free(imag_tmp);
	} else {

		CUDNN_CALL(cudnnTransformTensor(get_handle(), &alpha, comp_desc, (const float*)comp + 0, &beta, real_desc, real));
		CUDNN_CALL(cudnnTransformTensor(get_handle(), &alpha, comp_desc, (const float*)comp + 1, &beta, real_desc, imag));
	}
}

static void cudnn_tensor_transform_combine_complex(float alpha, float beta, cudnnTensorDescriptor_t comp_desc, complex float* comp, cudnnTensorDescriptor_t real_desc, const float* real, const float* imag)
{
	int nbDims = 8;
	cudnnDataType_t dataType;
	int dimA_comp[nbDims];
	int strA_comp[nbDims];

	CUDNN_ERROR(cudnnGetTensorNdDescriptor(comp_desc, nbDims, &dataType, &nbDims, dimA_comp, strA_comp));

	bool decomp_bart = true;

	long dims[nbDims];
	long strs[nbDims];

	for (int i = 0; i < nbDims; i++) {

		dims[i] = dimA_comp[i];
		strs[i] = strA_comp[i] * FL_SIZE;

		decomp_bart = decomp_bart && ((1 == dimA_comp[i]) || (0 == strA_comp[i] % 2));

		strA_comp[i] = (1 == dimA_comp[i]) ? 1 : strA_comp[i] / 2;
	}

	long (*tstrs[1])[nbDims] = {(long (*)[nbDims])strs};
	decomp_bart = decomp_bart && (1 == optimize_dims_gpu(1, nbDims, dims, tstrs));

	if (decomp_bart) {

		float* real_tmp = md_alloc_gpu(1, dims, FL_SIZE);
		float* imag_tmp = md_alloc_gpu(1, dims, FL_SIZE);

		if (0 != beta) {

			md_real(1, dims, real_tmp, comp);
			md_imag(1, dims, imag_tmp, comp);
		}

		cudnnTensorDescriptor_t tDesc;
		CUDNN_ERROR(cudnnCreateTensorDescriptor(&tDesc));
		CUDNN_ERROR(cudnnSetTensorNdDescriptor(tDesc, dataType, nbDims, dimA_comp, strA_comp));

		CUDNN_CALL(cudnnTransformTensor(get_handle(), &alpha, real_desc, real, &beta, tDesc, real_tmp));
		CUDNN_CALL(cudnnTransformTensor(get_handle(), &alpha, real_desc, imag, &beta, tDesc, imag_tmp));

		CUDNN_ERROR(cudnnDestroyTensorDescriptor(tDesc));

		md_zcmpl(1, dims, comp, real_tmp, imag_tmp);

		md_free(real_tmp);
		md_free(imag_tmp);
	} else {

		CUDNN_CALL(cudnnTransformTensor(get_handle(), &alpha, real_desc, real, &beta, comp_desc, (float*)comp + 0));
		CUDNN_CALL(cudnnTransformTensor(get_handle(), &alpha, real_desc, imag, &beta, comp_desc, (float*)comp + 1));
	}
}
// *_split methodes compute four real convolutions
// to compute one complex
static bool cudnn_zconvcorr_fwd_split(
 			struct conv_desc_s bcd,
			const complex float* in,
 			const complex float* krn,
			complex float* out,
			cudnnTensorFormat_t format
			)
{
	cudnnConvolutionDescriptor_t conv_desc = get_conv_descriptor(bcd);

	struct cudnn_tensor_s in_desc = get_tensor_descriptor(bcd, false, format);
	struct cudnn_tensor_s out_desc = get_tensor_descriptor(bcd, true, format);
	struct cudnn_filter_s krn_desc = get_filter_descriptor(bcd, format);

	float* krn_real = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
	float* krn_imag = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);

	cudnn_tensor_transform_split_complex(1., 0., krn_desc.transformed_filter_tensor_desc, krn_real, krn_imag, krn_desc.input_filter_tensor_desc, krn);


	const float* in_real = (const float*)in + 0;
	const float* in_imag = (const float*)in + 1;

	float* out_real = (float*)out + 0;
	float* out_imag = (float*)out + 1;

	bool direct = false; // if true, cudnn tries to perform convolution with out transformation

	direct = direct && cudnn_convcorr_fwd_int(1. , 1., conv_desc, in_desc.input_tensor_desc, in_real, krn_desc.filter_desc, krn_real, out_desc.input_tensor_desc, out_real);
	direct = direct && cudnn_convcorr_fwd_int(1. , 1., conv_desc, in_desc.input_tensor_desc, in_imag, krn_desc.filter_desc, krn_real, out_desc.input_tensor_desc, out_imag);

	direct = direct && cudnn_convcorr_fwd_int(1. , 1., conv_desc, in_desc.input_tensor_desc, in_real, krn_desc.filter_desc, krn_imag, out_desc.input_tensor_desc, out_imag);
	direct = direct && cudnn_convcorr_fwd_int(-1. , 1., conv_desc, in_desc.input_tensor_desc, in_imag, krn_desc.filter_desc, krn_imag, out_desc.input_tensor_desc, out_real);

	bool success = true;

	if (!direct) {

		float* in_real2 = md_alloc_gpu(bcd.N, bcd.idims, FL_SIZE);
		float* in_imag2 = md_alloc_gpu(bcd.N, bcd.idims, FL_SIZE);

		float* out_real2 = md_alloc_gpu(bcd.N, bcd.odims, FL_SIZE);
		float* out_imag2 = md_alloc_gpu(bcd.N, bcd.odims, FL_SIZE);

		cudnn_tensor_transform_split_complex(1., 0., in_desc.transformed_tensor_desc, in_real2, in_imag2, in_desc.input_tensor_desc, in);

		success = success && cudnn_convcorr_fwd_int(1. , 0., conv_desc, in_desc.transformed_tensor_desc, in_real2, krn_desc.filter_desc, krn_real, out_desc.transformed_tensor_desc, out_real2);
		success = success && cudnn_convcorr_fwd_int(1. , 0., conv_desc, in_desc.transformed_tensor_desc, in_imag2, krn_desc.filter_desc, krn_real, out_desc.transformed_tensor_desc, out_imag2);

		success = success && cudnn_convcorr_fwd_int( 1., 1., conv_desc, in_desc.transformed_tensor_desc, in_real2, krn_desc.filter_desc, krn_imag, out_desc.transformed_tensor_desc, out_imag2);
		success = success && cudnn_convcorr_fwd_int(-1., 1., conv_desc, in_desc.transformed_tensor_desc, in_imag2, krn_desc.filter_desc, krn_imag, out_desc.transformed_tensor_desc, out_real2);

		if (success)
			cudnn_tensor_transform_combine_complex(1., 1., out_desc.input_tensor_desc, out, out_desc.transformed_tensor_desc, out_real2, out_imag2);


		md_free(in_real2);
		md_free(in_imag2);
		md_free(out_real2);
		md_free(out_imag2);
	}

	md_free(krn_real);
	md_free(krn_imag);

	free_tensor_descriptor(in_desc);
	free_tensor_descriptor(out_desc);

	free_filter_descriptor(krn_desc);

	CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(conv_desc));

	return success;
}

static bool cudnn_zconvcorr_bwd_in_split(
 			struct conv_desc_s bcd,
			complex float* in,
 			const complex float* krn,
			const complex float* out,
			cudnnTensorFormat_t format
			)
{
	cudnnConvolutionDescriptor_t conv_desc = get_conv_descriptor(bcd);

	struct cudnn_tensor_s in_desc = get_tensor_descriptor(bcd, false, format);
	struct cudnn_tensor_s out_desc = get_tensor_descriptor(bcd, true, format);
	struct cudnn_filter_s krn_desc = get_filter_descriptor(bcd, format);

	float* krn_real = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
	float* krn_imag = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);

	cudnn_tensor_transform_split_complex(1., 0., krn_desc.transformed_filter_tensor_desc, krn_real, krn_imag, krn_desc.input_filter_tensor_desc, krn);

	float* in_real = (float*)in + 0;
	float* in_imag = (float*)in + 1;

	const float* out_real = (const float*)out + 0;
	const float* out_imag = (const float*)out + 1;

	bool direct = false; // if true, cudnn tries to perform convolution with out transformation

	direct = direct && cudnn_convcorr_bwd_in_int( 1. , 1., conv_desc, in_desc.input_tensor_desc, in_real, krn_desc.filter_desc, krn_real, out_desc.input_tensor_desc, out_real);
	direct = direct && cudnn_convcorr_bwd_in_int( 1. , 1., conv_desc, in_desc.input_tensor_desc, in_imag, krn_desc.filter_desc, krn_real, out_desc.input_tensor_desc, out_imag);

	direct = direct && cudnn_convcorr_bwd_in_int(-1. , 1., conv_desc, in_desc.input_tensor_desc, in_real, krn_desc.filter_desc, krn_imag, out_desc.input_tensor_desc, out_imag);
	direct = direct && cudnn_convcorr_bwd_in_int( 1. , 1., conv_desc, in_desc.input_tensor_desc, in_imag, krn_desc.filter_desc, krn_imag, out_desc.input_tensor_desc, out_real);

	bool success = true;

	if (!direct) {

		float* out_real2 = md_alloc_gpu(1, MD_DIMS(1), out_desc.size_transformed);
		float* out_imag2 = md_alloc_gpu(1, MD_DIMS(1), out_desc.size_transformed);

		float* in_real2 = md_alloc_gpu(1, MD_DIMS(1), in_desc.size_transformed);
		float* in_imag2 = md_alloc_gpu(1, MD_DIMS(1), in_desc.size_transformed);

		cudnn_tensor_transform_split_complex(1., 0., out_desc.transformed_tensor_desc, out_real2, out_imag2, out_desc.input_tensor_desc, out);

		success = success && cudnn_convcorr_bwd_in_int(1. , 0., conv_desc, in_desc.transformed_tensor_desc, in_real2, krn_desc.filter_desc, krn_real, out_desc.transformed_tensor_desc, out_real2);
		success = success && cudnn_convcorr_bwd_in_int(1. , 0., conv_desc, in_desc.transformed_tensor_desc, in_imag2, krn_desc.filter_desc, krn_real, out_desc.transformed_tensor_desc, out_imag2);

		success = success && cudnn_convcorr_bwd_in_int(-1. , 1., conv_desc, in_desc.transformed_tensor_desc, in_real2, krn_desc.filter_desc, krn_imag, out_desc.transformed_tensor_desc, out_imag2);
		success = success && cudnn_convcorr_bwd_in_int(1. , 1., conv_desc, in_desc.transformed_tensor_desc, in_imag2, krn_desc.filter_desc, krn_imag, out_desc.transformed_tensor_desc, out_real2);

		if (success)
			cudnn_tensor_transform_combine_complex(1., 1., in_desc.input_tensor_desc, in, in_desc.transformed_tensor_desc, in_real2, in_imag2);

		md_free(in_real2);
		md_free(in_imag2);
		md_free(out_real2);
		md_free(out_imag2);
	}

	md_free(krn_real);
	md_free(krn_imag);

	free_tensor_descriptor(in_desc);
	free_tensor_descriptor(out_desc);

	free_filter_descriptor(krn_desc);

	CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(conv_desc));

	return success;
}


static bool cudnn_zconvcorr_bwd_krn_split(
			struct conv_desc_s bcd,
			const complex float* in,
 			complex float* krn,
			const complex float* out,
			cudnnTensorFormat_t format
			)
{
	cudnnConvolutionDescriptor_t conv_desc = get_conv_descriptor(bcd);

	struct cudnn_tensor_s in_desc = get_tensor_descriptor(bcd, false, format);
	struct cudnn_tensor_s out_desc = get_tensor_descriptor(bcd, true, format);
	struct cudnn_filter_s krn_desc = get_filter_descriptor(bcd, format);

	float* krn_real = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
	float* krn_imag = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);

	const float* in_real = (const float*)in + 0;
	const float* in_imag = (const float*)in + 1;

	const float* out_real = (const float*)out + 0;
	const float* out_imag = (const float*)out + 1;

	bool direct = false; // if true, cudnn tries to perform convolution with out transformation

	direct = direct && cudnn_convcorr_bwd_krn_int( 1. , 0., conv_desc, in_desc.input_tensor_desc, in_real, krn_desc.filter_desc, krn_real, out_desc.input_tensor_desc, out_real);
	direct = direct && cudnn_convcorr_bwd_krn_int(-1. , 1., conv_desc, in_desc.input_tensor_desc, in_imag, krn_desc.filter_desc, krn_real, out_desc.input_tensor_desc, out_imag);

	direct = direct && cudnn_convcorr_bwd_krn_int( 1. , 0., conv_desc, in_desc.input_tensor_desc, in_real, krn_desc.filter_desc, krn_imag, out_desc.input_tensor_desc, out_imag);
	direct = direct && cudnn_convcorr_bwd_krn_int( 1. , 1., conv_desc, in_desc.input_tensor_desc, in_imag, krn_desc.filter_desc, krn_imag, out_desc.input_tensor_desc, out_real);

	bool success = true;

	if (!direct) {

		float* out_real2 = md_alloc_gpu(1, MD_DIMS(1), out_desc.size_transformed);
		float* out_imag2 = md_alloc_gpu(1, MD_DIMS(1), out_desc.size_transformed);

		float* in_real2 = md_alloc_gpu(1, MD_DIMS(1), in_desc.size_transformed);
		float* in_imag2 = md_alloc_gpu(1, MD_DIMS(1), in_desc.size_transformed);

		cudnn_tensor_transform_split_complex(1., 0., out_desc.transformed_tensor_desc, out_real2, out_imag2, out_desc.input_tensor_desc, out);
		cudnn_tensor_transform_split_complex(1., 0., in_desc.transformed_tensor_desc, in_real2, in_imag2, in_desc.input_tensor_desc, in);

		success = success && cudnn_convcorr_bwd_krn_int( 1. , 0., conv_desc, in_desc.transformed_tensor_desc, in_real2, krn_desc.filter_desc, krn_real, out_desc.transformed_tensor_desc, out_real2);
		success = success && cudnn_convcorr_bwd_krn_int(-1. , 1., conv_desc, in_desc.transformed_tensor_desc, in_imag2, krn_desc.filter_desc, krn_real, out_desc.transformed_tensor_desc, out_imag2);

		success = success && cudnn_convcorr_bwd_krn_int( 1. , 0., conv_desc, in_desc.transformed_tensor_desc, in_real2, krn_desc.filter_desc, krn_imag, out_desc.transformed_tensor_desc, out_imag2);
		success = success && cudnn_convcorr_bwd_krn_int( 1. , 1., conv_desc, in_desc.transformed_tensor_desc, in_imag2, krn_desc.filter_desc, krn_imag, out_desc.transformed_tensor_desc, out_real2);

		md_free(in_real2);
		md_free(in_imag2);
		md_free(out_real2);
		md_free(out_imag2);
	}

	if (success)
		cudnn_tensor_transform_combine_complex(1., 1., krn_desc.input_filter_tensor_desc, krn, krn_desc.transformed_filter_tensor_desc, krn_real, krn_imag);

	md_free(krn_real);
	md_free(krn_imag);

	free_tensor_descriptor(in_desc);
	free_tensor_descriptor(out_desc);

	free_filter_descriptor(krn_desc);

	CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(conv_desc));

	return success;
}

// these methodes compute real convolutions
static bool cudnn_convcorr_fwd(
 			struct conv_desc_s bcd,
			const float* in,
 			const float* krn,
			float* out,
			cudnnTensorFormat_t format
			)
{
	cudnnConvolutionDescriptor_t conv_desc = get_conv_descriptor(bcd);

	struct cudnn_tensor_s in_desc = get_tensor_descriptor(bcd, false, format);
	struct cudnn_tensor_s out_desc = get_tensor_descriptor(bcd, true, format);
	struct cudnn_filter_s krn_desc = get_filter_descriptor(bcd, format);

	float* krn_tmp = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
	cudnn_tensor_transform(1, 0, krn_desc.transformed_filter_tensor_desc, krn_tmp, krn_desc.input_filter_tensor_desc, krn);

	bool direct = false; // if true, cudnn tries to perform convolution with out transformation
	direct = direct && cudnn_convcorr_fwd_int(1. , 1., conv_desc, in_desc.input_tensor_desc, in, krn_desc.filter_desc, krn_tmp, out_desc.input_tensor_desc, out);

	bool success = true;

	if (!direct) {

		float* in_tmp = md_alloc_gpu(bcd.N, bcd.idims, FL_SIZE);
		float* out_tmp = md_alloc_gpu(bcd.N, bcd.odims, FL_SIZE);

		cudnn_tensor_transform(1., 0., in_desc.transformed_tensor_desc, in_tmp, in_desc.input_tensor_desc, in);
		success = success && cudnn_convcorr_fwd_int(1. , 0., conv_desc, in_desc.transformed_tensor_desc, in_tmp, krn_desc.filter_desc, krn_tmp, out_desc.transformed_tensor_desc, out_tmp);
		if (success)
			cudnn_tensor_transform(1., 1., out_desc.input_tensor_desc, out, out_desc.transformed_tensor_desc, out_tmp);

		md_free(in_tmp);
		md_free(out_tmp);
	}

	md_free(krn_tmp);

	free_tensor_descriptor(in_desc);
	free_tensor_descriptor(out_desc);

	free_filter_descriptor(krn_desc);

	CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(conv_desc));

	return success;
}

static bool cudnn_convcorr_bwd_in(
 			struct conv_desc_s bcd,
			float* in,
 			const float* krn,
			const float* out,
			cudnnTensorFormat_t format
			)
{
	cudnnConvolutionDescriptor_t conv_desc = get_conv_descriptor(bcd);

	struct cudnn_tensor_s in_desc = get_tensor_descriptor(bcd, false, format);
	struct cudnn_tensor_s out_desc = get_tensor_descriptor(bcd, true, format);
	struct cudnn_filter_s krn_desc = get_filter_descriptor(bcd, format);

	float* krn_tmp = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
	cudnn_tensor_transform(1, 0, krn_desc.transformed_filter_tensor_desc, krn_tmp, krn_desc.input_filter_tensor_desc, krn);

	bool direct = false; // if true, cudnn tries to perform convolution with out transformation
	direct = direct && cudnn_convcorr_bwd_in_int( 1. , 1., conv_desc, in_desc.input_tensor_desc, in, krn_desc.filter_desc, krn_tmp, out_desc.input_tensor_desc, out);
	bool success = true;

	if (!direct) {

		float* out_tmp = md_alloc_gpu(1, MD_DIMS(1), out_desc.size_transformed);
		float* in_tmp = md_alloc_gpu(1, MD_DIMS(1), in_desc.size_transformed);

		cudnn_tensor_transform(1., 0., out_desc.transformed_tensor_desc, out_tmp, out_desc.input_tensor_desc, out);
		success = success && cudnn_convcorr_bwd_in_int(1. , 0., conv_desc, in_desc.transformed_tensor_desc, in_tmp, krn_desc.filter_desc, krn_tmp, out_desc.transformed_tensor_desc, out_tmp);
		if (success)
			cudnn_tensor_transform(1., 1., in_desc.input_tensor_desc, in, in_desc.transformed_tensor_desc, in_tmp);

		md_free(in_tmp);
		md_free(out_tmp);
	}

	md_free(krn_tmp);

	free_tensor_descriptor(in_desc);
	free_tensor_descriptor(out_desc);

	free_filter_descriptor(krn_desc);

	CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(conv_desc));

	return success;
}


static bool cudnn_convcorr_bwd_krn(
			struct conv_desc_s bcd,
			const float* in,
 			float* krn,
			const float* out,
			cudnnTensorFormat_t format
			)
{
	cudnnConvolutionDescriptor_t conv_desc = get_conv_descriptor(bcd);

	struct cudnn_tensor_s in_desc = get_tensor_descriptor(bcd, false, format);
	struct cudnn_tensor_s out_desc = get_tensor_descriptor(bcd, true, format);
	struct cudnn_filter_s krn_desc = get_filter_descriptor(bcd, format);

	float* krn_tmp = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);

	cudnn_tensor_transform(1., 0., krn_desc.transformed_filter_tensor_desc, krn_tmp, krn_desc.input_filter_tensor_desc, krn);

	bool direct = false; // if true, cudnn tries to perform convolution with out transformation
	direct = direct && cudnn_convcorr_bwd_krn_int( 1. , 0., conv_desc, in_desc.input_tensor_desc, in, krn_desc.filter_desc, krn_tmp, out_desc.input_tensor_desc, out);

	bool success = true;

	if (!direct) {

		float* out_tmp = md_alloc_gpu(1, MD_DIMS(1), out_desc.size_transformed);
		float* in_tmp = md_alloc_gpu(1, MD_DIMS(1), in_desc.size_transformed);

		cudnn_tensor_transform(1., 0., out_desc.transformed_tensor_desc, out_tmp, out_desc.input_tensor_desc, out);
		cudnn_tensor_transform(1., 0., in_desc.transformed_tensor_desc, in_tmp, in_desc.input_tensor_desc, in);

		success = success && cudnn_convcorr_bwd_krn_int( 1. , 0., conv_desc, in_desc.transformed_tensor_desc, in_tmp, krn_desc.filter_desc, krn_tmp, out_desc.transformed_tensor_desc, out_tmp);

		md_free(in_tmp);
		md_free(out_tmp);
	}

	if (success)
		cudnn_tensor_transform(1., 0., krn_desc.input_filter_tensor_desc, krn, krn_desc.transformed_filter_tensor_desc, krn_tmp);

	md_free(krn_tmp);

	free_tensor_descriptor(in_desc);
	free_tensor_descriptor(out_desc);

	free_filter_descriptor(krn_desc);

	CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(conv_desc));

	return success;
}

// *_kernel methodes merge the complex channel in the convolution channel
// and create a real kernel including the complex multiplication
// this seems to be faster as only one
// 2 * in_channel x 2 * out_channel convolution is invoked instead of
// four in_channel x out_channel convolutions
static bool cudnn_zconvcorr_fwd_kernel(
 			struct conv_desc_s bcd,
			const complex float* in,
 			const complex float* krn,
			complex float* out,
			cudnnTensorFormat_t format
			)
{
	if (1 < bitcount(bcd.channel_in_flags))
		return false;
	if (1 < bitcount(bcd.channel_out_flags))
		return false;

	if (0 == bitcount(bcd.channel_out_flags))
		for (unsigned int i = 0; (i < bcd.N) && (0 == bcd.channel_out_flags); i++)
			if ((1 == bcd.odims[i]) && (1 == bcd.idims[i]) && (1 == bcd.kdims[i]))
				bcd.channel_out_flags = MD_BIT(i);

	if (0 == bitcount(bcd.channel_in_flags))
		for (unsigned int i = 0; (i < bcd.N) && (0 == bcd.channel_in_flags); i++)
			if ((1 == bcd.odims[i]) && (1 == bcd.idims[i]) && (1 == bcd.kdims[i]) && !(MD_IS_SET(bcd.channel_out_flags, i)))
				bcd.channel_in_flags = MD_BIT(i);

	if (1 != bitcount(bcd.channel_in_flags))
		return false;
	if (1 != bitcount(bcd.channel_out_flags))
		return false;

	if ((CFL_SIZE != bcd.istrs[flag_to_index(bcd.channel_in_flags)]) && (1 != bcd.idims[flag_to_index(bcd.channel_in_flags)]))
		return false;
	if ((CFL_SIZE != bcd.ostrs[flag_to_index(bcd.channel_out_flags)]) && (1 != bcd.odims[flag_to_index(bcd.channel_out_flags)]))
		return false;

	long rkstrs[bcd.N];
	md_calc_strides(bcd.N, rkstrs, bcd.kdims, FL_SIZE);

	float* krn_real = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
	float* krn_imag = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);

	md_real2(bcd.N, bcd.kdims, rkstrs, krn_real, bcd.kstrs, krn);
	md_imag2(bcd.N, bcd.kdims, rkstrs, krn_imag, bcd.kstrs, krn);

	struct conv_desc_s rbcd = bcd;
	rbcd.idims[flag_to_index(rbcd.channel_in_flags)] *= 2;
	rbcd.istrs[flag_to_index(rbcd.channel_in_flags)] /= 2;
	rbcd.odims[flag_to_index(rbcd.channel_out_flags)] *= 2;
	rbcd.ostrs[flag_to_index(rbcd.channel_out_flags)] /= 2;

	if (2 == rbcd.idims[flag_to_index(rbcd.channel_in_flags)])
		rbcd.istrs[flag_to_index(rbcd.channel_in_flags)] = FL_SIZE;
	if (2 == rbcd.odims[flag_to_index(rbcd.channel_out_flags)])
		rbcd.ostrs[flag_to_index(rbcd.channel_out_flags)] = FL_SIZE;

	rbcd.kdims[flag_to_index(rbcd.channel_in_flags)] *= 2;
	rbcd.kdims[flag_to_index(rbcd.channel_out_flags)] *= 2;
	md_calc_strides(rbcd.N, rbcd.kstrs, rbcd.kdims, FL_SIZE);


	long nkstrs_cp[bcd.N];
	md_copy_strides(bcd.N, nkstrs_cp, rbcd.kstrs);
	nkstrs_cp[flag_to_index(bcd.channel_in_flags)] *= 2;
	nkstrs_cp[flag_to_index(bcd.channel_out_flags)] *= 2;

	float* nkrn = md_alloc_gpu(bcd.N, rbcd.kdims, FL_SIZE);

	long pos[bcd.N];
	md_singleton_strides(bcd.N, pos);

	pos[flag_to_index(bcd.channel_out_flags)] = 0;
	pos[flag_to_index(bcd.channel_in_flags)]  = 0;
	//md_copy2(bcd.N, bcd.kdims, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), rkstrs, krn_real, FL_SIZE);
	cudnn_smul2(bcd.N, bcd.kdims, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), rkstrs, krn_real, 1);

	pos[flag_to_index(bcd.channel_out_flags)] = 1;
	pos[flag_to_index(bcd.channel_in_flags)]  = 1;
	//md_copy2(bcd.N, bcd.kdims, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), rkstrs, krn_real, FL_SIZE);
	cudnn_smul2(bcd.N, bcd.kdims, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), rkstrs, krn_real, 1);

	pos[flag_to_index(bcd.channel_out_flags)] = 1;
	pos[flag_to_index(bcd.channel_in_flags)]  = 0;
	//md_copy2(bcd.N, bcd.kdims, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), rkstrs, krn_imag, FL_SIZE);
	cudnn_smul2(bcd.N, bcd.kdims, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), rkstrs, krn_imag, 1);

	md_smul(bcd.N, bcd.kdims, krn_imag, krn_imag, -1.);
	pos[flag_to_index(bcd.channel_out_flags)] = 0;
	pos[flag_to_index(bcd.channel_in_flags)]  = 1;
	//md_copy2(bcd.N, bcd.kdims, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), rkstrs, krn_imag, FL_SIZE);
	cudnn_smul2(bcd.N, bcd.kdims, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), rkstrs, krn_imag, 1);

	md_free(krn_imag);
	md_free(krn_real);

	bool success = cudnn_convcorr_fwd(rbcd, (const float*)in, nkrn, (float*)out, format);

	md_free(nkrn);

	return success;
}

static bool cudnn_zconvcorr_bwd_in_kernel(
 			struct conv_desc_s bcd,
			complex float* in,
 			const complex float* krn,
			const complex float* out,
			cudnnTensorFormat_t format
			)
{
	if (1 < bitcount(bcd.channel_in_flags))
		return false;
	if (1 < bitcount(bcd.channel_out_flags))
		return false;

	if (0 == bitcount(bcd.channel_out_flags))
		for (unsigned int i = 0; (i < bcd.N) && (0 == bcd.channel_out_flags); i++)
			if ((1 == bcd.odims[i]) && (1 == bcd.idims[i]) && (1 == bcd.kdims[i]))
				bcd.channel_out_flags = MD_BIT(i);

	if (0 == bitcount(bcd.channel_in_flags))
		for (unsigned int i = 0; (i < bcd.N) && (0 == bcd.channel_in_flags); i++)
			if ((1 == bcd.odims[i]) && (1 == bcd.idims[i]) && (1 == bcd.kdims[i]) && !(MD_IS_SET(bcd.channel_out_flags, i)))
				bcd.channel_in_flags = MD_BIT(i);

	if (1 != bitcount(bcd.channel_in_flags))
		return false;
	if (1 != bitcount(bcd.channel_out_flags))
		return false;

	if ((CFL_SIZE != bcd.istrs[flag_to_index(bcd.channel_in_flags)]) && (1 != bcd.idims[flag_to_index(bcd.channel_in_flags)]))
		return false;
	if ((CFL_SIZE != bcd.ostrs[flag_to_index(bcd.channel_out_flags)]) && (1 != bcd.odims[flag_to_index(bcd.channel_out_flags)]))
		return false;

	long rkstrs[bcd.N];
	md_calc_strides(bcd.N, rkstrs, bcd.kdims, FL_SIZE);

	float* krn_real = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
	float* krn_imag = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);

	md_real2(bcd.N, bcd.kdims, rkstrs, krn_real, bcd.kstrs, krn);
	md_imag2(bcd.N, bcd.kdims, rkstrs, krn_imag, bcd.kstrs, krn);

	struct conv_desc_s rbcd = bcd;
	rbcd.idims[flag_to_index(rbcd.channel_in_flags)] *= 2;
	rbcd.istrs[flag_to_index(rbcd.channel_in_flags)] /= 2;
	rbcd.odims[flag_to_index(rbcd.channel_out_flags)] *= 2;
	rbcd.ostrs[flag_to_index(rbcd.channel_out_flags)] /= 2;

	if (2 == rbcd.idims[flag_to_index(rbcd.channel_in_flags)])
		rbcd.istrs[flag_to_index(rbcd.channel_in_flags)] = FL_SIZE;
	if (2 == rbcd.odims[flag_to_index(rbcd.channel_out_flags)])
		rbcd.ostrs[flag_to_index(rbcd.channel_out_flags)] = FL_SIZE;

	rbcd.kdims[flag_to_index(rbcd.channel_in_flags)] *= 2;
	rbcd.kdims[flag_to_index(rbcd.channel_out_flags)] *= 2;
	md_calc_strides(rbcd.N, rbcd.kstrs, rbcd.kdims, FL_SIZE);


	long nkstrs_cp[bcd.N];
	md_copy_strides(bcd.N, nkstrs_cp, rbcd.kstrs);
	nkstrs_cp[flag_to_index(bcd.channel_in_flags)] *= 2;
	nkstrs_cp[flag_to_index(bcd.channel_out_flags)] *= 2;

	float* nkrn = md_alloc_gpu(bcd.N, rbcd.kdims, FL_SIZE);

	long pos[bcd.N];
	md_singleton_strides(bcd.N, pos);

	pos[flag_to_index(bcd.channel_out_flags)] = 0;
	pos[flag_to_index(bcd.channel_in_flags)]  = 0;
	//md_copy2(bcd.N, bcd.kdims, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), rkstrs, krn_real, FL_SIZE);
	cudnn_smul2(bcd.N, bcd.kdims, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), rkstrs, krn_real, 1.);

	pos[flag_to_index(bcd.channel_out_flags)] = 1;
	pos[flag_to_index(bcd.channel_in_flags)]  = 1;
	//md_copy2(bcd.N, bcd.kdims, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), rkstrs, krn_real, FL_SIZE);
	cudnn_smul2(bcd.N, bcd.kdims, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), rkstrs, krn_real, 1.);

	pos[flag_to_index(bcd.channel_out_flags)] = 0;
	pos[flag_to_index(bcd.channel_in_flags)]  = 1;
	//md_copy2(bcd.N, bcd.kdims, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), rkstrs, krn_imag, FL_SIZE);
	cudnn_smul2(bcd.N, bcd.kdims, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), rkstrs, krn_imag, 1.);

	md_smul(bcd.N, bcd.kdims, krn_imag, krn_imag, -1.);
	pos[flag_to_index(bcd.channel_out_flags)] = 1;
	pos[flag_to_index(bcd.channel_in_flags)]  = 0;
	//md_copy2(bcd.N, bcd.kdims, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), rkstrs, krn_imag, FL_SIZE);
	cudnn_smul2(bcd.N, bcd.kdims, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), rkstrs, krn_imag, 1.);

	md_free(krn_imag);
	md_free(krn_real);

	bool success = cudnn_convcorr_bwd_in(rbcd, (float*)in, nkrn, (const float*)out, format);

	md_free(nkrn);

	return success;
}

static bool cudnn_zconvcorr_bwd_krn_kernel(
 			struct conv_desc_s bcd,
			const complex float* in,
 			complex float* krn,
			const complex float* out,
			cudnnTensorFormat_t format
			)
{
	if (1 < bitcount(bcd.channel_in_flags))
		return false;
	if (1 < bitcount(bcd.channel_out_flags))
		return false;

	if (0 == bitcount(bcd.channel_out_flags))
		for (unsigned int i = 0; (i < bcd.N) && (0 == bcd.channel_out_flags); i++)
			if ((1 == bcd.odims[i]) && (1 == bcd.idims[i]) && (1 == bcd.kdims[i]))
				bcd.channel_out_flags = MD_BIT(i);

	if (0 == bitcount(bcd.channel_in_flags))
		for (unsigned int i = 0; (i < bcd.N) && (0 == bcd.channel_in_flags); i++)
			if ((1 == bcd.odims[i]) && (1 == bcd.idims[i]) && (1 == bcd.kdims[i]) && !(MD_IS_SET(bcd.channel_out_flags, i)))
				bcd.channel_in_flags = MD_BIT(i);

	if (1 != bitcount(bcd.channel_in_flags))
		return false;
	if (1 != bitcount(bcd.channel_out_flags))
		return false;

	if ((CFL_SIZE != bcd.istrs[flag_to_index(bcd.channel_in_flags)]) && (1 != bcd.idims[flag_to_index(bcd.channel_in_flags)]))
		return false;
	if ((CFL_SIZE != bcd.ostrs[flag_to_index(bcd.channel_out_flags)]) && (1 != bcd.odims[flag_to_index(bcd.channel_out_flags)]))
		return false;

	long rkstrs[bcd.N];
	md_calc_strides(bcd.N, rkstrs, bcd.kdims, FL_SIZE);

	float* krn_real = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
	float* krn_imag = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);

	md_real2(bcd.N, bcd.kdims, rkstrs, krn_real, bcd.kstrs, krn);
	md_imag2(bcd.N, bcd.kdims, rkstrs, krn_imag, bcd.kstrs, krn);

	struct conv_desc_s rbcd = bcd;
	rbcd.idims[flag_to_index(rbcd.channel_in_flags)] *= 2;
	rbcd.istrs[flag_to_index(rbcd.channel_in_flags)] /= 2;
	rbcd.odims[flag_to_index(rbcd.channel_out_flags)] *= 2;
	rbcd.ostrs[flag_to_index(rbcd.channel_out_flags)] /= 2;

	if (2 == rbcd.idims[flag_to_index(rbcd.channel_in_flags)])
		rbcd.istrs[flag_to_index(rbcd.channel_in_flags)] = FL_SIZE;
	if (2 == rbcd.odims[flag_to_index(rbcd.channel_out_flags)])
		rbcd.ostrs[flag_to_index(rbcd.channel_out_flags)] = FL_SIZE;

	rbcd.kdims[flag_to_index(rbcd.channel_in_flags)] *= 2;
	rbcd.kdims[flag_to_index(rbcd.channel_out_flags)] *= 2;
	md_calc_strides(rbcd.N, rbcd.kstrs, rbcd.kdims, FL_SIZE);


	long nkstrs_cp[bcd.N];
	md_copy_strides(bcd.N, nkstrs_cp, rbcd.kstrs);
	nkstrs_cp[flag_to_index(bcd.channel_in_flags)] *= 2;
	nkstrs_cp[flag_to_index(bcd.channel_out_flags)] *= 2;

	float* nkrn = md_alloc_gpu(bcd.N, rbcd.kdims, FL_SIZE);
	md_clear(bcd.N, rbcd.kdims, nkrn, FL_SIZE);

	bool success = cudnn_convcorr_bwd_krn(rbcd, (const float*)in, nkrn, (const float*)out, format);

	float* krn_tmp = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);

	long pos[bcd.N];
	md_singleton_strides(bcd.N, pos);

	pos[flag_to_index(bcd.channel_out_flags)] = 0;
	pos[flag_to_index(bcd.channel_in_flags)]  = 0;
	//md_copy2(bcd.N, bcd.kdims, rkstrs, krn_tmp, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), FL_SIZE);
	cudnn_smul2(bcd.N, bcd.kdims, rkstrs, krn_tmp, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), 1);
	md_add(bcd.N, bcd.kdims, krn_real, krn_real, krn_tmp);

	pos[flag_to_index(bcd.channel_out_flags)] = 1;
	pos[flag_to_index(bcd.channel_in_flags)]  = 1;
	//md_copy2(bcd.N, bcd.kdims, rkstrs, krn_tmp, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), FL_SIZE);
	cudnn_smul2(bcd.N, bcd.kdims, rkstrs, krn_tmp, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), 1);
	md_sub(bcd.N, bcd.kdims, krn_real, krn_real, krn_tmp);

	pos[flag_to_index(bcd.channel_out_flags)] = 1;
	pos[flag_to_index(bcd.channel_in_flags)]  = 0;
	//md_copy2(bcd.N, bcd.kdims, rkstrs, krn_tmp, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), FL_SIZE);
	cudnn_smul2(bcd.N, bcd.kdims, rkstrs, krn_tmp, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), 1);
	md_add(bcd.N, bcd.kdims, krn_imag, krn_imag, krn_tmp);

	pos[flag_to_index(bcd.channel_out_flags)] = 0;
	pos[flag_to_index(bcd.channel_in_flags)]  = 1;
	//md_copy2(bcd.N, bcd.kdims, rkstrs, krn_tmp, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), FL_SIZE);
	cudnn_smul2(bcd.N, bcd.kdims, rkstrs, krn_tmp, nkstrs_cp, &MD_ACCESS(bcd.N, rbcd.kstrs, pos, nkrn), 1);
	md_add(bcd.N, bcd.kdims, krn_imag, krn_imag, krn_tmp);

	md_zcmpl2(bcd.N, bcd.kdims, bcd.kstrs, krn, rkstrs, krn_real, rkstrs, krn_imag);

	md_free(krn_imag);
	md_free(krn_real);
	md_free(krn_tmp);
	md_free(nkrn);

	return success;
}



bool zconvcorr_fwd_cudnn(	int N,
				long odims[N], long ostrs[N], complex float* out,
				long idims[N], long istrs[N], const complex float* in,
				long kdims[N], long kstrs[N], const complex float* krn,
				unsigned long flags, const long dilation[N], const long strides[N], bool conv)
{
	if (MAX_DIMS < N)
		return false;

	struct conv_desc_s bart_conv_desc = create_conv_desc(	N,
								odims, ostrs,
								idims, istrs,
								kdims, kstrs,
								dilation, strides,
								flags, conv);

	if (!check_cudnn_convcorr(bart_conv_desc))
		return false;

	if (cudnn_zconvcorr_fwd_kernel(bart_conv_desc, in, krn, out, CUDNN_TENSOR_NCHW)) {

		debug_printf(DP_DEBUG3, "conv by %s -> 1\n", __func__);
		return true;
	}

	if (cudnn_zconvcorr_fwd_split(bart_conv_desc, in, krn, out, CUDNN_TENSOR_NCHW)){

		debug_printf(DP_DEBUG3, "conv by %s -> 2 \n", __func__);
		return true;
	}

	return false;
}

bool zconvcorr_bwd_in_cudnn(	int N,
				long odims[N], long ostrs[N], const complex float* out,
				long idims[N], long istrs[N], complex float* in,
				long kdims[N], long kstrs[N], const complex float* krn,
				unsigned long flags, const long dilation[N], const long strides[N], bool conv)
{
	if (MAX_DIMS < N)
		return false;

	struct conv_desc_s bart_conv_desc = create_conv_desc(	N,
								odims, ostrs,
								idims, istrs,
								kdims, kstrs,
								dilation, strides,
								flags, conv);

	if (!check_cudnn_convcorr(bart_conv_desc))
		return false;

	if (cudnn_zconvcorr_bwd_in_kernel(bart_conv_desc, in, krn, out, CUDNN_TENSOR_NCHW)) {

		debug_printf(DP_DEBUG3, "conv by %s -> 1\n", __func__);
		return true;
	}

	if (cudnn_zconvcorr_bwd_in_split(bart_conv_desc, in, krn, out, CUDNN_TENSOR_NCHW)){

		debug_printf(DP_DEBUG3, "conv by %s -> 2 \n", __func__);
		return true;
	}

	return false;
}

bool zconvcorr_bwd_krn_cudnn(	int N,
				long odims[N], long ostrs[N], const complex float* out,
				long idims[N], long istrs[N], const complex float* in,
				long kdims[N], long kstrs[N], complex float* krn,
				unsigned long flags, const long dilation[N], const long strides[N], bool conv)
{
	if (MAX_DIMS < N)
		return false;

	struct conv_desc_s bart_conv_desc = create_conv_desc(	N,
								odims, ostrs,
								idims, istrs,
								kdims, kstrs,
								dilation, strides,
								flags, conv);

	if (!check_cudnn_convcorr(bart_conv_desc))
		return false;


	if (cudnn_zconvcorr_bwd_krn_kernel(bart_conv_desc, in, krn, out, CUDNN_TENSOR_NCHW)) {

		debug_printf(DP_DEBUG3, "conv by %s -> 1\n", __func__);
		return true;
	}

	if (cudnn_zconvcorr_bwd_krn_split(bart_conv_desc, in, krn, out, CUDNN_TENSOR_NCHW)){

		debug_printf(DP_DEBUG3, "conv by %s -> 2 \n", __func__);
		return true;
	}

	return false;
}



#endif
#endif