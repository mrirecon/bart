/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2025 Moritz Blumenthal
 */

#include <cstdint>
#include <vector>
#include <stdbool.h>

#include "torch/torch.h"
#include "torch/script.h"

#include "misc/debug.h"
#include "misc/misc.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "pytorch_cpp_wrapper.h"


struct pytorch_wrapper_s {

	int OO;
	int II;

	std::vector<uint64_t> osize;
	std::vector<uint64_t> isize;

	std::vector<torch::Tensor> otensor;
	std::vector<torch::Tensor> itensor;

	torch::jit::script::Module module;

	std::vector<std::vector<int64_t>> odims;
};

struct pytorch_wrapper_s* pytorch_wrapper_create(const char* path, int II, const int DI[], const long* idims[], int device)
{
	struct pytorch_wrapper_s* ret = new(struct pytorch_wrapper_s);

	try {
		
		ret->II = II;

		for (int i = 0; i < II; i++) {

			std::vector<int64_t> dims;

			uint64_t size = sizeof(_Complex float);

			for (int j = DI[i] - 1; j >= 0; j--) {

				size *= idims[i][j];
				dims.push_back(idims[i][j]);
			}

			auto options = torch::TensorOptions().dtype(torch::kComplexFloat).requires_grad(true);
#ifdef USE_CUDA
			if (-1 < device)
				options = options.device(torch::Device(torch::kCUDA, device));
#endif
			torch::Tensor tensor = torch::ones(dims, options);

			ret->isize.push_back(size);
			ret->itensor.push_back(tensor);
		}

#ifdef USE_CUDA
		if (-1 < device)
			ret->module = torch::jit::load(path, torch::Device(torch::kCUDA, device));
		else
#endif
		ret->module = torch::jit::load(path, torch::kCPU);

		//tracing for out dims
		std::vector<torch::jit::IValue> inputs;

		for (int i = 0; i < II; i++)		
			inputs.push_back(ret->itensor[i]);
		
		auto outputs = ret->module.forward(inputs);

		if (outputs.isTensor())
			ret->OO = 1;
		else
			ret->OO = outputs.toTuple()->elements().size();

		
		for (int i = 0; i < ret->OO; i++) {

			torch::Tensor output;

			if (outputs.isTuple())
				output = outputs.toTuple()->elements()[i].toTensor();
			else
				output = outputs.toTensor();
			
			uint64_t size = sizeof(_Complex float);

			std::vector<int64_t> odims;

			for (int j = 0; j < output.sizes().size(); j++) {

				size *= output.sizes()[j];
				odims.push_back(output.sizes()[j]);
			}

			ret->osize.push_back(size);
			ret->odims.push_back(odims);
		}

	} catch (const c10::Error& e) {

		error("PYTorch model at %s could not be loaded!:\n %s\n", path, e.what_without_backtrace());
	}

	return ret;
}

void pytorch_wrapper_free(const struct pytorch_wrapper_s* data)
{
	delete data;
}


int pytorch_wrapper_number_outputs(const struct pytorch_wrapper_s* data)
{
	return data->odims.size();
}

int pytorch_wrapper_rank_output(const struct pytorch_wrapper_s* data, int o)
{
	return data->odims[o].size();
}

void pytorch_wrapper_dims_output(const struct pytorch_wrapper_s* data, int o, int N, long dims[__VLA(N)])
{
	assert(N == data->odims[o].size());
	
	for (int i = 0; i < N; i++)
		dims[i] = data->odims[o][N - 1 - i];
}


void pytorch_wrapper_apply_unchecked(struct pytorch_wrapper_s* data, int N, _Complex float* args[__VLA(N)], int device)
{
	try {
		std::vector<torch::jit::IValue> inputs;

		for (int i = 0; i < data->II; i++) {

#ifdef USE_CUDA
			if (-1 == device) {

				data->itensor[i] = data->itensor[i].to(torch::kCPU);

			} else {

				data->itensor[i] = data->itensor[i].to(torch::Device(torch::kCUDA, device));
			}
		
			if ((-1 < device) || cuda_ondevice(args[i + data->OO]))
				cuda_memcpy(data->isize[i], data->itensor[i].data_ptr(), args[i + data->OO]);
			else
#endif
			memcpy(data->itensor[i].data_ptr(), args[i + data->OO], data->isize[i]);

			inputs.push_back(data->itensor[i]);
		}

#ifdef USE_CUDA
		if (-1 == device) {

			data->module.to(torch::kCPU);
			cuda_sync_device();

		} else
			data->module.to(torch::Device(torch::kCUDA, device));
#endif

		auto outputs = data->module.forward(inputs);

		data->otensor.clear();

		for (int i = 0; i < data->OO; i++) {

			torch::Tensor output;

			if (outputs.isTuple()) {

				output = outputs.toTuple()->elements()[i].toTensor().to(torch::kComplexFloat);

			} else {

				if (0 != i)
					error("PyTorch has only one output!\n");

				output = outputs.toTensor().to(torch::kComplexFloat);
			}

			data->otensor.push_back(output);
#ifdef USE_CUDA
			if ((-1 < device) || cuda_ondevice(args[i])) {

				cuda_sync_device();
				cuda_memcpy(data->osize[i], args[i], data->otensor[i].data_ptr());

			} else
	#endif
				memcpy(args[i], data->otensor[i].data_ptr(), data->osize[i]);
		}

	} catch (const c10::Error& e) {

		error("PYTorch apply failed!:\n %s\n", e.what_without_backtrace());
	}
}

void pytorch_wrapper_adjoint_unchecked(struct pytorch_wrapper_s* data, int o, int i, _Complex float* dst, const _Complex float* src)
{
	data->itensor[i].mutable_grad() = torch::Tensor();

	torch::Tensor tsrc = torch::empty_like(data->otensor[o]);

#ifdef USE_CUDA
	if ((tsrc.is_cuda()) || cuda_ondevice(src)) {

		cuda_memcpy(data->osize[o], tsrc.data_ptr(), src);
		cuda_sync_stream();

	} else
#endif
		memcpy(tsrc.data_ptr(), src, data->osize[o]);

	data->otensor[o].backward(tsrc, true, false, data->itensor[i]);

#ifdef USE_CUDA
	if ((data->itensor[i].grad().is_cuda()) || cuda_ondevice(dst)) {

		cuda_sync_device();
		cuda_memcpy(data->isize[i], dst, data->itensor[i].grad().data_ptr());

	} else
#endif
		memcpy(dst, data->itensor[i].grad().data_ptr(), data->isize[i]);
	
	data->itensor[i].mutable_grad() = torch::Tensor();
}

void pytorch_wrapper_derivative_unchecked(struct pytorch_wrapper_s* data, int o, int i, _Complex float* dst, const _Complex float* src)
{
	assert(0);
}

