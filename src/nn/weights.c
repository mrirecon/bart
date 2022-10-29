/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <complex.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"

#include "num/multind.h"
#include "num/iovec.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/const.h"
#include "nlops/chain.h"

#include "nn/nn.h"
#include "nn/init.h"

#include "weights.h"

const struct nn_weights_s* create_multi_md_array(int N, int D[N], const long* dimensions[N], const _Complex float* x[N], size_t sizes[N])
{
	const struct iovec_s* iovs[N];
	for (int i = 0; i < N; i++)
		iovs[i] = iovec_create(D[i], dimensions[i], sizes[i]);

	auto result = nn_weights_create(N, iovs);

	for (int i = 0; i < N; i++) {

		md_copy(D[i], dimensions[i], result->tensors[i], x[i], sizes[i]);
		iovec_free(iovs[i]);
	}

	return result;
}

void free_multi_md_array(const struct nn_weights_s* array)
{
	nn_weights_free((struct nn_weights_s*)array);
}



nn_weights_t nn_weights_create(int N, const struct iovec_s* iovs[N])
{
	PTR_ALLOC(struct nn_weights_s, result);
	result->N = N;

	PTR_ALLOC(const struct iovec_s*[N], niov);
	PTR_ALLOC(_Complex float*[N], ntensors);

	for (int i = 0; i < N; i++) {

		(*niov)[i] = iovec_create2(iovs[i]->N, iovs[i]->dims, iovs[i]->strs, iovs[i]->size);
		(*ntensors)[i] = md_alloc(iovs[i]->N, iovs[i]->dims, iovs[i]->size);
	}

	result->iovs = *PTR_PASS(niov);
	result->tensors = *PTR_PASS(ntensors);
	return PTR_PASS(result);
}

/**
 * Load weights from file
 *
 * @param name file name of weights (without extension)
 *
 * @param returns pointer to struct holding loaded weights
 *
 */
nn_weights_t load_nn_weights(const char *name)
{
	int N_max = 64;
	int D_max = 64;
	int D[N_max];
	long dimensions[N_max][D_max];
	complex float* args[N_max];

	int N = load_multi_cfl(name, N_max, D_max, D, dimensions, args);

	PTR_ALLOC(struct nn_weights_s, result);
	result->N = N;

	PTR_ALLOC(const struct iovec_s*[N], niov);
	PTR_ALLOC(_Complex float*[N], ntensors);

	const long* dimensions_unmap[N];

	for (int i = 0; i < N; i++) {

		(*niov)[i] = iovec_create(D[i], dimensions[i], sizeof(_Complex float));
		(*ntensors)[i] = md_alloc(D[i], dimensions[i], sizeof(_Complex float));
		md_copy(D[i], dimensions[i], (*ntensors)[i], args[i], sizeof(_Complex float));

		dimensions_unmap[i] = dimensions[i];
	}

	result->iovs = *PTR_PASS(niov);
	result->tensors = *PTR_PASS(ntensors);

	unmap_multi_cfl(result->N, D, dimensions_unmap, args);

	return PTR_PASS(result);
}

/**
 * Stores weights to file
 *
 * @param name file name of weights (without extension)
 * @param weights pointer to struct holding the weights
 *
 */
void dump_nn_weights(const char *name, nn_weights_t weights) {

	int D[weights->N];
	const long* dims[weights->N];
	for (int i = 0; i < weights->N; i++) {

		D[i] = weights->iovs[i]->N;
		dims[i] = weights->iovs[i]->dims;
	}

	dump_multi_cfl(name, weights->N, D, dims, (const complex float**)weights->tensors);
}

/**
 * Move weights to gpu
 *
 * @param weights pointer to struct holding the weights
 */
void move_gpu_nn_weights(nn_weights_t weights){
#ifdef USE_CUDA
	for (int i = 0; i < weights->N; i++) {

		auto iov = weights->iovs[i];
		complex float* tmp = md_alloc_gpu(iov->N, iov->dims, iov->size);
		md_copy(iov->N, iov->dims, tmp, weights->tensors[i], iov->size);
		md_free(weights->tensors[i]);
		weights->tensors[i] = tmp;
	}
#else
	error("Compiled without gpu support!");
	UNUSED(weights);
#endif
}

/**
 * Check if weights are copied to gpu
 *
 * @param weights pointer to struct holding the weights
 *
 * @returns boolean if weights are copied to gpu
 */
bool nn_weights_on_gpu(nn_weights_t weights)
{
#ifdef USE_CUDA
	return cuda_ondevice(weights->tensors[0]);
#else
	UNUSED(weights);
	return false;
#endif
}

/**
 * Free all memory related to weights (dimensions and md arrays)
 *
 * @param weights pointer to struct holding the weights
 */
void nn_weights_free(nn_weights_t weights){

	for (int i = 0; i < weights->N; i++) {

		iovec_free(weights->iovs[i]);
		md_free(weights->tensors[i]);
	}

	xfree(weights->tensors);
	xfree(weights->iovs);

	xfree(weights);
}

/**
 * Initialize weights for all inputs of a nn_t which have an initializer
 *
 * @param op nn_t struct
 * @param weights pointer to struct holding the weights, the struct and memory for the weights must be allocated
 *
 * @note The number of initializers must not exceed the number of weight tensors in the struct.
 * @note The dimensions of the inputs with an initializer must coincide the dimensions of the weights.
 */
void nn_init(nn_t op, nn_weights_t weights)
{
	for (int i = 0, ip = 0; i < nn_get_nr_in_args(op); i++){

		if(NULL != op->initializers[i]) {

			assert((int)ip < weights->N);
			auto iov = nlop_generic_domain(op->nlop, i);
			iovec_check(weights->iovs[ip], iov->N, iov->dims, iov->strs);
			initializer_apply(op->initializers[i], iov->N, iov->dims, weights->tensors[ip++]);
		}
	}
}

/**
 * Create a nn_weights_t having a tensor for each input of a nn_t with an initializer
 *
 * @param op nn_t struct
 *
 * @returns nn_weight_t
 */
nn_weights_t nn_weights_create_from_nn(nn_t x)
{
	int N = nn_get_nr_weights(x);
	const struct iovec_s* iovs[N];

	for (int i = 0, ip = 0; i < nlop_get_nr_in_args(x->nlop); i++)
		if (NULL != x->initializers[i])
			iovs[ip++] = nlop_generic_domain(x->nlop, i);

	return nn_weights_create(N, iovs);
}

/**
 * Create a nn_t whose inputs corresponding to weights are set to the weights provided
 *
 * This function can be used to create a nlop which can be used for inference
 *
 * @param op nn_t struct
 * @param weights
 * @param copy if true: weights are copied into nlop; else: only pointer is stored
 *
 * @returns nn_t used for inference
 */
nn_t nn_get_wo_weights(nn_t op, nn_weights_t weights, bool copy)
{
	assert(weights->N == nn_get_nr_weights(op));

	auto nlop_result = nlop_clone(op->nlop);

	for (int i = (int)nn_get_nr_out_args(op) - 1; i >= 0; i--)
		if (OUT_BATCHNORM == op->out_types[i])
			nlop_result = nlop_del_out_F(nlop_result, i);

	for (int i = (int)nn_get_nr_in_args(op) - 1, ip = weights->N - 1; i >= 0; i--)
		if ((IN_OPTIMIZE == op->in_types[i]) || (IN_BATCHNORM == op->in_types[i])) {

			auto iov = weights->iovs[ip];
			nlop_result = nlop_set_input_const_F(nlop_result, i, iov->N, iov->dims, copy, weights->tensors[ip--]);
		}

	auto result = nn_from_nlop_F(nlop_result);

	for (int i = 0, j = 0; i < nn_get_nr_in_args(result); i++, j++) {

		while ((IN_OPTIMIZE == op->in_types[i]) || (IN_BATCHNORM == op->in_types[i]))
			j++;
		nn_clone_arg_i_from_i(result, i, op, j);
	}

	for (int i = 0, j = 0; i < nn_get_nr_out_args(result); i++, j++) {

		while (OUT_BATCHNORM == op->out_types[i])
			j++;
		nn_clone_arg_o_from_o(result, i, op, j);
	}

	return result;
}

/**
 * Create a nn_t whose inputs corresponding to weights are set to the weights provided and free nn_t
 *
 * This function can be used to create a nlop which can be used for inference
 *
 * @param op nn_t struct
 * @param weights
 * @param copy if true: weights are copied into nlop; else: only pointer is stored
 *
 * @returns nn_t used for inference
 */
nn_t nn_get_wo_weights_F(nn_t op, nn_weights_t weights, bool copy)
{
	auto result = nn_get_wo_weights(op, weights, copy);
	nn_free(op);

	return result;
}

/**
 * Create a nlop whose inputs corresponding to weights are set to the weights provided
 *
 * This function can be used to create a nlop which can be used for inference
 *
 * @param op nn_t struct
 * @param weights
 * @param copy if true: weights are copied into nlop; else: only pointer is stored
 *
 * @returns nlop used for inference
 */
const struct nlop_s* nn_get_nlop_wo_weights(nn_t op, nn_weights_t weights, bool copy)
{
	auto nn_result = nn_get_wo_weights(op, weights, copy);
	auto result = nlop_clone(nn_get_nlop(nn_result));
	nn_free(nn_result);

	return result;
}

/**
 * Create a nlop whose inputs corresponding to weights are set to the weights provided and free nn_t
 *
 * This function can be used to create a nlop which can be used for inference
 *
 * @param op nn_t struct
 * @param weights
 * @param copy if true: weights are copied into nlop; else: only pointer is stored
 *
 * @returns nlop used for inference
 */
const struct nlop_s* nn_get_nlop_wo_weights_F(nn_t op, nn_weights_t weights, bool copy)
{
	auto nn_result = nn_get_wo_weights_F(op, weights, copy);
	auto result = nlop_clone(nn_get_nlop(nn_result));

	nn_free(nn_result);

	return result;
}

/**
 * Copy weights from one struct into another
 *
 * If a src dimension is one and the corresponding dst dim not, the array is repeated along this dimension.
 * Other deviations in the dimensions of dst and src are not allowed.
 *
 * @param dst
 * @param src
 */
void nn_weights_copy(nn_weights_t dst, nn_weights_t src)
{
	assert(dst->N == src->N);

	for (int i = 0; i < src->N; i++){

		auto iovd = dst->iovs[i];
		auto iovs = src->iovs[i];

		assert(iovd->N == iovs->N);
		assert(iovd->size == iovs->size);

		for (int j = 0; j < iovd->N; j++)
			assert((1 == iovs->dims[j] ) || (iovs->dims[j] == iovs->dims[j]));


		md_copy2(iovd->N, iovd->dims,
			MD_STRIDES(iovd->N, iovd->dims, iovd->size), dst->tensors[i],
			MD_STRIDES(iovs->N, iovs->dims, iovs->size), src->tensors[i],
			iovs->size);
	}
}
