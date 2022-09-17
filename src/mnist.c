/* Copyright 2021-2022. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#include "grecon/opt_iter6.h"
#include "grecon/losses.h"
#include "grecon/network.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "iter/iter6.h"
#include "iter/italgos.h"
#include "iter/batch_gen.h"

#include "nn/weights.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "misc/debug.h"
#include "misc/opts.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/types.h"

#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"

#include "nn/nn.h"
#include "nn/chain.h"
#include "nn/layers.h"
#include "nn/layers_nn.h"
#include "nn/activation.h"
#include "nn/activation_nn.h"
#include "nn/losses.h"

#include "networks/nnet.h"
#include "networks/unet.h"
#include "networks/losses.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif




/**
 * Generate nn-struct representing neural network for MNIST dataset
 *
 * @param odims output dimensions { Classification, Batchsize }
 * @param idims input dimensions { X, Y, Batchsize }
 * @param status Network status, i.e. training or inference (different behavior of dropot layer)
 */
static nn_t network_mnist_create(const long odims[2], const long idims[3], enum NETWORK_STATUS status)
{
	assert(10 == odims[0]);		// 10 classes 
	assert(idims[2] == odims[1]);	// batch size of input and output equals
	
	long dims[5] = {1, idims[0], idims[1], 1, idims[2]};	// input dimensions in the channel NHWC format, i.e. { Channels, X, Y, Z, Batch }
	bool nhwc = true;					// we use NHWC layout

	long kernel_size[] = { 3, 3, 1} ;	// size of 3D convolution kernel
	long strides[] = { 1, 1, 1 };
	long dilation[] = { 1, 1, 1 };
	
	long chan1 = 32;
	long chan2 = 64;
	long chan3 = 128;
	long chan4 = 10;

	const struct initializer_s* init = NULL; // fallback to default initializer
	
	long pool_size[] = { 2, 2, 1 };

	bool conv = false;	// we usecross correlation not convolution, i.e. as usual in deep-learning the convolution kernels are not flipped

	unsigned long bias_flag = MD_BIT(0); //bitmask to select dimensions of bias (channel dimension)

	// we initialize the input with a reshaping operator
	nn_t network = nn_from_nlop_F(nlop_from_linop(linop_reshape_create(5, dims, 3, idims)));	// reshape input

	// we append layers to the network, we always append to the single output with index "0, NULL" (c.f. src/nn/nn.h for indexing)
	network = nn_append_convcorr_layer(network, 0, NULL, "conv_", chan1, kernel_size, conv, PAD_VALID, nhwc, strides, dilation, init);
	network = nn_append_activation_bias(network, 0, NULL, "conv_bias_", ACT_RELU, bias_flag);
	network = nn_append_convcorr_layer(network, 0, NULL, "conv_", chan2, kernel_size, conv, PAD_VALID, nhwc, strides, dilation, init);
	network = nn_append_activation_bias(network, 0, NULL, "conv_bias_", ACT_RELU, bias_flag);
	network = nn_append_maxpool_layer(network, 0, NULL, pool_size, PAD_VALID, true);

	network = nn_append_flatten_layer(network, 0, NULL);
	network = nn_append_dropout_layer(network, 0, NULL, 0.25, status);
	network = nn_append_dense_layer(network, 0, NULL, "dense_", chan3, init);
	network = nn_append_activation_bias(network, 0, NULL, "dense_bias_", ACT_RELU, bias_flag);
	network = nn_append_dropout_layer(network, 0, NULL, 0.5, status);
	network = nn_append_dense_layer(network, 0, NULL, "dense_", chan4, init);
	network = nn_append_activation_bias(network, 0, NULL, "dense_bias_", ACT_SOFTMAX, bias_flag);

	debug_printf(DP_INFO, "MNIST-Network created:");
	nn_debug(DP_INFO, network);

	return network;
}


static const char help_str[] = "Trains or applies a MNIST network.\nThis network is to demonstrate how a neural network can be implemented in BART.";


int main_mnist(int argc, char* argv[argc])
{
	bool apply = false;
	bool train = false;

	bool gpu = false;

	const struct opt_s opts[] = {

		OPTL_SET('a', "apply", &apply, "apply nnet"),
		OPTL_SET('t', "train", &train, "trains network"),

		OPTL_SET('g', "gpu", &(gpu), "run on gpu"),
	};

	const char* filename_in;
	const char* filename_weights;
	const char* filename_out;

	struct arg_s args[] = {

		ARG_INFILE(true, &(filename_in), "input"),
		ARG_INOUTFILE(true, &filename_weights, "weights"),
		ARG_INOUTFILE(true, &(filename_out), "ref/output"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);


#ifdef USE_CUDA
	if (gpu) {

		num_init_gpu();
		cuda_use_global_memory();

	} else
#endif
	{
		num_init();
	}


	if (apply && train)
		error("Either train or apply!");

	long NI = 3;
	long NO = 2;

	long dims_in[NI];
	complex float* in = load_cfl(filename_in, NI, dims_in);

	long Nb = MIN(128, dims_in[NI - 1]);


	if (train) {

		long dims_out[NO];
		complex float* out = load_cfl(filename_out, NO, dims_out);

		long bdims_in[] = { dims_in[0], dims_in[1], Nb };
		long bdims_out[] = { dims_out[0], Nb };

		long Nt = dims_out[1];	 //dataset size
		assert(Nt == dims_in[2]);


		nn_t net = network_mnist_create(bdims_out, bdims_in, STAT_TRAIN);
		nn_t loss = nn_from_nlop_F(nlop_cce_create(2, bdims_out, ~MD_BIT(0)));

		nn_t train_op = nn_chain2_FF(net, 0, NULL, loss, 0, NULL);
		train_op = nn_set_out_type_F(train_op, 0, NULL, OUT_OPTIMIZE);

		nn_weights_t weights = nn_weights_create_from_nn(train_op);
		nn_init(train_op, weights);

		if (gpu)
			move_gpu_nn_weights(weights);
		
		const struct nlop_s* batch_generator = batch_gen_create(2, (int [2]){ NO, NI },
									(const long*[2]){ bdims_out, bdims_in},
									(const long*[2]){ dims_out, dims_in },
									(const complex float*[2]){ out, in },
									0, BATCH_GEN_SHUFFLE_DATA, 123);

		//setup for iter algorithm
		int II = nn_get_nr_in_args(train_op);
		int OO = nn_get_nr_out_args(train_op);

		float* src[II];	
		src[0] = NULL; //reference output -> allocated and filled by training algorithm/batch generator
		src[1] = NULL; //reference input -> allocated and filled by training algorithm/batch generator

		for (int i = 0; i < weights->N; i++)
			src[i + 2] = (float*)weights->tensors[i];

		enum IN_TYPE in_type[II];
		enum OUT_TYPE out_type[OO];

		nn_get_in_types(train_op, II, in_type);
		nn_get_out_types(train_op, OO, out_type);

		in_type[0] = IN_BATCH_GENERATOR;
		in_type[1] = IN_BATCH_GENERATOR;

		struct iter6_adadelta_conf train_conf = iter6_adadelta_conf_defaults;
		iter6_adadelta(CAST_UP(&train_conf), nn_get_nlop(train_op), II, in_type, NULL, src, OO, out_type, Nb, Nt / Nb, batch_generator, NULL);


		dump_nn_weights(filename_weights, weights);
		nn_weights_free(weights);

		unmap_cfl(NO, dims_out, out);
	}


	if (apply) {

		long dims_out[] = { 10, dims_in[2] };
		complex float* out = create_cfl(filename_out, NO, dims_out);

		nn_t net = network_mnist_create(dims_out, dims_in, STAT_TEST);
		
		nn_weights_t weights = load_nn_weights(filename_weights);

		if (gpu)
			move_gpu_nn_weights(weights);

		net = nn_get_wo_weights_F(net, weights, false);	//set inputs corresponding to weights to the loaded weights

		nlop_generic_apply_sameplace(nn_get_nlop(net),
			1, (int[1]){ 2 }, (const long*[1]){ dims_out }, (complex float* [1]){ out },
			1, (int[1]){ 3 }, (const long*[1]){ dims_in }, (const complex float*[1]){ in },
			weights->tensors[0]);

		unmap_cfl(NO, dims_out, out);

		nn_weights_free(weights);
	}


	unmap_cfl(NI, dims_in, in);

	return 0;
}
