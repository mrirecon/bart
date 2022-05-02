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

#include "nn/weights.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "misc/debug.h"
#include "misc/opts.h"
#include "misc/mmio.h"
#include "misc/misc.h"

#include "networks/nnet.h"
#include "networks/unet.h"
#include "networks/losses.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char help_str[] = "Trains or applies a neural network.";



static const struct nn_weights_s* get_validation_files(int NO, const char* out_name, int NI, const char* in_name)
{
	if (((NULL != out_name) && (NULL == in_name)) || ((NULL != in_name) && (NULL == out_name)))
		error("Only input or output for validation is provided");

	if (NULL == out_name)
		return NULL;

	long dims_out[NO];
	complex float* out = load_cfl(out_name, NO, dims_out);

	long dims_in[NI];
	complex float* in = load_cfl(in_name, NI, dims_in);

	auto result = create_multi_md_array(2, (int[2]){NI, NO}, (const long*[2]){dims_in, dims_out}, (const complex float*[2]){in, out}, (size_t[2]){CFL_SIZE, CFL_SIZE});

	unmap_cfl(NI, dims_in, in);
	unmap_cfl(NO, dims_out, out);

	return result;
}


int main_nnet(int argc, char* argv[argc])
{
	opts_iter6_init();

	bool apply = false;
	bool train = false;
	bool eval = false;

	bool load_mem = false;

	long N_batch = 0;

	const char* graph_filename = NULL;
	const char* filename_weights_load = NULL;

	int NI = -1;

	bool mnist_default = false;
	long N_segm_labels = -1;
	int label_index = 0;

	struct nnet_s config = nnet_init;

	struct opt_s network_opts[] = {

		OPTL_SET('M', "mnist", &(mnist_default), "use basic MNIST Network"),
		OPTL_LONG('U', "unet-segm", &(N_segm_labels), "labels", "use U-Net for segmentation"),
	};

	const char* validation_in = NULL;
	const char* validation_out = NULL;

	struct opt_s validation_opts[] = {

		OPTL_STRING('i', "in", &(validation_in), "file", "input for validation"),
		OPTL_STRING('o', "out", &(validation_out), "file", "reference for validation"),
	};

	const struct opt_s opts[] = {

		OPTL_SET('a', "apply", &apply, "apply nnet"),
		OPTL_SET( 'e', "eval", &eval, "evaluate nnet"),
		OPTL_SET('t', "train", &train, "trains network"),

		OPTL_SET('g', "gpu", &(config.gpu), "run on gpu"),

		OPTL_LONG('b', "batch-size", &(N_batch), "batchsize", "size of mini batches"),
		OPTL_INFILE('l', "load", (const char**)(&(filename_weights_load)), "<weights-init>", "load weights for continuing training"),

		OPTL_SUBOPT('N', "network", "...", "select neural network", ARRAY_SIZE(network_opts), network_opts),
		OPTL_SUBOPT('U', "unet-segm", "...", "configure U-Net for segmentation", N_unet_segm_opts, unet_segm_opts),

		OPTL_SUBOPT(0, "train-loss", "...", "configure the training loss", N_loss_opts, loss_opts),
		OPTL_SUBOPT(0, "valid-loss", "...", "configure the validation loss", N_val_loss_opts, val_loss_opts),
		OPTL_SUBOPT(0, "valid-data", "...", "provide validation data", ARRAY_SIZE(validation_opts), validation_opts),

		OPTL_SUBOPT('T', "train-algo", "...", "configure general training parmeters", N_iter6_opts, iter6_opts),
		OPTL_SUBOPT(0, "adam", "...", "configure Adam", N_iter6_adam_opts, iter6_adam_opts),
		//OPTL_SUBOPT(0, "iPALM", "...", "configure iPALM", N_iter6_ipalm_opts, iter6_ipalm_opts),

		OPTL_SET(0, "load-memory", &(load_mem), "load files into memory"),

		OPTL_STRING(0, "export-graph", (const char**)(&(graph_filename)), "<file.dot>", "export graph for visualization"),
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



	if (train)
		config.train_conf = iter6_get_conf_from_opts();

	config.valid_loss = get_val_loss_from_option();
	config.train_loss = get_loss_from_option();

	if (mnist_default)
		nnet_init_mnist_default(&config);

	if (-1 != N_segm_labels) {

		nnet_init_unet_segm_default(&config, N_segm_labels);

		if (-1 == NI)
			NI = 5;
	}

	config.train_loss->label_index = label_index;
	config.valid_loss->label_index = label_index;

	if (train) {

		if (NULL == config.train_conf) {

			debug_printf(DP_WARN, "No training algorithm selected. Fallback to Adam!");

			iter_6_select_algo = ITER6_ADAM;
			config.train_conf = iter6_get_conf_from_opts();

		} else {

			iter6_copy_config_from_opts(config.train_conf);
		}
	}

	if (NULL == config.network)
		error("No network selected!");

	if ((0 < config.train_conf->dump_mod) && (NULL == config.train_conf->dump_filename))
		config.train_conf->dump_filename = filename_weights;


#ifdef USE_CUDA
	if (config.gpu) {

		num_init_gpu();
		cuda_use_global_memory();
	}

	else
#endif
		num_init();


	if (apply && (train || eval))
		error("Application would overwrite training data! Either train or apply!");

	if (NULL != filename_weights_load) {

		if (apply)
			error("Weights should only be loaded for training using -l option!");

		config.weights = load_nn_weights(filename_weights_load);
	}

	config.graph_file = graph_filename;


	long dims_in[DIMS];
	complex float* in = load_cfl(filename_in, (-1 == NI) ? (int)DIMS : NI, dims_in);

	if (-1 == NI) {

		NI = DIMS;

		while ((NI > 0) && (1 == dims_in[NI - 1]))
			NI--;
	}


	if (N_batch == 0)
		N_batch = MIN(128, dims_in[NI - 1]);


	if (train) {

		long NO = config.get_no_odims(&config, NI, dims_in);
		long dims_out[NO];

		complex float* out = load_cfl(filename_out, NO, dims_out);

		complex float* in_mem = NULL;
		complex float* out_mem = NULL;

		if (load_mem) {

			in_mem = md_alloc(NI, dims_in, CFL_SIZE);
			md_copy(NI, dims_in, in_mem, in, CFL_SIZE);
			out_mem = md_alloc(NI, dims_out, CFL_SIZE);
			md_copy(NO, dims_out, out_mem, out, CFL_SIZE);
		}

		auto valid_files = get_validation_files(NO, validation_out, NI, validation_in);

		train_nnet(&config, NO, dims_out, load_mem ? out_mem : out, NI, dims_in, load_mem ? in_mem : in,  N_batch, valid_files);

		if (NULL != valid_files)
			free_multi_md_array(valid_files);

		dump_nn_weights(filename_weights, config.weights);

		md_free(in_mem);
		md_free(out_mem);

		unmap_cfl(NO, dims_out, out);
	}

	if (eval) {

		long NO = config.get_no_odims(&config, NI, dims_in);
		long dims_out[NO];
		complex float* out = load_cfl(filename_out, NO, dims_out);

		if (NULL == config.weights)
			config.weights = load_nn_weights(filename_weights);

		eval_nnet(&config, NO, dims_out, out, NI, dims_in, in,  N_batch);

		unmap_cfl(NO, dims_out, out);
	}

	if (apply) {

		long NO = config.get_no_odims(&config, NI, dims_in);
		long dims_out[NO];
		config.get_odims(&config, NO, dims_out, NO, dims_in);

		complex float* out = create_cfl(filename_out, NO, dims_out);

		config.weights = load_nn_weights(filename_weights);

		apply_nnet_batchwise(&config, NO, dims_out, out, NI, dims_in, in,  N_batch);

		unmap_cfl(NO, dims_out, out);
	}

	if (NULL != config.weights)
		nn_weights_free(config.weights);

	unmap_cfl(NI, dims_in, in);

	return 0;
}
