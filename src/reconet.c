/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>
#include <stdlib.h>
#include <complex.h>
#include <string.h>

#include "grecon/opt_iter6.h"
#include "grecon/losses.h"
#include "grecon/network.h"

#include "noncart/nufft.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"
#include "misc/opts.h"
#include "misc/mmio.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"
#include "num/mem.h"
#include "num/fft.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "iter/iter6.h"
#include "iter/iter.h"

#include "nn/data_list.h"
#include "nn/weights.h"

#include "networks/cnn.h"
#include "networks/unet.h"
#include "networks/tf.h"
#include "networks/reconet.h"
#include "networks/losses.h"
#include "networks/misc.h"

#include "nlops/mri_ops.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char help_str[] = "Trains or appplies a neural network for reconstruction.";

int main_reconet(int argc, char* argv[argc])
{
	opts_iter6_init();

	struct reconet_s config = reconet_config_opts;

	bool train = false;
	bool apply = false;
	bool eval = false;

	char* filename_weights_load = NULL;

	long Nb = 0;

	bool load_mem = false;

	bool normalize = false;

	bool varnet_default = false;
	bool modl_default = false;
	bool unet_default = false;

	bool test_defaults = false;

	enum NETWORK_SELECT net = NETWORK_NONE;

	const char* graph_filename = NULL;

	struct network_data_s data = network_data_empty;
	struct network_data_s valid_data = network_data_empty;

	const char* filename_mask = NULL;
	const char* filename_mask_val = NULL;


	struct opt_s dc_opts[] = {

		OPTL_FLOAT(0, "fix-lambda", &(config.dc_lambda_fixed), "f", "fix lambda to specified value (-1 means train lambda)"),
		OPTL_FLOAT(0, "lambda-init", &(config.dc_lambda_init), "f", "initialize lambda with specified value"),
		OPTL_SET(0, "gradient-step", &(config.dc_gradient), "use gradient steps for data-consistency"),
		OPTL_SET(0, "gradient-max-eigen", &(config.dc_scale_max_eigen), "scale stepsize by inverse max eigen value of A^HA"),
		OPTL_SET(0, "proximal-mapping", &(config.dc_proxmap), "use proximal mapping for data-consistency"),
		OPTL_INT(0, "max-cg-iter", &(config.dc_max_iter), "d", "number of cg steps for proximal mapping"),
	};

	struct opt_s init_opts[] = {

		OPTL_SET(0, "tickhonov", &(config.sense_init), "(init network with l2 regularized SENSE reconstruction instead of adjoint reconstruction)"), //used in webinar
		OPTL_SET(0, "sense", &(config.sense_init), "init network with l2 regularized SENSE reconstruction instead of adjoint reconstruction"),
		OPTL_INT(0, "max-cg-iter", &(config.init_max_iter), "d", "number of cg steps for Tikhonov regularized reconstruction"),
		OPTL_FLOAT(0, "fix-lambda", &(config.init_lambda_fixed), "f", "fix lambda to specified value (-1 means train lambda)"),
		OPTL_FLOAT(0, "lambda-init", &(config.init_lambda_init), "f", "initialize lambda with specified value"),
	};

	struct opt_s valid_opts[] = {

		OPTL_INFILE('t', "trajectory", &(valid_data.filename_trajectory), "<file>", "validation data trajectory"),
		OPTL_INFILE('p', "pattern", &(valid_data.filename_pattern), "<file>", "validation data sampling pattern / psf in kspace"),
		OPTL_INFILE('k', "kspace", &(valid_data.filename_kspace), "<file>", "validation data kspace"),
		OPTL_INFILE('c', "coil", &(valid_data.filename_coil), "<file>", "validation data sensitivity maps"),
		OPTL_INFILE('r', "ref", &(valid_data.filename_out), "<file>", "validation data reference"),
		OPTL_INOUTFILE('a', "adjoint", &(valid_data.filename_adjoint), "<file>", "(validation data adjoint (load or export))"),
		OPTL_INOUTFILE('P', "psf", &(valid_data.filename_psf), "<file>", "(validation data psf (load or export))"),
		OPTL_SET('e', "export", &(valid_data.export), "(export psf and adjoint reconstruction)"),
		OPTL_INFILE(0, "mask", &(filename_mask_val), "<mask>", "mask for computation of loss"),
	};

	struct opt_s network_opts[] = {

		OPTL_SET(0, "modl", &(modl_default), "use MoDL Network (also sets train and data-consistency default values)"),
		OPTL_SET(0, "varnet", &(varnet_default), "use Variational Network (also sets train and data-consistency default values)"),
		//OPTL_SET(0, "unet", &(unet_default), "use U-Net (also sets train and data-consistency default values)"),

		//OPTL_SELECT(0, "resnet-block", enum NETWORK_SELECT, &net, NETWORK_RESBLOCK, "use residual block (overwrite default)"),
		//OPTL_SELECT(0, "varnet-block", enum NETWORK_SELECT, &net, NETWORK_VARNET, "use variational block (overwrite default)"),
	};

	const struct opt_s opts[] = {

		OPTL_SET('t', "train", &train, "train reconet"),
		OPTL_SET('e', "eval", &eval, "evaluate reconet"),
		OPTL_SET('a', "apply", &apply, "apply reconet"),

		OPTL_SET('g', "gpu", &(config.gpu), "run on gpu"),

		OPTL_INFILE('l', "load", (const char**)(&(filename_weights_load)), "<weights-init>", "load weights for continuing training"),
		OPTL_LONG('b', "batch-size", &(Nb), "", "size of mini batches"),

		OPTL_LONG('I', "iterations", &(config.Nt), "", "number of unrolled iterations"),

		OPTL_SET('n', "normalize", &(config.normalize), "normalize data with maximum magnitude of adjoint reconstruction"),

		OPTL_SUBOPT('N', "network", "...", "select neural network", ARRAY_SIZE(network_opts), network_opts),
		OPTL_SUBOPT(0, "resnet-block", "...", "configure residual block", N_res_block_opts, res_block_opts),
		OPTL_SUBOPT(0, "varnet-block", "...", "configure variational block", N_variational_block_opts, variational_block_opts),
		OPTL_SUBOPT(0, "tensorflow", "...", "configure tensorflow as network", N_tensorflow_opts, network_tensorflow_opts),
		OPTL_SUBOPT(0, "unet", "...", "configure U-Net block", N_unet_reco_opts, unet_reco_opts),

		OPTL_SUBOPT(0, "data-consistency", "...", "configure data-consistency method", ARRAY_SIZE(dc_opts), dc_opts),
		OPTL_SUBOPT(0, "initial-reco", "...", "configure initialization", ARRAY_SIZE(init_opts), init_opts),

		OPTL_SELECT(0, "shared-weights", enum BOOL_SELECT, &(config.share_weights_select), BOOL_TRUE, "share weights across iterations"),
		OPTL_SELECT(0, "no-shared-weights", enum BOOL_SELECT, &(config.share_weights_select), BOOL_FALSE, "share weights across iterations"),
		OPTL_SELECT(0, "shared-lambda", enum BOOL_SELECT, &(config.share_lambda_select), BOOL_TRUE, "share lambda across iterations"),
		OPTL_SELECT(0, "no-shared-lambda", enum BOOL_SELECT, &(config.share_lambda_select), BOOL_FALSE, "share lambda across iterations"),

		OPTL_SET(0, "rss-norm", &(config.normalize_rss), "scale output image to rss normalization"),

		OPTL_INFILE(0, "trajectory", &(data.filename_trajectory), "<traj>", "trajectory"),
		OPTL_INFILE(0, "pattern", &(data.filename_pattern), "<pattern>", "sampling pattern / psf in kspace"),
		OPTL_INOUTFILE(0, "adjoint", &(data.filename_adjoint), "<file>", "(validation data adjoint (load or export))"),
		OPTL_INOUTFILE(0, "psf", &(data.filename_psf), "<file>", "(psf (load or export))"),
		OPTL_SET(0, "export", &(data.export), "(export psf and adjoint reconstruction)"),

		OPTL_INFILE(0, "mask", &(filename_mask), "<mask>", "mask for computation of loss"),

		OPTL_SUBOPT(0, "valid-data", "...", "provide validation data", ARRAY_SIZE(valid_opts),valid_opts),

		OPTL_SUBOPT(0, "train-loss", "...", "configure the training loss", N_loss_opts, loss_opts),
		OPTL_SUBOPT(0, "valid-loss", "...", "configure the validation loss", N_val_loss_opts, val_loss_opts),

		OPTL_SUBOPT('T', "train-algo", "...", "configure general training parmeters", N_iter6_opts, iter6_opts),
		OPTL_SUBOPT(0, "adam", "...", "configure Adam", N_iter6_adam_opts, iter6_adam_opts),
		OPTL_SUBOPT(0, "iPALM", "...", "configure iPALM", N_iter6_ipalm_opts, iter6_ipalm_opts),

		OPTL_SET(0, "load-memory", &(load_mem), "copy training data into memory"),
		OPTL_SET(0, "lowmem", &(config.low_mem), "reduce memory usage by checkpointing"),

		OPTL_SET(0, "test", &(test_defaults), "very small network for tests"),
		OPTL_STRING(0, "export-graph", (const char**)(&(graph_filename)), "<file.dot>", "export graph for visualization"),

		OPT_INFILE('B', &(data.filename_basis), "file", "(temporal (or other) basis)"),
	};

	const char* filename_weights;

	struct arg_s args[] = {

		ARG_INFILE(true, &(data.filename_kspace), "kspace"),
		ARG_INFILE(true, &(data.filename_coil), "sensitivities"),
		ARG_INOUTFILE(true, &filename_weights, "weights"),
		ARG_INOUTFILE(true, &(data.filename_out), "ref/out"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if (train)
		config.train_conf = iter6_get_conf_from_opts();

	config.valid_loss = get_val_loss_from_option();
	config.train_loss = get_loss_from_option();

	config.network = get_default_network(net);
	if (NULL != network_tensorflow_default.model_path)
		config.network = CAST_UP(&network_tensorflow_default);

	if (test_defaults) {

		if (modl_default)
			reconet_init_modl_test_default(&config);

		if (varnet_default)
			reconet_init_varnet_test_default(&config);

		if (unet_default)
			reconet_init_unet_test_default(&config);

	} else {

		if (modl_default)
			reconet_init_modl_default(&config);

		if (varnet_default)
			reconet_init_varnet_default(&config);

		if (unet_default)
			reconet_init_unet_default(&config);
	}

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

	if (0 == Nb)
		Nb = 10;

	if (normalize)
		config.normalize = true;

	if (0 < config.train_conf->dump_mod)
		config.train_conf->dump_filename = filename_weights;


	if (((train || eval) && apply) || (!train && !apply && ! eval))
		error("Network must be either trained (-t) or applied(-a)!\n");

#ifdef USE_CUDA
	if (config.gpu) {

		num_init_gpu();
		cuda_use_global_memory();

	} else
#endif
		num_init();

	if (apply)
		data.create_out = true;

	data.load_mem = load_mem;
	load_network_data(&data);

	Nb = MIN(Nb, network_data_get_tot(&data));

	if (config.sense_init && (-1. != config.init_lambda_fixed)) {

		network_data_compute_init(&data, config.init_lambda_fixed, config.init_max_iter);
		config.external_initialization = true;
	}

	if (config.normalize)
		network_data_normalize(&data);
	
	network_data_slice_dim_to_batch_dim(&data);


	bool use_valid_data = false;
	long Nt_val = 0;

	if (   (NULL != valid_data.filename_coil)
	    && (NULL != valid_data.filename_kspace)
	    && (NULL != valid_data.filename_out)   ) {

		assert(train);

		use_valid_data = true;
		valid_data.filename_basis = data.filename_basis;
		
		load_network_data(&valid_data);
		network_data_slice_dim_to_batch_dim(&valid_data);
		
		if (config.sense_init && (-1. != config.init_lambda_fixed))
			network_data_compute_init(&valid_data, config.init_lambda_fixed, config.init_max_iter);

		if (config.normalize)
			network_data_normalize(&valid_data);

		Nt_val = network_data_get_tot(&valid_data);
	}

	config.graph_file = graph_filename;

	if (NULL != filename_weights_load)
		config.weights = load_nn_weights(filename_weights_load);

	if (NULL != data.filename_trajectory) {

		config.mri_config->noncart = true;
		config.mri_config->nufft_conf = data.nufft_conf;
	}

	if (NULL != data.filename_basis)
		config.mri_config->basis_flags = TE_FLAG | COEFF_FLAG;

	if (train) {

		auto train_data_list = network_data_get_named_list(&data);

		complex float* mask = NULL;
		long mask_dims[DIMS];

		if (NULL != filename_mask) {

			mask = load_cfl(filename_mask, DIMS, mask_dims);
			config.train_loss->mask_flags = md_nontriv_dims(DIMS, mask_dims);
			named_data_list_append(train_data_list, DIMS, mask_dims, mask, "loss_mask");
		}

		complex float* mask_val = NULL;
		long mask_dims_val[DIMS];

		struct named_data_list_s* valid_data_list = NULL;

		if (use_valid_data) {

			valid_data_list = network_data_get_named_list(&valid_data);

			if (NULL != filename_mask_val) {

				mask_val = load_cfl(filename_mask_val, DIMS, mask_dims_val);
				config.valid_loss->mask_flags = md_nontriv_dims(DIMS, mask_dims_val);
				named_data_list_append(valid_data_list, DIMS, mask_dims_val, mask_val, "loss_mask");
			}
		}

		train_reconet(&config, data.N, data.max_dims, data.ND, data.psf_dims, Nb, train_data_list, Nt_val, valid_data_list);
		dump_nn_weights(filename_weights, config.weights);

		named_data_list_free(train_data_list);

		if (NULL != valid_data_list)
			named_data_list_free(valid_data_list);

		if (NULL != mask)
			unmap_cfl(DIMS, mask_dims, mask);

		if (NULL != mask_val)
			unmap_cfl(DIMS, mask_dims_val, mask_val);
	}

	if (eval) {

		auto eval_data_list = network_data_get_named_list(&data);

		if (NULL == config.weights)
			config.weights = load_nn_weights(filename_weights);
		eval_reconet(&config, data.N, data.max_dims, data.ND, data.psf_dims, eval_data_list);
	
		named_data_list_free(eval_data_list);
	}

	if (apply) {

		auto apply_data_list = network_data_get_named_list(&data);

		config.weights = load_nn_weights(filename_weights);
		apply_reconet(&config, data.N, data.max_dims, data.ND, data.psf_dims, apply_data_list);

		named_data_list_free(apply_data_list);
	}

	nn_weights_free(config.weights);

	free_network_data(&data);

	xfree(config.train_conf);

	return 0;
}
