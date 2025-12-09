/* Copyright 2021-2022. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2023. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 **/

#include <assert.h>
#include <stdbool.h>

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/types.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/ops.h"
#include "num/mpi_ops.h"
#include "num/init.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/sum.h"

#include "iter/proj.h"
#include "iter/iter.h"
#include "iter/iter6.h"
#include "iter/monitor_iter6.h"
#include "iter/batch_gen.h"
#include "iter/italgos.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/tenmul.h"
#include "nlops/someops.h"
#include "nlops/mri_ops.h"
#include "nlops/const.h"
#include "nlops/stack.h"

#include "nn/layers.h"
#include "nn/losses.h"
#include "nn/init.h"
#include "nn/data_list.h"
#include "nn/const.h"
#include "nn/weights.h"
#include "nn/nn.h"
#include "nn/chain.h"


#include "networks/misc.h"
#include "networks/losses.h"
#include "networks/cnn.h"
#include "networks/unet.h"

#include "reconet.h"

struct reconet_s reconet_config_opts = {

	.network = NULL,

	.Nt = 10,

	.share_weights_select = BOOL_DEFAULT,
	.share_lambda_select = BOOL_DEFAULT,
	.share_weights = false,
	.share_lambda = false,

	.sense_config = NULL,
	.one_channel_per_map = false,
	.external_initialization = false,

	//data consistency config
	.dc_lambda_fixed = -1.,
	.dc_lambda_init = -1.,
	.dc_gradient = false,
	.dc_scale_max_eigen = false,
	.dc_proxmap = false,
	.dc_max_iter = 10,

	//network initialization
	.normalize = false,
	.sense_init = false,
	.init_max_iter = -1,
	.init_lambda_fixed = -2,
	.init_lambda_init = -1,

	.weights = NULL,
	.train_conf = NULL,

	.train_loss = NULL,
	.valid_loss = NULL,

	.gpu = false,
	.low_mem = false,

	.graph_file = NULL,

	.coil_image = false,
	.ref_is_kspace = false,

	.normalize_rss = false,

	.ksp_training = false,

	.precomp = true,
};

static void reconet_init_default(struct reconet_s* reconet) {

	//network initialization
	reconet->init_max_iter = (-1 == reconet->init_max_iter) ? (reconet->dc_proxmap ? reconet->dc_max_iter : 30) : reconet->init_max_iter;
	reconet->init_lambda_fixed = (-2 == reconet->init_lambda_fixed) ? (reconet->dc_proxmap ? reconet->dc_lambda_fixed : 0) : reconet->init_lambda_fixed;
	reconet->init_lambda_init = (-1 == reconet->init_lambda_init) ? (reconet->dc_proxmap ? reconet->dc_lambda_init : 0.1) : reconet->init_lambda_init;
}

void reconet_init_modl_default(struct reconet_s* reconet)
{
	if (NULL == reconet->train_conf) {

		PTR_ALLOC(struct iter6_adam_conf, train_conf);

		*train_conf = iter6_adam_conf_defaults;
		reconet->train_conf = CAST_UP(PTR_PASS(train_conf));
	}

	if (NULL == reconet->network)
		reconet->network = CAST_UP(&network_resnet_default);

	reconet->share_weights = (reconet->share_weights_select == BOOL_DEFAULT) || (reconet->share_weights_select == BOOL_TRUE);
	reconet->share_lambda = (reconet->share_lambda_select == BOOL_DEFAULT) || (reconet->share_lambda_select == BOOL_TRUE);

	//data consistency config
	reconet->dc_lambda_init = (-1 == reconet->dc_lambda_init) ? 0.05 : reconet->dc_lambda_init;

	if (!reconet->dc_proxmap && !reconet->dc_gradient)
		reconet->dc_proxmap = true;

	if (NULL == reconet->train_loss) {

		reconet->train_loss =  &loss_option;
		reconet->train_loss->weighting_mse = 1.;
	}

	if (NULL == reconet->valid_loss)
		reconet->valid_loss = &loss_image_valid;

	reconet_init_default(reconet);
}

void reconet_init_varnet_default(struct reconet_s* reconet)
{
	if (NULL == reconet->train_conf) {

		PTR_ALLOC(struct iter6_iPALM_conf, train_conf);

		*train_conf = iter6_iPALM_conf_defaults;
		reconet->train_conf = CAST_UP(PTR_PASS(train_conf));
	}

	if (NULL == reconet->network)
		reconet->network = CAST_UP(&network_varnet_default);

	reconet->share_weights = reconet->share_weights_select == BOOL_TRUE;
	reconet->share_lambda = reconet->share_lambda_select == BOOL_TRUE;

	//data consistency config
	if (-1. == reconet->dc_lambda_init)
		reconet->dc_lambda_init = 0.2;

	if (!reconet->dc_proxmap && !reconet->dc_gradient)
		reconet->dc_gradient = true;

	if (NULL == reconet->train_loss) {

		reconet->train_loss = &loss_option;
		reconet->train_loss->weighting_mse_rss = 1.;
	}

	if (NULL == reconet->valid_loss)
		reconet->valid_loss = &loss_image_valid;

	reconet_init_default(reconet);
}

void reconet_init_modl_test_default(struct reconet_s* reconet)
{
	reconet_init_modl_default(reconet);

	reconet->Nt = 2;
	CAST_DOWN(network_resnet_s, reconet->network)->Nl = 3;
	CAST_DOWN(network_resnet_s, reconet->network)->Nf = 8;
}

void reconet_init_varnet_test_default(struct reconet_s* reconet)
{
	reconet_init_varnet_default(reconet);

	reconet->Nt = 2;
	CAST_DOWN(network_varnet_s, reconet->network)->Nf = 5;
	CAST_DOWN(network_varnet_s, reconet->network)->Kx = 3;
	CAST_DOWN(network_varnet_s, reconet->network)->Ky = 3;
	CAST_DOWN(network_varnet_s, reconet->network)->Nw = 5;
}

void reconet_init_unet_default(struct reconet_s* reconet)
{
	if (NULL == reconet->network)
		reconet->network = CAST_UP(&network_unet_default_reco);

	reconet_init_modl_default(reconet);
}

void reconet_init_unet_test_default(struct reconet_s* reconet)
{
	reconet_init_unet_default(reconet);

	reconet->Nt = 2;
	CAST_DOWN(network_unet_s, reconet->network)->Nf = 4;
	CAST_DOWN(network_unet_s, reconet->network)->N_level = 2;
}


static nn_t reconet_sort_args(nn_t reconet)
{
	const char* data_names[] = {
		"reference",
		"kspace",
		"adjoint",
		"initialization",
		"coil",
		"psf",
		"pattern",
		"trajectory",
		"scale",
		"loss_mask"
	};

	int N = nn_get_nr_named_in_args(reconet);
	const char* sorted_names[N + (int)ARRAY_SIZE(data_names) + 2];

	nn_get_in_names_copy(N, sorted_names + ARRAY_SIZE(data_names) + 2, reconet);

	for (int i = 0; i < (int)ARRAY_SIZE(data_names); i++)
		sorted_names[i] = data_names[i];

	sorted_names[ARRAY_SIZE(data_names)] = "lambda_init";
	sorted_names[ARRAY_SIZE(data_names) + 1] = "lambda";

	reconet = nn_sort_inputs_by_list_F(reconet, N + (int)ARRAY_SIZE(data_names) + 2, sorted_names);

	for (int i = 0; i < N; i++)
		xfree(sorted_names[i + (int)ARRAY_SIZE(data_names) + 2]);

	for (int i = 0; i < (int)ARRAY_SIZE(data_names); i++)
		if (nn_is_name_in_in_args(reconet, data_names[i]))
			reconet = nn_set_in_type_F(reconet, 0, data_names[i], IN_BATCH_GENERATOR);

	return reconet;
}

//add "scale" input for normalization
static nn_t reconet_normalization(nn_t network)
{
	const char* norm_names_in[] = {
		"initialization",
		"adjoint",
		"kspace",
		"reference"
	};

	bool scale = false;

	for (int i = 0; i < (int)ARRAY_SIZE(norm_names_in); i++) {

		const char* name = norm_names_in[i];

		if (!nn_is_name_in_in_args(network, name))
			continue;

		auto iov = nn_generic_domain(network, 0, name);

		long sdims[iov->N];
		md_select_dims(iov->N, BATCH_FLAG, sdims, iov->dims);

		auto nn_scale = nn_from_nlop_F(nlop_tenmul_create(iov->N, iov->dims, iov->dims, sdims));

		nn_scale = nn_set_in_type_F(nn_scale, 1, NULL, IN_BATCH_GENERATOR);
		nn_scale = nn_set_input_name_F(nn_scale, 1, "scale");

		if (scale)
			nn_scale = nn_mark_dup_F(nn_scale, "scale");

		nn_scale = nn_set_in_type_F(nn_scale, 0, NULL, IN_BATCH_GENERATOR);
		nn_scale = nn_set_input_name_F(nn_scale, 0, name);

		network = nn_chain2_FF(nn_scale, 0, NULL, network, 0, name);
		network = nn_stack_dup_by_name_F(network);

		scale = true;
	}

	const char* norm_names_out[] = {
		"reconstruction"
	};

	for (int i = 0; i < (int)ARRAY_SIZE(norm_names_out); i++) {

		const char* name = norm_names_out[i];

		if (!nn_is_name_in_out_args(network, name))
			continue;

		auto iov = nn_generic_codomain(network, 0, name);

		long sdims[iov->N];
		md_select_dims(iov->N, BATCH_FLAG, sdims, iov->dims);

		auto nlop_scale = nlop_tenmul_create(iov->N, iov->dims, iov->dims, sdims);
		nlop_scale = nlop_chain2_FF(nlop_zinv_create(iov->N, sdims), 0, nlop_scale, 1);

		auto nn_scale = nn_from_nlop_F(nlop_scale);

		nn_scale = nn_set_in_type_F(nn_scale, 1, NULL, IN_BATCH_GENERATOR);
		nn_scale = nn_set_input_name_F(nn_scale, 1, "scale");

		if (scale)
			nn_scale = nn_mark_dup_F(nn_scale, "scale");

		nn_scale = nn_set_output_name_F(nn_scale, 0, name);

		network = nn_chain2_FF(network, 0, name, nn_scale, 0, NULL);
		network = nn_stack_dup_by_name_F(network);

		scale = true;
	}

	return network;
}


/**
 * Returns dataconsistency block using Tikhonov regularization
 *
 * Out	= argmin_x ||Ax - y||^2 + Lambda||x - In||^2
 *	= (A^HA + Lambda)^-1[A^Hy + Lambda In]
 *
 * Input tensors:
 *
 * INDEX_0: 	idims
 * adjoint:	idims
 * lambda:	ldims
 *
 * Output tensors:
 *
 * INDEX_0:	idims
 */
static nn_t data_consistency_tikhonov_create(const struct reconet_s* config, int Nb, struct sense_model_s* models[Nb])
{
	struct iter_conjgrad_conf iter_conf = iter_conjgrad_defaults;
	iter_conf.l2lambda = 0.;
	iter_conf.maxiter = config->dc_max_iter;

	const struct nlop_s* nlop_dc = nlop_sense_dc_prox_create(Nb, models, &iter_conf, BATCH_FLAG); // in: input, adjoint, lambda; out: output

	auto result = nn_from_nlop_F(nlop_dc);
	result = nn_set_input_name_F(result, 1, "adjoint");

	result = nn_set_input_name_F(result, 1, "lambda");
	result = nn_set_in_type_F(result, 0, "lambda", IN_OPTIMIZE);
	result = nn_set_initializer_F(result, 0, "lambda", init_const_create(config->dc_lambda_init));

	auto iov = nn_generic_domain(result, 0, "lambda");
	auto prox_conv = operator_project_pos_real_create(iov->N, iov->dims);
	result = nn_set_prox_op_F(result, 0, "lambda", prox_conv);

	return result;// in:  input, adjoint, lambda; out: output
}


/**
 * Returns dataconsistency block using a gradient step
 *
 * Out	= lambda (A^HA In - A^H kspace)
 *
 * Input tensors:
 * INDEX_0	idims
 * adjoint:	idims
 * lambda:	ldims
 *
 * Output tensors:
 *
 * INDEX_0	idims
 */
static nn_t data_consistency_gradientstep_create(const struct reconet_s* config, int Nb, struct sense_model_s* models[Nb])
{
	nn_t result = nn_from_nlop_F(nlop_sense_dc_grad_create(Nb, models, BATCH_FLAG));

	result = nn_set_input_name_F(result, 1, "adjoint");
	result = nn_set_input_name_F(result, 1, "lambda");
	result = nn_set_initializer_F(result, 0, "lambda", init_const_create(config->dc_lambda_init));
	result = nn_set_in_type_F(result, 0, "lambda", IN_OPTIMIZE);

	auto iov = nn_generic_domain(result, 0, "lambda");
	auto prox_conv = operator_project_pos_real_create(iov->N, iov->dims);

	result = nn_set_prox_op_F(result, 0, "lambda", prox_conv);

	nn_debug(DP_DEBUG3, result);

	return result;
}




/**
 * Returns operator computing the initialization
 * for a network with trainable lambda
 *
 * @param mri_init
 * @param N
 * @param dims
 * @param idims
 *
 * Input tensors:
 * adjoint:	idims
 * lambda:	sdims
 *
 * Output tensors:
 * init		idims
 */
static nn_t nn_init_create(const struct reconet_s* config, int Nb, struct sense_model_s* models[Nb])
{
	assert(config->sense_init);

	int N = sense_model_get_N(config->sense_config);

	long img_dims[N];
	long scl_dims[N];

	sense_model_get_img_dims(config->sense_config, N, img_dims);
	img_dims[BATCH_DIM] = Nb;

	md_select_dims(N,  BATCH_FLAG, scl_dims, img_dims);


	struct iter_conjgrad_conf iter_conf = iter_conjgrad_defaults;
	iter_conf.l2lambda = 0.;
	iter_conf.maxiter = config->init_max_iter;

	auto nlop_result = nlop_sense_normal_inv_create(Nb, models, &iter_conf, BATCH_FLAG); //in: adjoint, lambda; out: (A^HA + l)^-1 adjoint
	auto nn_result = nn_from_nlop_F(nlop_result);

	if (-1 == config->init_lambda_fixed) {

		bool same_lambda = config->dc_proxmap;
		same_lambda = same_lambda && (config->dc_lambda_init == config->init_lambda_init);
		same_lambda = same_lambda && (config->dc_lambda_fixed == config->init_lambda_fixed);
		same_lambda = same_lambda && (config->dc_max_iter == config->init_max_iter);

		const char* lambda_name = (same_lambda && config->share_lambda) ? "lambda" : "lambda_init";

		nn_result = nn_set_input_name_F(nn_result, 1, lambda_name);
	} else {

		complex float fix = config->init_lambda_fixed;
		auto iov = nn_generic_domain(nn_result, 1, NULL);
		nn_result = nn_set_input_const_F2(nn_result, 1, NULL, iov->N, iov->dims, MD_SINGLETON_STRS(iov->N), true, &fix);
	}

	nn_result = nn_set_output_name_F(nn_result, 0, "init");
	nn_result = nn_set_input_name_F(nn_result, 0, "adjoint");

	return nn_result;
}


/**
 * Returns network block
 *
 * Input tensors:
 *
 * INDEX_0: 	idims:	(Ux, Uy, Uz, 1, Nb)
 * weights as listed in "sorted_weight_names"
 *
 * Output tensors:
 *
 * INDEX_0:	idims:	(Ux, Uy, Uz, 1, Nb)
 * [batchnorm output]
 */
static nn_t network_block_create(const struct reconet_s* config, int N, const long img_dims[N], enum NETWORK_STATUS status)
{
	long timg_dims[N];
	md_copy_dims(N, timg_dims, img_dims);

	if (!config->one_channel_per_map)
		timg_dims[MAPS_DIM] = 1;

	nn_t result = NULL;

	result = network_create(config->network, N, timg_dims, N, timg_dims, status);

	int N_in_names = nn_get_nr_named_in_args(result);
	int N_out_names = nn_get_nr_named_out_args(result);

	const char* in_names[N_in_names?:1];
	const char* out_names[N_out_names?:1];

	nn_get_in_names_copy(N_in_names, in_names, result);
	nn_get_out_names_copy(N_out_names, out_names, result);

	for (int i = 0; i < N_in_names; i++) {

		result = nn_append_singleton_dim_in_F(result, 0, in_names[i]);
		xfree(in_names[i]);
	}

	for (int i = 0; i < N_out_names; i++) {

		result = nn_append_singleton_dim_out_F(result, 0, out_names[i]);
		xfree(out_names[i]);
	}

	if (timg_dims[MAPS_DIM] != img_dims[MAPS_DIM]) {

		result = nn_chain2_FF(nn_from_nlop_F(nlop_from_linop_F(linop_expand_create(N, timg_dims, img_dims))), 0, NULL, result, 0, NULL);

		if (config->network->residual) {

			long pos[N];
			long res_dims[N];

			for (int i = 0; i < N; i++)
				pos[i] = 0;

			pos[MAPS_DIM] = 1;

			md_copy_dims(N, res_dims, img_dims);
			res_dims[MAPS_DIM] -= 1;

			auto lop = linop_extract_create(N, pos, res_dims, img_dims);
			result = nn_combine_FF(result, nn_from_nlop_F(nlop_from_linop_F(lop)));
			result = nn_dup_F(result, 0, NULL, 1, NULL);
			result = nn_stack_outputs_F(result, 0, NULL, 1, NULL, MAPS_DIM);

		} else {

			result = nn_chain2_FF(result, 0, NULL, nn_from_nlop_F(nlop_from_linop_F(linop_expand_create(N, img_dims, timg_dims))), 0, NULL);
		}
	}

	return nn_checkpoint_F(result, true, config->low_mem);
}


/**
 * Returns one cell of reconet iteration
 *
 * Input tensors:
 *
 * INDEX_0: 	idims
 * adjoint:	idims
 * coil:	dims
 * pattern:	pdims
 * lambda:	bdims
 * weights as listed in "sorted_weight_names"
 *
 * Output tensors:
 *
 * INDEX_0:	idims
 * [batchnorm output]
 */
static nn_t reconet_cell_create(const struct reconet_s* config, int Nb, struct sense_model_s* models[Nb], enum NETWORK_STATUS status)
{
	int N = sense_model_get_N(config->sense_config);

	long img_dims[N];
	sense_model_get_img_dims(config->sense_config, N, img_dims);
	img_dims[BATCH_DIM] = Nb;

	auto result = network_block_create(config, N, img_dims, status);

	if (config->dc_proxmap) {

		auto dc = data_consistency_tikhonov_create(config, Nb, models);
		result = nn_chain2_FF(result, 0, NULL, dc, 0, NULL);
	}

	if (config->dc_gradient) {

		auto dc = data_consistency_gradientstep_create(config, Nb, models);

		result = nn_combine_FF(dc, result);
		result = nn_dup_F(result, 0, NULL, 1, NULL);
		result = nn_combine_FF(nn_from_nlop_F(nlop_zaxpbz_create(N, img_dims, -1., 1)),result);
		result = nn_link_F(result, 1, NULL, 0, NULL);
		result = nn_link_F(result, 1, NULL, 0, NULL);
	}


	if (nn_is_name_in_in_args(result, "lambda"))
		result = nn_append_singleton_dim_in_F(result, 0, "lambda");

	return result;
}


static nn_t reconet_iterations_create(const struct reconet_s* config, int Nb, struct sense_model_s* models[Nb], enum NETWORK_STATUS status)
{
	auto result = reconet_cell_create(config, Nb, models, status);

	int N_in_names = nn_get_nr_named_in_args(result);
	int N_out_names = nn_get_nr_named_out_args(result);

	const char* in_names[N_in_names?:1];
	const char* out_names[N_out_names?:1];

	nn_get_in_names_copy(N_in_names, in_names, result);
	nn_get_out_names_copy(N_out_names, out_names, result);

	for (int i = 1; i < config->Nt; i++) {

		auto tmp = reconet_cell_create(config, Nb, models, status);

		tmp = nn_mark_dup_if_exists_F(tmp, "adjoint");
		tmp = nn_mark_dup_if_exists_F(tmp, "coil");
		tmp = nn_mark_dup_if_exists_F(tmp, "psf");

		tmp = (config->share_lambda ? nn_mark_dup_if_exists_F : nn_mark_stack_input_if_exists_F)(tmp, "lambda");

		// batchnorm weights are always stacked
		for (int i = 0; i < N_in_names; i++) {

			if (!nn_is_name_in_in_args(tmp, in_names[i]))
				continue;

			if (nn_get_dup(tmp, 0, in_names[i]) && config->share_weights)
				tmp = nn_mark_dup_F(tmp, in_names[i]);
			else
				tmp = nn_mark_stack_input_F(tmp, in_names[i]);
		}

		for (int i = 0; i < N_out_names; i++)
			tmp = nn_mark_stack_output_if_exists_F(tmp, out_names[i]);

		result = nn_chain2_FF(result, 0, NULL, tmp, 0, NULL);

		result = nn_stack_dup_by_name_F(result);
	}

	result = nn_sort_inputs_by_list_F(result, N_in_names, in_names);
	result = nn_sort_outputs_by_list_F(result, N_out_names, out_names);

	for (int i = 0; i < N_in_names; i++)
		xfree(in_names[i]);

	for (int i = 0; i < N_out_names; i++)
		xfree(out_names[i]);

	return result;
}


static nn_t reconet_create(const struct reconet_s* config, int Nb, enum NETWORK_STATUS status)
{
	struct sense_model_s* models[Nb];
	memset(models, 0, sizeof models);	// -fanalyzer uninitialized

	for (int i = 0; i < Nb; i++)
			models[i] = sense_model_create(config->sense_config);

	int N = sense_model_get_N(config->sense_config);

	auto network = reconet_iterations_create(config, Nb, models, status);

	if (!config->external_initialization) {

		if (config->sense_init) {

			auto nn_init = nn_init_create(config, Nb, models);

			nn_init = nn_mark_dup_F(nn_init, "adjoint");

			if (nn_is_name_in_in_args(nn_init, "lambda")) {

				nn_init = nn_append_singleton_dim_in_F(nn_init, 0, "lambda");
				network = ((config->share_lambda) ? nn_mark_dup_if_exists_F : nn_mark_stack_input_if_exists_F)(network, "lambda");
			}

			network = nn_chain2_swap_FF(nn_init, 0, "init", network, 0, NULL);
			network = nn_stack_dup_by_name_F(network);

		} else {

			network = nn_dup_F(network, 0, "adjoint", 0, NULL);
		}

	} else {

		network = nn_set_input_name_F(network, 0, "initialization");
		network = nn_set_in_type_F(network, 0, "initialization", IN_BATCH_GENERATOR);
	}


	if (nn_is_name_in_in_args(network, "lambda")) {

		long out_dims[N + 1];
		long lam_dims[N + 1]; // different lambda for different iteration

		md_copy_dims(N + 1, out_dims, nn_generic_domain(network, 0, "lambda")->dims);
		md_select_dims(N + 1, MD_BIT(N), lam_dims, out_dims);

		if (config->dc_gradient && config->dc_scale_max_eigen) {

			network = nn_chain2_swap_FF(nn_from_nlop_F(nlop_sense_scale_maxeigen_create(Nb, models, N + 1, out_dims)), 0, NULL, network, 0, "lambda");

			auto nn_set_data = nn_from_nlop_F(nlop_sense_model_set_data_batch_create(N + 1, out_dims, Nb, models));
			nn_set_data = nn_set_input_name_F(nn_set_data, 1, "coil");
			nn_set_data = nn_set_input_name_F(nn_set_data, 1, "psf");
			network = nn_chain2_swap_FF(nn_set_data, 0, NULL, network, 0, NULL);
			network = nn_set_input_name_F(network, 0, "lambda");

			network = nn_mark_dup_F(network, "coil");
			network = nn_mark_dup_F(network, "psf");
		}

		if (!md_check_equal_dims(N + 1, lam_dims, out_dims, ~0UL)) {

			network = nn_chain2_swap_FF(nn_from_nlop_F(nlop_from_linop_F(linop_repmat_create(N + 1, out_dims, ~MD_BIT(N)))), 0, NULL, network, 0, "lambda");
			network = nn_set_input_name_F(network, 0, "lambda");
		}

		if (-1 != config->dc_lambda_fixed) {

			complex float lambda = config->dc_lambda_fixed;
			network = nn_set_input_const_F2(network, 0, "lambda", N + 1, lam_dims, MD_SINGLETON_STRS(N + 1), true, &lambda);

		} else {

			network = nn_set_prox_op_F(network, 0, "lambda", operator_project_pos_real_create(N + 1, lam_dims));
			network = nn_set_in_type_F(network, 0, "lambda", IN_OPTIMIZE);
			network = nn_set_initializer_F(network, 0, "lambda", init_const_create(config->dc_lambda_init));

			network = nn_reshape_in_F(network, 0, "lambda", 1, lam_dims + N);
		}
	}

	if (nn_is_name_in_in_args(network, "lambda_init")) {

		long ldims[N];

		md_copy_dims(N, ldims, nn_generic_domain(network, 0, "lambda_init")->dims);

		complex float one = 1;
		auto scale_lambda = nn_from_nlop_F(nlop_tenmul_create(N, ldims, MD_SINGLETON_DIMS(N), ldims));
		scale_lambda = nn_set_input_const_F2(scale_lambda, 1, NULL, N, ldims, MD_SINGLETON_STRS(N), true, &one);

		network = nn_chain2_swap_FF(scale_lambda, 0, NULL, network, 0, "lambda_init");
		network = nn_set_input_name_F(network, 0, "lambda_init");

		network = nn_reshape_in_F(network, 0, "lambda_init", 1, MD_SINGLETON_DIMS(1));

		network = nn_set_prox_op_F(network, 0, "lambda_init", operator_project_pos_real_create(1, MD_SINGLETON_DIMS(1)));
		network = nn_set_in_type_F(network, 0, "lambda_init", IN_OPTIMIZE);
		network = nn_set_initializer_F(network, 0, "lambda_init", init_const_create(config->init_lambda_init));
	}

	long img_dims[N];
	sense_model_get_img_dims(config->sense_config, N, img_dims);
	img_dims[BATCH_DIM] = Nb;

	if (config->precomp) {

		auto nn_set_data = nn_from_nlop_F(nlop_sense_model_set_data_batch_create(N, img_dims, Nb, models));

		nn_set_data = nn_set_input_name_F(nn_set_data, 1, "coil");
		nn_set_data = nn_set_input_name_F(nn_set_data, 1, "psf");

		network = nn_chain2_swap_FF(nn_set_data, 0, NULL, network, 0, "adjoint");
		network = nn_set_input_name_F(network, 0, "adjoint");
		network = nn_stack_dup_by_name_F(network);
	} else {

		auto nn_adjoint = nn_from_nlop_F(nlop_sense_adjoint_create(Nb, models, false));

		nn_adjoint = nn_set_input_name_F(nn_adjoint, 0, "kspace");
		nn_adjoint = nn_set_input_name_F(nn_adjoint, 0, "coil");
		nn_adjoint = nn_set_input_name_F(nn_adjoint, 0, "pattern");

		if (1 == nn_get_nr_unnamed_in_args(nn_adjoint))
			nn_adjoint = nn_set_input_name_F(nn_adjoint, 0, "trajectory");

		network = nn_chain2_swap_FF(nn_adjoint, 0, NULL, network, 0, "adjoint");
	}

	network = nn_set_output_name_F(network, 0 , "reconstruction");

	if (config->coil_image || config->ksp_training) {

		assert(!config->ksp_training || !config->precomp || nn_is_name_in_in_args(network, "trajectory"));

		const struct nlop_s* loss_trafo = nlop_mri_loss_create(config->ksp_training, Nb, models);
		network = nn_chain2_FF(network, 0, "reconstruction", nn_from_nlop_F(loss_trafo), 0, NULL);
		network = nn_set_output_name_F(network, 0, "reconstruction");
	}

	network = reconet_sort_args(network);

	for (int i = 0; i < Nb; i++)
		sense_model_free(models[i]);

	return network;
}



static nn_t reconet_train_create(const struct reconet_s* config, int Nb, bool valid)
{
	static bool recursive = false;

	if (   !recursive
	    && (1 < mpi_get_num_procs())
	    && (1 < Nb)
	    && (!valid || (0 == Nb % mpi_get_num_procs()))
	    && network_is_diagonal(config->network)) {

		if (0 != Nb % mpi_get_num_procs())
			error("Batch size must be multiple of number of ranks!\n");

		recursive = true;

		nn_t tmp[Nb];

		for (int i = 0; i < Nb; i++) {

			tmp[i] = reconet_train_create(config, 1, valid);

			for (int j = 1; j < mpi_get_num_procs(); j++) {

				i++;
				tmp[i] = nn_clone(tmp[i - 1]);
			}
		}

		return nn_stack_multigpu_F(Nb, tmp, BATCH_DIM);
	}

	if (!recursive && (1 < mpi_get_num_procs())) {

#ifdef NON_DETERMINISTIC
		assert(0);
#endif
		mri_ops_activate_multigpu();
	}

	auto train_op = reconet_create(config, Nb, valid ? STAT_TEST : STAT_TRAIN);

	const struct iovec_s* cod = nn_generic_codomain(train_op, 0, "reconstruction");

	int N = cod->N;
	long out_dims[N];
	md_copy_dims(N, out_dims, cod->dims);

	long scl_dims[N];
	md_select_dims(N, BATCH_FLAG, scl_dims, out_dims);

	auto loss_op = valid 	? val_measure_create(config->valid_loss, N, out_dims)
				: train_loss_create(config->train_loss, N, out_dims);

	loss_op = nn_set_input_name_F(loss_op, 1, "reference");

	train_op = nn_chain2_FF(train_op, 0, "reconstruction", loss_op, 0, NULL);

	if (valid)
		train_op = nn_del_out_bn_F(train_op);

	if (config->normalize)
		train_op = reconet_normalization(train_op);

	train_op = reconet_sort_args(train_op);

	return train_op;
}

static nn_t reconet_valid_create(struct reconet_s* config, int Nb, struct named_data_list_s* valid_data)
{
	auto ref_iov = named_data_list_get_iovec(valid_data, "reference");
	config->coil_image = (1 != ref_iov->dims[COIL_DIM]);

	iovec_free(ref_iov);

	auto valid_loss = reconet_train_create(config, Nb, true);

	return nn_valid_create(valid_loss, valid_data);
}


static nn_t reconet_apply_op_create(const struct reconet_s* config)
{
	auto nn_apply = reconet_create(config, 1, STAT_TEST);

	if (config->normalize)
		nn_apply = reconet_normalization(nn_apply);

	nn_apply = reconet_sort_args(nn_apply);
	nn_apply = nn_get_wo_weights_F(nn_apply, config->weights, false);

	int N = sense_model_get_N(config->sense_config);

	if (config->coil_image) {

		long cim_dims[N];
		long img_dims[N];
		long col_dims[N];

		sense_model_get_cim_dims(config->sense_config, N, cim_dims);
		sense_model_get_img_dims(config->sense_config, N, img_dims);
		sense_model_get_col_dims(config->sense_config, N, col_dims);

		nn_apply = nn_chain2_FF(nn_apply , 0, "reconstruction", nn_from_nlop_F(nlop_tenmul_create(N, cim_dims, img_dims, col_dims)), 0, NULL);
		nn_apply = nn_dup_F(nn_apply , 0, "coil", 0, NULL);
		nn_apply = nn_set_output_name_F(nn_apply , 0, "reconstruction");
	}

	debug_printf(DP_INFO, "Apply RecoNet\n");
	nn_debug(DP_INFO, nn_apply);

	return nn_apply;
}



void train_reconet(	struct reconet_s* config,
			long Nb_train, struct named_data_list_s* train_data,
			long Nb_valid, struct named_data_list_s* valid_data)
{
	unsigned long bat_flags = BATCH_FLAG;

	auto ref_iov = named_data_list_get_iovec(train_data, "reference");

	config->coil_image = (1 != ref_iov->dims[COIL_DIM]);
	long ntot = ref_iov->dims[BATCH_DIM];

	iovec_free(ref_iov);

	auto nn_train = reconet_train_create(config, Nb_train, false);

	if (config->ref_is_kspace) {

		//save GPU memory
		nn_train = nn_mark_dup_if_exists_F(nn_train, "kspace");
		nn_train = nn_rename_input_F(nn_train, "kspace", "reference");
		nn_train = nn_stack_dup_by_name_F(nn_train);
	}

	debug_printf(DP_INFO, "Train Reconet\n");
	nn_debug(DP_INFO, nn_train);

	if (NULL == config->weights) {

		config->weights = nn_weights_create_from_nn(nn_train);
		nn_init(nn_train, config->weights);

	} else {

		auto tmp_weights = nn_weights_create_from_nn(nn_train);
		nn_weights_copy(tmp_weights, config->weights);
		nn_weights_free(config->weights);
		config->weights = tmp_weights;
	}

	if (config->gpu)
		move_gpu_nn_weights(config->weights);

	//create batch generator
	struct bat_gen_conf_s batgen_config = bat_gen_conf_default;
	batgen_config.bat_flags = bat_flags;
	batgen_config.seed = config->train_conf->batch_seed;
	batgen_config.type = config->train_conf->batchgen_type;

	auto batch_generator = nn_batchgen_create(&batgen_config, nn_train, train_data);

	//setup for iter algorithm
	int NI = nn_get_nr_in_args(nn_train);
	int NO = nn_get_nr_out_args(nn_train);

	float* src[NI];

	enum IN_TYPE in_type[NI];
	nn_get_in_types(nn_train, NI, in_type);

	const struct operator_p_s* projections[NI];
	nn_get_prox_ops(nn_train, NI, projections);

	enum OUT_TYPE out_type[NO];
	nn_get_out_types(nn_train, NO, out_type);

	int weight_index = 0;

	for (int i = 0; i < NI; i++) {

		switch (in_type[i]) {

		case IN_BATCH_GENERATOR:

			src[i] = NULL;
			break;

		case IN_UNIFORM_RAND:
		case IN_GAUSSIAN_RAND:
		case IN_BATCH:
		case IN_UNDEFINED:
		case IN_STATIC:

			assert(0);
			break;

		case IN_OPTIMIZE:
		case IN_BATCHNORM:

			auto iov_weight = config->weights->iovs[weight_index];
			auto iov_train_op = nlop_generic_domain(nn_get_nlop(nn_train), i);

			assert(md_check_equal_dims(iov_weight->N, iov_weight->dims, iov_train_op->dims, ~0UL));

			src[i] = (float*)config->weights->tensors[weight_index];
			weight_index++;
		}
	}

	int num_monitors = 0;
	const struct monitor_value_s* value_monitors[3];

	if (NULL != valid_data) {

		auto nn_validation_loss = reconet_valid_create(config, Nb_valid, valid_data);

		const char* val_names[nn_get_nr_out_args(nn_validation_loss)];

		for (int i = 0; i < nn_get_nr_out_args(nn_validation_loss); i++)
			val_names[i] = nn_get_out_name_from_arg_index(nn_validation_loss, i, false);

		value_monitors[num_monitors] = monitor_iter6_nlop_create(nn_get_nlop(nn_validation_loss), false, nn_get_nr_out_args(nn_validation_loss), val_names);

		nn_free(nn_validation_loss);

		num_monitors += 1;
	}

	bool monitor_lambda = true;

	if (monitor_lambda && nn_is_name_in_in_args(nn_train, "lambda_init")) {

		int index_lambda = nn_get_in_arg_index(nn_train, 0, "lambda_init");

		const char* lams[1] = { "li" };
		auto lambda_i = nlop_from_linop_F(linop_identity_create(1, MD_DIMS(1)));

		for(int i = 0; i < index_lambda; i++)
			lambda_i  = nlop_combine_FF(nlop_del_out_create(1, MD_DIMS(1)), lambda_i );

		for(int i = index_lambda + 1; i < NI; i++)
			lambda_i  = nlop_combine_FF(lambda_i , nlop_del_out_create(1, MD_DIMS(1)));

		value_monitors[num_monitors] = monitor_iter6_nlop_create(lambda_i , true, 1, lams);

		nlop_free(lambda_i);

		num_monitors += 1;
	}


	if (monitor_lambda && nn_is_name_in_in_args(nn_train, "lambda")) {

		int index_lambda = nn_get_in_arg_index(nn_train, 0, "lambda");
		int num_lambda = nn_generic_domain(nn_train, 0, "lambda")->dims[0];

		const char* lams[num_lambda];

		for (int i = 0; i < num_lambda; i++)
			lams[i] = ptr_printf("l%d", i);

		auto destack_lambda = nlop_from_linop_F(linop_identity_create(2, MD_DIMS(1, num_lambda)));

		for (int i = num_lambda - 1; 0 < i; i--)
			destack_lambda = nlop_chain2_FF(destack_lambda, 0, nlop_destack_create(2, MD_DIMS(1, i), MD_DIMS(1, 1), MD_DIMS(1, i + 1), 1), 0);

		for(int i = 0; i < index_lambda; i++)
			destack_lambda = nlop_combine_FF(nlop_del_out_create(1, MD_DIMS(1)), destack_lambda);

		for(int i = index_lambda + 1; i < NI; i++)
			destack_lambda = nlop_combine_FF(destack_lambda, nlop_del_out_create(1, MD_DIMS(1)));

		value_monitors[num_monitors] = monitor_iter6_nlop_create(destack_lambda, true, num_lambda, lams);
		nlop_free(destack_lambda);

		for (int i = 0; i < num_lambda; i++)
			xfree(lams[i]);

		num_monitors += 1;
	}

	struct monitor_iter6_s* monitor = monitor_iter6_create(true, true, num_monitors, value_monitors);

	iter6_by_conf(config->train_conf, nn_get_nlop(nn_train), NI, in_type, projections, src, NO, out_type, Nb_train, ntot / Nb_train, batch_generator, monitor);

	nn_free(nn_train);
	nlop_free(batch_generator);

	monitor_iter6_free(monitor);
}

void apply_reconet(struct reconet_s* config, struct named_data_list_s* data)
{
	if (config->gpu)
		move_gpu_nn_weights(config->weights);

	auto ref_iov = named_data_list_get_iovec(data, "reconstruction");
	config->coil_image = (1 != ref_iov->dims[COIL_DIM]);
	iovec_free(ref_iov);

	auto nn_apply = reconet_apply_op_create(config);

	nn_apply_named_list(nn_apply, data, config->weights->tensors[0]);

	nn_free(nn_apply);

	if (config->normalize_rss) {

		auto dom_rec =  named_data_list_get_iovec(data, "reconstruction");
		auto dom_col =  named_data_list_get_iovec(data, "coil");

		assert(dom_col->N == dom_rec->N);
		assert(!config->coil_image);

		complex float* tmp = md_alloc(dom_rec->N, dom_rec->dims, CFL_SIZE);

		md_zrss(dom_col->N, dom_col->dims, COIL_FLAG, tmp, named_data_list_get_data(data, "coil"));
		md_zmul(dom_rec->N, dom_rec->dims, named_data_list_get_data(data, "reconstruction"), named_data_list_get_data(data, "reconstruction"), tmp);

		md_free(tmp);
	}
}

void eval_reconet(struct reconet_s* config, struct named_data_list_s* data)
{
	auto dom_rec = named_data_list_get_iovec(data, "reference");

	complex float* tmp_out = md_alloc(dom_rec->N, dom_rec->dims, CFL_SIZE);

	named_data_list_append(data, dom_rec->N, dom_rec->dims, tmp_out, "reconstruction");

	apply_reconet(config, data);

	int N = sense_model_get_N(config->sense_config);

	long tout_dims[N];
	md_select_dims(N, ~BATCH_FLAG, tout_dims, dom_rec->dims);
	auto loss = val_measure_create(config->valid_loss, N, tout_dims);
	int NL = nn_get_nr_out_args(loss);

	const struct nlop_s* loss_op = nlop_clone(nn_get_nlop(loss));

	for (int i = 1; i < NL; i++)
		loss_op = nlop_stack_outputs_F(loss_op, 0, 1, 0);

	for (int i = 0; i < N - 1; i++)
		loss_op = nlop_append_singleton_dim_out_F(loss_op, 0);

	long tloss_dims[N];
	md_select_dims(N, BATCH_FLAG, tloss_dims, dom_rec->dims);
	tloss_dims[0] = NL;

	complex float* tloss = md_alloc(N, tloss_dims, CFL_SIZE);

	int DO[1] = { N };
	const long* odims[1] = { tloss_dims };
	complex float* loss_arr[1] = { tloss };

	int DI[] = { N, N };
	const long* idims[2] = { dom_rec->dims, dom_rec->dims };
	const complex float* input_arr[2] = { tmp_out, named_data_list_get_data(data, "reference") };

	nlop_generic_apply_loop(loss_op, BATCH_FLAG, 1, DO , odims, loss_arr, 2, DI, idims, input_arr);
	nlop_free(loss_op);

	complex float losses[NL];
	md_zavg(N, tloss_dims, ~1UL, losses, tloss);

	for (int i = 0; i < NL; i++)
		debug_printf(DP_INFO, "%s: %e\n", nn_get_out_name_from_arg_index(loss, i, false), crealf(losses[i]));

	nn_free(loss);
	md_free(tloss);
	md_free(tmp_out);

	iovec_free(dom_rec);
}

