#include <assert.h>
#include <float.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "iter/italgos.h"

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"
#include "misc/types.h"

#include "nn/const.h"
#include "nn/losses_nn.h"
#include "nn/weights.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/fft.h"
#include "num/ops.h"
#include "num/rand.h"

#include "iter/proj.h"
#include <math.h>
#include <string.h>

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/linop.h"
#include "linops/someops.h"

#include "iter/iter.h"
#include "iter/iter6.h"
#include "iter/monitor_iter6.h"
#include "iter/batch_gen.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/tenmul.h"
#include "nlops/someops.h"
#include "nlops/mri_ops.h"
#include "nlops/const.h"
#include "nlops/stack.h"

#include "nn/activation.h"
#include "nn/layers.h"
#include "nn/losses.h"
#include "nn/init.h"

#include "nn/init.h"

#include "nn/nn.h"
#include "nn/chain.h"
#include "nn/layers_nn.h"
#include "nn/activation_nn.h"

#include "nn/nn_ops.h"

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

	.mri_config = NULL,

	//data consistency config
	.dc_lambda_fixed = -1.,
	.dc_lambda_init = -1.,
	.dc_gradient = false,
	.dc_scale_max_eigen = false,
	.dc_tickhonov = false,
	.dc_max_iter = 10,

	//network initialization
	.normalize = false,
	.tickhonov_init = false,
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

	.normalize_rss = false,
};

static void reconet_init_default(struct reconet_s* reconet) {

	//network initialization
	reconet->init_max_iter = (-1 == reconet->init_max_iter) ? (reconet->dc_tickhonov ? reconet->dc_max_iter : 30) : reconet->init_max_iter;
	reconet->init_lambda_fixed = (-2 == reconet->init_lambda_fixed) ? (reconet->dc_tickhonov ? reconet->dc_lambda_fixed : 0) : reconet->init_lambda_fixed;
	reconet->init_lambda_init = (-1 == reconet->init_lambda_init) ? (reconet->dc_tickhonov ? reconet->dc_lambda_init : 0.1) : reconet->init_lambda_init;
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

	reconet->mri_config = (NULL == reconet->mri_config) ? &conf_nlop_mri_simple: reconet->mri_config;

	//data consistency config
	reconet->dc_lambda_init = (-1 == reconet->dc_lambda_init) ? 0.05 : reconet->dc_lambda_init;
	if (!reconet->dc_tickhonov && !reconet->dc_gradient)
		reconet->dc_tickhonov = true;

	if (NULL == reconet->train_loss) {

		reconet->train_loss =  &loss_option;
		reconet->train_loss->weighting_mse = 1.;
	}

	if (NULL == reconet->valid_loss)
		reconet->valid_loss =  &loss_image_valid;

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

	reconet->mri_config = (NULL == reconet->mri_config) ? &conf_nlop_mri_simple: reconet->mri_config;

	//data consistency config
	reconet->dc_lambda_init = (-1 == reconet->dc_lambda_init) ? 0.2 : reconet->dc_lambda_init;
	if (!reconet->dc_tickhonov && !reconet->dc_gradient)
		reconet->dc_gradient = true;

	if (NULL == reconet->train_loss) {

		reconet->train_loss =  &loss_option;
		reconet->train_loss->weighting_mse_rss = 1.;
	}

	if (NULL == reconet->valid_loss)
		reconet->valid_loss =  &loss_image_valid;

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

/**
 * Returns dataconsistency block using Tickhonov regularization
 *
 * Out	= argmin_x ||Ax - y||^2 + Lambda||x - In||^2
 *	= (A^HA + Lambda)^-1[A^Hy + Lambda In]
 *
 * Input tensors:
 *
 * INDEX_0: 	idims
 * adjoint:	idims
 * coil:	cdims
 * pattern:	pdims
 * lambda:	ldims
 *
 * Output tensors:
 *
 * INDEX_0:	idims
 */
static nn_t data_consistency_tickhonov_create(const struct reconet_s* config, unsigned int N, const long max_dims[N], unsigned int ND, const long psf_dims[ND])
{
	struct iter_conjgrad_conf iter_conf = iter_conjgrad_defaults;
	iter_conf.l2lambda = 0.;
	iter_conf.maxiter = config->dc_max_iter;

	long ldims[N];
	md_select_dims(N, config->mri_config->batch_flags, ldims, max_dims);

	long img_dims[N];
	md_select_dims(N, config->mri_config->image_flags, img_dims, max_dims);

	const struct nlop_s* nlop_dc = nlop_mri_dc_prox_create(N, max_dims, ldims, ND, psf_dims, config->mri_config, &iter_conf); // in: input, adjoint, coil, pattern, lambda; out: output

	auto result = nn_from_nlop_F(nlop_dc);
	result = nn_set_input_name_F(result, 1, "adjoint");
	result = nn_set_input_name_F(result, 1, "coil");
	result = nn_set_input_name_F(result, 1, "psf");

	result = nn_set_input_name_F(result, 1, "lambda");
	result = nn_set_in_type_F(result, 0, "lambda", IN_OPTIMIZE);
	result = nn_set_initializer_F(result, 0, "lambda", init_const_create(config->dc_lambda_init));

	auto iov = nn_generic_domain(result, 0, "lambda");
	auto prox_conv = operator_project_pos_real_create(iov->N, iov->dims);
	result = nn_set_prox_op_F(result, 0, "lambda", prox_conv);

	nn_debug(DP_DEBUG3, result);

	return result;// in:  input, adjoint, coil, pattern, lambda; out: output
}


/**
 * Returns dataconsistency block using a gradient step
 *
 * Out	= lambda (A^HA In - A^H kspace)
 *
 * Input tensors:
 * INDEX_0	idims
 * adjoint:	idims
 * coil:	dims
 * pattern:	pdims
 * lambda:	ldims
 *
 * Output tensors:
 *
 * INDEX_0	idims
 */
static nn_t data_consistency_gradientstep_create(const struct reconet_s* config, unsigned int N, const long max_dims[N], unsigned int ND, const long psf_dims[ND])
{
	long img_dims[N];
	long ldims[N];

	md_select_dims(N, config->mri_config->image_flags, img_dims, max_dims);
	md_select_dims(N, config->mri_config->batch_flags, ldims, img_dims);


	const struct nlop_s* nlop_result = nlop_mri_normal_create(N, max_dims, ND, psf_dims, config->mri_config);
	nlop_result = nlop_chain2_FF(nlop_result, 0, nlop_zaxpbz_create(N, img_dims, 1, -1.), 0);

	const struct nlop_s* nlop_scale = nlop_tenmul_create(N, img_dims, img_dims, ldims);
	nlop_result = nlop_chain2_swap_FF(nlop_result, 0, nlop_scale, 0);

	nn_t result = nn_from_nlop_F(nlop_result);

	result = nn_set_input_name_F(result, 0, "adjoint");
	result = nn_set_input_name_F(result, 1, "coil");
	result = nn_set_input_name_F(result, 1, "psf");

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
 * for a network
 * [and normalization scale]
 *
 * The output init is either the adjoint reconstruction AHb or the regularized SENSE reconstruction
 * (AHA + lambda)^-1AHb
 *
 * The output is normalized such that the maximum absolute value of the adjoint or SENSE reconstruction is one.
 * If lambda is trainable, the data is always normalized with respect to the adjoint reconstruction.
 *
 * @param mri_init
 * @param N
 * @param dims
 * @param idims
 *
 * Input tensors:
 * adjoint:	idims
 * [coils:	cdims]
 * [pattern:	pdims]
 * [lambda:	sdims]
 *
 * Output tensors:
 * adjoint	idims; (normalized)
 * init		idims;
 * [scale: 	sdims]
 */
static nn_t nn_init_create(const struct reconet_s* config, int N, const long max_dims[N], int ND, const long psf_dims[N])
{
	long img_dims[N];
	long scl_dims[N];

	md_select_dims(N, config->mri_config->image_flags, img_dims, max_dims);
	md_select_dims(N, config->mri_config->batch_flags, scl_dims, img_dims);

	if (!config->tickhonov_init) {

		auto nlop_result = nlop_combine_FF(nlop_from_linop_F(linop_identity_create(N, img_dims)), nlop_from_linop_F(linop_identity_create(N, img_dims)));
		nlop_result = nlop_dup_F(nlop_result, 0, 1);

		auto nn_result = nn_from_nlop_F(nlop_result);
		nn_result = nn_set_output_name_F(nn_result, 0, "adjoint");
		nn_result = nn_set_output_name_F(nn_result, 0, "init");

		if (config->normalize) {

			auto nn_normalize = nn_from_nlop_F(nlop_norm_max_abs_create(N, img_dims, config->mri_config->batch_flags));
			nn_normalize = nn_set_output_name_F(nn_normalize, 1, "scale");
			nn_result = nn_chain2_FF(nn_normalize, 0, NULL, nn_result, 0, NULL);
		}

		nn_result = nn_set_input_name_F(nn_result, 0, "adjoint");

		return nn_result;
	}

	struct iter_conjgrad_conf iter_conf = iter_conjgrad_defaults;
	iter_conf.l2lambda = 0.;
	iter_conf.maxiter = config->init_max_iter;

	auto nlop_result = nlop_mri_normal_inv_create(N, max_dims, scl_dims, ND, psf_dims, config->mri_config, &iter_conf); //in: adjoint, coil, pattern, lambda; out: (A^HA + l)^-1 adjoint
	nlop_result = nlop_combine_FF(nlop_from_linop_F(linop_identity_create(N, img_dims)),  nlop_result);
	nlop_result = nlop_dup_F(nlop_result, 0, 1); //in: adjoint, coil, pattern, lambda; out: adjoint, (A^HA + l)^-1 adjoint

	auto nn_result = nn_from_nlop_F(nlop_result);
	nn_result = nn_set_input_name_F(nn_result, 1, "coil");
	nn_result = nn_set_input_name_F(nn_result, 1, "psf");

	bool same_lambda = config->dc_tickhonov;
	same_lambda = same_lambda && (config->dc_lambda_init == config->init_lambda_init);
	same_lambda = same_lambda && (config->dc_lambda_fixed == config->init_lambda_fixed);
	same_lambda = same_lambda && (config->dc_max_iter == config->init_max_iter);

	const char* lambda_name = (same_lambda && config->share_lambda) ? "lambda" : "lambda_init";
	nn_result = nn_set_input_name_F(nn_result, 1, lambda_name);


	if (-1 != config->init_lambda_fixed) {

		complex float lambda = config->init_lambda_fixed;
		nn_result = nn_set_input_const_F2(nn_result, 0, lambda_name, N, scl_dims, MD_SINGLETON_STRS(N), true, &lambda);
	}

	if (config->normalize) {

		auto nn_normalize = nn_from_nlop_F(nlop_norm_max_abs_create(N, img_dims, config->mri_config->batch_flags));
		nn_normalize = nn_set_output_name_F(nn_normalize, 1, "scale");

		if (-1 == config->init_lambda_fixed) {

			nn_result = nn_chain2_FF(nn_normalize, 0, NULL, nn_result, 0, NULL);
			nn_result = nn_set_output_name_F(nn_result, 0, "adjoint");
			nn_result = nn_set_output_name_F(nn_result, 0, "init");
		} else {

			nn_normalize = nn_set_output_name_F(nn_normalize, 0, "init");
			nn_result = nn_chain2_FF(nn_result, 1, NULL, nn_normalize, 0, NULL);

			const struct nlop_s* scale = nlop_tenmul_create(N, img_dims, img_dims, scl_dims);
			scale = nlop_chain2_FF(nlop_zinv_create(N, scl_dims), 0, scale, 1); //in: adjoint, scale; out: adjoint / scale

			auto nn_scale = nn_from_nlop_F(scale);
			nn_scale = nn_set_output_name_F(nn_scale, 0, "adjoint");

			nn_result = nn_chain2_keep_FF(nn_result, 0, "scale", nn_scale, 1, NULL);
			nn_result = nn_link_F(nn_result, 0, NULL, 0, NULL);
		}
	} else {

		nn_result = nn_set_output_name_F(nn_result, 0, "adjoint");
		nn_result = nn_set_output_name_F(nn_result, 0, "init");
	}

	nn_result = nn_set_input_name_F(nn_result, 0, "adjoint");

	return nn_result;
}


/**
 * Returns network block
 *
 * Input tensors:
 *
 * INDEX_0: 	idims:	(Ux, Uy, Uz, 1, Nb)
 * reinsert:	idims:	(Ux, Uy, Uz, 1, Nb) [Optional]
 * weights as listed in "sorted_weight_names"
 *
 * Output tensors:
 *
 * INDEX_0:	idims:	(Ux, Uy, Uz, 1, Nb)
 * [batchnorm output]
 */
static nn_t network_block_create(const struct reconet_s* config, unsigned int N, const long img_dims[N], enum NETWORK_STATUS status)
{

	assert(1 == bitcount(config->mri_config->batch_flags));
	int bat_dim = md_max_idx(config->mri_config->batch_flags);

	unsigned long channel_flag = (~(FFT_FLAGS | config->mri_config->batch_flags)) & (md_nontriv_dims(N, img_dims));
	assert(config->mri_config->batch_flags > channel_flag);

	long chn_dims[N];
	md_select_dims(N, channel_flag, chn_dims, img_dims);
	long channel = md_calc_size(N, chn_dims);

	long dims[5] = {img_dims[0], img_dims[1], img_dims[2], channel, img_dims[bat_dim]};
	long dims_net[5] = {channel, img_dims[0], img_dims[1], img_dims[2], img_dims[bat_dim]};

	nn_t result = NULL;

	result = network_create(config->network, 5, dims_net, 5, dims_net, status);

	if (1 != channel) {

		unsigned int iperm[5] = {3, 0, 1, 2, 4};
		unsigned int operm[5] = {1, 2, 3, 0, 4};

		result = nn_chain2_swap_FF(nn_from_nlop_F(nlop_from_linop_F(linop_permute_create(5, iperm, dims))), 0, NULL, result, 0, NULL);
		result = nn_chain2_swap_FF(result, 0, NULL, nn_from_nlop_F(nlop_from_linop_F(linop_permute_create(5, operm, dims_net))), 0, NULL);
	}


	result = nn_reshape_in_F(result, 0, NULL, N, img_dims);
	result = nn_reshape_out_F(result, 0, NULL, N, img_dims);

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

	return nn_checkpoint_F(result, true, (1 < config->Nt) && config->low_mem);
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
 * init:	idims	[optional]
 * weights as listed in "sorted_weight_names"
 *
 * Output tensors:
 *
 * INDEX_0:	idims
 * [batchnorm output]
 */
static nn_t reconet_cell_create(const struct reconet_s* config, unsigned int N, const long max_dims[N], int ND, const long psf_dims[ND], enum NETWORK_STATUS status)
{
	long img_dims[N];
	md_select_dims(N, config->mri_config->image_flags, img_dims, max_dims);

	auto result = network_block_create(config, N, img_dims, status);

	int N_in_names = nn_get_nr_named_in_args(result);
	int N_out_names = nn_get_nr_named_out_args(result);

	const char* sorted_in_names[6 + N_in_names];
	const char* sorted_out_names[N_out_names?:1];

	sorted_in_names[0] = "kspace";
	sorted_in_names[1] = "adjoint";
	sorted_in_names[2] = "coils";
	sorted_in_names[3] = "psf";
	sorted_in_names[4] = "lambda_init";
	sorted_in_names[5] = "lambda";
	nn_get_in_names_copy(N_in_names, sorted_in_names + 6, result);
	nn_get_out_names_copy(N_out_names, sorted_out_names, result);


	if (config->dc_tickhonov) {

		auto dc = data_consistency_tickhonov_create(config, N, max_dims, ND, psf_dims);
		result = nn_chain2_FF(result, 0, NULL, dc, 0, NULL);
	}

	if (config->dc_gradient) {

		auto dc = data_consistency_gradientstep_create(config, N, max_dims, ND, psf_dims);
		result = nn_combine_FF(dc, result);
		result = nn_dup_F(result, 0, NULL, 1, NULL);
		result = nn_combine_FF(nn_from_nlop_F(nlop_zaxpbz_create(N, img_dims, -1., 1)),result);
		result = nn_link_F(result, 1, NULL, 0, NULL);
		result = nn_link_F(result, 1, NULL, 0, NULL);
	}

	result = nn_sort_inputs_by_list_F(result, N_in_names + 6, sorted_in_names);
	result = nn_sort_outputs_by_list_F(result, N_out_names, sorted_out_names);

	for (int i = 6; i < N_in_names + 6; i++)
		xfree(sorted_in_names[i]);
	for (int i = 0; i < N_out_names; i++)
		xfree(sorted_out_names[i]);

	if (nn_is_name_in_in_args(result, "lambda"))
		result = nn_append_singleton_dim_in_F(result, 0, "lambda");

	return result;
}


static nn_t reconet_iterations_create(const struct reconet_s* config, int N, const long max_dims[N], int ND, const long psf_dims[N], enum NETWORK_STATUS status)
{
	auto result = reconet_cell_create(config, N, max_dims, ND, psf_dims, status);

	int N_in_names = nn_get_nr_named_in_args(result);
	int N_out_names = nn_get_nr_named_out_args(result);

	const char* in_names[N_in_names?:1];
	const char* out_names[N_out_names?:1];

	nn_get_in_names_copy(N_in_names, in_names, result);
	nn_get_out_names_copy(N_out_names, out_names, result);

	for (int i = 1; i < config->Nt; i++) {

		auto tmp = reconet_cell_create(config, N, max_dims, ND, psf_dims, status);

		tmp = nn_mark_dup_if_exists_F(tmp, "adjoint");
		tmp = nn_mark_dup_if_exists_F(tmp, "coil");
		tmp = nn_mark_dup_if_exists_F(tmp, "psf");
		tmp = nn_mark_dup_if_exists_F(tmp, "reinsert");

		tmp = (config->share_lambda ? nn_mark_dup_if_exists_F : nn_mark_stack_input_if_exists_F)(tmp, "lambda");

		// batchnorm weights are always stacked
		for (int i = 0; i < N_in_names; i++) {

			if (nn_is_name_in_in_args(tmp, in_names[i])) {

				if (nn_get_dup(tmp, 0, in_names[i]) && config->share_weights)
					tmp = nn_mark_dup_F(tmp, in_names[i]);
				else
					tmp = nn_mark_stack_input_F(tmp, in_names[i]);
			}
		}

		for (int i = 0; i < N_out_names; i++)
			tmp = nn_mark_stack_output_if_exists_F(tmp, out_names[i]);

		result = nn_chain2_FF(result, 0, NULL, tmp, 0, NULL);


		result = nn_stack_dup_by_name_F(result);
	}

	if (nn_is_name_in_in_args(result, "reinsert"))
		result = nn_dup_F(result, 0, NULL, 0, "reinsert");

	result = nn_sort_inputs_by_list_F(result, N_in_names, in_names);
	result = nn_sort_outputs_by_list_F(result, N_out_names, out_names);

	for (int i = 0; i < N_in_names; i++)
		xfree(in_names[i]);
	for (int i = 0; i < N_out_names; i++)
		xfree(out_names[i]);

	return result;
}

static bool is_name_in_list(int N, const char* names[N], const char* name)
{
	bool result = false;
	for (int i = 0; i < N; i++)
		result |= (NULL == names[i]) ? false : (0 == strcmp(names[i], name));
	return result;
}

static nn_t reconet_create(const struct reconet_s* config, int N, const long max_dims[N], int ND, const long psf_dims[N], enum NETWORK_STATUS status)
{
	auto network = reconet_iterations_create(config, N, max_dims, ND, psf_dims, status);

	int N_in_names = nn_get_nr_named_in_args(network);
	const char* in_names[N_in_names + 6];

	int i = 0;
	in_names[i++] = "kspace";
	in_names[i++] = "adjoint";
	in_names[i++] = "coil";
	in_names[i++] = "psf";
	in_names[i++] = "lambda_init";
	in_names[i++] = "lambda";

	const char* tnames[N_in_names];
	nn_get_in_names_copy(N_in_names, tnames, network);
	for (int j = 0; j < N_in_names; j++)
		if (!is_name_in_list(i, in_names, tnames[j]))
			in_names[i++] = tnames[j];

	auto nn_init = nn_init_create(config, N, max_dims, ND, psf_dims);

	if (nn_is_name_in_in_args(network, "coil"))
		nn_init = nn_mark_dup_if_exists_F(nn_init, "coil");

	if (nn_is_name_in_in_args(network, "psf"))
		nn_init = nn_mark_dup_if_exists_F(nn_init, "psf");

	if (nn_is_name_in_in_args(nn_init, "lambda")) {

		nn_init = nn_append_singleton_dim_in_F(nn_init, 0, "lambda");
		network = ((config->share_lambda) ? nn_mark_dup_if_exists_F : nn_mark_stack_input_if_exists_F)(network, "lambda");
	}

	nn_t result = nn_chain2_swap_FF(nn_init, 0, "adjoint", network, 0, "adjoint");
	result = nn_link_F(result, 0, "init", 0, NULL);
	result = nn_stack_dup_by_name_F(result);

	if (nn_is_name_in_in_args(result, "lambda")) {

		long out_dims[N + 1]; // N + 1 for stacking
		long bat_dims[N + 1]; // different scalings for different batches
		long lam_dims[N + 1]; // different lambda for different iteration

		md_copy_dims(N + 1, out_dims, nn_generic_domain(result, 0, "lambda")->dims);
		md_select_dims(N + 1, config->mri_config->batch_flags, bat_dims, out_dims);
		md_select_dims(N + 1, ~config->mri_config->batch_flags, lam_dims, out_dims);

		auto scale_lambda = nn_from_nlop_F(nlop_tenmul_create(N + 1, out_dims, lam_dims, bat_dims));
		result = nn_chain2_swap_FF(scale_lambda, 0, NULL, result, 0, "lambda");

		result = nn_set_input_name_F(result, 0, "lambda");
		result = nn_set_input_name_F(result, 0, "lambda_scale");

		if (-1 != config->dc_lambda_fixed) {

			complex float lambda = config->dc_lambda_fixed;
			result = nn_set_input_const_F2(result, 0, "lambda", N + 1, lam_dims, MD_SINGLETON_STRS(N + 1), true, &lambda);
		} else {

			result = nn_set_prox_op_F(result, 0, "lambda", operator_project_pos_real_create(N + 1, lam_dims));
			result = nn_set_in_type_F(result, 0, "lambda", IN_OPTIMIZE);
			result = nn_set_initializer_F(result, 0, "lambda", init_const_create(config->dc_lambda_init));

			result = nn_reshape_in_F(result, 0, "lambda", 1, lam_dims + N);
		}

		if (config->dc_gradient && config->dc_scale_max_eigen) {

			auto nlop_max_eigen = nlop_mri_normal_max_eigen_create(N, max_dims, ND, psf_dims, config->mri_config);
			nlop_max_eigen = nlop_reshape_out_F(nlop_max_eigen, 0, N + 1, bat_dims);
			nlop_max_eigen = nlop_chain2_FF(nlop_max_eigen, 0, nlop_zinv_create(N + 1, bat_dims), 0);

			auto nn_max_eigen = nn_from_nlop_F(nlop_max_eigen);
			nn_max_eigen = nn_set_input_name_F(nn_max_eigen, 0, "coil");
			nn_max_eigen = nn_set_input_name_F(nn_max_eigen, 0, "psf");

			nn_max_eigen = nn_mark_dup_F(nn_max_eigen, "coil");
			nn_max_eigen = nn_mark_dup_F(nn_max_eigen, "psf");

			result = nn_chain2_FF(nn_max_eigen, 0, NULL, result, 0, "lambda_scale");

			result = nn_stack_dup_by_name_F(result);

		} else {

			complex float one = 1;
			result = nn_set_input_const_F2(result, 0, "lambda_scale", N + 1, bat_dims, MD_SINGLETON_STRS(N + 1), true, &one);
		}
	}

	if (nn_is_name_in_in_args(result, "lambda_init")) {

		long ldims[N];

		md_copy_dims(N, ldims, nn_generic_domain(result, 0, "lambda_init")->dims);

		complex float one = 1;
		auto scale_lambda = nn_from_nlop_F(nlop_tenmul_create(N, ldims, MD_SINGLETON_DIMS(N), ldims));
		scale_lambda = nn_set_input_const_F2(scale_lambda, 1, NULL, N, ldims, MD_SINGLETON_STRS(N), true, &one);

		result = nn_chain2_swap_FF(scale_lambda, 0, NULL, result, 0, "lambda_init");
		result = nn_set_input_name_F(result, 0, "lambda_init");

		result = nn_set_prox_op_F(result, 0, "lambda_init", operator_project_pos_real_create(N + 1, ldims));
		result = nn_set_in_type_F(result, 0, "lambda_init", IN_OPTIMIZE);
		result = nn_set_initializer_F(result, 0, "lambda_init", init_const_create(config->init_lambda_init));

		result = nn_reshape_in_F(result, 0, "lambda_init", 1, MD_SINGLETON_DIMS(1));
	}

	if (config->coil_image) {

		long cim_dims[N];
		long img_dims[N];
		long col_dims[N];

		md_select_dims(N, config->mri_config->coil_image_flags, cim_dims, max_dims);
		md_select_dims(N, config->mri_config->image_flags, img_dims, max_dims);
		md_select_dims(N, config->mri_config->coil_flags, col_dims, max_dims);

		result = nn_chain2_FF(result, 0, NULL, nn_from_nlop_F(nlop_tenmul_create(N, cim_dims, img_dims, col_dims)), 0, NULL);
		result = nn_dup_F(result, 0, "coil", 0, NULL);
	}

	result = nn_sort_inputs_by_list_F(result, i, in_names);
	for (int j = 0; j < N_in_names; j++)
		xfree(tnames[j]);

	return result;
}

static nn_t reconet_train_create(const struct reconet_s* config, int N, const long max_dims[N], int ND, const long psf_dims[N], bool valid)
{
	long out_dims[N];
	long scl_dims[N];

	md_select_dims(N, config->coil_image ? config->mri_config->coil_image_flags : config->mri_config->image_flags, out_dims, max_dims);
	md_select_dims(N, config->mri_config->batch_flags, scl_dims, out_dims);

	auto train_op = reconet_create(config, N, max_dims, ND, psf_dims, valid ? STAT_TEST : STAT_TRAIN);

	if(config->normalize) {

		auto nn_norm_ref = nn_from_nlop_F(nlop_chain2_FF(nlop_zinv_create(N, scl_dims), 0, nlop_tenmul_create(N, out_dims, out_dims, scl_dims), 1));
		train_op = nn_chain2_FF(train_op, 0, "scale", nn_norm_ref, 1, NULL);
	} else {

		train_op = nn_combine_FF(nn_from_nlop_F(nlop_from_linop_F(linop_identity_create(N, out_dims))), train_op);
	}

	if (valid) {

		auto loss = val_measure_create(config->valid_loss, N, out_dims);

		train_op = nn_chain2_FF(train_op, 1, NULL, loss, 0, NULL);
		train_op = nn_link_F(train_op, 0, NULL, 0, NULL);
		train_op = nn_del_out_bn_F(train_op);
	} else {


		auto loss = train_loss_create(config->train_loss, N, out_dims);
		train_op = nn_chain2_FF(train_op, 1, NULL, loss, 0, NULL);
		train_op = nn_link_F(train_op, 0, NULL, 0, NULL);
	}

	return train_op;
}

static nn_t reconet_valid_create(struct reconet_s* config, struct network_data_s* vf)
{
	load_network_data(vf);

	config->mri_config->pattern_flags = md_nontriv_dims(vf->ND, vf->psf_dims);
	config->coil_image = (1 != vf->out_dims[COIL_DIM]);

	auto valid_loss = reconet_train_create(config, vf->N, vf->max_dims, vf->ND, vf->psf_dims, true);

	valid_loss = nn_ignore_input_F(valid_loss, 0, NULL,vf-> N, vf->out_dims, true, vf->out);
	valid_loss = nn_ignore_input_F(valid_loss, 0, "adjoint", vf->N, vf->img_dims, true, vf->adjoint);
	valid_loss = nn_ignore_input_F(valid_loss, 0, "coil", vf->N, vf->col_dims, true, vf->coil);
	valid_loss = nn_ignore_input_F(valid_loss, 0, "psf", vf->ND, vf->psf_dims, true, vf->psf);

	free_network_data(vf);

	return valid_loss;
}


static nn_t reconet_apply_op_create(const struct reconet_s* config, int N, const long max_dims[N], int ND, const long psf_dims[N])
{
	long img_dims[N];
	long scl_dims[N];

	md_select_dims(N, config->mri_config->image_flags, img_dims, max_dims);
	md_select_dims(N, config->mri_config->batch_flags, scl_dims, img_dims);

	auto nn_apply = reconet_create(config, N, max_dims, ND, psf_dims, STAT_TEST);

	if(config->normalize) {

		auto nn_norm_ref = nn_from_nlop_F(nlop_tenmul_create(N, img_dims, img_dims, scl_dims));

		nn_apply = nn_chain2_FF(nn_apply, 0, NULL, nn_norm_ref, 0, NULL);
		nn_apply = nn_link_F(nn_apply, 0, "scale", 0, NULL);
	}

	debug_printf(DP_INFO, "Apply RecoNet\n");
	nn_debug(DP_INFO, nn_apply);

	return nn_get_wo_weights_F(nn_apply, config->weights, false);
}



void train_reconet(	struct reconet_s* config, unsigned int N, const long max_dims[N],
			const long out_dims[N], _Complex float* ref,
			const long img_dims[N], const complex float* adjoint,
			const long col_dims[N], const _Complex float* coil,
			int ND, const long psf_dims[ND], const _Complex float* pattern,
			long Nb, struct network_data_s* valid_files)
{
	assert(1 == bitcount(config->mri_config->batch_flags));
	int bat_dim = md_max_idx(config->mri_config->batch_flags);

	long Nt = max_dims[bat_dim]; // number datasets
	Nb = MIN(Nb, Nt);

	long bat_max_dims[N];
	long bat_out_dims[N];
	long bat_img_dims[N];
	long bat_col_dims[N];
	long bat_psf_dims[ND];

	md_copy_dims(N, bat_max_dims, max_dims);
	md_copy_dims(N, bat_out_dims, out_dims);
	md_copy_dims(N, bat_img_dims, img_dims);
	md_copy_dims(N, bat_col_dims, col_dims);
	md_copy_dims(ND, bat_psf_dims, psf_dims);

	bat_max_dims[bat_dim] = Nb;
	bat_out_dims[bat_dim] = Nb;
	bat_img_dims[bat_dim] = Nb;
	bat_col_dims[bat_dim] = Nb;
	bat_psf_dims[bat_dim] = (1 == psf_dims[bat_dim]) ? 1 : Nb;


	config->mri_config->pattern_flags = md_nontriv_dims(ND, psf_dims);
	config->coil_image = (1 != out_dims[COIL_DIM]);

	auto nn_train = reconet_train_create(config, N, bat_max_dims, ND, bat_psf_dims, false);

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
	const complex float* train_data[] = {ref, adjoint, coil, pattern};
	const long* bat_dims[] = { bat_out_dims, bat_img_dims, bat_col_dims, bat_psf_dims };
	const long* tot_dims[] = { out_dims, img_dims, col_dims, psf_dims };

	auto batch_generator = batch_gen_create_from_iter(config->train_conf, 4, (const int[4]){ N, N ,N , ND }, bat_dims, tot_dims, train_data, 0);

	//setup for iter algorithm
	int NI = nn_get_nr_in_args(nn_train);
	int NO = nn_get_nr_out_args(nn_train);

	float* src[NI];

	for (int i = 0; i < config->weights->N; i++) {

		auto iov_weight = config->weights->iovs[i];
		auto iov_train_op = nlop_generic_domain(nn_get_nlop(nn_train), i + 4);
		assert(md_check_equal_dims(iov_weight->N, iov_weight->dims, iov_train_op->dims, ~0));
		src[i + 4] = (float*)config->weights->tensors[i];
	}

	enum IN_TYPE in_type[NI];
	const struct operator_p_s* projections[NI];
	enum OUT_TYPE out_type[NO];

	nn_get_in_types(nn_train, NI, in_type);
	nn_get_out_types(nn_train, NO, out_type);

	for (int i = 0; i < 4; i++) {

		src[i] = NULL;
		in_type[i] = IN_BATCH_GENERATOR;
	}

	for (int i = 0; i < NI; i++)
		projections[i] = nn_get_prox_op_arg_index(nn_train, i);

	int num_monitors = 0;
	const struct monitor_value_s* value_monitors[3];

	if (NULL != valid_files) {

		auto nn_validation_loss = reconet_valid_create(config, valid_files);
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

		const char* lams[1] = {"li"};

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

		const char* lam = "l";
		const char* lams[num_lambda];

		for (int i = 0; i < num_lambda; i++)
			lams[i] = lam;

		auto destack_lambda = nlop_from_linop_F(linop_identity_create(2, MD_DIMS(1, num_lambda)));
		for (int i = num_lambda - 1; 0 < i; i--)
			destack_lambda = nlop_chain2_FF(destack_lambda, 0, nlop_destack_create(2, MD_DIMS(1, i), MD_DIMS(1, 1), MD_DIMS(1, i + 1), 1), 0);

		for(int i = 0; i < index_lambda; i++)
			destack_lambda = nlop_combine_FF(nlop_del_out_create(1, MD_DIMS(1)), destack_lambda);
		for(int i = index_lambda + 1; i < NI; i++)
			destack_lambda = nlop_combine_FF(destack_lambda, nlop_del_out_create(1, MD_DIMS(1)));

		value_monitors[num_monitors] = monitor_iter6_nlop_create(destack_lambda, true, num_lambda, lams);
		nlop_free(destack_lambda);
		num_monitors += 1;
	}

	struct monitor_iter6_s* monitor = monitor_iter6_create(true, true, num_monitors, value_monitors);

	iter6_by_conf(config->train_conf, nn_get_nlop(nn_train), NI, in_type, projections, src, NO, out_type, Nb, Nt / Nb, batch_generator, monitor);

	nn_free(nn_train);
	nlop_free(batch_generator);

	monitor_iter6_free(monitor);
}

void apply_reconet(	struct reconet_s* config, unsigned int N, const long max_dims[N],
			const long out_dims[N], _Complex float* out,
			const long img_dims[N], const complex float* adjoint,
			const long col_dims[N], const _Complex float* coil,
			int ND, const long psf_dims[ND], const _Complex float* psf)
{
	if (config->gpu)
		move_gpu_nn_weights(config->weights);

	int DO[1] = { N };
	int DI[3] = { N, N, ND };

	const long* odims[1] = { out_dims };
	const long* idims[3] = { img_dims, col_dims, psf_dims };

	complex float* dst[1] = { out };
	const complex float* src[3] = { adjoint, coil, psf };

	long max_dims1[N];
	long psf_dims1[ND];

	md_select_dims(N, ~BATCH_FLAG, max_dims1, max_dims);
	md_select_dims(ND, ~BATCH_FLAG, psf_dims1, psf_dims);

	config->coil_image = (1 != out_dims[COIL_DIM]);
	auto nn_apply = reconet_apply_op_create(config, N, max_dims1, ND, psf_dims1);

	nlop_generic_apply_loop_sameplace(nn_get_nlop(nn_apply), BATCH_FLAG, 1, DO, odims, dst, 3, DI, idims, src, config->weights->tensors[0]);

	nn_free(nn_apply);

	if (config->normalize_rss) {

		assert(md_check_equal_dims(N, img_dims, out_dims, ~0));

		complex float* tmp = md_alloc_sameplace(N, img_dims, CFL_SIZE, out);
		md_zrss(N, col_dims, COIL_FLAG, tmp, coil);
		md_zmul(N, img_dims, out, out, tmp);
		md_free(tmp);
	}
}

void eval_reconet(	struct reconet_s* config, unsigned int N, const long max_dims[N],
			const long out_dims[N], const complex float* out,
			const long img_dims[N], const complex float* adjoint,
			const long col_dims[N], const complex float* coil,
			int ND, const long psf_dims[ND], const complex float* psf)
{
	complex float* tmp_out = md_alloc(N, out_dims, CFL_SIZE);

	auto loss = val_measure_create(config->valid_loss, N, out_dims);
	int NL = nn_get_nr_out_args(loss);
	complex float losses[NL];
	md_clear(1, MD_DIMS(NL), losses, CFL_SIZE);

	apply_reconet(config, N, max_dims, out_dims, tmp_out, img_dims, adjoint, col_dims, coil, ND, psf_dims, psf);

	complex float* args[NL + 2];
	for (int i = 0; i < NL; i++)
		args[i] = losses + i;

	complex float* tmp_ref = md_alloc_sameplace(N, out_dims, CFL_SIZE, out);

	md_copy(N, out_dims, tmp_ref, out, CFL_SIZE);

	if (config->normalize_rss) {

		assert(md_check_equal_dims(N, img_dims, out_dims, ~0));

		complex float* tmp = md_alloc_sameplace(N, img_dims, CFL_SIZE, out);
		md_zrss(N, col_dims, COIL_FLAG, tmp, coil);
		md_zmul(N, img_dims, tmp_ref, tmp_ref, tmp);

		md_free(tmp);
	}

	args[NL] = tmp_out;
	args[NL + 1] = tmp_ref;

	nlop_generic_apply_select_derivative_unchecked(nn_get_nlop(loss), NL + 2, (void**)args, 0, 0);
	for (int i = 0; i < NL ; i++)
		debug_printf(DP_INFO, "%s: %e\n", nn_get_out_name_from_arg_index(loss, i, false), crealf(losses[i]));

	nn_free(loss);
	md_free(tmp_out);
	md_free(tmp_ref);
}
