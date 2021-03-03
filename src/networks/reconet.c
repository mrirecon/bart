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

	.reinsert = false,

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
	.multi_loss = 0,

	.gpu = false,
	.low_mem = false,

	.graph_file = NULL,
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
		reconet->train_loss->weighting_mse_sa = 1.;
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
static nn_t data_consistency_tickhonov_create(const struct reconet_s* config, unsigned int N, const long dims[N], const long idims[N])
{
	struct iter_conjgrad_conf iter_conf = iter_conjgrad_defaults;
	iter_conf.l2lambda = 1.;
	iter_conf.maxiter = config->dc_max_iter;

	long ldims[N];
	md_select_dims(N, config->mri_config->batch_flags, ldims, dims);

	auto nlop_dc = mri_normal_inversion_create(N, dims, idims, config->mri_config, &iter_conf); // in: lambda * input + adjoint, coil, pattern, lambda; out: output
	nlop_dc = nlop_chain2_swap_FF(nlop_zaxpbz_create(N, idims, 1., 1.), 0, nlop_dc, 0); // in: lambda * input, adjoint, coil, pattern, lambda; out: output

	const struct nlop_s* nlop_scale_lambda = nlop_tenmul_create(N, idims, idims, ldims);
	nlop_scale_lambda = nlop_chain2_FF(nlop_from_linop_F(linop_zreal_create(N, ldims)), 0, nlop_scale_lambda, 1); // in: input, lambda; out: lambda * input

	nlop_dc = nlop_chain2_FF(nlop_scale_lambda, 0, nlop_dc, 0); // in: adjoint, coil, pattern, lambda, input, lambda; out: output
	nlop_dc = nlop_dup_F(nlop_dc, 3, 5); // in: adjoint, coil, pattern, lambda, input; out: output
	nlop_dc = nlop_shift_input_F(nlop_dc, 0, 4); // in: input, adjoint, coil, pattern, lambda; out: output

	// nlop_dc : in: input, adjoint, coil, pattern[, lambda]; out: output

	auto result = nn_from_nlop_F(nlop_dc);
	result = nn_set_input_name_F(result, 1, "adjoint");
	result = nn_set_input_name_F(result, 1, "coil");
	result = nn_set_input_name_F(result, 1, "pattern");

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
 * kspace:	dims
 * coil:	dims
 * pattern:	pdims
 * lambda:	ldims
 *
 * Output tensors:
 *
 * INDEX_0	idims
 */
static nn_t data_consistency_gradientstep_create(const struct reconet_s* config, unsigned int N, const long dims[N], const long idims[N])
{
	const struct nlop_s* nlop_result = nlop_mri_normal_create(N, dims, idims, config->mri_config);
	nlop_result = nlop_chain2_FF(nlop_result, 0, nlop_zaxpbz_create(N, idims, 1, -1.), 0);

	long ldims[N];
	md_select_dims(N, config->mri_config->batch_flags, ldims, dims);

	const struct nlop_s* nlop_scale = nlop_tenmul_create(N, idims, idims, ldims);
	nlop_result = nlop_chain2_swap_FF(nlop_result, 0, nlop_scale, 0);

	nn_t result = nn_from_nlop_F(nlop_result);

	result = nn_set_input_name_F(result, 0, "adjoint");
	result = nn_set_input_name_F(result, 1, "coil");
	result = nn_set_input_name_F(result, 1, "pattern");

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
 * @param mri_init
 * @param N
 * @param dims
 * @param idims
 *
 * Input tensors:
 * kspace:	kdims
 * coils:	cdims
 * pattern:	pdims
 * lambda:	(1))
 *
 * Output tensors:
 * INDEX_0	idims
 * [adjoint	idims; if INDEX0 is tickhonov regularized inverse]
 * [scale: 	(1,  1,  1,  1,  Nb)]
 */
static nn_t nn_init_create(const struct reconet_s* config, unsigned int N, const long dims[N], const long idims[N])
{
	auto nlop_zf = nlop_mri_adjoint_create(N, dims, idims, config->mri_config);
	auto result = nn_from_nlop_F(nlop_zf);
	result = nn_set_input_name_F(result, 0, "kspace");
	result = nn_set_input_name_F(result, 0, "coil");
	result = nn_set_input_name_F(result, 0, "pattern");

	// normalization based on tickhonov regularized input only makes sense if lambda is fixed
	if (config->normalize && !(config->tickhonov_init && (-1 != config->init_lambda_fixed))) {

		auto nn_normalize = nn_from_nlop_F(nlop_norm_max_abs_create(N, idims, config->mri_config->batch_flags));
		nn_normalize = nn_set_output_name_F(nn_normalize, 1, "scale");
		result = nn_chain2_FF(result, 0, NULL, nn_normalize, 0, NULL);
	}

	if (config->tickhonov_init) {

		struct iter_conjgrad_conf iter_conf = iter_conjgrad_defaults;
		iter_conf.l2lambda = 1.;
		iter_conf.maxiter = config->init_max_iter;

		auto nlop_dc = mri_normal_inversion_create(N, dims, idims, config->mri_config, &iter_conf); //in: adjoint, coil, pattern, lambda; out: (A^HA + l)^-1 adjoint

		if (config->normalize && (-1 != config->init_lambda_fixed)) {

			long sdims[N];
			md_select_dims(N, config->mri_config->batch_flags, sdims, idims);

			complex float lambda = config->init_lambda_fixed;
			nlop_dc = nlop_set_input_const_F2(nlop_dc, 3, N, sdims, MD_SINGLETON_STRS(N), true, &lambda); //in: adjoint, coil, pattern; out: (A^HA + l)^-1 adjoint
			nlop_dc = nlop_chain2_FF(nlop_dc, 0, nlop_norm_max_abs_create(N, idims, config->mri_config->batch_flags), 0); //in: adjoint, coil, pattern; out: (A^HA + l)^-1 adjoint / scale, scale

			const struct nlop_s* scale = nlop_tenmul_create(N, idims, idims, sdims);
			scale = nlop_chain2_FF(nlop_zinv_create(N, sdims), 0, scale, 1); //in: adjoint, scale; out: adjoint / scale

			nlop_dc = nlop_chain2_keep_FF(nlop_dc, 1, scale, 1); //in: adjoint, adjoint, coil pattern; out: adjoint / scale, (A^HA + l)^-1 adjoint / scale, scale
			nlop_dc = nlop_dup_F(nlop_dc, 0, 1); //in: adjoint, coil pattern; out: adjoint / scale, (A^HA + l)^-1 adjoint / scale, scale
		} else {

			nlop_dc = nlop_combine_FF(nlop_from_linop_F(linop_identity_create(N, idims)), nlop_dc);
			nlop_dc = nlop_dup_F(nlop_dc, 0, 1); // in: adjoint, coil, pattern, lambda; out: adjoint, (A^HA + lambda) adjoint
		}

		auto nn_dc = nn_from_nlop_F(nlop_dc);
		nn_dc = nn_set_input_name_F(nn_dc, 1, "coil");
		nn_dc = nn_set_input_name_F(nn_dc, 1, "pattern");
		nn_dc = nn_mark_dup_F(nn_dc, "coil");
		nn_dc = nn_mark_dup_F(nn_dc, "pattern");

		if (-1 == config->init_lambda_fixed) {

			bool same_lambda = config->dc_tickhonov;
			same_lambda = same_lambda && (config->dc_lambda_init == config->init_lambda_init);
			same_lambda = same_lambda && (config->dc_lambda_fixed == config->init_lambda_fixed);
			same_lambda = same_lambda && (config->dc_max_iter == config->init_max_iter);

			const char* name = (same_lambda && config->share_lambda) ? "lambda" : "lambda_init";

			nn_dc = nn_set_input_name_F(nn_dc, 1, name);
			nn_dc = nn_set_in_type_F(nn_dc, 0, name, IN_OPTIMIZE);
			nn_dc = nn_set_initializer_F(nn_dc, 0, name, init_const_create(config->init_lambda_init));

			auto iov = nn_generic_domain(nn_dc, 0, name);
			auto prox_conv = operator_project_pos_real_create(iov->N, iov->dims);
			nn_dc = nn_set_prox_op_F(nn_dc, 0, name, prox_conv);
		}

		if (config->normalize && (-1 != config->init_lambda_fixed))
			nn_dc = nn_set_output_name_F(nn_dc, 2, "scale");

		nn_dc = nn_set_output_name_F(nn_dc, 0, "adjoint");

		result = nn_chain2_FF(result, 0, NULL, nn_dc, 0, NULL);
		result = nn_stack_dup_by_name_F(result);
	}

	return result;
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
static nn_t network_block_create(const struct reconet_s* config, unsigned int N, const long idims[N], enum NETWORK_STATUS status)
{
	assert(5 == N);
	assert(1 == idims[3]);

	long idims_w[5] = {1, idims[0], idims[1], idims[2], idims[4]};
	long odims_w[5] = {1, idims[0], idims[1], idims[2], idims[4]};

	nn_t result = NULL;

	if (config->reinsert) {

		long idims_w2[5] = {2, idims[0], idims[1], idims[2], idims[4]};
		const struct nlop_s* nlop_init = nlop_stack_create(5, idims_w2, idims_w, idims_w, 0);
		nlop_init = nlop_reshape_in_F(nlop_init, 0, 5, idims);
		nlop_init = nlop_reshape_in_F(nlop_init, 1, 5, idims);

		result = nn_from_nlop_F(nlop_init);
		result = nn_set_input_name_F(result, 1, "reinsert");

		result = nn_chain2_FF(result, 0, NULL, config->network->create(config->network, N, odims_w, N, idims_w2, status), 0, NULL);

	} else {

		result = config->network->create(config->network, N, odims_w, N, idims_w, status);
		result = nn_reshape_in_F(result, 0, NULL, 5, idims);
	}

	result = nn_reshape_out_F(result, 0, NULL, 5, idims);

	int N_in_names = nn_get_nr_named_in_args(result);
	int N_out_names = nn_get_nr_named_out_args(result);

	const char* in_names[N_in_names];
	const char* out_names[N_out_names];

	nn_get_in_names_copy(N_in_names, in_names, result);
	nn_get_out_names_copy(N_out_names, out_names, result);

	for (int i = 0; i < N_in_names; i++) {

		if (0 != strcmp("reinsert", in_names[i]))
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
static nn_t reconet_cell_create(const struct reconet_s* config, unsigned int N, const long dims[N], const long idims[N], enum NETWORK_STATUS status)
{
	auto result = network_block_create(config, N, idims, status);

	int N_in_names = nn_get_nr_named_in_args(result);
	int N_out_names = nn_get_nr_named_out_args(result);

	const char* sorted_in_names[6 + N_in_names];
	const char* sorted_out_names[N_out_names];

	sorted_in_names[0] = "kspace";
	sorted_in_names[1] = "adjoint";
	sorted_in_names[2] = "coils";
	sorted_in_names[3] = "pattern";
	sorted_in_names[4] = "lambda_init";
	sorted_in_names[5] = "lambda";
	nn_get_in_names_copy(N_in_names, sorted_in_names + 6, result);
	nn_get_out_names_copy(N_out_names, sorted_out_names, result);

	if (config->dc_tickhonov) {

		auto dc = data_consistency_tickhonov_create(config, N, dims, idims);
		result = nn_chain2_FF(result, 0, NULL, dc, 0, NULL);
	}

	if (config->dc_gradient) {

		auto dc = data_consistency_gradientstep_create(config, N, dims, idims);
		result = nn_combine_FF(dc, result);
		result = nn_dup_F(result, 0, NULL, 1, NULL);
		result = nn_combine_FF(nn_from_nlop_F(nlop_zaxpbz_create(N, idims, -1., 1)),result);
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


static nn_t reconet_iterations_create(const struct reconet_s* config, unsigned int N, const long dims[N], const long idims[N], enum NETWORK_STATUS status)
{
	auto result = reconet_cell_create(config, N, dims, idims, status);

	int N_in_names = nn_get_nr_named_in_args(result);
	int N_out_names = nn_get_nr_named_out_args(result);

	const char* in_names[N_in_names];
	const char* out_names[N_out_names];

	nn_get_in_names_copy(N_in_names, in_names, result);
	nn_get_out_names_copy(N_out_names, out_names, result);

	for (int i = 1; i < config->Nt; i++) {

		auto tmp = reconet_cell_create(config, N, dims, idims, status);

		tmp = nn_mark_dup_if_exists_F(tmp, "adjoint");
		tmp = nn_mark_dup_if_exists_F(tmp, "coil");
		tmp = nn_mark_dup_if_exists_F(tmp, "pattern");
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

		char out_name[30];
		sprintf(out_name, "out_%d", i);
		result = nn_set_output_name_F(result, 0, out_name);
		result = nn_set_out_type_F(result, 0, out_name, OUT_STATIC);
		result = nn_chain2_keep_FF(result, 0, out_name, tmp, 0, NULL);

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

static nn_t reconet_create(const struct reconet_s* config, unsigned int N, const long dims[N], const long idims[N], enum NETWORK_STATUS status)
{
	auto network = reconet_iterations_create(config, N, dims, idims, status);

	int N_in_names = nn_get_nr_named_in_args(network);
	const char* in_names[N_in_names + 6];

	int i= 0;
	in_names[i++] = "kspace";
	in_names[i++] = "adjoint";
	in_names[i++] = "coil";
	in_names[i++] = "pattern";
	in_names[i++] = "lambda_init";
	in_names[i++] = "lambda";

	const char* tnames[N_in_names];
	nn_get_in_names_copy(N_in_names, tnames, network);
	for (int j = 0; j < N_in_names; j++)
		if (!is_name_in_list(i, in_names, tnames[j]))
			in_names[i++] = tnames[j];

	network = nn_mark_dup_if_exists_F(network, "coil");
	network = nn_mark_dup_if_exists_F(network, "pattern");
	network = nn_mark_dup_if_exists_F(network, "lambda_init");

	auto init = nn_init_create(config, N, dims, idims);
	if (nn_is_name_in_in_args(init, "lambda")) {

		init = nn_append_singleton_dim_in_F(init, 0, "lambda");
		network = ((config->share_lambda) ? nn_mark_dup_if_exists_F : nn_mark_stack_input_if_exists_F)(network, "lambda");
	}

	if (nn_is_name_in_in_args(init, "lambda_init"))
		init = nn_append_singleton_dim_in_F(init, 0, "lambda_init");

	nn_t result = NULL;

	if (nn_is_name_in_out_args(init, "adjoint")) {

		result = nn_chain2_swap_FF(init, 0, NULL, network, 0, NULL);
		result = nn_link_F(result, 0, "adjoint", 0, "adjoint");
	} else {

		result = nn_dup_F(network, 0, NULL, 0, "adjoint");
		result = nn_chain2_swap_FF(init, 0, NULL, result, 0, NULL);
	}

	result = nn_stack_dup_by_name_F(result);

	if (nn_is_name_in_in_args(result, "lambda")) {

		long odims[N + 1];
		long ldims[N + 1];
		long bdims[N + 1];

		md_copy_dims(N + 1, odims, nn_generic_domain(result, 0, "lambda")->dims);
		md_select_dims(N + 1, MD_BIT(N), ldims, odims);
		md_select_dims(N + 1, ~MD_BIT(N), bdims, odims);

		auto scale_lambda = nn_from_nlop_F(nlop_tenmul_create(N + 1, odims, ldims, bdims));
		result = nn_chain2_swap_FF(scale_lambda, 0, NULL, result, 0, "lambda");

		result = nn_set_input_name_F(result, 0, "lambda");
		result = nn_set_input_name_F(result, 0, "lambda_scale");

		if (-1 != config->dc_lambda_fixed) {

			complex float lambda = config->dc_lambda_fixed;
			result = nn_set_input_const_F2(result, 0, "lambda", N + 1, ldims, MD_SINGLETON_STRS(N + 1), true, &lambda);
		} else {

			result = nn_set_prox_op_F(result, 0, "lambda", operator_project_pos_real_create(N + 1, ldims));
			result = nn_set_in_type_F(result, 0, "lambda", IN_OPTIMIZE);
			result = nn_set_initializer_F(result, 0, "lambda", init_const_create(config->dc_lambda_init));

			result = nn_reshape_in_F(result, 0, "lambda", 1, ldims + N);
		}

		if (config->dc_gradient && config->dc_scale_max_eigen) {

			auto nlop_max_eigen = mri_normal_max_eigen_create(N, dims, idims, config->mri_config);
			nlop_max_eigen = nlop_reshape_out_F(nlop_max_eigen, 0, N + 1, bdims);
			nlop_max_eigen = nlop_chain2_FF(nlop_max_eigen, 0, nlop_zinv_create(N + 1, bdims), 0);

			auto nn_max_eigen = nn_from_nlop_F(nlop_max_eigen);
			nn_max_eigen = nn_set_input_name_F(nn_max_eigen, 0, "coil");
			nn_max_eigen = nn_set_input_name_F(nn_max_eigen, 0, "pattern");

			nn_max_eigen = nn_mark_dup_F(nn_max_eigen, "coil");
			nn_max_eigen = nn_mark_dup_F(nn_max_eigen, "pattern");

			result = nn_chain2_FF(nn_max_eigen, 0, NULL, result, 0, "lambda_scale");

			result = nn_stack_dup_by_name_F(result);

		} else {

			complex float one = 1;
			result = nn_set_input_const_F2(result, 0, "lambda_scale", N + 1, bdims, MD_SINGLETON_STRS(N + 1), true, &one);
		}
	}

	if (nn_is_name_in_in_args(result, "lambda_init")) {

		long odims[N + 1];
		long ldims[N + 1];
		long bdims[N + 1];

		md_copy_dims(N + 1, odims, nn_generic_domain(result, 0, "lambda_init")->dims);
		md_select_dims(N + 1, MD_BIT(N), ldims, odims);
		md_select_dims(N + 1, ~MD_BIT(N), bdims, odims);

		auto scale_lambda = nn_from_nlop_F(nlop_tenmul_create(N + 1, odims, ldims, bdims));
		result = nn_chain2_swap_FF(scale_lambda, 0, NULL, result, 0, "lambda_init");

		result = nn_set_input_name_F(result, 0, "lambda_init");
		result = nn_set_input_name_F(result, 0, "lambda_init_scale");

		if (-1 != config->dc_lambda_fixed) {

			complex float lambda = config->dc_lambda_fixed;
			result = nn_set_input_const_F2(result, 0, "lambda_init", N + 1, ldims, MD_SINGLETON_STRS(N + 1), true, &lambda);
		} else {

			result = nn_set_prox_op_F(result, 0, "lambda_init", operator_project_pos_real_create(N + 1, ldims));
			result = nn_set_in_type_F(result, 0, "lambda_init", IN_OPTIMIZE);
			result = nn_set_initializer_F(result, 0, "lambda_init", init_const_create(config->dc_lambda_init));

			result = nn_reshape_in_F(result, 0, "lambda_init", 1, ldims + N);
		}

		complex float one = 1;
		result = nn_set_input_const_F2(result, 0, "lambda_init_scale", N + 1, bdims, MD_SINGLETON_STRS(N + 1), true, &one);
	}

	result = nn_sort_inputs_by_list_F(result, i, in_names);
	for (int j = 0; j < N_in_names; j++)
		xfree(tnames[j]);

	return result;
}


static nn_t reconet_train_create(const struct reconet_s* config, unsigned int N, const long dims[N], const long idims[N], bool valid)
{
	auto train_op = reconet_create(config, N, dims, idims, valid ? STAT_TEST : STAT_TRAIN);

	if(config->normalize) {

		long sdims[N];
		md_select_dims(N, config->mri_config->batch_flags, sdims, dims);
		auto nn_norm_ref = nn_from_nlop_F(nlop_chain2_FF(nlop_zinv_create(N, sdims), 0, nlop_tenmul_create(N, idims, idims, sdims), 1));

		train_op = nn_chain2_FF(train_op, 0, "scale", nn_norm_ref, 1, NULL);
	} else {

		train_op = nn_combine_FF(nn_from_nlop_F(nlop_from_linop_F(linop_identity_create(N, idims))), train_op);
	}

	if (valid) {

		for (int i = 1; i < config->Nt; i++) {

			char out_name[30];
			sprintf(out_name, "out_%d", i);
			train_op = nn_del_out_F(train_op, 0, out_name);
		}

		auto loss = val_measure_create(config->valid_loss, N, idims);

		train_op = nn_chain2_FF(train_op, 1, NULL, loss, 0, NULL);
		train_op = nn_link_F(train_op, 0, NULL, 0, NULL);
	} else {

		auto loss = train_loss_multi_create(config->train_loss, N, idims, config->Nt, config->multi_loss);

		train_op = nn_chain2_FF(train_op, 1, NULL, loss, config->Nt - 1, NULL);

		for (int i = 1; i < config->Nt; i++) {

			char out_name[30];
			sprintf(out_name, "out_%d", i);
			train_op = nn_link_F(train_op, 0, out_name, 0, NULL);
		}

		train_op = nn_link_F(train_op, 0, NULL, 0, NULL);
	}

	if (valid)
		train_op = nn_del_out_bn_F(train_op);

	return train_op;
}

static nn_t reconet_valid_create(const struct reconet_s* config, unsigned int N, struct network_data_s* vf)
{
	load_network_data(vf);

	config->mri_config->pattern_flags = FFT_FLAGS;
	if (1 != vf->pdims[4])
		config->mri_config->pattern_flags |= MD_BIT(4);

	auto valid_loss = reconet_train_create(config, N, vf->kdims, vf->idims, true);

	valid_loss = nn_ignore_input_F(valid_loss, 0, NULL, N, vf->idims, true, vf->out);
	valid_loss = nn_ignore_input_F(valid_loss, 0, "kspace", N, vf->kdims, true, vf->kspace);
	valid_loss = nn_ignore_input_F(valid_loss, 0, "coil", N, vf->cdims, true, vf->coil);
	valid_loss = nn_ignore_input_F(valid_loss, 0, "pattern", N, vf->pdims, true, vf->pattern);

	free_network_data(vf);

	return valid_loss;
}


static nn_t reconet_apply_op_create(const struct reconet_s* config, unsigned int N, const long dims[N], const long idims[N])
{
	auto nn_apply = reconet_create(config, N, dims, idims, STAT_TEST);

	for (int i = 1; i < config->Nt; i++) {

		char out_name[30];
		sprintf(out_name, "out_%d", i);
		nn_apply = nn_del_out_F(nn_apply, 0, out_name);
	}

	if(config->normalize) {

		long sdims[N];
		md_select_dims(N, config->mri_config->batch_flags, sdims, dims);
		auto nn_norm_ref = nn_from_nlop_F(nlop_tenmul_create(N, idims, idims, sdims));

		nn_apply = nn_chain2_FF(nn_apply, 0, NULL, nn_norm_ref, 0, NULL);
		nn_apply = nn_link_F(nn_apply, 0, "scale", 0, NULL);
	}
	debug_printf(DP_INFO, "Apply RecoNet\n");
	nn_debug(DP_INFO, nn_apply);

	return nn_get_wo_weights_F(nn_apply, config->weights, false);
}



void train_reconet(	struct reconet_s* config, unsigned int N,
			const long idims[N], _Complex float* ref,
			const long kdims[N], _Complex float* kspace,
			const long cdims[N], const _Complex float* coil,
			const long pdims[N], const _Complex float* pattern,
			long Nb, struct network_data_s* valid_files)
{
	assert(5 == N);
	long Nt = kdims[4]; // number datasets

	long nkdims[N];
	long nidims[N];
	long ncdims[N];

	md_copy_dims(N, nkdims, kdims);
	md_copy_dims(N, nidims, idims);
	md_copy_dims(N, ncdims, cdims);

	nkdims[4] = Nb;
	nidims[4] = Nb;
	ncdims[4] = Nb;

	config->mri_config->pattern_flags = FFT_FLAGS;
	if (1 != pdims[4])
		config->mri_config->pattern_flags |= MD_BIT(4);

	auto nn_train = reconet_train_create(config, N, nkdims, nidims, false);

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
	const complex float* train_data[] = {ref, kspace, coil, pattern};
	const long* train_dims[] = {	nn_generic_domain(nn_train, 0, NULL)->dims,
					nn_generic_domain(nn_train, 0, "kspace")->dims,
					nn_generic_domain(nn_train, 0, "coil")->dims,
					nn_generic_domain(nn_train, 0, "pattern")->dims};

	assert(md_check_equal_dims(N, ncdims, train_dims[2], ~0));

	auto batch_generator = batch_gen_create_from_iter(config->train_conf, 4, N, train_dims, train_data, Nt, 0);

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

		auto nn_validation_loss = reconet_valid_create(config, N, valid_files);
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

static const struct nn_s* nn_apply_reconet = NULL;
static long nn_apply_batch_size = -1;

void apply_reconet(	const struct reconet_s* config, unsigned int N,
			const long idims[N], _Complex float* out,
			const long kdims[N], const _Complex float* kspace,
			const long cdims[N], const _Complex float* coil,
			const long pdims[N], const _Complex float* pattern)
{

	if ((NULL == nn_apply_reconet) || kdims[4] != nn_apply_batch_size) {

		if (config->gpu)
			move_gpu_nn_weights(config->weights);

		if (NULL != nn_apply_reconet)
			nn_free(nn_apply_reconet);
		nn_apply_batch_size = kdims[4];

		config->mri_config->pattern_flags = FFT_FLAGS;
		if (1 != pdims[4])
			config->mri_config->pattern_flags |= MD_BIT(4);

		nn_apply_reconet = reconet_apply_op_create(config, N, kdims, idims);

		static bool export = true;
		if (export && NULL != config->graph_file) {

			nn_export_graph(config->graph_file, nn_apply_reconet);
			export = false;
		}
	}

	complex float* out_tmp = md_alloc_sameplace(N, idims, CFL_SIZE, config->weights->tensors[0]);
	complex float* kspace_tmp = md_alloc_sameplace(N, kdims, CFL_SIZE, config->weights->tensors[0]);
	complex float* coil_tmp = md_alloc_sameplace(N, cdims, CFL_SIZE, config->weights->tensors[0]);
	complex float* pattern_tmp = md_alloc_sameplace(N, pdims, CFL_SIZE, config->weights->tensors[0]);

	md_copy(5, kdims, kspace_tmp, kspace, CFL_SIZE);
	md_copy(5, cdims, coil_tmp, coil, CFL_SIZE);
	md_copy(5, pdims, pattern_tmp, pattern, CFL_SIZE);

	complex float* args[4];

	args[0] = out_tmp;
	args[1] = kspace_tmp;
	args[2] = coil_tmp;
	args[3] = pattern_tmp;

	nlop_generic_apply_select_derivative_unchecked(nn_get_nlop(nn_apply_reconet), 4, (void**)args, 0, 0);

	md_copy(5, idims, out, out_tmp, CFL_SIZE);

	md_free(out_tmp);
	md_free(kspace_tmp);
	md_free(coil_tmp);
	md_free(pattern_tmp);
}

void apply_reconet_batchwise(	const struct reconet_s* config, unsigned int N,
				const long idims[N], _Complex float* out,
				const long kdims[N], const _Complex float* kspace,
				const long cdims[N], const _Complex float* coil,
				const long pdims[N], const _Complex float* pattern,
				long Nb)
{
	long Nt = kdims[4];
	while (0 < Nt) {

		long kdims1[N];
		long cdims1[N];
		long idims1[N];
		long pdims1[N];

		md_copy_dims(N, kdims1, kdims);
		md_copy_dims(N, cdims1, cdims);
		md_copy_dims(N, idims1, idims);
		md_copy_dims(N, pdims1, pdims);

		long Nb_tmp = MIN(Nt, Nb);

		kdims1[4] = Nb_tmp;
		cdims1[4] = Nb_tmp;
		idims1[4] = Nb_tmp;
		pdims1[4] = MIN(pdims1[4], Nb_tmp);

		apply_reconet(config, N, idims1, out, kdims1, kspace, cdims1, coil, pdims1, pattern);

		out += md_calc_size(N, idims1);
		kspace += md_calc_size(N, kdims1);
		coil += md_calc_size(N, cdims1);
		if (1 < pdims[4])
			pattern += md_calc_size(N, pdims1);

		Nt -= Nb_tmp;
	}

	if (NULL != nn_apply_reconet)
		nn_free(nn_apply_reconet);
	nn_apply_reconet = NULL;
	nn_apply_batch_size = -1;
}

void eval_reconet(	const struct reconet_s* config, unsigned int N,
			const long idims[N], const _Complex float* out,
			const long kdims[N], const _Complex float* kspace,
			const long cdims[N], const _Complex float* coil,
			const long pdims[N], const _Complex float* pattern,
			long Nb)
{
	complex float* tmp_out = md_alloc(N, idims, CFL_SIZE);

	auto loss = val_measure_create(config->valid_loss, N, idims);
	int NL = nn_get_nr_out_args(loss);
	complex float losses[NL];
	md_clear(1, MD_DIMS(NL), losses, CFL_SIZE);

	apply_reconet_batchwise( config, N, idims, tmp_out, kdims, kspace, cdims, coil, pdims, pattern, Nb);

	complex float* args[NL + 2];
	for (int i = 0; i < NL; i++)
		args[i] = losses + i;

	args[NL] = tmp_out;
	args[NL + 1] = (complex float*)out;

	nlop_generic_apply_select_derivative_unchecked(nn_get_nlop(loss), NL + 2, (void**)args, 0, 0);
	for (int i = 0; i < NL ; i++)
		debug_printf(DP_INFO, "%s: %e\n", nn_get_out_name_from_arg_index(loss, i, false), crealf(losses[i]));

	nn_free(loss);
	md_free(tmp_out);
}