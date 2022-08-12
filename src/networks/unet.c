/* Copyright 2021-2022. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 **/

#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>

#include "misc/debug.h"
#include "misc/opts.h"
#include "misc/types.h"
#include "misc/mri.h"

#include "num/multind.h"
#include "num/iovec.h"
#include "num/fft.h"
#include "num/flpmath.h"
#include "num/ops.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/someops.h"
#include "nlops/stack.h"
#include "nlops/tenmul.h"

#include "nn/nn.h"
#include "nn/activation_nn.h"
#include "nn/chain.h"
#include "nn/layers_nn.h"
#include "nn/nn_ops.h"
#include "nn/activation.h"
#include "nn/layers.h"
#include "nn/weights.h"
#include "nn/init.h"

#include "networks/cnn.h"
#include "unet.h"


DEF_TYPEID(network_unet_s);

struct network_unet_s network_unet_default_reco = {

	.INTERFACE.TYPEID = &TYPEID2(network_unet_s),

	.INTERFACE.create = network_unet_create,

	.INTERFACE.low_mem = false,

	.INTERFACE.norm = NORM_NONE,
	.INTERFACE.norm_batch_flag = MD_BIT(4),

	.INTERFACE.debug = false,
	.INTERFACE.bart_to_channel_first = true,

	.INTERFACE.prefix = NULL,

	.N = 5,
	.kdims = {[0 ... DIMS -1] = 0},
	.dilations = {[0 ... DIMS -1] = 1},

	.Nf = 32,
	.Kx = 3,
	.Ky = 3,
	.Kz = 1,
	.Ng = 1,

	.conv_flag = 14,
	.channel_flag = 1,
	.group_flag = 0,
	.batch_flag = 16,

	.N_level = 4,

	.channel_factor = 1.,	//number channels on lower level
	.reduce_factor = 2.,	//reduce resolution of lower level

	.Nl_before = 2,
	.Nl_after = 2,
	.Nl_lowest = 4,

	.real_constraint = false,

	.init_real = false,
	.init_zeros_residual = false,

	.use_bn = false,
	.use_bias = true,

	.activation = ACT_RELU,
	.activation_output = ACT_LIN,

	.padding = PAD_SAME,

	.ds_method = UNET_DS_STRIDED_CONV,
	.us_method = UNET_US_STRIDED_CONV,

	.combine_method = UNET_COMBINE_ADD,

	.INTERFACE.residual = true,

	.adjoint = false,
};

struct network_unet_s network_unet_default_segm = {

	.INTERFACE.TYPEID = &TYPEID2(network_unet_s),
	.INTERFACE.create = network_unet_create,
	.INTERFACE.low_mem = false,

	.INTERFACE.debug = false,
	.INTERFACE.bart_to_channel_first = false,

	.N = 5,

	.kdims = {[0 ... DIMS -1] = 0},
	.dilations = {[0 ... DIMS -1] = 1},

	.Nf = 64,
	.Kx = 3,
	.Ky = 3,
	.Kz = 1,
	.Ng = 1,

	.conv_flag = 14,
	.channel_flag = 1,
	.group_flag = 0,
	.batch_flag = 16,

	.N_level = 6,

	.channel_factor = 1.,	//number channels on lower level
	.reduce_factor = 2.,	//reduce resolution of lower level

	.Nl_before = 1,
	.Nl_after = 1,
	.Nl_lowest = 2,

	.real_constraint = true,

	.use_bn = false,
	.use_bias = true,

	.activation = ACT_RELU,
	.activation_output = ACT_SOFTMAX,

	.padding = PAD_SAME,

	.ds_method = UNET_DS_STRIDED_CONV,
	.us_method = UNET_US_STRIDED_CONV,

	.combine_method = UNET_COMBINE_ADD,

	.INTERFACE.residual = false,

	.adjoint = false,

	.init_real = false,
	.init_zeros_residual = false,
};

static nn_t unet_sort_names(nn_t network, struct network_unet_s* unet)
{
	const char* prefixes[] = {"_init", "_before", "_down", "", "_up", "_after", "_last"};
	const char* weights[] = {"conv", "corr", "conv_adj", "corr_adj", "bn", "bn_gamma", "bias"};

	const char* names[unet->N_level][ARRAY_SIZE(prefixes)][ARRAY_SIZE(weights)];

	for (int i = 0; i < unet->N_level; i++) {
		for (unsigned int j = 0; j < ARRAY_SIZE(prefixes); j++) {
			for (unsigned int k = 0; k < ARRAY_SIZE(weights); k++) {

				names[i][j][k] = ptr_printf("level_%d%s_%s", i, prefixes[j], weights[k]);

				if (unet->real_constraint && nn_is_name_in_in_args(network, names[i][j][k]))
					network = nn_real_input_F(network, 0, names[i][j][k]);
			}
		}
	}

	int N = unet->N_level * ARRAY_SIZE(prefixes) * ARRAY_SIZE(weights);

	network = nn_sort_inputs_by_list_F(network, N, &(names[0][0][0]));
	network = nn_sort_outputs_by_list_F(network, N, &(names[0][0][0]));

	for (int i = 0; i < N; i++)
		xfree((&(names[0][0][0]))[i]);

	return network;
}

struct nn_conv_block_s {

	unsigned long conv_flag;
	unsigned long channel_flag;
	unsigned long group_flag;

	bool adjoint;
	bool conv;

	enum PADDING padding;
	enum ACTIVATION activation;

	bool use_bias;
	bool use_bn;
	bool use_bn_gamma;

	bool stack;
	const char* name_prefix;

	bool init_real;
	bool init_zero;
};

static nn_t nn_unet_append_conv_block(	nn_t network, int o, const char* oname,
					struct nn_conv_block_s* config,
					unsigned int N, const long kdims[N], const long strides[N], const long dilations[N],
					enum NETWORK_STATUS status)
{
	if (config->init_zero && config->use_bn)
		assert(config->use_bn_gamma);

	const char* name_prefix = (NULL == config->name_prefix) ? "" : config->name_prefix;
	const char* name_postfix = config->stack ? "" : "_";

	bool stack;
	const char* name_working;

	const char* layer_name = ptr_printf("%s%s", config->conv ? "conv" : "corr", (config->adjoint) ? "_adj" : "");
	const char* name = ptr_printf("%s%s%s", name_prefix, layer_name, name_postfix);
	const char* name_tmp = ptr_printf("%s%s_tmp", name_prefix, layer_name);

	xfree(layer_name);

	stack = config->stack && nn_is_name_in_in_args(network, name);
	name_working = stack ? name_tmp : name;

	unsigned long in_flag = (config->adjoint ? out_flag_conv_generic : in_flag_conv_generic)(N, config->conv_flag, config->channel_flag, config->group_flag);

	const struct initializer_s* init_conv = NULL;

	if (config->init_zero && !config->use_bn)
		init_conv = init_const_create(0);
	else
		init_conv = init_kaiming_create(in_flag, config->init_real, false, 0);

	if (config->adjoint)
		network = nn_append_transposed_convcorr_layer_generic(network, o, oname, name_working, config->conv_flag, config->channel_flag, config->group_flag, N, kdims, strides, dilations, config->conv, config->padding, config->adjoint, init_conv);
	else
		network = nn_append_convcorr_layer_generic(network, o, oname, name_working, config->conv_flag, config->channel_flag, config->group_flag, N, kdims, strides, dilations, config->conv, config->padding, init_conv);

	if (config->stack) {

		network = nn_append_singleton_dim_in_F(network, 0, name_working);

		if (stack)
			network = nn_stack_inputs_F(network, 0, name, 0, name_tmp, -1);
	}

	xfree(name);
	xfree(name_tmp);

	if (config->use_bn) {

		const char* name = ptr_printf("%sbn%s", name_prefix, name_postfix);
		const char* name_tmp = ptr_printf("%sbn_tmp", name_prefix);

		stack = config->stack && nn_is_name_in_in_args(network, name);
		name_working = stack ? name_tmp : name;

		network = nn_append_batchnorm_layer(network, o, oname, name_working, ~(config->channel_flag | config->group_flag), status, NULL);

		if (config->stack) {

			network = nn_append_singleton_dim_in_F(network, 0, name_working);
			network = nn_append_singleton_dim_out_F(network, 0, name_working);

			if (stack) {

				network = nn_stack_inputs_F(network, 0, name, 0, name_tmp, -1);
				network = nn_stack_outputs_F(network, 0, name, 0, name_tmp, -1);
			}
		}

		xfree(name);
		xfree(name_tmp);

		if (config->use_bn_gamma) {

			const char* name = ptr_printf("%sbn_gamma%s", name_prefix, name_postfix);
			const char* name_tmp = ptr_printf("%sbn_gamma_tmp", name_prefix);

			stack = config->stack && nn_is_name_in_in_args(network, name);
			name_working = stack ? name_tmp : name;

			//append gamma for batchnorm
			auto iov = nn_generic_codomain(network, 0, NULL);
			long gdims [iov->N];
			md_select_dims(iov->N, config->channel_flag | config->group_flag, gdims, iov->dims);

			auto nn_scale_gamma = nn_from_nlop_F(nlop_tenmul_create(iov->N, iov->dims, iov->dims, gdims));
			network = nn_chain2_swap_FF(network, o, oname, nn_scale_gamma, 0, NULL);
			network = nn_set_input_name_F(network, -1, name_working);
			network = nn_set_initializer_F(network, 0, name_working, init_const_create(config->init_zero ? 0 : 1));
			network = nn_set_in_type_F(network, 0, name_working, IN_OPTIMIZE);
			network = nn_set_dup_F(network, 0, name_working, false);

			if (NULL == oname)
				network = nn_shift_output_F(network, o, NULL, 0, NULL);
			else
				network = nn_set_output_name_F(network, 0, oname);

			if(config->stack)
				network = nn_append_singleton_dim_in_F(network, 0, name_working);

			if (stack)
				network = nn_stack_inputs_F(network, 0, name, 0, name_tmp, -1);

			xfree(name);
			xfree(name_tmp);
		}
	}

	if (config->use_bias) {

		const char* name = ptr_printf("%sbias%s", name_prefix, name_postfix);
		const char* name_tmp = ptr_printf("%sbias_tmp", name_prefix);

		stack = config->stack && nn_is_name_in_in_args(network, name);
		name_working = stack ? name_tmp : name;

		network = nn_append_activation_bias(network, o, oname, name_working, config->activation, (config->channel_flag | config->group_flag));
		network = nn_append_singleton_dim_in_F(network, 0, name_working);

		if(config->stack)
			network = nn_append_singleton_dim_in_F(network, 0, name_working);

		if (stack)
			network = nn_stack_inputs_F(network, 0, name, 0, name_tmp, -1);

		xfree(name);
		xfree(name_tmp);

	} else {

		network = nn_append_activation(network, o, oname, config->activation);
	}

	return network;
}

static bool get_init_zero(struct network_unet_s* unet, unsigned int level, bool last_layer)
{
	if (!last_layer)
		return false;

	if (!unet->init_zeros_residual)
		return false;

	if (0 == level)
		return unet->INTERFACE.residual;

	return true;
}

static nn_t unet_append_conv_block(	nn_t network, struct network_unet_s* unet,
					unsigned int N, const long kdims[N], enum ACTIVATION activation,
					unsigned int level, bool after, bool last_layer,
					const char* name_prefix, enum NETWORK_STATUS status)
{
	struct nn_conv_block_s config;

	if (last_layer)
		assert(after);

	config.conv_flag = unet->conv_flag;
	config.channel_flag = unet->channel_flag;
	config.group_flag = unet->group_flag;

	config.adjoint = after && unet->adjoint;
	config.conv = false;

	config.padding = unet->padding;
	config.activation = activation;

	config.use_bias = unet->use_bias;
	config.use_bn = unet->use_bn;
	config.use_bn_gamma = unet->use_bn && last_layer;

	config.stack = true;
	config.name_prefix = name_prefix;

	config.init_real = unet->init_real || unet->real_constraint;
	config.init_zero = get_init_zero(unet, level, last_layer && after);

	return nn_unet_append_conv_block(network, 0, NULL, &config, N, kdims, MD_SINGLETON_DIMS(N), unet->dilations, status);
}


static nn_t unet_sample_fft_create(struct network_unet_s* unet, unsigned int N, const long dims[N], long down_dims[N], bool up, enum NETWORK_STATUS status)
{
	UNUSED(status);

	for (unsigned int i = 0; i < N; i++)
		down_dims[i] = MD_IS_SET(unet->conv_flag, i) ? MAX(1, round(dims[i] / unet->reduce_factor)) : dims[i];

	const struct linop_s* linop_result = linop_fftc_create(N, dims, unet->conv_flag);

	linop_result = linop_chain_FF(linop_result, linop_resize_center_create(N, down_dims, dims));
	linop_result = linop_chain_FF(linop_result, linop_ifftc_create(N, down_dims, unet->conv_flag));

	if (unet->real_constraint) {

		linop_result = linop_chain_FF(linop_result, linop_zreal_create(N, down_dims));
		linop_result = linop_chain_FF(linop_zreal_create(N, dims), linop_result);
	}

	if (up) {

		auto tmp = linop_get_adjoint(linop_result);
		linop_free(linop_result);
		linop_result = tmp;
	}

	return nn_from_nlop_F(nlop_from_linop_F(linop_result));
}

static nn_t unet_sample_conv_strided_create(struct network_unet_s* unet, unsigned int N, const long dims[N], long down_dims[N], bool up, unsigned int level, enum NETWORK_STATUS status)
{
	long kdims[N];
	long strides[N];
	long dilations[N];

	md_singleton_dims(N, kdims);
	md_singleton_dims(N, strides);
	md_singleton_dims(N, dilations);

	md_copy_dims(N, down_dims, dims);

	if (unet->reduce_factor != roundf(unet->reduce_factor))
		error("Convolution can only be used for integer downsampling");

	long stride = lroundf(unet->reduce_factor);

	for (unsigned int i = 0; i < N; i++) {

		if (MD_IS_SET(unet->conv_flag, i) && (1 < dims[i])) {

			assert(0 == dims[i] % stride);
			down_dims[i] = dims[i] / stride;
			strides[i] = stride;
			kdims[i] = 3;
		}

		if (MD_IS_SET(unet->channel_flag, i)) {

			kdims[i] = up ? dims[i] : dims[i] * unet->channel_factor;
			down_dims[i] = dims[i] * unet->channel_factor;
		}

		if (MD_IS_SET(unet->group_flag, i)) {

			kdims[i] = dims[i];
		}
	}

	struct nn_conv_block_s config;

	config.conv_flag = unet->conv_flag;
	config.channel_flag = unet->channel_flag;
	config.group_flag = unet->group_flag;

	config.adjoint = up;
	config.conv = false;

	config.padding = PAD_SAME;
	config.activation = ACT_LIN;

	config.use_bias = unet->use_bias;
	config.use_bn = unet->use_bn;
	config.use_bn_gamma = unet->use_bn && up;

	config.stack = true;
	config.name_prefix = ptr_printf("level_%u_%s_", level, up ? "up" : "down");

	config.init_real = unet->init_real || unet->real_constraint;
	config.init_zero = get_init_zero(unet, level, up);

	auto result = nn_from_nlop_F(nlop_from_linop_F(linop_identity_create(N, up ? down_dims : dims)));
	result = nn_unet_append_conv_block(result, 0, NULL, &config, N, kdims, strides, dilations, status);

	xfree(config.name_prefix);

	return result;
}

static nn_t unet_downsample_create(struct network_unet_s* unet, unsigned int N, const long dims[N], long down_dims[N], unsigned int level, enum NETWORK_STATUS status)
{
	switch (unet->ds_method) {

	case UNET_DS_FFT:
		return unet_sample_fft_create(unet, N, dims, down_dims, false, status);

	case UNET_DS_STRIDED_CONV:
		return unet_sample_conv_strided_create(unet, N, dims, down_dims, false, level, status);
	}

	assert(0);
}

static nn_t unet_upsample_create(struct network_unet_s* unet, unsigned int N, const long dims[N], long down_dims[N], unsigned int level, enum NETWORK_STATUS status)
{
	switch (unet->us_method) {

	case UNET_US_FFT:
		return unet_sample_fft_create(unet, N, dims, down_dims, true, status);

	case UNET_US_STRIDED_CONV:
		return unet_sample_conv_strided_create(unet, N, dims, down_dims, true, level, status);
	}

	assert(0);
}

static void unet_get_kdims(const struct network_unet_s* config, unsigned int N, long kdims[N], unsigned int level)
{
	if (0 != md_calc_size(config->N, config->kdims)) {

		md_copy_dims(N, kdims, config->kdims);

	} else {

		assert(1 == bitcount(config->channel_flag));
		assert(3 >= bitcount(config->conv_flag));

		long tdims[3] = {config->Kx, config->Ky, config->Kz};
		long* tdim = tdims;

		for (unsigned int i = 0; i < N; i++) {

			kdims[i] = 1;

			if (MD_IS_SET(config->conv_flag, i)) {

				kdims[i] = *tdim;
				tdim += 1;
			}

			if (MD_IS_SET(config->channel_flag, i))
				kdims[i] = config->Nf;

			if (MD_IS_SET(config->group_flag, i))
				kdims[i] = config->Ng;
		}
	}

	for (unsigned int i = 0; i < N; i++) {

		if (!MD_IS_SET(config->channel_flag, i))
			continue;

		for (unsigned int j = 0; j < level; j++)
			kdims[i] = lroundf(kdims[i] * config->channel_factor);
	}
}

static nn_t unet_lowest_level_create(struct network_unet_s* unet, unsigned int N, const long odims[N], const long idims[N], unsigned int level, enum NETWORK_STATUS status)
{
	assert(0 < level);
	long kdims[N];
	unet_get_kdims(unet, N, kdims, level);

	long Nl = unet->Nl_lowest;

	long okdims[N];
	md_copy_dims(N, okdims, kdims);

	for(unsigned int i = 0; i < N; i++) {

		if (MD_IS_SET(unet->channel_flag, i) || MD_IS_SET(unet->group_flag, i))
			okdims[i] = odims[i];
	}

	//if dim for group index are not equal in the last layer, we make it a channel dim
	unsigned long ichannel_flag = unet->channel_flag;
	unsigned long igroup_flag = unet->group_flag;
	unsigned long ochannel_flag = unet->channel_flag;
	unsigned long ogroup_flag = unet->group_flag;

	//we try to stack as many weights as possible
	//if the shape of the first conv block equals the following (init_same == true), it is stacked
	//if the shape of the last conv block equals the ones befor (last_same == true), it is stacked
	bool init_same = true;
	bool last_same = true;

	for (unsigned int i = 0; i < N; i++){

		if (MD_IS_SET(igroup_flag, i) && (kdims[i] != idims[i])) {

			igroup_flag = MD_CLEAR(igroup_flag, i);
			ichannel_flag = MD_SET(ichannel_flag, i);
			init_same = false;
			error ("Groups in unet are currently not supported");
		}

		if (MD_IS_SET(ichannel_flag, i) && (kdims[i] != idims[i]))
			init_same = false;

		if (MD_IS_SET(ogroup_flag, i) && (okdims[i] != odims[i])) {

			ogroup_flag = MD_CLEAR(ogroup_flag, i);
			ochannel_flag = MD_SET(ochannel_flag, i);
			error ("Groups in unet are currently not supported");
		}
	}

	last_same = last_same && md_check_equal_dims(N, kdims, okdims, ~0);
	last_same = last_same && (ogroup_flag == unet->group_flag) && (ochannel_flag == unet->channel_flag);

	Nl -= init_same ? 0 : 1;
	Nl -= last_same ? 0 : 1;
	assert(0 <= Nl);

	auto result = nn_from_nlop_F(nlop_from_linop_F(linop_identity_create(N, idims)));

	if (!init_same) {

		int prefix_len = snprintf(NULL, 0, "level_%u_init_", level);
		char prefix[prefix_len + 1];
		sprintf(prefix, "level_%u_init_", level);

		result = unet_append_conv_block(result, unet, N, kdims, unet->activation, level, false, false, prefix, status);
	}

	for (int i = 0; i < Nl; i++){

		int prefix_len = snprintf(NULL, 0, "level_%u_", level);
		char prefix[prefix_len + 1];
		sprintf(prefix, "level_%u_", level);

		bool last_layer = (unet->us_method != UNET_US_STRIDED_CONV) && last_same && (i + 1 == Nl);
		result = unet_append_conv_block(result, unet, N, kdims, last_layer ? ACT_LIN : unet->activation, level, 2 * (init_same ? i : i + 1) >= unet->Nl_lowest, last_layer, prefix, status);
	}

	if (!last_same) {

		int prefix_len = snprintf(NULL, 0, "level_%u_last_", level);

		char prefix[prefix_len + 1];
		sprintf(prefix, "level_%u_last_", level);

		bool last_layer = (unet->us_method != UNET_US_STRIDED_CONV);

		result = unet_append_conv_block(result, unet, N, okdims, last_layer ? ACT_LIN : unet->activation, level, true, last_layer, prefix, status);
	}


	debug_printf(DP_DEBUG1, "U-Net lowest level (%u) created:\n", level);
	nn_debug(DP_DEBUG1, result);

	return result;
}


static nn_t unet_level_create(struct network_unet_s* unet, unsigned int N, const long odims[N], const long idims[N], unsigned int level, enum NETWORK_STATUS status)
{
	if (level + 1 == unet->N_level)
		return unet_lowest_level_create(unet, N, odims, idims, level, status);

	long kdims[N];
	unet_get_kdims(unet, N, kdims, level);


	long okdims[N];
	md_copy_dims(N, okdims, kdims);

	for (unsigned int i = 0; i < N; i++) {

		if (MD_IS_SET(unet->channel_flag, i) || MD_IS_SET(unet->group_flag, i))
			okdims[i] = odims[i];
	}

	//if dim for group index are not equal in the last layer, we make it a channel dim
	unsigned long ichannel_flag = unet->channel_flag;
	unsigned long igroup_flag = unet->group_flag;
	unsigned long ochannel_flag = unet->channel_flag;
	unsigned long ogroup_flag = unet->group_flag;

	//we try to stack as many weights as possible
	//if the shape of the first conv block equals the following (init_same == true), it is stacked
	//if the shape of the last conv block equals the ones befor (last_same == true), it is stacked
	bool init_same = true;
	bool last_same = true;

	for (unsigned int i = 0; i < N; i++){

		if (MD_IS_SET(igroup_flag, i) && (kdims[i] != idims[i])) {

			igroup_flag = MD_CLEAR(igroup_flag, i);
			ichannel_flag = MD_SET(ichannel_flag, i);
			init_same = false;
			error ("Groups in unet are currently not supported");
		}

		if (MD_IS_SET(ichannel_flag, i) && (kdims[i] != idims[i]))
			init_same = false;

		if (MD_IS_SET(ogroup_flag, i) && (okdims[i] != odims[i])) {

			ogroup_flag = MD_CLEAR(ogroup_flag, i);
			ochannel_flag = MD_SET(ochannel_flag, i);
			error ("Groups in unet are currently not supported");
		}
	}

	last_same = last_same && md_check_equal_dims(N, kdims, okdims, ~0);
	last_same = last_same && (ogroup_flag == unet->group_flag) && (ochannel_flag == unet->channel_flag);

	long Nl_before = init_same ? unet->Nl_before : unet->Nl_before - 1;
	long Nl_after = last_same ? unet->Nl_after : unet->Nl_after - 1;


	// create first block of convolution
	auto result = nn_from_nlop_F(nlop_from_linop_F(linop_identity_create(N, idims)));

	if (!init_same) {

		int prefix_len = snprintf(NULL, 0, "level_%u_init_", level);
		char prefix[prefix_len + 1];
		sprintf(prefix, "level_%u_init_", level);

		result = unet_append_conv_block(result, unet, N, kdims, unet->activation, level, false, false, prefix, status);
	}

	for (int i = 0; i < Nl_before; i++){

		int prefix_len = snprintf(NULL, 0, "level_%u_before_", level);
		char prefix[prefix_len + 1];
		sprintf(prefix, "level_%u_before_", level);

		result = unet_append_conv_block(result, unet, N, kdims, unet->activation, level, false, false, prefix, status);
	}

	//create lower level unet
	long down_dims_in[N];
	auto nn_ds = unet_downsample_create(unet, N, nn_generic_codomain(result, 0, NULL)->dims, down_dims_in, level, status);
	long down_dims_out[N];
	auto nn_us = unet_upsample_create(unet, N, nn_generic_codomain(result, 0, NULL)->dims, down_dims_out, level, status);

	//FIXME: currently, a level is not allowed to change spatial dimensions (valid convolution)
	//While the upsampling opperator should define the channel dimensions, the lower level should define the spatial dims

	auto lower_level = unet_level_create(unet, N, down_dims_out, down_dims_in, level + 1, status);
	lower_level = nn_chain2_swap_FF(nn_ds, 0, NULL, lower_level, 0, NULL);
	lower_level = nn_chain2_swap_FF(lower_level, 0, NULL, nn_us, 0, NULL);

	long tdims[N];
	md_copy_dims(N, tdims, nn_generic_codomain(result, 0, NULL)->dims);

	const struct nlop_s* nlop_join = NULL;

	if (UNET_COMBINE_ATTENTION_SIGMOID == unet->combine_method)
		nlop_join = nlop_tenmul_create(N, tdims, tdims, tdims);

	if (UNET_COMBINE_ADD == unet->combine_method)
		nlop_join = nlop_zaxpbz_create(N, nn_generic_codomain(result, 0, NULL)->dims, 1, 1);

	lower_level = nn_chain2_swap_FF(lower_level, 0, NULL, nn_from_nlop_F(nlop_join), 0, NULL);
	lower_level = nn_dup_F(lower_level, 0, NULL, 1, NULL);

	result = nn_chain2_swap_FF(result, 0, NULL, lower_level, 0, NULL);

	enum ACTIVATION activation_last_layer = unet->activation_output;

	if (0 != level) {

		activation_last_layer = ACT_LIN;

		if (UNET_COMBINE_ATTENTION_SIGMOID == unet->combine_method)
			activation_last_layer = ACT_SIGMOID;
	}

	//create conv blocks after lower level
	for (int i = 0; i < Nl_after; i++){

		int prefix_len = snprintf(NULL, 0, "level_%u_after_", level);
		char prefix[prefix_len + 1];
		sprintf(prefix, "level_%u_after_", level);

		bool last_layer = ((unet->us_method != UNET_US_STRIDED_CONV) || (0 == level)) && last_same && (i + 1 == Nl_after);
		result = unet_append_conv_block(result, unet, N, kdims, last_layer ? activation_last_layer : unet->activation, level, true, last_layer, prefix, status);
	}

	if (!last_same) {

		int prefix_len = snprintf(NULL, 0, "level_%u_last_", level);
		char prefix[prefix_len + 1];
		sprintf(prefix, "level_%u_last_", level);

		bool last_layer = ((unet->us_method != UNET_US_STRIDED_CONV) || (0 == level));
		result = unet_append_conv_block(result, unet, N, okdims, last_layer ? activation_last_layer : unet->activation, level, true, last_layer, prefix, status);
	}

	debug_printf(DP_DEBUG1, "U-Net level %u created:\n", level);
	nn_debug(DP_DEBUG1, result);

	return result;
}

nn_t network_unet_create(const struct network_s* _unet, unsigned int NO, const long odims[NO], unsigned int NI, const long idims[NI], enum NETWORK_STATUS status)
{
	assert(NO == NI);
	unsigned int N = NO;

	auto unet = CAST_DOWN(network_unet_s, _unet);

	if (PAD_VALID == unet->padding) {

		bool same_dims = true;
		same_dims = same_dims && (0 == unet->Nl_lowest % 2);
		same_dims = same_dims && (unet->Nl_after == unet->Nl_before);
		same_dims = same_dims && (unet->adjoint);

		if (!same_dims)
			error("U-Net: Spatial dims are not allowed to change in a level!");
	}

	auto result = unet_level_create(unet, N, odims, idims, 0, status);

	result = unet_sort_names(result, unet);

	debug_printf(DP_DEBUG1, "U-Net created:\n");
	nn_debug(DP_DEBUG1, result);

	return result;
}
