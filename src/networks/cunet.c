/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author: Moritz Blumenthal
 */


#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>

#include "misc/misc.h"

#include "num/multind.h"
#include "num/iovec.h"
#include "num/fft.h"
#include "num/flpmath.h"
#include "num/ops.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/sum.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/someops.h"
#include "nlops/stack.h"
#include "nlops/tenmul.h"
#include "nlops/const.h"
#include "nlops/zexp.h"

#include "nn/nn.h"
#include "nn/activation_nn.h"
#include "nn/chain.h"
#include "nn/layers_nn.h"
#include "nn/activation.h"
#include "nn/weights.h"
#include "nn/init.h"
#include "nn/batchnorm.h"


#include "cunet.h"

struct nn_cunet_conf_s cunet_defaults = {

	.levels = 5,

	.padding = PAD_SAME,
	.activation = ACT_RELU,

	.num_filters = 48,
	.cunits = 256,
	.conditional = true,

	.ksizes = { 3, 3, 3 },
	.dilations = { 2, 2, 2 },
	.strides = { 2, 2, 2 },
};

static nn_t instance_norm_plus_create(const long dims[5], const char* prefix)
{
	unsigned long flags = MD_BIT(1) | MD_BIT(2) | MD_BIT(3);

	long sdims[5];
	long wdims[5];
	md_select_dims(5, ~flags, sdims, dims);
	md_select_dims(5, MD_BIT(0), wdims, dims);

	const struct nlop_s* ret = nlop_norm_avg_create(5, dims, flags);
	ret = nlop_append_FF(ret, 0, nlop_del_out_F(nlop_norm_std_create(5, dims, flags, 1.e-7), 1));

	const struct nlop_s* adjusted_mean = nlop_del_out_F(nlop_norm_avg_create(5, sdims, MD_BIT(0)), 1);
	adjusted_mean = nlop_chain_FF(adjusted_mean, nlop_del_out_F(nlop_norm_std_create(5, sdims, MD_BIT(0), 1.e-7), 1));
	adjusted_mean = nlop_prepend_FF(adjusted_mean, nlop_tenmul_create(5, sdims, sdims, wdims), 0);
	adjusted_mean = nlop_reshape_in_F(adjusted_mean, 1, 1, wdims);

	ret = nlop_chain2_swap_FF(ret, 1, adjusted_mean, 0);
	ret = nlop_chain2_FF(ret, 0, nlop_zaxpbz2_create(5, dims, ~0UL, 1., ~flags, 1.), 1);
	ret = nlop_link_F(ret, 1, 0);

	const char* aname = ptr_printf("%s_alpha", prefix);

	nn_t nn_ret = nn_from_nlop_F(ret);
	nn_ret = nn_set_initializer_F(nn_ret, 1, NULL, init_const_create(1.));
	nn_ret = nn_set_in_type_F(nn_ret, 1, NULL, IN_OPTIMIZE);
	nn_ret = nn_set_input_name_F(nn_ret, 1, aname);

	xfree(aname);

	return nn_ret;
}


static nn_t cond_res_block_create(struct nn_cunet_conf_s* conf, const long dims[5], const long dilations[3], const char* prefix, int index)
{
	const char* bname1 = ptr_printf("%s_cres%d_bias1", prefix, index);
	const char* bname2 = ptr_printf("%s_cres%d_bias2", prefix, index);

	const char* cname1 = ptr_printf("%s_cres%d_conv1", prefix, index);
	const char* cname2 = ptr_printf("%s_cres%d_conv2", prefix, index);

	const char* nname1 = ptr_printf("%s_cres%d_norm1", prefix, index);
	const char* nname2 = ptr_printf("%s_cres%d_norm2", prefix, index);

	const char* einame = ptr_printf("emb");
	const char* ecname = ptr_printf("%s_cres%d_cond", prefix, index);

	auto network = nn_from_linop_F(linop_identity_create(5, dims));

	network = nn_chain2_FF(network, 0, NULL, instance_norm_plus_create(dims, nname1), 0, NULL);
	network = nn_append_activation(network, 0, NULL, conf->activation, MD_BIT(0));
	network = nn_append_convcorr_layer(network, 0, NULL, cname1, dims[0], conf->ksizes, false, conf->padding, true, NULL, dilations, NULL);
	network = nn_append_activation_bias(network, 0, NULL, bname1, ACT_LIN, MD_BIT(0));

	network = nn_chain2_FF(network, 0, NULL, instance_norm_plus_create(dims, nname2), 0, NULL);

	if (0 < conf->cunits) {

		long cdims[2] = { conf->cunits, dims[4] };

		auto cond = nn_from_linop_F(linop_identity_create(2, cdims));
		cond = nn_append_dense_layer(cond, 0, NULL, ecname, dims[0], NULL);
		cond = nn_set_input_name_F(cond, 0, einame);

		cdims[0] = dims[0];
		auto nlop_add = nlop_zaxpbz2_create(5, dims, ~0UL, 1., MD_BIT(0) | MD_BIT(4), 1.);
		nlop_add = nlop_reshape_in_F(nlop_add, 1, 2, cdims);

		cond = nn_chain2_FF(cond, 0, NULL, nn_from_nlop_F(nlop_add), 1, NULL);
		network = nn_chain2_FF(network, 0, NULL, cond, 0, NULL);
	}

	network = nn_append_activation(network, 0, NULL, conf->activation, MD_BIT(0));
	network = nn_append_convcorr_layer(network, 0, NULL, cname2, dims[0], conf->ksizes, false, conf->padding, true, NULL, dilations, NULL);
	network = nn_append_activation_bias(network, 0, NULL, bname2, ACT_LIN, MD_BIT(0));

	network = nn_chain2_FF(network, 0, NULL, nn_from_nlop_F(nlop_zaxpbz_create(5, dims, 1., 1.)), 0, NULL);
	network = nn_dup_F(network, 0, NULL, 1, NULL);

	//network = nn_checkpoint_F(network, false, false);

	xfree(bname1);
	xfree(bname2);
	xfree(cname1);
	xfree(cname2);
	xfree(nname1);
	xfree(nname2);
	xfree(einame);
	xfree(ecname);

	return network;
}



static nn_t cunet_level_create(struct nn_cunet_conf_s* conf, int level, const long dims[5])
{
	const char* prefix = ptr_printf("level_%d", level);

	nn_t network = NULL;

	if (level == conf->levels) {

		network = cond_res_block_create(conf, dims, NULL, prefix, 0);
		network = nn_mark_dup_if_exists_F(network, "emb");

		network = nn_chain2_FF(network, 0, NULL, cond_res_block_create(conf, dims, conf->dilations, prefix, 1), 0, NULL);
		network = nn_stack_dup_by_name_F(network);

		network = nn_mark_dup_if_exists_F(network, "emb");
		network = nn_chain2_FF(network, 0, NULL, cond_res_block_create(conf, dims, NULL, prefix, 2), 0, NULL);
		network = nn_stack_dup_by_name_F(network);
	} else {

		long low_channel = MIN(256, 2 * dims[0]);
		const char* dname = ptr_printf("%s_down_conv", prefix);
		const char* uname = ptr_printf("%s_up_conv", prefix);

		auto low = nn_from_linop_F(linop_identity_create(5, dims));
		low = nn_append_activation(low, 0, NULL, conf->activation, MD_BIT(0));
		low = nn_append_convcorr_layer(low, 0, NULL, dname, low_channel, conf->strides, false, conf->padding, true, conf->strides, NULL, NULL);

		auto iov = nn_generic_codomain(low, 0, NULL);
		low = nn_chain2_FF(low, 0, NULL, cunet_level_create(conf, level + 1, iov->dims), 0, NULL);
		low = nn_append_transposed_convcorr_layer(low, 0, NULL, uname, dims[0], conf->strides, false, false, conf->padding, true, conf->strides, NULL, NULL);

		iov = nn_generic_codomain(low, 0, NULL);
		low = nn_chain2_FF(low, 0, NULL, nn_from_nlop_F(nlop_zaxpbz_create(5, iov->dims, 1, 1)), 0, NULL);
		low = nn_dup_F(low, 0, NULL, 1, NULL);
		low = nn_chain2_FF(low, 0, NULL, instance_norm_plus_create(dims, prefix), 0, NULL);

		network = cond_res_block_create(conf, dims, NULL, prefix, 0);

		network = nn_mark_dup_if_exists_F(network, "emb");
		network = nn_chain2_FF(network, 0, NULL, low, 0, NULL);
		network = nn_stack_dup_by_name_F(network);

		network = nn_mark_dup_if_exists_F(network, "emb");
		network = nn_chain2_FF(network, 0, NULL, cond_res_block_create(conf, dims, NULL, prefix, 1), 0, NULL);
		network = nn_stack_dup_by_name_F(network);

		xfree(dname);
		xfree(uname);
	}

	xfree(prefix);

	return network;
}

static nn_t cunet_emb_create(struct nn_cunet_conf_s* conf, long Nb)
{
	long edims[2] = { 1, Nb };

	auto nn_emb = nn_from_linop_F(linop_zreal_create(2, edims));
	nn_emb = nn_reshape_in_F(nn_emb, 0, NULL, 1, edims + 1);
	nn_emb = nn_append_dense_layer(nn_emb, 0, NULL, "emb_0", conf->num_filters, init_std_normal_create(true, 30 * M_PI, 0));

	edims[0] = conf->num_filters;
	nn_emb = nn_chain2_FF(nn_emb, 0, NULL, nn_from_linop_F(linop_zreal_create(2, edims)), 0, NULL);
	nn_emb = nn_chain2_FF(nn_emb, 0, NULL, nn_from_linop_F(linop_scale_create(2, edims, 1.i)), 0, NULL);
	nn_emb = nn_chain2_FF(nn_emb, 0, NULL, nn_from_nlop_F(nlop_zexp_create(2, edims)), 0, NULL);

	nn_emb = nn_append_dense_layer(nn_emb, 0, NULL, "emb_1", 2 * conf->num_filters, NULL);
	nn_emb = nn_append_activation_bias(nn_emb, 0, NULL, "emb_1_bias", conf->activation, MD_BIT(0));
	nn_emb = nn_append_dense_layer(nn_emb, 0, NULL, "emb_2", conf->cunits, NULL);
	nn_emb = nn_append_activation_bias(nn_emb, 0, NULL, "emb_2_bias", conf->activation, MD_BIT(0));

	return nn_emb;
}


nn_t cunet_create(struct nn_cunet_conf_s* conf, int N, const long dims[N])
{
	struct nn_cunet_conf_s tconf = *conf;

	assert(5 == N);

	for (int i = 0; i < 3; i++) {

		tconf.ksizes[i] = 1 < dims[i + 1] ? tconf.ksizes[i] : 1;
		tconf.dilations[i] = 1 < dims[i + 1] ? tconf.dilations[i] : 1;
		tconf.strides[i] = 1 < dims[i + 1] ? tconf.strides[i] : 1;
	}

	conf = &tconf;
	conf->cunits = conf->conditional ? 4 * conf->num_filters : 0;

	auto network = nn_from_linop_F(linop_identity_create(N, dims));
	network = nn_append_convcorr_layer(network, 0, NULL, "base_conv1", conf->num_filters, conf->ksizes, false, conf->padding, true, NULL, NULL, NULL);
	network = nn_append_activation_bias(network, 0, NULL, "base_bias1", ACT_LIN, MD_BIT(0));

	auto iov = nn_generic_codomain(network, 0, NULL);

	auto low = cunet_level_create(conf, 0, iov->dims);
	//low = nn_chain2_FF(low, 0, NULL, nn_from_nlop_F(nlop_zaxpbz_create(iov->N, iov->dims, 1, 1)), 0, NULL);
	//low = nn_dup_F(low, 0, NULL, 1, NULL);

	network = nn_chain2_FF(network, 0, NULL, low, 0, NULL);

	iov = nn_generic_codomain(network, 0, NULL);
	network = nn_chain2_FF(network, 0, NULL, instance_norm_plus_create(iov->dims, "base"), 0, NULL);
	network = nn_append_activation(network, 0, NULL, conf->activation, MD_BIT(0));
	network = nn_append_convcorr_layer(network, 0, NULL, "base_conv2", dims[0], conf->ksizes, false, conf->padding, true, NULL, NULL, NULL);
	network = nn_append_activation_bias(network, 0, NULL, "base_bias2", ACT_LIN, MD_BIT(0));

	if (conf->conditional) {

		auto nn_emb = cunet_emb_create(conf, dims[4]);
		network = nn_chain2_FF(nn_emb, 0, NULL, network, 0, "emb");
	} else {

		network = nn_combine_FF(network, nn_from_nlop_F(nlop_del_out_create(2, MD_DIMS(1, dims[4]))));
	}

	network = nn_optimize_graph_F(network);
	network = nn_sort_args_F(network);

	return network;
}

