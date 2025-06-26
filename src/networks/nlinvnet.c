/* Copyright 2023-2024. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * Moritz Blumenthal
 */

#include <assert.h>
#include <complex.h>
#include <float.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"

#include "nn/layers.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/init.h"
#include "num/mpi_ops.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/sum.h"
#include "linops/fmac.h"
#include "linops/grad.h"

#include "iter/proj.h"
#include "iter/iter.h"
#include "iter/iter6.h"
#include "iter/monitor_iter6.h"
#include "iter/batch_gen.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/tenmul.h"
#include "nlops/someops.h"
#include "nlops/const.h"
#include "nlops/stack.h"

#include "nn/nn.h"
#include "nn/chain.h"
#include "nn/nn_ops.h"
#include "nn/data_list.h"
#include "nn/const.h"
#include "nn/weights.h"

#include "networks/misc.h"
#include "networks/losses.h"
#include "networks/cnn.h"

#include "grecon/losses.h"

#include "noir/model2.h"
#include "noir/recon2.h"
#include "noir/model_net.h"

#include "nlinvnet.h"


struct nlinvnet_s nlinvnet_config_opts = {

	// Training configuration
	.train_conf = NULL,
	.train_loss = &loss_option,
	.valid_loss = &val_loss_option,
	.l2loss_reg = 0.,
	.time_mask = { -1, -1 },
	.avg_coils_loss = 1,

	// Self-Supervised k-Space
	.ksp_training = false,
	.ksp_split = -1.,
	.ksp_shared_dims = 0.,
	.use_reco_file = NULL,
	.ksp_leaky = 0.,

	// Network block
	.network = NULL,
	.weights = NULL,
	.share_weights = true,
	.lambda = -0.01,
	.lambda_sens = 0.,
	.filter_flags = 0,
	.filter = NULL,

	.conv_time = 0,
	.conv_padding = PAD_SAME,
	
	// NLINV configuration
	.conf = NULL,
	.model = NULL,
	.iter_conf = NULL,
	.iter_conf_net = NULL,
	.cgtol = 0.1,
	.iter_net = 3,		//# of iterations with network
	.oversampling_coils = 2.,
	.senssize = 32,

	.fix_coils = false,
	.ref_init_img = false,
	.ref_init_col = false,
	.ref_init_col_rt = false,
	.scaling = -100.,
	.real_time_init = false,
	.temp_damp = 0.9,
	.debug = false,

	.normalize_rss = false,
};

void nlinvnet_init(struct nlinvnet_s* nlinvnet, int N,
	const long trj_dims[N],
	const long wgh_dims[N],
	const long bas_dims[N], const complex float* basis,
	const long ksp_dims[N],
	const long cim_dims[N],
	const long img_dims[N],
	const long col_dims[N])
{
	nlinvnet->iter_conf_net = TYPE_ALLOC(struct iter_conjgrad_conf);

	*nlinvnet->iter_conf_net = iter_conjgrad_defaults;
	nlinvnet->iter_conf_net->super.alpha = 0.;
	nlinvnet->iter_conf_net->l2lambda = 0.;
	nlinvnet->iter_conf_net->maxiter = nlinvnet->conf->cgiter;
	nlinvnet->iter_conf_net->tol = 0.;

	nlinvnet->iter_conf = TYPE_ALLOC(struct iter_conjgrad_conf);
	*nlinvnet->iter_conf = iter_conjgrad_defaults;
	nlinvnet->iter_conf->super.alpha = 0.;
	nlinvnet->iter_conf->l2lambda = 0.;
	nlinvnet->iter_conf->maxiter = nlinvnet->conf->cgiter;
	nlinvnet->iter_conf->tol = nlinvnet->cgtol;

	if (NULL == get_loss_from_option())
		nlinvnet->train_loss->weighting_mse = 1.;

	if (NULL == get_val_loss_from_option())
		nlinvnet->valid_loss = &loss_image_valid;

	assert(0 == nlinvnet->iter_conf_net->tol);


	struct noir2_model_conf_s model_conf = noir2_model_conf_defaults;
	model_conf.fft_flags = (nlinvnet->conf->sms) ? FFT_FLAGS | SLICE_FLAG : FFT_FLAGS;
	model_conf.rvc = nlinvnet->conf->rvc;
	model_conf.a = nlinvnet->conf->a;
	model_conf.b = nlinvnet->conf->b;
	model_conf.noncart = nlinvnet->conf->noncart;
	model_conf.nufft_conf = nlinvnet->conf->nufft_conf;
	model_conf.oversampling_coils = nlinvnet->oversampling_coils;

	long tcol_dims[N];
	md_copy_dims(N, tcol_dims, col_dims);

	for (int i = 0; i < 3; i++)
		if ((0 < nlinvnet->senssize) && (1 < tcol_dims[i]))
			tcol_dims[i] = nlinvnet->senssize;

	nlinvnet->model = noir2_net_config_create(N, trj_dims, wgh_dims, bas_dims, basis, NULL, NULL, ksp_dims, cim_dims, img_dims, tcol_dims, TIME_FLAG, &model_conf);
}

static nn_t nlinvnet_sort_args_F(nn_t net)
{
	const char* data_names[] = {
		"ref",
		"ksp",
		"pat",
		"trj",
		"loss_mask",
		"prev_frames"
	};

	int N = nn_get_nr_named_in_args(net);
	const char* sorted_names[N + (int)ARRAY_SIZE(data_names) + 3];

	nn_get_in_names_copy(N, sorted_names + ARRAY_SIZE(data_names) + 3, net);

	for (int i = 0; i < (int)ARRAY_SIZE(data_names); i++)
		sorted_names[i] = data_names[i];

	sorted_names[ARRAY_SIZE(data_names) + 0] = "lam";
	sorted_names[ARRAY_SIZE(data_names) + 1] = "lam_sens";
	sorted_names[ARRAY_SIZE(data_names) + 2] = "alp";

	net = nn_sort_inputs_by_list_F(net, N + (int)ARRAY_SIZE(data_names) + 3, sorted_names);

	for (int i = 0; i < N; i++)
		xfree(sorted_names[i + (int)ARRAY_SIZE(data_names) + 3]);

	for (int i = 0; i < (int)ARRAY_SIZE(data_names); i++)
		if (nn_is_name_in_in_args(net, data_names[i]))
			net = nn_set_in_type_F(net, 0, data_names[i], IN_BATCH_GENERATOR);

	const char* out_names_nlinv[] = { "ksp", "cim", "img", "col" };

	N = nn_get_nr_named_out_args(net);
	const char* out_names[(int)ARRAY_SIZE(out_names_nlinv) + N];

	for (int i = 0; i < (int)ARRAY_SIZE(out_names_nlinv); i++)
		out_names[i] = out_names_nlinv[i];


	nn_get_out_names_copy(N, out_names + ARRAY_SIZE(out_names_nlinv), net);

	net = nn_sort_outputs_by_list_F(net, (int)ARRAY_SIZE(out_names_nlinv) + N, out_names);

	for (int i = 0; i < N; i++)
		xfree(out_names[(int)ARRAY_SIZE(out_names_nlinv) + i]);

	net = nn_sort_inputs_F(net);
	net = nn_sort_outputs_F(net);

	if (nn_is_name_in_in_args(net, "lam")) {

		const struct iovec_s* iov = nn_generic_domain(net, 0, "lam");
		auto prox_conv = operator_project_pos_real_create(iov->N, iov->dims);
		net = nn_set_prox_op_F(net, 0, "lam", prox_conv);
	}

	if (nn_is_name_in_in_args(net, "lam_sens")) {

		const struct iovec_s* iov = nn_generic_domain(net, 0, "lam_sens");
		auto prox_conv = operator_project_pos_real_create(iov->N, iov->dims);
		net = nn_set_prox_op_F(net, 0, "lam_sens", prox_conv);
	}

	return net;
}


static nn_t nlinvnet_network_create(const struct nlinvnet_s* nlinvnet, int N, const long _img_dims[N], enum NETWORK_STATUS status)
{
	nn_t network = NULL;

	if (0 < nlinvnet->conv_time) {

		int window_dim = TIME_DIM;

		while (1 != _img_dims[++window_dim])
			assert(BATCH_DIM > 1);

		const struct linop_s* lop_prep = NULL;
		long pos = 0; //position of the feed through frame (residual network)
		
		if (PAD_CAUSAL == nlinvnet->conv_padding) {

			pos = nlinvnet->conv_time - 1;
			lop_prep = linop_padding_create_onedim(N, _img_dims, PAD_CAUSAL, TIME_DIM, nlinvnet->conv_time - 1, 0);
		}
		
		if (PAD_SAME == nlinvnet->conv_padding) {

			assert(1 == nlinvnet->conv_time % 2);
			pos = nlinvnet->conv_time / 2;
			lop_prep = linop_padding_create_onedim(N, _img_dims, PAD_SAME, TIME_DIM, nlinvnet->conv_time / 2, nlinvnet->conv_time / 2);
		}

		lop_prep = linop_chain_FF(lop_prep, linop_hankelization_create(N, linop_codomain(lop_prep)->dims, TIME_DIM, window_dim, nlinvnet->conv_time));
		
		const struct nlop_s* nlop_prep = nlop_from_linop_F(lop_prep);

		lop_prep = linop_transpose_create(N, TIME_DIM, window_dim, nlop_generic_codomain(nlop_prep, 0)->dims);

		long img_dims[N];
		md_copy_dims(N, img_dims, linop_codomain(lop_prep)->dims);
		img_dims[BATCH_DIM] *= img_dims[window_dim];
		img_dims[window_dim] = 1;

		lop_prep = linop_chain_FF(lop_prep, linop_reshape2_create(N, MD_BIT(window_dim) | BATCH_FLAG, img_dims, linop_codomain(lop_prep)->dims));
		nlop_prep = nlop_chain2_FF(nlop_prep, 0, nlop_from_linop_F(lop_prep), 0);

		network = network_create(nlinvnet->network, N, img_dims, N, img_dims, status);
		network = nn_chain2_FF(nn_from_nlop_F(nlop_prep), 0, NULL, network, 0, NULL);

		network = nn_chain2_FF(network, 0, NULL, nn_from_linop_F(linop_slice_one_create(N, TIME_DIM, pos, nn_generic_codomain(network, 0, NULL)->dims)), 0, NULL);
		network = nn_chain2_FF(network, 0, NULL, nn_from_linop_F(linop_reshape2_create(N, BATCH_FLAG | TIME_FLAG , _img_dims, nn_generic_codomain(network, 0, NULL)->dims)), 0, NULL);

	} else {

		network = network_create(nlinvnet->network, N, _img_dims, N, _img_dims, status);
	}

	if (0 != nlinvnet->filter_flags) {

		auto lop_post = linop_fftc_create(N, _img_dims, nlinvnet->filter_flags);
		auto ifft = linop_get_adjoint(lop_post);

		lop_post = linop_chain_FF(lop_post, linop_cdiag_create(N, _img_dims, nlinvnet->filter_flags, nlinvnet->filter));
		lop_post = linop_chain_FF(lop_post, ifft);
		network = nn_chain2_FF(network, 0, NULL, nn_from_linop_F(lop_post), 0, NULL);
	}

	return network;
}

static nn_t nlinvnet_get_network_step(const struct nlinvnet_s* nlinvnet, struct noir2_net_s* model, enum NETWORK_STATUS status)
{
	int N = noir2_net_get_N(model);
	assert(N == DIMS);

	long img_dims[N];
	noir2_net_get_img_dims(model, N, img_dims);

	if (nlinvnet->ref_init_img) 
		img_dims[COIL_DIM] *= 2;

	auto network = nlinvnet_network_create(nlinvnet, N, img_dims, status);

	int N_in_names = nn_get_nr_named_in_args(network);
	int N_out_names = nn_get_nr_named_out_args(network);
	const char* in_names[N_in_names];
	const char* out_names[N_out_names];
	nn_get_in_names_copy(N_in_names, in_names, network);
	nn_get_out_names_copy(N_out_names, out_names, network);

	for (int i = 0; i < N_in_names; i++) {

		network = nn_append_singleton_dim_in_F(network, 0, in_names[i]);

		xfree(in_names[i]);
	}

	for (int i = 0; i < N_out_names; i++) {

		network = nn_append_singleton_dim_out_F(network, 0, out_names[i]);

		xfree(out_names[i]);
	}

	long img_one_dims[N];
	noir2_net_get_img_dims(model, N, img_one_dims);

	if (nlinvnet->ref_init_img) {

		network = nn_chain2_FF(network, 0, NULL, nn_from_linop_F(linop_slice_one_create(N, COIL_DIM, 0, img_dims)), 0, NULL);
		network = nn_chain2_FF(nn_from_nlop_F(nlop_stack_create(N, img_dims, img_one_dims, img_one_dims, COIL_DIM)), 0, NULL, network, 0, NULL);
		network = nn_set_input_name_F(network, 0, "ref_img");

	} else {

		auto dummy = nn_from_nlop_F(nlop_del_out_create(N, img_one_dims));
		dummy = nn_set_input_name_F(dummy, 0, "ref_img");
		network = nn_combine_FF(network, dummy);		
	}

	nn_t join = nn_from_nlop_F(noir_join_create(model));

	if (nlinvnet->ref_init_col || nlinvnet->ref_init_col_rt) {

		auto col_dom = nn_generic_domain(join, 1, NULL);
		struct linop_s* lop_shift;

		if (nlinvnet->ref_init_col_rt)
			lop_shift = linop_chain_FF(linop_scale_create(col_dom->N, col_dom->dims, nlinvnet->temp_damp), linop_shift_create(col_dom->N, col_dom->dims, TIME_DIM, 1, PAD_VALID));
		else
			lop_shift = linop_identity_create(col_dom->N, col_dom->dims);

		auto nn_shift = nn_from_linop_F(lop_shift);
		nn_shift = nn_set_input_name_F(nn_shift, 0, "ref_col");

		join = nn_chain2_FF(nn_shift, 0, NULL, join, 1, NULL);

	} else {

		auto dom = nn_generic_domain(join, 1, NULL);
		auto dummy = nn_from_nlop_F(nlop_del_out_create(dom->N, dom->dims));
		dummy = nn_set_input_name_F(dummy, 0, "ref_col");
		
		complex float zero[1] = { 0. };

		join = nn_set_input_const_F2(join, 1, NULL, dom->N, dom->dims, MD_SINGLETON_STRS(dom->N), true, zero);
		join = nn_combine_FF(dummy, join);	
	}

	network = nn_chain2_FF(network, 0, NULL, join, 0, NULL);

	nn_t split = nn_from_nlop_F(noir_extract_img_create(model));
	network = nn_chain2_FF(split, 0, NULL, network, 0, NULL);

	nn_t split_ref = nn_from_nlop_F(noir_split_create(model));
	split_ref = nn_set_input_name_F(split_ref, 0, "ref_x");
	split_ref = nn_set_output_name_F(split_ref, 0, "ref_img");
	split_ref = nn_set_output_name_F(split_ref, 0, "ref_col");

	network = nn_chain2_FF(split_ref, 0, "ref_img", network, 0, "ref_img");
	network = nn_link_F(network, 0, "ref_col", 0, "ref_col");

	return network;
}


static nn_t nlinvnet_gn_reg(const struct nlinvnet_s* nlinvnet, struct noir2_net_s* model, enum NETWORK_STATUS status)
{
	auto nlop_dc = (nlinvnet->fix_coils ? noir_sense_recon_create : noir_gauss_newton_step_create)(model, nlinvnet->iter_conf_net);

	if (nlinvnet->debug) {

		static int counter = 0;

		const char* ni = ptr_printf("gn_%d_in", counter);
		const char* no = ptr_printf("gn_%d_out", counter);
		const char* nr = ptr_printf("gn_%d_ref", counter);

		nlop_dc = nlop_prepend_FF(noir_nlop_dump_create(model, ni), nlop_dc, 1);
		nlop_dc = nlop_prepend_FF(noir_nlop_dump_create(model, nr), nlop_dc, 2);

		nlop_dc = nlop_append_FF(nlop_dc, 0, noir_nlop_dump_create(model, no));

		counter++;
	}

	nn_t result = nn_from_nlop_F(nlop_dc);
	result = nn_set_input_name_F(result, 0, "y");
	result = nn_set_input_name_F(result, 1, "x_0");
	result = nn_set_input_name_F(result, 1, "alp");

	long reg_dims[2];
	md_copy_dims(2, reg_dims, nn_generic_domain(result, 0, "x_0")->dims);

	auto network = nlinvnet_get_network_step(nlinvnet, model, status);

	int N_in_names_gn = nn_get_nr_named_in_args(result);
	int N_in_names_net = nn_get_nr_named_in_args(network);

	const char* in_names[N_in_names_gn + N_in_names_net];
	nn_get_in_names_copy(N_in_names_gn, in_names, result);
	nn_get_in_names_copy(N_in_names_net, in_names + N_in_names_gn, network);

	auto nlop_reg = noir_join_create(model);
		
	auto dom = nlop_generic_domain(nlop_reg, 0);
	nlop_reg = nlop_prepend_FF(nlop_from_linop_F(linop_repmat_create(dom->N, dom->dims, ~0UL)), nlop_reg, 0);
		
	dom = nlop_generic_domain(nlop_reg, 1);
	nlop_reg = nlop_prepend_FF(nlop_from_linop_F(linop_repmat_create(dom->N, dom->dims, ~0UL)), nlop_reg, 1);

	nlop_reg = nlop_reshape_in_F(nlop_reg, 0, 1, MD_SINGLETON_DIMS(1));
	nlop_reg = nlop_reshape_in_F(nlop_reg, 1, 1, MD_SINGLETON_DIMS(1));

	nlop_reg = nlop_chain2_FF(nlop_zaxpbz_create(1, MD_SINGLETON_DIMS(1), 1, 1), 0, nlop_reg, 0);
	nlop_reg = nlop_chain2_FF(nlop_zaxpbz_create(1, MD_SINGLETON_DIMS(1), 1, 1), 0, nlop_reg, 0);
	nlop_reg = nlop_dup_F(nlop_reg, 0, 2);

	auto nn_reg = nn_from_nlop_F(nlop_reg);
	nn_reg = nn_set_input_name_F(nn_reg, 0, "alp");
	nn_reg = nn_set_input_name_F(nn_reg, 0, "lam");
	nn_reg = nn_set_input_name_F(nn_reg, 0, "lam_sens");

	result = nn_chain2_swap_FF(nn_reg, 0, NULL, result, 0, "alp");
	result = nn_mark_dup_F(result, "lam");
	result = nn_mark_dup_F(result, "lam_sens");

	//make lambda dummy input of network
	nn_t tmp = nn_from_nlop_F(nlop_del_out_create(1, MD_DIMS(1)));
	tmp = nn_set_input_name_F(tmp, 0, "lam");
	tmp = nn_set_in_type_F(tmp, 0, "lam", IN_OPTIMIZE);;
	tmp = nn_set_initializer_F(tmp, 0, "lam", init_const_create(fabsf(nlinvnet->lambda)));
	network = nn_combine_FF(tmp, network);

	//make lambda sens dummy input of network
	tmp = nn_from_nlop_F(nlop_del_out_create(1, MD_DIMS(1)));
	tmp = nn_set_input_name_F(tmp, 0, "lam_sens");
	tmp = nn_set_in_type_F(tmp, 0, "lam_sens", IN_OPTIMIZE);;
	tmp = nn_set_initializer_F(tmp, 0, "lam_sens", init_const_create(fabsf(nlinvnet->lambda_sens)));
	network = nn_combine_FF(tmp, network);

	result = nn_chain2_FF(network, 0, NULL, result, 0, "x_0");
	result = nn_dup_F(result, 0, NULL, 1, NULL);
	result = nn_stack_dup_by_name_F(result);
	result = nn_sort_inputs_by_list_F(result, N_in_names_gn + N_in_names_net, in_names);

	for (int i = 0; i < N_in_names_gn + N_in_names_net; i++)
		xfree(in_names[i]);

	return result;
}


static nn_t nlinvnet_chain_alpha(nn_t network, float redu, float alpha_min)
{
	auto nlop_scale = nlop_from_linop_F(linop_scale_create(1, MD_SINGLETON_DIMS(1), 1. / redu));
	nlop_scale = nlop_chain_FF(nlop_zsadd_create(1, MD_SINGLETON_DIMS(1), -alpha_min), nlop_scale);
	nlop_scale = nlop_chain_FF(nlop_scale, nlop_zsadd_create(1, MD_SINGLETON_DIMS(1), alpha_min));

	auto scale = nn_from_nlop_F(nlop_scale);
	network = nn_chain2_FF(scale, 0, NULL, network, 0, "alp");
	network = nn_set_input_name_F(network, -1, "alp");

	network = nlinvnet_sort_args_F(network);

	return network;
}

static nn_t nlinvnet_create(const struct nlinvnet_s* nlinvnet, struct noir2_net_s* model, enum NETWORK_STATUS status)
{
	auto result = nlinvnet_gn_reg(nlinvnet, model, status);

	for (int i = 1; i < nlinvnet->iter_net; i++) {

		result = nlinvnet_chain_alpha(result, nlinvnet->conf->redu, nlinvnet->conf->alpha_min);

		result = nn_mark_dup_if_exists_F(result, "y");
		result = nn_mark_dup_if_exists_F(result, "x_0");
		result = nn_mark_dup_if_exists_F(result, "alp");
		result = nn_mark_dup_if_exists_F(result, "ref_x");

		auto tmp = nlinvnet_gn_reg(nlinvnet, model, status);

		int N_in_names = nn_get_nr_named_in_args(tmp);
		int N_out_names = nn_get_nr_named_out_args(tmp);

		const char* in_names[N_in_names];
		const char* out_names[N_out_names];

		nn_get_in_names_copy(N_in_names, in_names, tmp);
		nn_get_out_names_copy(N_out_names, out_names, tmp);

		// batchnorm weights are always stacked
		for (int i = 0; i < N_in_names; i++) {

			if (nn_is_name_in_in_args(result, in_names[i])) {

				if (nn_get_dup(result, 0, in_names[i]) && nlinvnet->share_weights)
					result = nn_mark_dup_F(result, in_names[i]);
				else
					result = nn_mark_stack_input_F(result, in_names[i]);
			}
		}

		for (int i = 0; i < N_out_names; i++)
			result = nn_mark_stack_output_if_exists_F(result, out_names[i]);

		result = nn_chain2_FF(tmp, 0, NULL, result, 0, NULL);
		result = nn_stack_dup_by_name_F(result);

		for (int i = 0; i < N_in_names; i++)
			xfree(in_names[i]);

		for (int i = 0; i < N_out_names; i++)
			xfree(out_names[i]);
	}

	result = nn_dup_F(result, 0, NULL, 0, "ref_x");

	for (int i = nlinvnet->iter_net; i < (int)(nlinvnet->conf->iter); i++)
		result = nlinvnet_chain_alpha(result, nlinvnet->conf->redu, nlinvnet->conf->alpha_min);

	// initialization reco
	const struct nlop_s* nlop_init_reco;
	if (nlinvnet->real_time_init)
		nlop_init_reco = noir_rtnlinv_iter_create(model, nlinvnet->iter_conf, (int)nlinvnet->conf->iter - nlinvnet->iter_net, 0, nlinvnet->conf->redu, nlinvnet->conf->alpha_min, nlinvnet->temp_damp);
	else
		nlop_init_reco = noir_gauss_newton_iter_create_create(model, nlinvnet->iter_conf, (int)nlinvnet->conf->iter - nlinvnet->iter_net, nlinvnet->conf->redu, nlinvnet->conf->alpha_min);
	
	nlop_init_reco = nlop_set_input_scalar_F(nlop_init_reco, 2, 0);

	auto dom_alp = nlop_generic_domain(nlop_init_reco, 2);
	nlop_init_reco = nlop_prepend_FF(nlop_from_linop_F(linop_repmat_create(dom_alp->N, dom_alp->dims, ~0UL)), nlop_init_reco, 2);
	nlop_init_reco = nlop_reshape_in_F(nlop_init_reco, 2, 1, MD_DIMS(1));
	

	auto nn_init_reco = nn_from_nlop_F(nlop_init_reco);
	nn_init_reco = nn_set_input_name_F(nn_init_reco, 0, "y");
	nn_init_reco = nn_set_input_name_F(nn_init_reco, 1, "alp");
	nn_init_reco = nn_mark_dup_F(nn_init_reco, "alp");
	nn_init_reco = nn_mark_dup_F(nn_init_reco, "y");

	result = nn_chain2_FF(nn_init_reco, 0, NULL, result, 0, NULL);
	result = nn_stack_dup_by_name_F(result);
	result = nlinvnet_sort_args_F(result);

	complex float alpha = nlinvnet->conf->alpha;
	result = nn_set_input_const_F2(result, 0, "alp", 1, MD_SINGLETON_DIMS(1), MD_SINGLETON_STRS(1), true, &alpha);	// in: y, xn, x0


	//init image with one and coils with zero
	complex float one = 1;
	complex float zero = 0;

	auto nlop_init = noir_join_create(model);
	auto dom = nlop_generic_domain(nlop_init, 0);
	nlop_init = nlop_set_input_const_F2(nlop_init, 0, dom->N, dom->dims, MD_SINGLETON_STRS(dom->N), true, &one);
	dom = nlop_generic_domain(nlop_init, 0);
	nlop_init = nlop_set_input_const_F2(nlop_init, 0, dom->N, dom->dims, MD_SINGLETON_STRS(dom->N), true, &zero);
	
	auto d1 = nlop_generic_codomain(nlop_init, 0);
	auto d2 = nn_generic_domain(result, 0, NULL);

	nlop_init = nlop_chain2_FF(nlop_init, 0, nlop_from_linop_F(linop_expand_create(d1->N, d2->dims, d1->dims)), 0);
	result = nn_chain2_FF(nn_from_nlop_F(nlop_init), 0, NULL, result, 0, NULL);


	// normalization of input
	float scale = -nlinvnet->scaling;

	int N = noir2_net_get_N(model);
	long cim_dims[N];
	long sdims[N];

	noir2_net_get_cim_dims(model, N, cim_dims);	
	md_select_dims(N, BATCH_FLAG, sdims, cim_dims);

	const struct nlop_s* nlop_scale = NULL;
	if (0 > nlinvnet->scaling) {

		nlop_scale = nlop_norm_znorm_create(N, cim_dims, BATCH_FLAG);

	} else {

		complex float one[1] = { 1. };
		nlop_scale = nlop_const_create2(N, sdims, MD_SINGLETON_STRS(N), true, one);
		nlop_scale = nlop_combine_FF(nlop_from_linop_F(linop_identity_create(N, cim_dims)), nlop_scale);
	}

	nlop_scale = nlop_chain2_FF(nlop_scale, 1, nlop_from_linop_F(linop_scale_create(N, sdims, 1. / scale)), 0);
	nlop_scale = nlop_chain2_FF(nlop_scale, 1, nlop_from_linop_F(linop_scale_create(N, cim_dims, scale)), 0);
	nlop_scale = nlop_chain2_keep_FF(nlop_scale, 1, nlop_zsqrt_create(N, sdims), 0);
	nlop_scale = nlop_reshape_out_F(nlop_scale, 0, 2, (long[2]) { 1, sdims[BATCH_DIM] });

	auto nn_scale = nn_from_nlop_F(nlop_scale);
	nn_scale = nn_set_output_name_F(nn_scale, 0, "scale_sqrt");
	nn_scale = nn_set_output_name_F(nn_scale, 1, "scale");
	result = nn_chain2_FF(nn_scale, 0, NULL, result, 0, "y");

	const struct nlop_s* nlop_adj;

	if (nlinvnet->conf->noncart)
		nlop_adj = noir_adjoint_nufft_create(model);
	else
	 	nlop_adj = noir_adjoint_fft_create(model);

	long ksp_dims[N];
	long pat_dims[N];

	md_copy_dims(N, ksp_dims, nlop_generic_domain(nlop_adj, 0)->dims);
	md_copy_dims(N, pat_dims, nlop_generic_domain(nlop_adj, 1)->dims);

	nlop_adj = nlop_chain2_swap_FF(nlop_tenmul_create(N, ksp_dims, ksp_dims, pat_dims), 0, nlop_adj, 0);
	nlop_adj = nlop_dup_F(nlop_adj, 1, 2);

	result = nn_chain2_swap_FF(nn_from_nlop_F(nlop_adj), 0, NULL , result, 0, NULL);
	result = nn_set_input_name_F(result, 0, "ksp");
	result = nn_set_input_name_F(result, 0, "pat");

	if (nlinvnet->conf->noncart)
		result = nn_set_input_name_F(result, 0, "trj");

	// normalize output
	auto cod = nn_generic_codomain(result, 0, NULL);
	long cdims[2] = { cod->dims[0] * cod->dims[1] / sdims[BATCH_DIM], sdims[BATCH_DIM]};
	long tdims[2] = { cod->dims[0], cod->dims[1]};

	result = nn_reshape_out_F(result, 0, NULL, 2, cdims);
	result = nn_chain2_FF(result, 0, NULL, nn_from_nlop_F(nlop_tenmul_create(2, cdims, cdims, (long[2]){ 1, sdims[BATCH_DIM] })), 0, NULL);
	result = nn_link_F(result, 0, "scale_sqrt", 0, NULL); 
	result = nn_reshape_out_F(result, 0, NULL, 2, tdims);

	return result;
}


static nn_t nlinvnet_apply_op_create(const struct nlinvnet_s* nlinvnet, int Nb)
{
	struct noir2_net_s* model = noir2_net_create(nlinvnet->model, Nb);

	auto nn_apply = nlinvnet_create(nlinvnet, model, STAT_TEST);

	nn_apply = nn_chain2_FF(nn_apply, 0, NULL, nn_from_nlop_F(noir_decomp_create(model)), 0, NULL);
	nn_apply = nn_set_output_name_F(nn_apply, 0, "img");
	nn_apply = nn_set_output_name_F(nn_apply, 0, "col");

	nn_apply = nn_del_out_F(nn_apply, 0, "scale");

	if (!nn_is_name_in_in_args(nn_apply, "trj")) {

		nn_apply = nn_combine_FF(nn_from_nlop_F(nlop_del_out_create(DIMS, MD_SINGLETON_DIMS(DIMS))), nn_apply);
		nn_apply = nn_set_input_name_F(nn_apply, 0, "trj");
	}

	nn_apply = nlinvnet_sort_args_F(nn_apply);

	int N_weights = 0;
	for (int i = nn_get_nr_in_args(nn_apply) - 1; i >= 0; i--)
		if ((IN_OPTIMIZE == nn_apply->in_types[i]) || (IN_BATCHNORM == nn_apply->in_types[i]))
			N_weights++;

	complex float zero[1] = { 0 };

	if (nlinvnet->weights->N + 1 == N_weights)
		nn_apply = nn_set_input_const_F(nn_apply, 0, "lam_sens", 1, MD_DIMS(1), true, zero);

	return nn_get_wo_weights_F(nn_apply, nlinvnet->weights, false);
}

static nn_t nlinvnet_valid_create(const struct nlinvnet_s* nlinvnet, struct named_data_list_s* valid_data)
{
	auto ksp_iov = named_data_list_get_iovec(valid_data, "ksp");
	int Nb = ksp_iov->dims[BATCH_DIM];
	iovec_free(ksp_iov);

	struct noir2_net_s* model = noir2_net_create(nlinvnet->model, Nb);

	auto result = nlinvnet_create(nlinvnet, model, STAT_TEST);
	result = nn_del_out_bn_F(result);
	result = nn_del_out_F(result, 0, "scale");

	if (nn_is_name_in_out_args(result, "l2_reg_reco"))
		result = nn_del_out_F(result, 0, "l2_reg_reco");

	result = nn_chain2_FF(result, 0, NULL, nn_from_nlop_F(noir_decomp_create(model)), 0, NULL);

	auto cim_iov = named_data_list_get_iovec(valid_data, "ref");
	auto valid_loss = nn_chain2_FF(result, 0, NULL, val_measure_create(nlinvnet->valid_loss, cim_iov->N, cim_iov->dims), 0, NULL);
	iovec_free(cim_iov);

	valid_loss = nn_set_input_name_F(valid_loss, 0, "ref");
	valid_loss = nlinvnet_sort_args_F(valid_loss);

	return nn_valid_create(valid_loss, valid_data);
}


static nn_t nlinvnet_train_loss_create(const struct nlinvnet_s* nlinvnet, int Nb)
{
	struct noir2_net_s* model = noir2_net_create(nlinvnet->model, Nb);

	auto nn_train = nlinvnet_create(nlinvnet, model, STAT_TRAIN);

	if (0 < nlinvnet->l2loss_reg) {

		unsigned long mask_flag = (-1. != nlinvnet->time_mask[0]) || (-1. != nlinvnet->time_mask[1]) ? TIME_FLAG : 0;
		auto nlop_reg = noir_nlinv_regularization_create(model, mask_flag);

		if ((-1. != nlinvnet->time_mask[0]) || (-1. != nlinvnet->time_mask[1])) {

			long time = nlop_generic_domain(nlop_reg, 1)->dims[TIME_DIM];
			
			int N = nlop_generic_domain(nlop_reg, 1)->N;

			long tdims[N];
			md_singleton_dims(N, tdims);
			tdims[TIME_DIM] = time;

			complex float mask[time];

			for (int i = 0; i < time; i++) {

				if ((i >= nlinvnet->time_mask[0]) && ((-1 == nlinvnet->time_mask[1]) || (i < nlinvnet->time_mask[1])))
					mask[i] = 1;
				else
					mask[i] = 0;
			}

			nlop_reg = nlop_set_input_const_F(nlop_reg, 1, N, tdims, true, mask);
		}

		auto cod = nn_generic_codomain(nn_train, 0, "scale");

		nlop_reg = nlop_chain2_swap_FF(nlop_reg, 0, nlop_tenmul_create(cod->N, cod->dims, cod->dims, cod->dims), 0);
		nlop_reg = nlop_flatten_out_F(nlop_reg, 0);
		auto reg = nn_from_nlop_F(nlop_reg);

		reg = nn_set_input_name_F(reg, 1, "scale");
		reg = nn_set_output_name_F(reg, 0, "l2_reg_reco");

		nn_train = nn_chain2_keep_FF(nn_train, 0, NULL, reg, 0, NULL);
		nn_train = nn_link_F(nn_train, 0, "scale", 0, "scale");

	} else {

		nn_train = nn_del_out_F(nn_train, 0, "scale");
	}

	if (1 < nlinvnet->avg_coils_loss)
		nn_train = nn_chain2_FF(nn_train, 0, NULL, nn_from_nlop_F(noir_nlinv_average_coils_create(model, nlinvnet->conv_padding, nlinvnet->avg_coils_loss)),0, NULL);

	nn_train = nn_chain2_FF(nn_train, 0, NULL, nn_from_nlop_F(noir_cim_create(model)),0, NULL);

	if (nlinvnet->ksp_training) {

		nn_t fft_op;
		if (nlinvnet->conf->noncart)
			fft_op = nn_from_nlop_F(noir_nufft_create(model));
		else
			fft_op = nn_from_nlop_F(noir_fft_create(model));

		if (nlinvnet->conf->noncart) {

			fft_op = nn_set_input_name_F(fft_op, 1, "trj");
			fft_op = nn_mark_dup_F(fft_op, "trj");
		}

		nn_train = nn_chain2_FF(nn_train, 0, NULL, fft_op, 0, NULL);
		nn_train = nn_stack_dup_by_name_F(nn_train);
	}

	int N = nn_generic_codomain(nn_train, 0, NULL)->N;

	long out_dims[N];
	long pat_dims[N];
	md_copy_dims(N, out_dims, nn_generic_codomain(nn_train, 0, NULL)->dims);
	md_copy_dims(N, pat_dims, nn_generic_domain(nn_train, 0, "pat")->dims);

	nn_t loss = train_loss_create(nlinvnet->train_loss, N, out_dims);

	/*const*/ char* loss_name = strdup(nn_get_out_name_from_arg_index(loss, 0, NULL));

	if (!loss_name)
		error("memory out");

	if ((-1. != nlinvnet->time_mask[0]) || (-1. != nlinvnet->time_mask[1])) {

		long tdims[N];
		md_select_dims(N, TIME_FLAG, tdims, out_dims);

		long time = tdims[TIME_DIM];

		complex float mask[time];

		for (int i = 0; i < time; i++) {

			if ((i >= nlinvnet->time_mask[0]) && ((-1 == nlinvnet->time_mask[1]) || (i < nlinvnet->time_mask[1])))
				mask[i] = 1;
			else
				mask[i] = 0;
		}

		loss = nn_chain2_FF(nn_from_linop_F(linop_cdiag_create(N, out_dims, TIME_FLAG, mask)), 0, NULL, loss, 0, NULL);
		loss = nn_chain2_FF(nn_from_linop_F(linop_cdiag_create(N, out_dims, TIME_FLAG, mask)), 0, NULL, loss, 0, NULL);
	}

	if (-1. != nlinvnet->ksp_split) {

		loss = nn_chain2_FF(nn_from_nlop_F(nlop_tenmul_create(N, out_dims, out_dims, pat_dims)), 0, NULL, loss, 0, NULL);
		loss = nn_chain2_FF(nn_from_nlop_F(nlop_tenmul_create(N, out_dims, out_dims, pat_dims)), 0, NULL, loss, 0, NULL);
		loss = nn_dup_F(loss, 1, NULL, 3, NULL);
		loss = nn_set_input_name_F(loss, 1, "pat_ref");
	}

	nn_train = nn_chain2_FF(nn_train, 0, NULL, loss, 0, NULL);
	nn_train = nn_set_input_name_F(nn_train, 0, "ref");

	if (0 < nlinvnet->l2loss_reg) {

		nn_train = nn_chain2_FF(nn_train, 0, loss_name, nn_from_nlop_F(nlop_zaxpbz_create(1, MD_DIMS(1), 1, nlinvnet->l2loss_reg)), 0, NULL);
		nn_train = nn_link_F(nn_train, 0, "l2_reg_reco", 0, NULL);
		nn_train = nn_set_out_type_F(nn_train, 0, NULL, OUT_OPTIMIZE);
		nn_train = nn_set_output_name_F(nn_train, 0, loss_name);
	}

	free(loss_name);

	nn_train = nlinvnet_sort_args_F(nn_train);

	return nn_train;
}



void train_nlinvnet(struct nlinvnet_s* nlinvnet, int Nb, struct named_data_list_s* train_data, struct named_data_list_s* valid_data)
{
	auto ref_iov = named_data_list_get_iovec(train_data, "ref");
	long Nt = ref_iov->dims[BATCH_DIM];

	iovec_free(ref_iov);

	Nb = MIN(Nb, Nt);

	assert(0 == Nb % mpi_get_num_procs());
	assert(1 == mpi_get_num_procs() || network_is_diagonal(nlinvnet->network));

	int M = 1;

	if (network_is_diagonal(nlinvnet->network))
		M = Nb;

	int R = mpi_get_num_procs();
	assert(R > 0);
	assert(0 == M % R);

	nn_t train_ops[M];

	for (int i = 0; i < M / R; i++) {

		train_ops[R * i] = nlinvnet_train_loss_create(nlinvnet, Nb / M);

		for (int j = 1; j < R; j++)
			train_ops[R * i + j] = nn_clone(train_ops[R * i]);
	}

	auto nn_train = (1 == M) ? train_ops[0] : nn_stack_multigpu_F(M, train_ops, -1);

	if (-1. != nlinvnet->ksp_split) {

		const struct nlop_s* nlop_rand_split;

		auto dom = nn_generic_domain(nn_train, 0, "pat_ref");
		int N = dom->N;
		long pat_dims[N];
		md_copy_dims(N, pat_dims, dom->dims);

		const complex float* use_reco = NULL;
		long use_reco_dims[DIMS];
		unsigned long use_reco_nontriv = 0UL;

		if (NULL != nlinvnet->use_reco_file) {

			use_reco = load_cfl(nlinvnet->use_reco_file, DIMS, use_reco_dims);

			use_reco_nontriv = md_nontriv_dims(DIMS, use_reco_dims);
		}

		nlop_rand_split = nlop_rand_split_fixed_create(N, pat_dims, nlinvnet->ksp_shared_dims, BATCH_FLAG | TIME_FLAG,
					nlinvnet->ksp_split, use_reco_nontriv, use_reco, nlinvnet->ksp_leaky);

		if (NULL != use_reco)
			unmap_cfl(DIMS, use_reco_dims, use_reco);
			
		auto split_op = nn_from_nlop_F(nlop_rand_split);
		split_op = nn_set_output_name_F(split_op, 0, "pat_trn");
		split_op = nn_set_output_name_F(split_op, 0, "pat_ref");

		nn_train = nn_chain2_swap_FF(split_op, 0, "pat_trn", nn_train, 0, "pat");
		nn_train = nn_link_F(nn_train, 0, "pat_ref", 0, "pat_ref");
		nn_train = nn_set_input_name_F(nn_train, 0, "pat");

		nn_train = nlinvnet_sort_args_F(nn_train);
	}

	debug_printf(DP_INFO, "Train nlinvnet\n");
	nn_debug(DP_INFO, nn_train);

	if (NULL == nlinvnet->weights) {

		nlinvnet->weights = nn_weights_create_from_nn(nn_train);
		nn_init(nn_train, nlinvnet->weights);

	} else {

		auto tmp_weights = nn_weights_create_from_nn(nn_train);

		nn_weights_copy(tmp_weights, nlinvnet->weights);

		nn_weights_free(nlinvnet->weights);

		nlinvnet->weights = tmp_weights;
	}

	if (bart_use_gpu)
		move_gpu_nn_weights(nlinvnet->weights);

	if (0 <= nlinvnet->lambda)
		nn_train = nn_set_in_type_F(nn_train, 0, "lam", IN_STATIC);

	if (0 < nlinvnet->lambda_sens)
		nn_train = nn_set_in_type_F(nn_train, 0, "lam_sens", IN_STATIC);

	//create batch generator
	struct bat_gen_conf_s batgen_config = bat_gen_conf_default;
	batgen_config.seed = nlinvnet->train_conf->batch_seed;
	batgen_config.type = nlinvnet->train_conf->batchgen_type;
	
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

		case IN_BATCH:
		case IN_UNDEFINED:
			error("Intype of arg %d not supported!\n", i);
			break;

		case IN_OPTIMIZE:
		case IN_STATIC:
		case IN_BATCHNORM:

			auto iov_weight = nlinvnet->weights->iovs[weight_index];
			auto iov_train_op = nlop_generic_domain(nn_get_nlop(nn_train), i);
			assert(md_check_equal_dims(iov_weight->N, iov_weight->dims, iov_train_op->dims, ~0UL));
			src[i] = (float*)nlinvnet->weights->tensors[weight_index];
			weight_index++;
			break;
		}
	}

	int num_monitors = 0;
	const struct monitor_value_s* value_monitors[3];

	if (NULL != valid_data) {

		auto nn_validation_loss = nlinvnet_valid_create(nlinvnet, valid_data);

		const char* val_names[nn_get_nr_out_args(nn_validation_loss)];

		for (int i = 0; i < nn_get_nr_out_args(nn_validation_loss); i++)
			val_names[i] = nn_get_out_name_from_arg_index(nn_validation_loss, i, false);

		value_monitors[num_monitors] = monitor_iter6_nlop_create(nn_get_nlop(nn_validation_loss), false, nn_get_nr_out_args(nn_validation_loss), val_names);
		nn_free(nn_validation_loss);

		num_monitors += 1;
	}

	if (nn_is_name_in_in_args(nn_train, "lam")) {

		int index_lambda = nn_get_in_arg_index(nn_train, 0, "lam");
		int num_lambda = nn_generic_domain(nn_train, 0, "lam")->dims[0];

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

	if (nn_is_name_in_in_args(nn_train, "lam_sens")) {

		int index_lambda = nn_get_in_arg_index(nn_train, 0, "lam_sens");
		int num_lambda = nn_generic_domain(nn_train, 0, "lam_sens")->dims[0];

		const char* lam = "ls";
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

	iter6_by_conf(nlinvnet->train_conf, nn_get_nlop(nn_train), NI, in_type, projections, src, NO, out_type, Nb, Nt / Nb, batch_generator, monitor);

	nn_free(nn_train);
	nlop_free(batch_generator);

	monitor_iter6_free(monitor);
}


void apply_nlinvnet(struct nlinvnet_s* nlinvnet, int N,
	const long img_dims[N], complex float* img,
	const long col_dims[N], complex float* col,
	const long ksp_dims[N], const complex float* ksp,
	const long pat_dims[N], const complex float* pat,
	const long trj_dims[N], const complex float* trj)
{
	if (bart_use_gpu)
		move_gpu_nn_weights(nlinvnet->weights);

	auto nn_apply = nlinvnet_apply_op_create(nlinvnet, 1);

	assert(DIMS == N);

	int DO[2] = { N, N };
	int DI[3] = { N, N, N };

	const long* odims[2] = { img_dims, col_dims };
	const long* idims[3] = { ksp_dims, pat_dims, trj_dims };

	complex float* dst[2] = { img, col };
	const complex float* src[5] = { ksp, pat, trj };

	const struct nlop_s* nlop_apply = nlop_optimize_graph(nlop_clone(nn_apply->nlop));

	nn_debug(DP_INFO, nn_apply);
	unsigned long batch_flags = md_nontriv_dims(N, img_dims)
				    & ~md_nontriv_dims(N, nn_generic_codomain(nn_apply, 0, "img")->dims);

	nn_free(nn_apply);

	nlop_unset_derivatives(nlop_apply);
	nlop_generic_apply_loop_sameplace(nlop_apply, batch_flags, 2, DO, odims, dst, 3, DI, idims, src, nlinvnet->weights->tensors[0]);

	nlop_free(nlop_apply);

	if (nlinvnet->normalize_rss) {

		long col_dims2[N];
		md_select_dims(N, ~COIL_FLAG, col_dims2, col_dims);

		complex float* tmp = md_alloc_sameplace(N, col_dims2, CFL_SIZE, img);

		md_zrss(N, col_dims, COIL_FLAG, tmp, col);
		md_zmul2(N, img_dims, MD_STRIDES(N, img_dims, CFL_SIZE), img, MD_STRIDES(N, img_dims, CFL_SIZE), img, MD_STRIDES(N, col_dims2, CFL_SIZE), tmp);

		md_free(tmp);
	}
}

