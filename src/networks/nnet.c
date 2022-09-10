/* Copyright 2021-2022. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 **/

#include <assert.h>
#include <stdbool.h>
#include <math.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "num/ops.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "iter/iter6.h"
#include "iter/monitor_iter6.h"
#include "iter/batch_gen.h"
#include "iter/italgos.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"
#include "nlops/const.h"
#include "nlops/someops.h"

#include "nn/misc.h"
#include "nn/nn.h"
#include "nn/const.h"
#include "nn/weights.h"
#include "nn/init.h"
#include "nn/losses.h"
#include "nn/chain.h"
#include "nn/weights.h"
#include "nn/layers.h"

#include "networks/unet.h"
#include "networks/losses.h"
#include "networks/cnn.h"


#include "nnet.h"

struct nnet_s nnet_init = {

	.network = NULL,

	.weights = NULL,
	.train_conf = NULL,

	.train_loss = NULL,
	.valid_loss = NULL,

	.low_mem = false,
	.gpu = false,

	.get_no_odims = NULL,
	.get_odims = NULL,

	.graph_file = NULL,

	.N_segm_labels = -1,
};

static unsigned int get_no_odims_mnist(const struct nnet_s* config, unsigned int NI, const long idims[NI])
{
	UNUSED(config);
	UNUSED(idims);

	return 2;
}

static void get_odims_mnist(const struct nnet_s* config, unsigned int NO, long odims[NO], unsigned int NI, const long idims[NI])
{
	UNUSED(config);

	odims[0] = 10;
	odims[1] = idims[2];
}

void nnet_init_mnist_default(struct nnet_s* nnet)
{
	if (NULL == nnet->train_conf) {

		PTR_ALLOC(struct iter6_adadelta_conf, train_conf);
		*train_conf = iter6_adadelta_conf_defaults;

		nnet->train_conf = CAST_UP(PTR_PASS(train_conf));
	}

	PTR_ALLOC(struct network_s, network);
	*network = network_mnist_default;

	nnet->network = PTR_PASS(network);

	nnet->get_no_odims = get_no_odims_mnist;
	nnet->get_odims = get_odims_mnist;

	if (NULL == nnet->train_loss) {

		nnet->train_loss =  &loss_option;
		nnet->train_loss->weighting_cce = 1.;
	}

	if (NULL == nnet->valid_loss)
		nnet->valid_loss =  &loss_classification_valid;
}

static unsigned int get_no_odims_segm(const struct nnet_s* config, unsigned int NI, const long idims[NI])
{
	UNUSED(config);
	UNUSED(idims);

	return NI;
}

static void get_odims_segm(const struct nnet_s* config, unsigned int NO, long odims[NO], unsigned int NI, const long idims[NI])
{
	UNUSED(config);

	assert(NO == NI);

	md_copy_dims(NO, odims, idims);

	odims[0] = config->N_segm_labels;
}

void nnet_init_unet_segm_default(struct nnet_s* nnet, long N_segm_labels)
{
	PTR_ALLOC(struct iter6_adam_conf, train_conf);
	*train_conf = iter6_adam_conf_defaults;

	nnet->train_conf = CAST_UP(PTR_PASS(train_conf));

	PTR_ALLOC(struct network_unet_s, network);
	*network = network_unet_default_segm;

	nnet->network = CAST_UP(PTR_PASS(network));

	nnet->get_no_odims = get_no_odims_segm;
	nnet->get_odims = get_odims_segm;

	if (NULL == nnet->train_loss) {

		nnet->train_loss =  &loss_option;
		nnet->train_loss->weighting_cce = 1.;
	}

	if (NULL == nnet->valid_loss)
		nnet->valid_loss =  &loss_classification_valid;

	nnet->N_segm_labels = N_segm_labels;
}


static nn_t nnet_network_create(const struct nnet_s* config, unsigned int NO, const long odims[NO], unsigned int NI, const long idims[NI], enum NETWORK_STATUS status)
{
	return network_create(config->network, NO, odims, NI, idims, status);
}


static nn_t nnet_train_create(const struct nnet_s* config, unsigned int NO, const long odims[NO], unsigned int NI, const long idims[NI])
{
	auto train_op = nnet_network_create(config, NO, odims, NI, idims, STAT_TRAIN);

	auto loss = train_loss_create(config->train_loss, NO, odims);

	train_op = nn_chain2_FF(train_op, 0, NULL, loss, 0, NULL);

	return train_op;
}

static nn_t nnet_apply_op_create(const struct nnet_s* config, unsigned int NO, const long odims[NO], unsigned int NI, const long idims[NI])
{
	auto nn_apply = nnet_network_create(config, NO, odims, NI, idims, STAT_TEST);

	return nn_get_wo_weights_F(nn_apply, config->weights, false);
}

static nn_t nnet_valid_create(const struct nnet_s* config, const struct nn_weights_s* vf)
{
	auto iov_o = vf->iovs[1];
	auto iov_i = vf->iovs[0];

	auto valid_loss = nnet_network_create(config, iov_o->N, iov_o->dims, iov_i->N, iov_i->dims, STAT_TEST);
	valid_loss = nn_del_out_bn_F(valid_loss);

	auto loss = val_measure_create(config->valid_loss, iov_o->N, iov_o->dims);
	valid_loss = nn_chain2_FF(valid_loss, 0, NULL, loss, 0, NULL);

	valid_loss = nn_ignore_input_F(valid_loss, 0, NULL, iov_o->N, iov_o->dims, true, vf->tensors[1]);
	valid_loss = nn_ignore_input_F(valid_loss, 1, NULL, iov_i->N, iov_i->dims, true, vf->tensors[0]);

	return valid_loss;
}



void train_nnet(struct nnet_s* config,
		unsigned int NO, const long odims[NO], const complex float* out,
		unsigned int NI, const long idims[NI], const complex float* in,
		long Nb, const struct nn_weights_s* valid_files)
{
	long Nt = odims[NO - 1];
	assert(Nt == idims[NI - 1]);

	long bodims[NO];
	long bidims[NI];

	md_copy_dims(NO, bodims, odims);
	md_copy_dims(NI, bidims, idims);

	bodims[NO - 1] = Nb;
	bidims[NI - 1] = Nb;

	auto nn_train = nnet_train_create(config, NO, bodims, NI, bidims);

	debug_printf(DP_INFO, "Train Network\n");
	nn_debug(DP_INFO, nn_train);

	if (NULL == config->weights) {

		config->weights = nn_weights_create_from_nn(nn_train);
		nn_init(nn_train, config->weights);
	}

	if (config->gpu)
		move_gpu_nn_weights(config->weights);


	const complex float* train_data[] = {out, in};
	const long* bat_dims[] = { bodims, bidims };
	const long* tot_dims[] = { odims, idims };
	auto batch_generator = batch_gen_create_from_iter(config->train_conf, 2, (const int[2]){NO, NI}, bat_dims, tot_dims, train_data, 0);

	//setup for iter algorithm
	int II = nn_get_nr_in_args(nn_train);
	int OO = nn_get_nr_out_args(nn_train);

	float* src[II];

	for (int i = 0; i < config->weights->N; i++) {

		auto iov_weight = config->weights->iovs[i];
		auto iov_train_op = nlop_generic_domain(nn_get_nlop(nn_train), i + 2);

		assert(md_check_equal_dims(iov_weight->N, iov_weight->dims, iov_train_op->dims, ~0));

		src[i + 2] = (float*)config->weights->tensors[i];
	}

	enum IN_TYPE in_type[II];
	const struct operator_p_s* projections[II];
	enum OUT_TYPE out_type[OO];

	nn_get_in_types(nn_train, II, in_type);
	nn_get_out_types(nn_train, OO, out_type);

	for (int i = 0; i < 2; i++) {

		src[i] = NULL;
		in_type[i] = IN_BATCH_GENERATOR;
	}

	for (int i = 0; i < II; i++)
		projections[i] = nn_get_prox_op_arg_index(nn_train, i);

	int num_monitors = 0;
	const struct monitor_value_s* value_monitors[1];

	if (NULL != valid_files) {

		auto nn_validation_loss = nnet_valid_create(config, valid_files);
		const char* val_names[nn_get_nr_out_args(nn_validation_loss)];

		for (int i = 0; i < nn_get_nr_out_args(nn_validation_loss); i++)
			val_names[i] = nn_get_out_name_from_arg_index(nn_validation_loss, i, false);

		value_monitors[num_monitors] = monitor_iter6_nlop_create(nn_get_nlop(nn_validation_loss), false, nn_get_nr_out_args(nn_validation_loss), val_names);

		nn_free(nn_validation_loss);

		num_monitors += 1;
	}

	struct monitor_iter6_s* monitor = monitor_iter6_create(true, true, num_monitors, value_monitors);

	iter6_by_conf(config->train_conf, nn_get_nlop(nn_train), II, in_type, projections, src, OO, out_type, Nb, Nt / Nb, batch_generator, monitor);

	if (NULL != config->graph_file)
		nn_export_graph(config->graph_file, nn_train);

	nn_free(nn_train);

	nlop_free(batch_generator);

	monitor_iter6_free(monitor);
}


void apply_nnet(	const struct nnet_s* config,
			unsigned int NO, const long odims[NO], complex float* out,
			unsigned int NI, const long idims[NI], const complex float* in)
{
	if (config->gpu)
		move_gpu_nn_weights(config->weights);

	auto nnet = nnet_apply_op_create(config, NO, odims, NI, idims);

	static bool export = true;
	if (export && NULL != config->graph_file) {

		nn_export_graph(config->graph_file, nnet);
		export = false;
	}

	complex float* out_tmp = md_alloc_sameplace(NO, odims, CFL_SIZE, config->weights->tensors[0]);
	complex float* in_tmp = md_alloc_sameplace(NI, idims, CFL_SIZE, config->weights->tensors[0]);

	md_copy(NI, idims, in_tmp, in, CFL_SIZE);

	complex float* args[2];

	args[0] = out_tmp;
	args[1] = in_tmp;

	nlop_generic_apply_select_derivative_unchecked(nn_get_nlop(nnet), 2, (void**)args, 0, 0);

	md_copy(NO, odims, out, out_tmp, CFL_SIZE);

	nn_free(nnet);

	md_free(in_tmp);
	md_free(out_tmp);
}


void apply_nnet_batchwise(	const struct nnet_s* config,
				unsigned int NO, const long odims[NO], complex float* out,
				unsigned int NI, const long idims[NI], const complex float* in,
				long Nb)
{
	long Nt = odims[NO - 1];

	while (0 < Nt) {

		long odims1[NO];
		long idims1[NI];

		md_copy_dims(NI, idims1, idims);
		md_copy_dims(NO, odims1, odims);

		long Nb_tmp = MIN(Nt, Nb);

		odims1[NO - 1] = Nb_tmp;
		idims1[NI - 1] = Nb_tmp;

		apply_nnet(config, NO, odims1, out, NI, idims1, in);

		out += md_calc_size(NO, odims1);
		in += md_calc_size(NI, idims1);

		Nt -= Nb_tmp;
	}
}


extern void eval_nnet(	struct nnet_s* nnet,
			unsigned int NO, const long odims[NO], const _Complex float* out,
			unsigned int NI, const long idims[NI], const _Complex float* in,
			long Nb)
{
	complex float* tmp_out = md_alloc(NO, odims, CFL_SIZE);

	auto loss = val_measure_create(nnet->valid_loss, NO, odims);
	int N = nn_get_nr_out_args(loss);

	complex float losses[N];

	md_clear(1, MD_DIMS(N), losses, CFL_SIZE);

	apply_nnet_batchwise(nnet, NO, odims, tmp_out, NI, idims, in, Nb);

	complex float* args[N + 2];

	for (int i = 0; i < N; i++)
		args[i] = losses + i;

	args[N] = tmp_out;
	args[N + 1] = (complex float*)out;

	nlop_generic_apply_select_derivative_unchecked(nn_get_nlop(loss), N + 2, (void**)args, 0, 0);

	for (int i = 0; i < N ; i++)
		debug_printf(DP_INFO, "%s: %e\n", nn_get_out_name_from_arg_index(loss, i, false), crealf(losses[i]));

	nn_free(loss);

	print_confusion_matrix(NO, odims, 0, tmp_out, out);

	md_free(tmp_out);
}
