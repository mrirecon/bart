/* Copyright 2022. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2022-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 **/

#include <assert.h>
#include <stdio.h>

#include "iter/italgos.h"

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/io.h"

#include "nn/nn.h"
#include "nn/weights.h"
#include "nn/tf_wrapper.h"

#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/multind.h"

#include "nlops/nlop.h"

#include "networks/cnn.h"

#include "tf.h"


DEF_TYPEID(network_tensorflow_s);

struct network_tensorflow_s network_tensorflow_default = {

	.super.TYPEID = &TYPEID2(network_tensorflow_s),
	.super.create = network_tensorflow_create,
	.super.low_mem = false,
	.super.norm = NORM_NONE,
	.super.norm_batch_flag = MD_BIT(4),
	.super.debug = false,
	.super.prefix = NULL,
	.super.residual = false,
	.super.bart_to_channel_first = false,

	.model_path = NULL,
	.tf_graph = NULL,
};



nn_t network_tensorflow_create(const struct network_s* _config, int NO, const long odims[NO], int NI, const long idims[NI], enum NETWORK_STATUS /*status*/)
{
	auto config = CAST_DOWN(network_tensorflow_s, _config);

	if (NULL == config->tf_graph) {

		config->tf_graph = tf_shared_graph_create(config->model_path, NULL);

		if (NULL == tf_shared_graph_get_init_path(config->tf_graph))
			debug_printf(DP_WARN, "All weights of TensorFlow v1 graph are initialized with zeros (if not loaded)!\n");
	}

	tf_shared_graph_set_batch_size(config->tf_graph, idims[NI - 1]);

	const struct nlop_s* nlop_net = nlop_tf_shared_create(config->tf_graph);

	assert(iovec_check(nlop_generic_domain(nlop_net, 0), NI, idims, MD_STRIDES(NI, idims, CFL_SIZE)));
	assert(iovec_check(nlop_generic_codomain(nlop_net, 0), NO, odims, MD_STRIDES(NI, odims, CFL_SIZE)));

	nn_t nn_net = nn_from_nlop_F(nlop_net);

	assert(1 == nn_get_nr_out_args(nn_net));
	assert(1 <= nn_get_nr_in_args(nn_net));

	int counter = 0;
	char wgh_name[30];

	nn_weights_t tf_init = NULL;

	if (NULL != tf_shared_graph_get_init_path(config->tf_graph)) {

		io_reserve_input(tf_shared_graph_get_init_path(config->tf_graph));

		tf_init = load_nn_weights(tf_shared_graph_get_init_path(config->tf_graph));
	}


	while (1 < nn_get_nr_unnamed_in_args(nn_net)) {

		nn_net = nn_set_in_type_F(nn_net, 1, NULL, IN_OPTIMIZE);

		if (NULL == tf_init)
			nn_net = nn_set_initializer_F(nn_net, 1, NULL, init_const_create(0));
		else
			nn_net = nn_set_initializer_F(nn_net, 1, NULL, init_array_create(tf_init->iovs[counter]->N, tf_init->iovs[counter]->dims, tf_init->tensors[counter]));

		snprintf(wgh_name, 30, "tf_weight_%d", counter++);

		nn_net = nn_set_input_name_F(nn_net, 1, wgh_name);
	}

	debug_printf(DP_DEBUG1, "TensorFlow Network created:\n");
	nn_debug(DP_DEBUG1, nn_net);

	return nn_net;
}
