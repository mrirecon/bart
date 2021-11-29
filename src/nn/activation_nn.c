/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include "nlops/nlop.h"
#include "nlops/chain.h"

#include "nn/activation.h"

#include "nn/nn.h"
#include "nn/init.h"

#include "activation_nn.h"


nn_t nn_append_activation(nn_t network, int o, const char* oname, enum ACTIVATION activation)
{
	o = nn_get_out_arg_index(network, o, oname);
	auto nlop = append_activation(nlop_clone(nn_get_nlop(network)), o, activation);
	auto result = nn_from_nlop_F(nlop);
	nn_clone_args(result, network);
	nn_free(network);


	return result;
}

nn_t nn_append_activation_bias(nn_t network, int o, const char* oname, const char* bname, enum ACTIVATION activation, unsigned long bflag)
{
	o = nn_get_out_arg_index(network, o, oname);
	auto nlop = append_activation_bias(nlop_clone(nn_get_nlop(network)), o, activation, bflag);
	auto result = nn_from_nlop_F(nlop);
	nn_clone_args(result, network);

	result = nn_set_in_type_F(result, -1, NULL, IN_OPTIMIZE);
	result = nn_set_initializer_F(result, -1, NULL, init_const_create(0.));

	if (NULL != bname)
		result = nn_set_input_name_F(result, -1, bname);

	nn_free(network);

	return result;
}