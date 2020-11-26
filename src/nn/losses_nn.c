/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */


#include "iter/italgos.h"

#include "num/iovec.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"

#include "nn/losses.h"

#include "nn/chain.h"
#include "nn/nn.h"

#include "losses_nn.h"

nn_t nn_loss_mse_append(nn_t network, int o, const char* oname, unsigned long mean_dims)
{
	int nlop_o = nn_get_out_arg_index(network, o, oname);

	auto nlop = nlop_clone(nn_get_nlop(network));
	auto iov = nlop_generic_codomain(nlop, nlop_o);
	nlop = nlop_chain2_swap_FF(nlop, nlop_o, nlop_mse_create(iov->N, iov->dims, mean_dims), 0);
	nlop = nlop_shift_output_F(nlop, nlop_o, 0);

	auto result = nn_from_nlop_F(nlop);

	nn_clone_args(result, network);
	nn_free(network);

	result = nn_shift_input_index_F(result, 0, nn_get_nr_in_args(result) - 1);

	result = nn_set_out_type_F(result, o, oname, OUT_OPTIMIZE);
	return result;
}

nn_t nn_loss_cce_append(nn_t network, int o, const char* oname, unsigned long scaling_flag)
{
	int nlop_o = nn_get_out_arg_index(network, o, oname);

	auto nlop = nlop_clone(nn_get_nlop(network));
	auto iov = nlop_generic_codomain(nlop, nlop_o);
	nlop = nlop_chain2_swap_FF(nlop, nlop_o, nlop_cce_create(iov->N, iov->dims, scaling_flag), 0);
	nlop = nlop_shift_output_F(nlop, nlop_o, 0);

	auto result = nn_from_nlop_F(nlop);

	nn_clone_args(result, network);
	nn_free(network);

	result = nn_shift_input_index_F(result, 0, nn_get_nr_in_args(result) - 1);

	result = nn_set_out_type_F(result, o, oname, OUT_OPTIMIZE);
	return result;
}

nn_t nn_loss_dice_append(nn_t network, int o, const char* oname, unsigned long label_flag, unsigned long mean_flag, float weighting_exponent, bool square_denominator)
{
	int nlop_o = nn_get_out_arg_index(network, o, oname);

	auto nlop = nlop_clone(nn_get_nlop(network));
	auto iov = nlop_generic_codomain(nlop, nlop_o);
	nlop = nlop_chain2_swap_FF(nlop, nlop_o, nlop_dice_create(iov->N, iov->dims, label_flag, mean_flag, weighting_exponent, square_denominator), 0);
	nlop = nlop_shift_output_F(nlop, nlop_o, 0);

	auto result = nn_from_nlop_F(nlop);

	nn_clone_args(result, network);
	nn_free(network);

	result = nn_shift_input_index_F(result, 0, nn_get_nr_in_args(result) - 1);

	result = nn_set_out_type_F(result, o, oname, OUT_OPTIMIZE);
	return result;
}
