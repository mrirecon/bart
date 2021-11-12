/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "iter/italgos.h"
#include "misc/misc.h"

#include "num/iovec.h"
#include "num/multind.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/const.h"

#include "nn/nn.h"
#include "const.h"

/**
 * Set one input of nn_t to a constant array, i.e. the resulting nn_t has one input less, and free nn_t
 *
 * @param op nn_t struct (will be freed)
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 * @param N no. of dimensions of the constant array
 * @param dims dimensions of constant array (must coincide with dimensions of input)
 * @param strs of array in mem
 * @param copy if true: store a copy of the array in the nn_t (nlop); else: only store a pointer
 * @param in constant input array
 *
 * @returns nn_t with one input set to the constant input array
 */
nn_t nn_set_input_const_F2(nn_t op, int i, const char* iname, int N, const long dims[N], const long strs[N], _Bool copy, const _Complex float* in)
{
	i = nn_get_in_arg_index(op, i, iname);
	auto result = nn_from_nlop_F(nlop_set_input_const2(nn_get_nlop(op), i, N, dims, strs, copy, in));

	for (int j = 0, jp = 0; j < nn_get_nr_in_args(result); j++) {

		if (i == (int)j) jp++;
		nn_clone_arg_i_from_i(result, j, op, jp);
		jp++;
	}

	for (int j = 0; j < nn_get_nr_out_args(result); j++)
		nn_clone_arg_o_from_o(result, j, op, j);

	nn_free(op);

	return result;
}

/**
 * Set one input of nn_t to a constant array, i.e. the resulting nn_t has one input less, and free nn_t
 *
 * @param op nn_t struct (will be freed)
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 * @param N no. of dimensions of the constant array
 * @param dims dimensions of constant array (must coincide with dimensions of input)
 * @param copy if true: store a copy of the array in the nn_t (nlop); else: only store a pointer
 * @param in constant input array
 *
 * @returns nn_t with one input set to the constant input array
 */
nn_t nn_set_input_const_F(nn_t op, int i, const char* iname, int N, const long dims[N], _Bool copy, const _Complex float* in)
{
	i = nn_get_in_arg_index(op, i, iname);
	auto result = nn_from_nlop_F(nlop_set_input_const(nn_get_nlop(op), i, N, dims, copy, in));

	for (int j = 0, jp = 0; j < nn_get_nr_in_args(result); j++) {

		if (i == (int)j) jp++;
		nn_clone_arg_i_from_i(result, j, op, jp);
		jp++;
	}

	for (int j = 0; j < nn_get_nr_out_args(result); j++)
		nn_clone_arg_o_from_o(result, j, op, j);

	nn_free(op);

	return result;
}

/**
 * Delete one output of nn_t, i.e. the resulting nn_t has one output less, and free nn_t
 *
 * @param op nn_t struct (will be freed)
 * @param o input index (ignored if oname != NULL)
 * @param oname name of input
 *
 * @returns nn_t with one output less
 */
nn_t nn_del_out_F(nn_t op, int o, const char* oname)
{
	o = nn_get_out_arg_index(op, o, oname);
	auto result = nn_from_nlop_F(nlop_del_out(nn_get_nlop(op), o));

	for (int j = 0, jp = 0; j < nn_get_nr_out_args(result); j++) {

		if (o == (int)j) jp++;
		nn_clone_arg_o_from_o(result, j, op, jp);
		jp++;
	}

	for (int i = 0; i < nn_get_nr_in_args(result); i++)
		nn_clone_arg_i_from_i(result, i, op, i);

	nn_free(op);

	return result;
}

/**
 * Delete all outputs of nn_t which are used to update floating mean/variance of batch normalization layers, and free nn_t
 * This function can be used to extract the forward operator needed for inference
 *
 * @param op nn_t struct (will be freed)
 *
 * @returns nn_t without outputs for batch normalization
 */
nn_t nn_del_out_bn_F(nn_t op)
{
	auto result = op;
	for (int o = nn_get_nr_out_args(op) - 1; o >= 0; o--)
		if (OUT_BATCHNORM == result->out_types[o])
			result = nn_del_out_F(result, nn_get_out_index_from_arg_index(result, o), nn_get_out_name_from_arg_index(result, o, false));

	return result;
}

/**
 * Set one input of nn_t to a constant array but keep the input as dummy input (when applied the array for the input will be ignored), and free nn_t
 *
 * @param op nn_t struct (will be freed)
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 * @param N no. of dimensions of the constant array
 * @param dims dimensions of constant array (must coincide with dimensions of input)
 * @param copy if true: store a copy of the array in the nn_t (nlop); else: only store a pointer
 * @param in constant input array
 *
 * @returns nn_t with one input set to the constant input array
 */
nn_t nn_ignore_input_F(nn_t op, int i, const char* iname, int N, const long dims[N], _Bool copy, const _Complex float* in)
{
	i = nn_get_in_arg_index(op, i, iname);

	long dims2[N];
	md_copy_dims(N, dims2, nlop_generic_domain(nn_get_nlop(op), i)->dims);

	assert(md_check_equal_dims(N, dims2, dims, md_nontriv_dims(N, dims)));

	auto nlop = nlop_set_input_const2(nn_get_nlop(op), i, N, dims2, MD_STRIDES(N, dims, sizeof(complex float)), copy, in);
	nlop = nlop_combine_FF(nlop_del_out_create(N, dims2), nlop);
	nlop = nlop_shift_input_F(nlop, i, 0);
	auto result = nn_from_nlop_F(nlop);

	for (int j = 0; j < nn_get_nr_in_args(result); j++)
		nn_clone_arg_i_from_i(result, j, op, j);

	for (int j = 0; j < nn_get_nr_out_args(result); j++)
		nn_clone_arg_o_from_o(result, j, op, j);

	nn_free(op);

	return result;
}

