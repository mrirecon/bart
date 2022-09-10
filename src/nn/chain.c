/* Copyright 2021-2022. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 **/

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "misc/misc.h"

#include "iter/prox.h"
#include "num/ops_p.h"

#include "nn/init.h"
#include "num/iovec.h"
#include "num/multind.h"

#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"

#include "nn/nn.h"
#include "chain.h"

/**
 * Add real value constraint to intput of nn_t
 *
 * @param op nn_t struct
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 * @returns nn_t with real input
 */
nn_t nn_real_input(nn_t op, int i, const char* iname)
{
	auto iov = nn_generic_domain(op, i, iname);

	i = nn_get_in_arg_index(op, i, iname);

	auto rvc = nlop_from_linop_F(linop_zreal_create(iov->N, iov->dims));
	auto nlop_result = nlop_chain2(rvc, 0, nn_get_nlop(op), i);
	nlop_free(rvc);
	nlop_result = nlop_shift_input_F(nlop_result, i, nlop_get_nr_in_args(nlop_result) - 1);

	auto result = nn_from_nlop_F(nlop_result);

	for (int i = 0; i < nn_get_nr_in_args(result); i++)
		nn_clone_arg_i_from_i(result, i, op, i);
	for (int i = 0; i < nn_get_nr_out_args(result); i++)
		nn_clone_arg_o_from_o(result, i, op, i);
	return result;
}

/**
 * Add real value constraint to output of nn_t
 *
 * @param op nn_t struct
 * @param o output index (ignored if oname != NULL)
 * @param oname name of output
 * @returns nn_t with real input
 */
nn_t nn_real_output(nn_t op, int o, const char* oname)
{
	auto iov = nn_generic_codomain(op, o, oname);

	o = nn_get_out_arg_index(op, o, oname);

	auto rvc = nlop_from_linop_F(linop_zreal_create(iov->N, iov->dims));
	auto nlop_result = nlop_chain2(nn_get_nlop(op), o, rvc, 0);
	nlop_free(rvc);
	nlop_result = nlop_shift_output_F(nlop_result, o, 0);

	auto result = nn_from_nlop_F(nlop_result);

	for (int i = 0; i < nn_get_nr_in_args(result); i++)
		nn_clone_arg_i_from_i(result, i, op, i);
	for (int i = 0; i < nn_get_nr_out_args(result); i++)
		nn_clone_arg_o_from_o(result, i, op, i);
	return result;
}

/**
 * Add real value constraint to intput of nn_t
 *
 * @param op nn_t struct
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 * @returns nn_t with real input
 */
nn_t nn_real_input_F(nn_t op, int i, const char* iname)
{
	auto result = nn_real_input(op, i, iname);
	nn_free(op);
	return result;
}

/**
 * Add real value constraint to output of nn_t
 *
 * @param op nn_t struct
 * @param o output index (ignored if oname != NULL)
 * @param oname name of output
 * @returns nn_t with real input
 */
nn_t nn_real_output_F(nn_t op, int o, const char* oname)
{
	auto result = nn_real_output(op, o, oname);
	nn_free(op);
	return result;
}


/**
 * Reshape output of nn_t
 *
 * @param op nn_t struct
 * @param o output index (ignored if oname != NULL)
 * @param oname name of output
 * @param N number of new output dimensions
 * @param odims new output dimensions
 *
 * @returns nn_t with reshaped output
 */
nn_t nn_reshape_out(nn_t op, int o, const char* oname, int N, const long odims[N])
{
	o = nn_get_out_arg_index(op, o, oname);
	auto result = nn_from_nlop_F(nlop_reshape_out(nn_get_nlop(op), o, N, odims));

	for (int i = 0; i < nn_get_nr_in_args(result); i++)
		nn_clone_arg_i_from_i(result, i, op, i);
	for (int i = 0; i < nn_get_nr_out_args(result); i++)
		nn_clone_arg_o_from_o(result, i, op, i);

	return result;
}

/**
 * Reshape input of nn_t
 *
 * @param op nn_t struct
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 * @param N number of new input dimensions
 * @param idims new input dimensions
 *
 * @returns nn_t with reshaped input
 */
nn_t nn_reshape_in(nn_t op, int i, const char* iname, int N, const long idims[N])
{
	i = nn_get_in_arg_index(op, i, iname);
	auto result = nn_from_nlop_F(nlop_reshape_in(nn_get_nlop(op), i, N, idims));

	for (int i = 0; i < nn_get_nr_in_args(result); i++)
		nn_clone_arg_i_from_i(result, i, op, i);
	for (int i = 0; i < nn_get_nr_out_args(result); i++)
		nn_clone_arg_o_from_o(result, i, op, i);

	auto iov = nlop_generic_domain(op->nlop, i);

	auto init_tmp = init_reshape_create(iov->N, iov->dims, result->initializers[i]);
	initializer_free(result->initializers[i]);
	result->initializers[i] = init_tmp;

	if (NULL != result->prox_ops[i])
		result->prox_ops[i] = operator_p_reshape_out_F(operator_p_reshape_in_F(result->prox_ops[i], N, idims), N, idims);

	return result;
}

/**
 * Reshape output of nn_t and free nn_t
 *
 * @param op nn_t struct (will be freed)
 * @param o output index (ignored if oname != NULL)
 * @param oname name of output
 * @param N number of new output dimensions
 * @param odims new output dimensions
 * @param clear clear acquired k-space
 *
 * @returns nn_t with reshaped output
 */
nn_t nn_reshape_out_F(nn_t op, int o, const char* oname, int NO, const long odims[NO])
{
	auto result = nn_reshape_out(op, o, oname, NO, odims);
	nn_free(op);
	return result;
}

/**
 * Reshape input of nn_t and free nn_t
 *
 * @param op nn_t struct (will be freed)
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 * @param N number of new input dimensions
 * @param idims new input dimensions
 *
 * @returns nn_t with reshaped input
 */
nn_t nn_reshape_in_F(nn_t op, int i, const char* iname, int NI, const long idims[NI])
{
	auto result = nn_reshape_in(op, i, iname, NI, idims);
	nn_free(op);
	return result;
}

/**
 * Reshape input of nn_t to have an additional singleton dimension and free nn_t
 * {dims[0], ..., dims[N-1]} -> {dims[0], ..., dims[N-1], 1}
 *
 * @param op nn_t struct (will be freed)
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 *
 * @returns nn_t with reshaped input
 */
nn_t nn_append_singleton_dim_in_F(nn_t op, int i, const char* iname)
{
	auto iov = nn_generic_domain(op, i, iname);
	long dims[iov->N + 1];
	md_copy_dims(iov->N, dims, iov->dims);
	dims[iov->N] = 1;

	return nn_reshape_in_F(op, i, iname, iov->N + 1, dims);
}

/**
 * Reshape output of nn_t to have an additional singleton dimension and free nn_t
 * {dims[0], ..., dims[N-1]} -> {dims[0], ..., dims[N-1], 1}
 *
 * @param op nn_t struct (will be freed)
 * @param o input index (ignored if oname != NULL)
 * @param oname name of output
 *
 * @returns nn_t with reshaped output
 */
nn_t nn_append_singleton_dim_out_F(nn_t op, int o, const char* oname)
{
	auto iov = nn_generic_codomain(op, o, oname);
	long dims[iov->N + 1];
	md_copy_dims(iov->N, dims, iov->dims);
	dims[iov->N] = 1;

	return nn_reshape_out_F(op, o, oname, iov->N + 1, dims);
}

/**
 * Permute inputs of nn_t
 * Input i of result = input perm[i] of op
 *
 * @param op nn_t struct
 * @param I2 no. of inputs of op (either total or unnamed)
 * @param perm permutation of inputs
 *
 * @returns nn_t with permuted inputs
 *
 * @note If I2 equals the number of unnamed inputs, only the unnamed inputs are permuted, i.e. the named ones stay at their positions
 */
nn_t nn_permute_inputs(nn_t op, int I2, const int perm[I2])
{
	assert((nn_get_nr_in_args(op) == I2) || (nn_get_nr_unnamed_in_args(op) == I2));

	int II = nn_get_nr_in_args(op);
	int nperm[II];

	for (int i = 0; i < II; i++)
		nperm[i] = i;

	if (nn_get_nr_unnamed_in_args(op) == I2) {

		for(int i = 0; i < nn_get_nr_unnamed_in_args(op); i++)
			nperm[nn_get_in_arg_index(op, i, NULL)] = nn_get_in_arg_index(op, perm[i], NULL);
	} else {

		for(int i = 0; i < nn_get_nr_in_args(op); i++)
			nperm[i] = perm[i];
	}

	auto result = nn_from_nlop_F(nlop_permute_inputs(nn_get_nlop(op), II, nperm));

	for (int i = 0; i < nn_get_nr_in_args(result); i++)
		nn_clone_arg_i_from_i(result, i, op, nperm[i]);
	for (int i = 0; i < nn_get_nr_out_args(result); i++)
		nn_clone_arg_o_from_o(result, i, op, i);

	return result;
}

/**
 * Permute outputs of nn_t
 * Output o of result = output perm[o] of op
 *
 * @param op nn_t struct
 * @param O2 no. of outputs of op (either total or unnamed)
 * @param perm permutation of outputs
 *
 * @returns nn_t with permuted outputs
 *
 * @note If O2 equals the number of unnamed outputs, only the unnamed outputs are permuted, i.e. the named ones stay at their positions
 */
nn_t nn_permute_outputs(nn_t op, int O2, const int perm[O2])
{
	assert((nn_get_nr_out_args(op) == O2) || (nn_get_nr_unnamed_out_args(op) == O2));

	int OO = nn_get_nr_out_args(op);
	int nperm[OO];

	for (int i = 0; i < OO; i++)
		nperm[i] = i;

	if (nn_get_nr_unnamed_out_args(op) == O2) {

		for(int i = 0; i < nn_get_nr_unnamed_out_args(op); i++)
			nperm[nn_get_out_arg_index(op, i, NULL)] = nn_get_out_arg_index(op, perm[i], NULL);
	} else {

		for(int i = 0; i < nn_get_nr_out_args(op); i++)
			nperm[i] = perm[i];
	}

	auto result = nn_from_nlop_F(nlop_permute_outputs(nn_get_nlop(op), OO, nperm));

	for (int i = 0; i < nn_get_nr_in_args(result); i++)
		nn_clone_arg_i_from_i(result, i, op, i);
	for (int i = 0; i < nn_get_nr_out_args(result); i++)
		nn_clone_arg_o_from_o(result, i, op, nperm[i]);

	return result;
}

/**
 * Permute inputs of nn_t and frees nn_t
 * Input i of result = input perm[i] of op
 *
 * @param op nn_t struct (will be freed)
 * @param I2 no of inputs (either total or unnamed)
 * @param perm permutation of inputs
 *
 * @returns nn_t with permuted inputs
 *
 * @note If I2 equals the number of unnamed inputs, only the unnamed inputs are permuted, i.e. the named ones stay at their positions
 */
nn_t nn_permute_inputs_F(nn_t op, int I2, const int perm[I2])
{
	auto result = nn_permute_inputs(op, I2, perm);
	nn_free(op);
	return result;
}

/**
 * Permute outputs of nn_t and frees nn_t
 * Output o of result = output perm[o] of op
 *
 * @param op nn_t struct (will be freed)
 * @param O2 no. of outputs of op (either total or unnamed)
 * @param perm permutation of outputs
 *
 * @returns nn_t with permuted outputs
 *
 * @note If O2 equals the number of unnamed outputs, only the unnamed outputs are permuted, i.e. the named ones stay at their positions
 */
nn_t nn_permute_outputs_F(nn_t op, int O2, const int perm[O2])
{
	auto result = nn_permute_outputs(op, O2, perm);
	nn_free(op);
	return result;
}

/**
 * Combine two nn_t's to one.
 *
 * The resulting nn_t will have a total amount of in- and outputs corresponding to the respective sum of in- and outputs of a and b.
 * The first in-/outputs correspond to a, the latter ones to b.
 * When the resulting nn_t is applied, the apply function of b is called first and the one of a afterwards, thus, if inputs of b depend on outputs of a, the result is undetermined.
 * a and b must not have mutual input names or output names
 *
 * Inputs:	ia1 ia2 ia3 ib1 ib2 ib3
 *		 |   |   |   |   |   |
 * Nlops:	(    a    ) (    b    )
 *		 |   |       |   |   |
 * Outputs:	oa1 oa2     ob1 ob2 ob3
 *
 * @param a nn_t struct
 * @param b nn_t struct
 *
 * @returns combined nn_t (a, b)
 */
nn_t nn_combine(nn_t a, nn_t b)
{
	int IIa = nn_get_nr_in_args(a);
	int IIb = nn_get_nr_in_args(b);
	int OOa = nn_get_nr_out_args(a);
	int OOb = nn_get_nr_out_args(b);

	for (int ia = 0; ia < IIa; ia++)
		for (int ib = 0; ib < IIb; ib++)
			assert(    (NULL == nn_get_in_names(a)[ia])
				|| (NULL == nn_get_in_names(b)[ib])
				|| (0 != strcmp(nn_get_in_names(a)[ia], nn_get_in_names(b)[ib])));

	for (int ia = 0; ia < OOa; ia++)
		for (int ib = 0; ib < OOb; ib++)
			assert(    (NULL == nn_get_out_names(a)[ia])
				|| (NULL == nn_get_out_names(b)[ib])
				|| (0 != strcmp(nn_get_out_names(a)[ia], nn_get_out_names(b)[ib])));

	auto result = nn_from_nlop_F(nlop_combine(nn_get_nlop(a), nn_get_nlop(b)));

	for (int i = 0; i < IIa; i++)
		nn_clone_arg_i_from_i(result, i, a, i);
	for (int i = 0; i < IIb; i++)
		nn_clone_arg_i_from_i(result, IIa + i, b, i);

	for (int i = 0; i < OOa; i++)
		nn_clone_arg_o_from_o(result, i, a, i);
	for (int i = 0; i < OOb; i++)
		nn_clone_arg_o_from_o(result, OOa + i, b, i);

	return result;
}

/**
 * Combine two nn_t's to one and free the former ones
 *
 * The resulting nn_t will have a total amount of in- and outputs corresponding to the respective sum of in- and outputs of a and b.
 * The first in-/outputs correspond to a, the latter ones to b.
 * When the resulting nn_t is applied, the apply function of b is called first and the one of a afterwards, thus, if inputs of b depend on outputs of a, the result is undetermined.
 * a and b must not have mutual input names or output names
 *
 * Inputs:	ia1 ia2 ia3 ib1 ib2 ib3
 *		 |   |   |   |   |   |
 * Nlops:	(    a    ) (    b    )
 *		 |   |       |   |   |
 * Outputs:	oa1 oa2     ob1 ob2 ob3
 *
 * @param a nn_t struct (will be freed)
 * @param b nn_t struct (will be freed)
 *
 * @returns combined nn_t (a, b)
 */
nn_t nn_combine_FF(nn_t a, nn_t b)
{
	auto result = nn_combine(a, b);
	nn_free(a);
	nn_free(b);
	return result;
}

/**
 * Link an output of nn_t op in one of its inputs
 *
 * The output must be computed before the input is accessed (c.f. nn_combine).
 *
 *		  |  |
 * Inputs:	 i1 i2 i3--<-
 * Nlops:	(    op   ) |
 * Outputs:	 o1 o2-->---
 *		  |
 *
 * @param op nn_t struct
 * @param o output index (ignored if oname != NULL)
 * @param oname name of output
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 *
 * @returns linked nn_t
 */
nn_t nn_link(nn_t op, int o, const char* oname, int i, const char* iname)
{
	int OO = nn_get_nr_out_args(op) - 1;
	int II = nn_get_nr_in_args(op) - 1;

	o = nn_get_out_arg_index(op, o, oname);
	i = nn_get_in_arg_index(op, i, iname);

	auto result = nn_from_nlop_F(nlop_link(nn_get_nlop(op), o, i));

	for (int ii = 0, ip = 0; ii < II; ii++){

		if (ii == i)
			ip++;
		nn_clone_arg_i_from_i(result, ii, op, ip);
		ip++;
	}

	for (int ii = 0, ip = 0; ii < OO; ii++){

		if (ii == o)
			ip++;
		nn_clone_arg_o_from_o(result, ii, op, ip);
		ip++;
	}

	return result;
}

/**
 * Link an output of nn_t in one of its inputs and frees nn_t
 *
 * The output must be computed before the input is accessed (c.f. nn_combine).
 *
 *		  |  |
 * Inputs:	 i1 i2 i3--<-
 * Nlops:	(    op   ) |
 * Outputs:	 o1 o2-->---
 *		  |
 *
 * @param op nn_t struct (will be freed)
 * @param o output index (ignored if oname != NULL)
 * @param oname name of output
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 *
 * @returns linked nn_t
 */
nn_t nn_link_F(nn_t x, int o, const char* oname, int i, const char* iname)
{
	auto result = nn_link(x, o, oname, i, iname);
	nn_free(x);
	return result;
}

/**
 * Chain output o of nn a in input i of nn b.
 *
 * Returned operator has
 * - inputs:  [b_0, ..., b_i-1, b_i+1, ..., b_n, a_0, ..., a_n]
 * - outputs: [b_0, ..., b_n, a_0, ..., a_o-1, a_o+1, ..., a_n]
 *
 * @param a nn_t struct
 * @param o output index (ignored if oname != NULL)
 * @param oname name of output
 * @param b nn_t struct
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 *
 * @returns chained nn_t
 */
nn_t nn_chain2(nn_t a, int o, const char* oname, nn_t b, int i, const char* iname)
{
	int IIa = nn_get_nr_in_args(a);
	int IIb = nn_get_nr_in_args(b);
	int OOa = nn_get_nr_out_args(a);
	int OOb = nn_get_nr_out_args(b);

	o = nn_get_out_arg_index(a, o, oname);
	i = nn_get_in_arg_index(b, i, iname);

	for (int ia = 0; ia < IIa; ia++)
		for (int ib = 0; ib < IIb; ib++)
			assert(    (i == ib)
				|| (NULL == nn_get_in_names(a)[ia])
				|| (NULL == nn_get_in_names(b)[ib])
				|| (0 != strcmp(nn_get_in_names(a)[ia], nn_get_in_names(b)[ib])));

	for (int ia = 0; ia < OOa; ia++)
		for (int ib = 0; ib < OOb; ib++)
			assert(    (o == ia)
				|| (NULL == nn_get_out_names(a)[ia])
				|| (NULL == nn_get_out_names(b)[ib])
				|| (0 != strcmp(nn_get_out_names(a)[ia], nn_get_out_names(b)[ib])));

	auto result = nn_from_nlop_F(nlop_chain2(a->nlop, o, b->nlop, i));

	for (int j = 0, jp = 0; j < IIb - 1; j++, jp++) {

		if (j == i)
			jp++;
		nn_clone_arg_i_from_i(result, j, b, jp);
	}

	for (int j = 0; j < IIa; j++)
		nn_clone_arg_i_from_i(result, j + IIb - 1, a, j);

	for (int j = 0; j < OOb; j++)
		nn_clone_arg_o_from_o(result, j, b, j);

	for (int j = 0, jp = 0; j < OOa - 1; j++, jp++) {

		if (j == o)
			jp++;
		nn_clone_arg_o_from_o(result, j + OOb, a, jp);
	}

	return result;
}

/**
 * Chain output o of nn a in input i of nn b.
 * Free a and b
 *
 * Returned operator has
 * - inputs:  [b_0, ..., b_i-1, b_i+1, ..., b_n, a_0, ..., a_n]
 * - outputs: [b_0, ..., b_n, a_0, ..., a_o-1, a_o+1, ..., a_n]
 *
 * @param a nn_t struct (will be freed)
 * @param o output index (ignored if oname != NULL)
 * @param oname name of output
 * @param nn_t struct (will be freed)
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 *
 * @returns chained nn_t
 */
nn_t nn_chain2_FF(nn_t a, int o, const char* oname, nn_t b, int i, const char* iname)
{
	auto result = nn_chain2(a, o, oname, b, i, iname);

	nn_free(a);
	nn_free(b);

	return result;
}

/**
 * Chain output o of nn a in input i of nn b.
 * Permute inputs.
 * Free a and b.
 *
 * Returned operator has
 * - inputs:  [a_0, ..., a_n, b_0, ..., b_i-1, b_i+1, ..., b_n]
 * - outputs: [b_0, ..., b_n, a_0, ..., a_o-1, a_o+1, ..., a_n]
 *
 * @param a nn_t struct (will be freed)
 * @param o output index (ignored if oname != NULL)
 * @param oname name of output
 * @param nn_t struct (will be freed)
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 *
 * @returns chained nn_t
 */
nn_t nn_chain2_swap_FF(nn_t a, int o, const char* oname, nn_t b, int i, const char* iname)
{
	int unIIa = nn_get_nr_in_args(a);

	auto result = nn_chain2_FF(a, o, oname, b, i, iname);

	int unII = nn_get_nr_in_args(result);
	int perm[unII];

	for (int i = 0; i < unII; i++)
		perm[(unIIa + i) % unII] = i;

	return nn_permute_inputs_F(result, unII, perm);
}

/**
 * Chain output o of nn a in input i of nn b.
 * Keep output of a
 *
 * Returned operator has
 * - inputs:  [b_0, ..., b_i-1, b_i+1, ..., b_n, a_0, ..., a_n]
 * - outputs: [b_0, ..., b_n, a_0, ..., a_o-1, a_o+1, ..., a_n]
 *
 * @param a nn_t struct
 * @param o output index (ignored if oname != NULL)
 * @param oname name of output
 * @param b nn_t struct
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 *
 * @returns chained nn_t
 */
nn_t nn_chain2_keep(nn_t a, int o, const char* oname, nn_t b, int i, const char* iname)
{
	int IIa = nn_get_nr_in_args(a);
	int IIb = nn_get_nr_in_args(b);
	int OOa = nn_get_nr_out_args(a);
	int OOb = nn_get_nr_out_args(b);

	o = nn_get_out_arg_index(a, o, oname);
	i = nn_get_in_arg_index(b, i, iname);

	auto result = nn_from_nlop_F(nlop_chain2_keep(a->nlop, o, b->nlop, i));

	for (int j = 0, jp = 0; j < IIb - 1; j++, jp++) {

		if (j == i)
			jp++;
		nn_clone_arg_i_from_i(result, j, b, jp);
	}

	for (int j = 0; j < IIa; j++)
		nn_clone_arg_i_from_i(result, j + IIb - 1, a, j);

	for (int j = 0; j < OOb; j++)
		nn_clone_arg_o_from_o(result, j, b, j);

	for (int j = 0; j < OOa; j++)
		nn_clone_arg_o_from_o(result, j + OOb, a, j);

	return result;
}

/**
 * Chain output o of nn a in input i of nn b.
 * Keep output of a.
 * Free a and b.
 *
 * Returned operator has
 * - inputs:  [b_0, ..., b_i-1, b_i+1, ..., b_n, a_0, ..., a_n]
 * - outputs: [b_0, ..., b_n, a_0, ..., a_o-1, a_o+1, ..., a_n]
 *
 * @param a nn_t struct (will be freed)
 * @param o output index (ignored if oname != NULL)
 * @param oname name of output
 * @param nn_t struct (will be freed)
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 *
 * @returns chained nn_t
 */
nn_t nn_chain2_keep_FF(nn_t a, int o, const char* oname, nn_t b, int i, const char* iname)
{
	auto result = nn_chain2_keep(a, o, oname, b, i, iname);

	nn_free(a);
	nn_free(b);

	return result;
}

/**
 * Chain output o of nn a in input i of nn b.
 *Keep output of a.
 * Permute inputs.
 * Free a and b.
 *
 * Returned operator has
 * - inputs:  [a_0, ..., a_n, b_0, ..., b_i-1, b_i+1, ..., b_n]
 * - outputs: [b_0, ..., b_n, a_0, ..., a_o-1, a_o, a_o+1, ..., a_n]
 *
 * @param a nn_t struct (will be freed)
 * @param o output index (ignored if oname != NULL)
 * @param oname name of output
 * @param nn_t struct (will be freed)
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 *
 * @returns chained nn_t
 */
nn_t nn_chain2_keep_swap_FF(nn_t a, int o, const char* oname, nn_t b, int i, const char* iname)
{
	int unIIa = nn_get_nr_in_args(a);

	auto result = nn_chain2_keep_FF(a, o, oname, b, i, iname);

	int unII = nn_get_nr_in_args(result);
	int perm[unII];

	for (int i = 0; i < unII; i++)
		perm[(unIIa + i) % unII] = i;

	return nn_permute_inputs_F(result, unII, perm);
}

/**
 * Duplicate two inputs of a nn_t, i.e. the input b will be set to equal the input a
 *
 * g(x_1, ... ,x_b-1, x_b+1, ... , x_n) = f(x_1, ... ,x_b-1, x_a, x_b+1, ... x_n)
 *
 * The duplicated input will take the position, name, etc. of the index a/aname.
 * Thus, if a and b are numeric indices and a>b, the duplicated index will be a-1, and a if a<b.
 * Note that this behaviour differs from nlop_dup(operator_dup)
 * Note that the initializers of a and b must be compatible, i.e. at least one is NULL or they equal
 *
 * @param op nn_t struct
 * @param a first input index (ignored if aname != NULL)
 * @param aname name of first input
 * @param b second input index (ignored if bname != NULL)
 * @param bname name of second input
 *
 * @returns nn_t with dupped inputs
 */
nn_t nn_dup(nn_t op, int a, const char* aname, int b, const char* bname)
{

	a = nn_get_in_arg_index(op, a, aname);
	b = nn_get_in_arg_index(op, b, bname);

	assert(op->dup[a] == op->dup[b]);

	int II = nn_get_nr_in_args(op);
	int OO = nn_get_nr_out_args(op);

	auto init_tmp = init_dup_create(op->initializers[a], op->initializers[b]);

	const struct operator_p_s* prox_tmp = NULL;
	if (NULL != op->prox_ops[a])
		prox_tmp = operator_p_ref(op->prox_ops[a]);
	else
		if (NULL != op->prox_ops[b])
			prox_tmp = operator_p_ref(op->prox_ops[b]);

	auto nlop = nlop_dup(nn_get_nlop(op), MIN(a , b), MAX(a, b));
	if (a > b)
		nlop = nlop_shift_input_F(nlop, a - 1, b);

	auto result = nn_from_nlop_F(nlop);

	for (int i = 0, ip = 0; i < II - 1; i++) {

		if (i == b) ip++;
		nn_clone_arg_i_from_i(result, i, op, ip++);
	}
	for (int i = 0; i < OO; i++)
		nn_clone_arg_o_from_o(result, i, op, i);

	initializer_free(result->initializers[(a > b) ? a - 1 : a]);
	result->initializers[(a > b) ? a - 1 : a] = init_tmp;

	operator_p_free(result->prox_ops[(a > b) ? a - 1 : a]);
	result->prox_ops[(a > b) ? a - 1 : a] = prox_tmp;

	return result;
}

/**
 * Duplicate two inputs of a nn_t, i.e. the input b will be set to equal the input a, and free the nn_t
 *
 * g(x_1, ... ,x_b-1, x_b+1, ... , x_n) = f(x_1, ... ,x_b-1, x_a, x_b+1, ... x_n)
 *
 * The duplicated input will take the position, name, etc. of the index a/aname.
 * Thus, if a and b are numeric indices and a>b, the duplicated index will be a-1, and a if a<b.
 * Note that this behaviour differs from nlop_dup(operator_dup)
 * Note that the initializers of a and b must be compatible, i.e. at least one is NULL or they equal
 *
 * @param op nn_t struct (will be freed)
 * @param a first input index (ignored if aname != NULL)
 * @param aname name of first input
 * @param b second input index (ignored if bname != NULL)
 * @param bname name of second input
 *
 * @returns nn_t with dupped inputs
 */
nn_t nn_dup_F(nn_t op, int a, const char* aname, int b, const char* bname)
{
	auto result = nn_dup(op, a, aname, b, bname);
	nn_free(op);
	return result;
}

/**
 * Stack two inputs of a nn_t, i.e. the new input will be destacked and chained in the inputs a and b
 *
 * g(x_1, ... ,x_b-1, x_b+1, ... x_s ... , x_n) = f(x_1, ... ,x_b-1, x_sb, x_b+1, ..., x_sa, ... x_n)
 * , where x_s equals x_sb stacked on x_sa
 *
 * The stacked input will take the position, name, etc. of the index a/aname.
 * Thus, if a and b are numeric indices and a>b, the duplicated index will be a-1, and a if a<b.
 *
 * @param op nn_t struct
 * @param a first input index (ignored if aname != NULL)
 * @param aname name of first input
 * @param b second input index (ignored if bname != NULL)
 * @param bname name of second input
 * @param stack_dim index at which dimension the two inputs should be stacked (can be negative meaning that it is counted backward starting from the last)
 *
 * @returns nn_t with stacked inputs
 */
nn_t nn_stack_inputs(nn_t op, int a, const char* aname, int b, const char* bname, int stack_dim)
{
	a = nn_get_in_arg_index(op, a, aname);
	b = nn_get_in_arg_index(op, b, bname);

	assert(op->dup[a] == op->dup[b]);

	int II = nn_get_nr_in_args(op);
	int OO = nn_get_nr_out_args(op);

	auto iova = nlop_generic_domain(op->nlop, a);
	auto iovb = nlop_generic_domain(op->nlop, b);
	assert(iova->N == iovb->N);
	auto init_tmp = init_stack_create(iova->N, stack_dim, iova->dims, op->initializers[a], iovb->dims, op->initializers[b]);

	if (0 > stack_dim)
		stack_dim += iova->N;

	const struct operator_p_s* prox_tmp = NULL;
	if ((NULL != op->prox_ops[a]) || (NULL != op->prox_ops[b])) {

		auto prox_a = (NULL == op->prox_ops[a]) ? prox_zero_create(iova->N, iova->dims) : operator_p_ref(op->prox_ops[a]);
		auto prox_b = (NULL == op->prox_ops[b]) ? prox_zero_create(iovb->N, iovb->dims) : operator_p_ref(op->prox_ops[b]);
		prox_tmp = operator_p_stack_FF(stack_dim, stack_dim, prox_a, prox_b);
	}

	auto nlop = nlop_stack_inputs(nn_get_nlop(op), a, b, stack_dim);
	if (a > b)
		nlop = nlop_shift_input_F(nlop, a - 1, b);
	auto result = nn_from_nlop_F(nlop);

	for (int i = 0, ip = 0; i < II - 1; i++) {

		if (i == b) ip++;
		nn_clone_arg_i_from_i(result, i, op, ip++);
	}
	for (int i = 0; i < OO; i++)
		nn_clone_arg_o_from_o(result, i, op, i);

	initializer_free(result->initializers[(a > b) ? a - 1 : a]);
	result->initializers[(a > b) ? a - 1 : a] = init_tmp;

	operator_p_free(result->prox_ops[(a > b) ? a - 1 : a]);
	result->prox_ops[(a > b) ? a - 1 : a] = prox_tmp;

	return result;
}

/**
 * Stack two inputs of a nn_t, i.e. the new input will be destacked and chained in the inputs a and b, and frees the nn_t
 *
 * g(x_1, ... ,x_b-1, x_b+1, ... x_s ... , x_n) = f(x_1, ... ,x_b-1, x_sb, x_b+1, ..., x_sa, ... x_n)
 * , where x_s equals x_sb stacked on x_sa
 *
 * The stacked input will take the position, name, etc. of the index a/aname.
 * Thus, if a and b are numeric indices and a>b, the duplicated index will be a-1, and a if a<b.
 *
 * @param op nn_t struct (will be freed)
 * @param a first input index (ignored if aname != NULL)
 * @param aname name of first input
 * @param b second input index (ignored if bname != NULL)
 * @param bname name of second input
 * @param stack_dim index at which dimension the two inputs should be stacked (can be negative meaning that it is counted backward starting from the last)
 *
 * @returns nn_t with stacked inputs
 */
nn_t nn_stack_inputs_F(nn_t op, int a, const char* aname, int b, const char* bname, int stack_dim)
{
	auto result = nn_stack_inputs(op, a, aname, b, bname, stack_dim);
	nn_free(op);
	return result;
}

/**
 * Stack two outputs of a nn_t, i.e. the two outputs a and b will be computed and the results of b is stacked on the result of a to form the new output
 *
 * The stacked input will take the position, name, etc. of the index a/aname.
 * Thus, if a and b are numeric indices and a>b, the duplicated index will be a-1, and a if a<b.
 *
 * @param op nn_t struct
 * @param a first output index (ignored if aname != NULL)
 * @param aname name of first output
 * @param b second output index (ignored if bname != NULL)
 * @param bname name of second output
 * @param stack_dim index at which dimension the two outputs should be stacked (can be negative meaning that it is counted backward starting from the last)
 *
 * @returns nn_t with stacked outputs
 */
nn_t nn_stack_outputs(nn_t op, int a, const char* aname, int b, const char* bname, int stack_dim)
{
	a = nn_get_out_arg_index(op, a, aname);
	b = nn_get_out_arg_index(op, b, bname);

	int II = nn_get_nr_in_args(op);
	int OO = nn_get_nr_out_args(op);

	auto nlop = nlop_stack_outputs(nn_get_nlop(op), a, b, stack_dim);
	if (a > b)
		nlop = nlop_shift_output_F(nlop, a - 1, b);
	auto result = nn_from_nlop_F(nlop);

	for (int i = 0; i < II; i++)
		nn_clone_arg_i_from_i(result, i, op, i);

	for (int i = 0, ip = 0; i < OO - 1; i++) {

		if (i == b) ip++;
		nn_clone_arg_o_from_o(result, i, op, ip++);
	}

	return result;
}

/**
 * Stack two outputs of a nn_t, i.e. the two outputs a and b will be computed and the results of b is stacked on the result of a to form the new output, and free nn_t
 *
 * The stacked input will take the position, name, etc. of the index a/aname.
 * Thus, if a and b are numeric indices and a>b, the duplicated index will be a-1, and a if a<b.
 *
 * @param op nn_t struct (will be freed)
 * @param a first output index (ignored if aname != NULL)
 * @param aname name of first output
 * @param b second output index (ignored if bname != NULL)
 * @param bname name of second output
 * @param stack_dim index at which dimension the two outputs should be stacked (can be negative meaning that it is counted backward starting from the last)
 *
 * @returns nn_t with stacked outputs
 */
nn_t nn_stack_outputs_F(nn_t op, int a, const char* aname, int b, const char* bname, int stack_dim)
{
	auto result = nn_stack_outputs(op, a, aname, b, bname, stack_dim);
	nn_free(op);
	return result;
}

/**
 * Permute inputs of nn_t such that the index o is shifted to position n and free nn_t
 *
 * @param op nn_t struct (will be freed)
 * @param n new position of input
 * @param o old position of input
 *
 * @returns nn_t with permuted inputs
 *
 * @note the indices n and o also count named arguments, use nn_shift_input_F if only unnamed inputs shall be shifted
 */
nn_t nn_shift_input_index_F(nn_t x, int n, int o)
{
	int new_index = n;
	int old_index = o;
	int II = nn_get_nr_in_args(x);
	assert(old_index < II);
	assert(new_index < II);

	int perm[II];
	for (int i = 0, ip = 0; i < II; i++, ip++) {

		perm[i] = ip;
		if (i == old_index) ip++;
		if (i == new_index) ip--;
		if (new_index > old_index)
			perm[i] = ip;
	}

	perm[new_index] = old_index;

	return nn_permute_inputs_F(x, II, perm);
}

/**
 * Permute outputs of nn_t such that the index o is shifted to position n and free nn_t
 *
 * @param op nn_t struct (will be freed)
 * @param n new position of input
 * @param o old position of input
 *
 * @returns nn_t with permuted outputs
 *
 * @note the indices n and o also count named arguments, use nn_shift_output_F if only unnamed inputs shall be shifted
 */
nn_t nn_shift_output_index_F(nn_t x, int n, int o)
{
	int new_index = n;
	int old_index = o;
	int OO = nn_get_nr_out_args(x);
	assert(old_index < OO);
	assert(new_index < OO);

	int perm[OO];
	for (int i = 0, ip = 0; i < OO; i++, ip++) {

		perm[i] = ip;
		if (i == old_index) ip++;
		if (i == new_index) ip--;
		if (new_index > old_index)
			perm[i] = ip;
	}

	perm[new_index] = old_index;

	return nn_permute_outputs_F(x, OO, perm);
}

/**
 * Permute inputs of nn_t such that the input at o is shifted to position of input n and free nn_t
 *
 * @param op nn_t struct (will be freed)
 * @param n new position of input
 * @param o old position of input
 *
 * @returns nn_t with permuted inputs
 */
nn_t nn_shift_input_F(nn_t x, int n, const char* nname, int o, const char* oname)
{
	n = nn_get_in_arg_index(x, n, nname);
	o = nn_get_in_arg_index(x, o, oname);

	return nn_shift_input_index_F(x, n, o);
}

/**
 * Permute outputs of nn_t such that the output o is shifted to position of output n and free nn_t
 *
 * @param op nn_t struct (will be freed)
 * @param n new position of input
 * @param o old position of input
 *
 * @returns nn_t with permuted outputs
 */
nn_t nn_shift_output_F(nn_t x, int n, const char* nname, int o, const char* oname)
{
	n = nn_get_out_arg_index(x, n, nname);
	o = nn_get_out_arg_index(x, o, oname);

	return nn_shift_output_index_F(x, n, o);
}

/**
 * Rename input with name to #DUP_name and free nn_t
 *
 * If the nn_t is combined with another having an input with the selected name,
 * nn_stack_dup_by_name_F can be used to dup the two inputs
 *
 * @param op nn_t struct (will be freed)
 * @param name old name
 *
 * @returns nn_t with new name
 */
nn_t nn_mark_dup_F(nn_t x, const char* name)
{
	char nname[strlen(name) + 6];
	sprintf(nname, "#DUP_%s", name);

	return nn_rename_input_F(x, nname, name);
}

/**
 * Rename input with name to #STACK_name and free nn_t
 *
 * If the nn_t is combined with another having an input with the selected name,
 * nn_stack_dup_by_name_F can be used to stack the two inputs
 *
 * @param op nn_t struct (will be freed)
 * @param name old name
 *
 * @returns nn_t with new name
 */
nn_t nn_mark_stack_input_F(nn_t x, const char* name)
{
	char nname[strlen(name) + 8];
	sprintf(nname, "#STACK_%s", name);

	return nn_rename_input_F(x, nname, name);
}

/**
 * Rename output with name to #STACK_name and free nn_t
 *
 * If the nn_t is combined with another having an input with the selected name,
 * nn_stack_dup_by_name_F can be used to stack the two outputs
 *
 * @param op nn_t struct (will be freed)
 * @param name old name
 *
 * @returns nn_t with new name
 */
nn_t nn_mark_stack_output_F(nn_t x, const char* name)
{
	char nname[strlen(name) + 8];
	sprintf(nname, "#STACK_%s", name);

	return nn_rename_output_F(x, nname, name);
}

static nn_t stack_in_by_name(nn_t x) {

	const char* stack_name = NULL;
	for (int i = 0; i < nn_get_nr_in_args(x); i ++)
		if (NULL != x->in_names[i] && 0 == strncmp(x->in_names[i], "#STACK_", 7))
			stack_name = x->in_names[i];

	if (NULL == stack_name)
		return x;

	return nn_stack_inputs_F(x, 0, stack_name + 7, 0, stack_name, -1);
}

static nn_t stack_out_by_name(nn_t x) {

	const char* stack_name = NULL;
	for (int i = 0; i < nn_get_nr_out_args(x); i ++)
		if (NULL != x->out_names[i] && 0 == strncmp(x->out_names[i], "#STACK_", 7))
			stack_name = x->out_names[i];

	if (NULL == stack_name)
		return x;

	return nn_stack_outputs_F(x, 0, stack_name + 7, 0, stack_name, -1);
}

static nn_t dup_by_name(nn_t x) {

	const char* stack_name = NULL;
	for (int i = 0; i < nn_get_nr_in_args(x); i ++)
		if (NULL != x->in_names[i] && 0 == strncmp(x->in_names[i], "#DUP_", 5))
			stack_name = x->in_names[i];

	if (NULL == stack_name)
		return x;

	return nn_dup_F(x, 0, stack_name + 5, 0, stack_name);
}

/**
 * Search for input/output names #DUP_%s or #STACK_%s. If such names are found, a nn_t with stacked (#STACK_%s on %s) and dupped inputs/outputs is returned
 *
 * @param op nn_t struct (will be freed)
 *
 * @returns nn_t with stacked and dupped arguments
 */
nn_t nn_stack_dup_by_name_F(nn_t op)
{
	nn_t result = op;

	nn_t prev = NULL;
	while (result != prev) {

		prev = result;
		result = dup_by_name(result);
	}

	prev = NULL;
	while (result != prev) {

		prev = result;
		result = stack_in_by_name(result);
	}

	prev = NULL;
	while (result != prev) {

		prev = result;
		result = stack_out_by_name(result);
	}

	return result;
}

static bool is_name_in_list(int N, const char* names[N], const char* name)
{
	if (NULL == name)
		return false;

	bool result = false;
	for (int i = 0; i < N; i++)
		result |= (NULL == names[i]) ? false : (0 == strcmp(names[i], name));
	return result;
}

static int names_remove_double(int N, const char* dst_names[N], const char* src_names[N])
{
	int NN = 0;
	for (int i = 0; i < N; i++)
		if (!is_name_in_list(NN, dst_names, src_names[i]))
			dst_names[NN++] = src_names[i];
	return NN;
}

/**
 * Permute inputs of nn_t such that all inputs with a name contained in the provided list are in the same order as in the list and free nn_t
 *
 * @param op nn_t struct (will be freed)
 * @param N no. of names in the list
 * @param sorted_names list of names
 *
 * @returns nn_t with sorted inputs
 *
 * @note not all input names must be provided in the list and vice versa
 */
nn_t nn_sort_inputs_by_list_F(nn_t x, int N, const char* sorted_names[N])
{
	int II = nn_get_nr_in_args(x);
	int nperm[II];

	int index = 0;

	const char* nnames[N];
	int NN = names_remove_double(N, nnames, sorted_names);

	for (int i = 0; i < II; i++){

		if (is_name_in_list(NN, nnames, nn_get_in_name_from_arg_index(x, i, false))) {

			while (! nn_is_name_in_in_args(x, nnames[index]))
				index++;

			nperm[i] = nn_get_in_arg_index(x, 0, nnames[index]);
			index++;
		} else {

			nperm[i] = i;
		}
	}

	return nn_permute_inputs_F(x, II, nperm);
}

/**
 * Permute inputs of nn_t such that all inputs with a name contained in the provided list are in the same order as in the list and free nn_t
 *
 * @param op nn_t struct (will be freed)
 * @param N no. of names in the list
 * @param sorted_names list of names
 *
 * @returns nn_t with sorted inputs
 *
 * @note not all input names must be provided in the list and vice versa
 */
nn_t nn_sort_outputs_by_list_F(nn_t x, int N, const char* sorted_names[N])
{
	int OO = nn_get_nr_out_args(x);
	int nperm[OO];

	int index = 0;

	const char* nnames[N?:1];
	int NN = names_remove_double(N, nnames, sorted_names);

	for (int i = 0; i < OO; i++){

		if (is_name_in_list(NN, nnames, nn_get_out_name_from_arg_index(x, i, false))) {

			while (! nn_is_name_in_out_args(x, nnames[index]))
				index++;

			nperm[i] = nn_get_out_arg_index(x, 0, nnames[index]);
			index++;
		} else {

			nperm[i] = i;
		}
	}

	return nn_permute_outputs_F(x, OO, nperm);
}


/**
 * Reshape input of nn_t to have an additional singleton dimension and free nn_t
 * {dims[0], ..., dims[N-1]} -> {dims[0], ..., dims[N-1], 1}
 * return op if name does not exist
 *
 * @param op nn_t struct (will be freed)
 * @param iname name of input
 *
 * @returns nn_t with reshaped input
 */
nn_t nn_append_singleton_dim_in_if_exists_F(nn_t op, const char* iname)
{
	assert(NULL != iname);
	if (nn_is_name_in_in_args(op, iname))
		return nn_append_singleton_dim_in_F(op, 0, iname);
	else
		return op;
}

/**
 * Reshape output of nn_t to have an additional singleton dimension and free nn_t
 * {dims[0], ..., dims[N-1]} -> {dims[0], ..., dims[N-1], 1}
 * return op if name does not exist
 *
 * @param op nn_t struct (will be freed)
 * @param oname name of output
 *
 * @returns nn_t with reshaped output
 */
nn_t nn_append_singleton_dim_out_if_exists_F(nn_t op, const char* oname)
{
	if (nn_is_name_in_out_args(op, oname))
		return nn_append_singleton_dim_out_F(op, 0, oname);
	else
		return op;
}


/**
 * Rename input with name to #DUP_name and free nn_t
 *
 * If the nn_t is combined with another having an input with the selected name,
 * nn_stack_dup_by_name_F can be used to dup the two inputs
 * return op if name does not exist
 *
 * @param op nn_t struct (will be freed)
 * @param name old name
 *
 * @returns nn_t with new name
 */
nn_t nn_mark_dup_if_exists_F(nn_t x, const char* name)
{
	if (nn_is_name_in_in_args(x, name))
		return nn_mark_dup_F(x, name);
	else
		return x;
}

/**
 * Rename input with name to #STACK_name and free nn_t
 *
 * If the nn_t is combined with another having an input with the selected name,
 * nn_stack_dup_by_name_F can be used to stack the two inputs
 * return op if name does not exist
 *
 * @param op nn_t struct (will be freed)
 * @param name old name
 *
 * @returns nn_t with new name
 */
nn_t nn_mark_stack_input_if_exists_F(nn_t x, const char* name)
{
	if (nn_is_name_in_in_args(x, name))
		return nn_mark_stack_input_F(x, name);
	else
		return x;
}

/**
 * Rename output with name to #STACK_name and free nn_t
 *
 * If the nn_t is combined with another having an input with the selected name,
 * nn_stack_dup_by_name_F can be used to stack the two outputs
 * return op if name does not exist
 *
 * @param op nn_t struct (will be freed)
 * @param name old name
 *
 * @returns nn_t with new name
 */
nn_t nn_mark_stack_output_if_exists_F(nn_t x, const char* name)
{
	if (nn_is_name_in_out_args(x, name))
		return nn_mark_stack_output_F(x, name);
	else
		return x;
}

/**
 * Permute inputs of nn_t such that all inputs without a name come before named inputs
 *
 * @param op nn_t struct (will be freed)
 *
 * @returns nn_t with sorted inputs
 **/
nn_t nn_sort_inputs_F(nn_t x)
{
	int II = nn_get_nr_in_args(x);
	int index_unnamed = 0;
	int index_named = nn_get_nr_unnamed_in_args(x);

	int nperm[II];

	for (int i = 0; i < II; i++){

		if (NULL == (x->in_names)[i])
			nperm[index_unnamed++] = i;
		else
			nperm[index_named++] = i;
	}

	return nn_permute_inputs_F(x, II, nperm);
}

/**
 * Permute inputs of nn_t such that all outputs without a name come before named outputs
 *
 * @param op nn_t struct (will be freed)
 *
 * @returns nn_t with sorted inputs
 **/
nn_t nn_sort_outputs_F(nn_t x)
{
	int OO = nn_get_nr_out_args(x);
	int index_unnamed = 0;
	int index_named = nn_get_nr_unnamed_out_args(x);

	int nperm[OO];

	for (int i = 0; i < OO; i++){

		if (NULL == (x->out_names)[i])
			nperm[index_unnamed++] = i;
		else
			nperm[index_named++] = i;
	}

	return nn_permute_outputs_F(x, OO, nperm);
}
