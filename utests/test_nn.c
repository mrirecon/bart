/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <complex.h>

#include "nn/chain.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/iovec.h"
#include "num/ops.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"

#include "nlops/nlop.h"
#include "nlops/const.h"
#include "nlops/chain.h"

#include "nn/nn.h"
#include "nn/chain.h"
#include "nn/init.h"
#include "nn/weights.h"

#include "utest.h"

static nn_t nn_get_default1(void)
{
	enum { OO = 6, II = 5 };

	const struct nlop_s* result_nlop = nlop_del_out_create(2, MD_DIMS(1, 1,));
	for (int i = 1; i < II; i++)
		result_nlop = nlop_combine_FF(result_nlop, nlop_del_out_create(2, MD_DIMS(1 + i, 1,)));

	complex float tmp[OO] = { 0. };
	for (int i = 0; i < OO; i++)
		result_nlop = nlop_combine_FF(result_nlop, nlop_const_create(2, MD_DIMS(1 + i, 1,), true, tmp));

	auto result = nn_from_nlop_F(result_nlop);

	result = nn_set_input_name_F(result, 3, "IN_INDEX3");
	result = nn_set_input_name_F(result, 2, "IN_INDEX2");

	result = nn_set_output_name_F(result, 4, "OUT_INDEX4");
	result = nn_set_output_name_F(result, 2, "OUT_INDEX2");

	return result;
}

#if 0
static nn_t nn_get_default2(void)
{
	enum { OO = 2, II = 2 };

	const struct nlop_s* result_nlop = nlop_del_out_create(2, MD_DIMS(1, 1,));
	for (int i = 1; i < II; i++)
		result_nlop = nlop_combine_FF(result_nlop, nlop_del_out_create(2, MD_DIMS(1 + i, 1,)));

	complex float tmp[OO] = { 0. };
	for (int i = 0; i < OO; i++)
		result_nlop = nlop_combine_FF(result_nlop, nlop_const_create(2, MD_DIMS(1 + i, 1,), true, tmp));

	auto result = nn_from_nlop_F(result_nlop);

	result = nn_set_input_name_F(result, 1, "IN_INDEX1");
	result = nn_set_output_name_F(result, 1, "OUT_INDEX1");
	result = nn_set_output_name_F(result, 0, "OUT_INDEX0");

	return result;
}

static nn_t nn_get_default3(void)
{
	enum { OO = 2, II = 2 };

	const struct nlop_s* result_nlop = nlop_del_out_create(2, MD_DIMS(1, 1,));
	for (int i = 1; i < II; i++)
		result_nlop = nlop_combine_FF(result_nlop, nlop_del_out_create(2, MD_DIMS(1 + i, 1,)));

	complex float tmp[OO] = { 0. };
	for (int i = 0; i < OO; i++)
		result_nlop = nlop_combine_FF(result_nlop, nlop_const_create(2, MD_DIMS(1 + i, 1,), true, tmp));

	auto result = nn_from_nlop_F(result_nlop);

	result = nn_set_input_name_F(result, 1, "2IN_INDEX1");
	result = nn_set_output_name_F(result, 1, "2OUT_INDEX1");
	result = nn_set_output_name_F(result, 0, "2OUT_INDEX0");

	return result;
}
#endif



static bool test_nn_indexing(void)
{
	auto nn = nn_get_default1();

	bool result = true;
	result &= (0 == nn_get_in_arg_index(nn, 0, NULL));
	result &= (1 == nn_get_in_arg_index(nn, 1, NULL));
	result &= (2 == nn_get_in_arg_index(nn, 0, "IN_INDEX2"));
	result &= (3 == nn_get_in_arg_index(nn, 0, "IN_INDEX3"));
	result &= (4 == nn_get_in_arg_index(nn, 2, NULL));

	result &= (0 == nn_get_in_arg_index(nn, -3, NULL));
	result &= (1 == nn_get_in_arg_index(nn, -2, NULL));
	result &= (4 == nn_get_in_arg_index(nn, -1, NULL));

	result &= (0 == nn_get_out_arg_index(nn, 0, NULL));
	result &= (1 == nn_get_out_arg_index(nn, 1, NULL));
	result &= (2 == nn_get_out_arg_index(nn, 0, "OUT_INDEX2"));
	result &= (3 == nn_get_out_arg_index(nn, 2, NULL));
	result &= (4 == nn_get_out_arg_index(nn, 0, "OUT_INDEX4"));
	result &= (5 == nn_get_out_arg_index(nn, 3, NULL));

	result &= (0 == nn_get_out_arg_index(nn, -4, NULL));
	result &= (1 == nn_get_out_arg_index(nn, -3, NULL));
	result &= (3 == nn_get_out_arg_index(nn, -2, NULL));
	result &= (5 == nn_get_out_arg_index(nn, -1, NULL));

	result &= (5 == nn_get_nr_in_args(nn));
	result &= (6 == nn_get_nr_out_args(nn));

	result &= (3 == nn_get_nr_unnamed_in_args(nn));
	result &= (4 == nn_get_nr_unnamed_out_args(nn));

	nn_free(nn);

	return result;
}

UT_REGISTER_TEST(test_nn_indexing);

static bool test_nn_sorting(void)
{
	auto nn = nn_get_default1();

	const char* sorted_names[4] = {
		"IN_INDEX3",
		"IN_INDEX2",
		"OUT_INDEX4",
		"OUT_INDEX2"
	};

	nn = nn_sort_inputs_by_list_F(nn, 4, sorted_names);
 	nn = nn_sort_outputs_by_list_F(nn, 4, sorted_names);

	bool result = true;
	result &= (0 == nn_get_in_arg_index(nn, 0, NULL));
	result &= (1 == nn_get_in_arg_index(nn, 1, NULL));
	result &= (2 == nn_get_in_arg_index(nn, 0, "IN_INDEX3"));
	result &= (3 == nn_get_in_arg_index(nn, 0, "IN_INDEX2"));
	result &= (4 == nn_get_in_arg_index(nn, 2, NULL));

	result &= (0 == nn_get_out_arg_index(nn, 0, NULL));
	result &= (1 == nn_get_out_arg_index(nn, 1, NULL));
	result &= (2 == nn_get_out_arg_index(nn, 0, "OUT_INDEX4"));
	result &= (3 == nn_get_out_arg_index(nn, 2, NULL));
	result &= (4 == nn_get_out_arg_index(nn, 0, "OUT_INDEX2"));
	result &= (5 == nn_get_out_arg_index(nn, 3, NULL));

	result &= (5 == nn_get_nr_in_args(nn));
	result &= (6 == nn_get_nr_out_args(nn));

	result &= (3 == nn_get_nr_unnamed_in_args(nn));
	result &= (4 == nn_get_nr_unnamed_out_args(nn));

	nn_free(nn);

	return result;
}

UT_REGISTER_TEST(test_nn_sorting);


static bool test_nn_rename(void)
{
	auto nn = nn_get_default1();

	nn = nn_rename_input_F(nn, "NEW_IN_INDEX2", "IN_INDEX2");
	nn = nn_rename_output_F(nn, "NEW_OUT_INDEX4", "OUT_INDEX4");

	bool result = true;
	result &= (2 == nn_get_in_arg_index(nn, 0, "NEW_IN_INDEX2"));
	result &= (3 == nn_get_in_arg_index(nn, 0, "IN_INDEX3"));
	result &= (2 == nn_get_out_arg_index(nn, 0, "OUT_INDEX2"));
	result &= (4 == nn_get_out_arg_index(nn, 0, "NEW_OUT_INDEX4"));


	nn_free(nn);

	return result;
}

UT_REGISTER_TEST(test_nn_rename);

static bool test_nn_unname(void)
{
	auto nn = nn_get_default1();

	nn = nn_unset_input_name_F(nn, "IN_INDEX2");
	nn = nn_unset_output_name_F(nn, "OUT_INDEX4");

	bool result = true;
	result &= iovec_check(nn_generic_domain(nn, -1, NULL), 2, MD_DIMS(3, 1), MD_DIMS(CFL_SIZE, 0));
	result &= iovec_check(nn_generic_codomain(nn, -1, NULL), 2, MD_DIMS(5, 1), MD_DIMS(CFL_SIZE, 0));

	nn_free(nn);

	return result;
}

UT_REGISTER_TEST(test_nn_unname);

static bool test_nn_init_and_weight_and_stack_inputs(void)
{
	auto nn = nn_get_default1();

	nn = nn_set_initializer_F(nn, 0, "IN_INDEX2", init_const_create(2));
	nn = nn_set_initializer_F(nn, 0, "IN_INDEX3", init_const_create(3));

	nn = nn_reshape_in_F(nn, 0, "IN_INDEX2", 1, MD_DIMS(3));
	nn = nn_append_singleton_dim_in_F(nn, 0, "IN_INDEX2");

	auto nn1 = nn_stack_inputs(nn, 0, "IN_INDEX2", 0, "IN_INDEX3", 0);
	auto nn2 = nn_stack_inputs(nn, 0, "IN_INDEX3", 0, "IN_INDEX2", 0);

	auto weights1 = nn_weights_create_from_nn(nn1);
	auto weights2 = nn_weights_create_from_nn(nn2);

	nn_init(nn1, weights1);
	nn_init(nn2, weights2);

	float err = 0.;
	complex float ref1[7] = {2, 2, 2, 3, 3, 3, 3};
	complex float ref2[7] = {3, 3, 3, 3, 2, 2, 2};
	err += md_zrmse(2, MD_DIMS(7, 1), weights1->tensors[0], ref1);
	err += md_zrmse(2, MD_DIMS(7, 1), weights2->tensors[0], ref2);

	bool result = (err < UT_TOL);
	result &= iovec_check(nn_generic_domain(nn1, 0, "IN_INDEX2"), weights1->iovs[0]->N, weights1->iovs[0]->dims, MD_STRIDES(weights1->iovs[0]->N, weights1->iovs[0]->dims, CFL_SIZE));
	result &= iovec_check(nn_generic_domain(nn2, 0, "IN_INDEX3"), weights2->iovs[0]->N, weights2->iovs[0]->dims, MD_STRIDES(weights2->iovs[0]->N, weights2->iovs[0]->dims, CFL_SIZE));

	result &= (2 == nn_get_in_arg_index(nn1, 0, "IN_INDEX2"));
	result &= (2 == nn_get_in_arg_index(nn2, 0, "IN_INDEX3"));

	nn_weights_free(weights1);
	nn_weights_free(weights2);

	nn_free(nn);
	nn_free(nn1);
	nn_free(nn2);

	return result;
}

UT_REGISTER_TEST(test_nn_init_and_weight_and_stack_inputs);


static bool test_nn_permute_inputs1(void)
{
	auto nn = nn_get_default1();

	nn = nn_set_initializer_F(nn, 0, "IN_INDEX2", init_const_create(2));
	nn = nn_set_initializer_F(nn, 0, "IN_INDEX3", init_const_create(3));

	nn = nn_reshape_in_F(nn, 0, "IN_INDEX2", 1, MD_DIMS(3));
	nn = nn_append_singleton_dim_in_F(nn, 0, "IN_INDEX2");

	nn = nn_permute_inputs_F(nn, 3, (int[3]){2, 1, 0});

	bool result = true;
	result &= iovec_check(nn_generic_domain(nn, -1, NULL), 2, MD_DIMS(1, 1), MD_DIMS(0, 0));
	result &= iovec_check(nn_generic_domain(nn, -2, NULL), 2, MD_DIMS(2, 1), MD_DIMS(CFL_SIZE, 0));
	result &= iovec_check(nn_generic_domain(nn, -3, NULL), 2, MD_DIMS(5, 1), MD_DIMS(CFL_SIZE, 0));

	result &= (2 == nn_get_in_arg_index(nn, 0, "IN_INDEX2"));
	result &= (3 == nn_get_in_arg_index(nn, 0, "IN_INDEX3"));

	nn_free(nn);

	return result;
}

UT_REGISTER_TEST(test_nn_permute_inputs1);



static bool test_nn_permute_inputs2(void)
{
	auto nn = nn_get_default1();

	nn = nn_set_initializer_F(nn, 0, "IN_INDEX2", init_const_create(2));
	nn = nn_set_initializer_F(nn, 0, "IN_INDEX3", init_const_create(3));

	nn = nn_reshape_in_F(nn, 0, "IN_INDEX2", 1, MD_DIMS(3));
	nn = nn_append_singleton_dim_in_F(nn, 0, "IN_INDEX2");

	nn = nn_permute_inputs_F(nn, 5, (int[5]){4, 3, 2, 1, 0});

	bool result = true;
	result &= iovec_check(nn_generic_domain(nn, -1, NULL), 2, MD_DIMS(1, 1), MD_DIMS(0, 0));
	result &= iovec_check(nn_generic_domain(nn, -2, NULL), 2, MD_DIMS(2, 1), MD_DIMS(CFL_SIZE, 0));
	result &= iovec_check(nn_generic_domain(nn, -3, NULL), 2, MD_DIMS(5, 1), MD_DIMS(CFL_SIZE, 0));

	result &= (2 == nn_get_in_arg_index(nn, 0, "IN_INDEX2"));
	result &= (1 == nn_get_in_arg_index(nn, 0, "IN_INDEX3"));

	nn_free(nn);

	return result;
}

UT_REGISTER_TEST(test_nn_permute_inputs2);

static bool test_nn_permute_outputs1(void)
{
	auto nn = nn_get_default1();

	nn = nn_permute_outputs_F(nn, 4, (int[4]){3, 2, 1, 0});

	bool result = true;
	result &= iovec_check(nn_generic_codomain(nn, -1, NULL), 2, MD_DIMS(1, 1), MD_DIMS(0, 0));
	result &= iovec_check(nn_generic_codomain(nn, -2, NULL), 2, MD_DIMS(2, 1), MD_DIMS(CFL_SIZE, 0));
	result &= iovec_check(nn_generic_codomain(nn, -3, NULL), 2, MD_DIMS(4, 1), MD_DIMS(CFL_SIZE, 0));
	result &= iovec_check(nn_generic_codomain(nn, -4, NULL), 2, MD_DIMS(6, 1), MD_DIMS(CFL_SIZE, 0));

	result &= (2 == nn_get_out_arg_index(nn, 0, "OUT_INDEX2"));
	result &= (4 == nn_get_out_arg_index(nn, 0, "OUT_INDEX4"));

	nn_free(nn);

	return result;
}

UT_REGISTER_TEST(test_nn_permute_outputs1);



static bool test_nn_permute_outputs2(void)
{
	auto nn = nn_get_default1();

	nn = nn_permute_outputs_F(nn, 6, (int[6]){5, 4, 3, 2, 1, 0});

	bool result = true;
	result &= iovec_check(nn_generic_codomain(nn, -1, NULL), 2, MD_DIMS(1, 1), MD_DIMS(0, 0));
	result &= iovec_check(nn_generic_codomain(nn, -2, NULL), 2, MD_DIMS(2, 1), MD_DIMS(CFL_SIZE, 0));
	result &= iovec_check(nn_generic_codomain(nn, -3, NULL), 2, MD_DIMS(4, 1), MD_DIMS(CFL_SIZE, 0));
	result &= iovec_check(nn_generic_codomain(nn, -4, NULL), 2, MD_DIMS(6, 1), MD_DIMS(CFL_SIZE, 0));

	result &= (3 == nn_get_out_arg_index(nn, 0, "OUT_INDEX2"));
	result &= (1 == nn_get_out_arg_index(nn, 0, "OUT_INDEX4"));

	nn_free(nn);

	return result;
}
UT_REGISTER_TEST(test_nn_permute_outputs2);
