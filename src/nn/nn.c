/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/shrdptr.h"
#include "misc/types.h"
#include "misc/graph.h"

#include "nn/init.h"
#include "nn/chain.h"
#include "num/multind.h"
#include "num/ops.h"
#include "num/ops_p.h"
#include "num/iovec.h"

#include "iter/italgos.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/const.h"
#include "nlops/checkpointing.h"

#include "nn.h"

/*
In this file, we define the struct nn_s and its type nn_t.

This struct is a wrapper around a non-linear operator which should simplify the composition of complex neural networks.
While inputs and outputs of non-linear operators are indexed with a positive index, the nn_t type supports both indexing using name strings and numeric indexing.
Note that named arguments are not counted for numeric indexing, but a named input might be between to numeric indices.
Negative indices can be used to start counting from the last index (this is not possible for nlops).

An example how the consecutive inputs of a nn_t might be accessed is:
[0, 1, "weight1", "bias27", 2, "pattern", -3, 4, "kspace" -1]

Functions working with a specific input/output are usually passed an integer and a string. The integer value is only used for indexing if the string points to NULL, else the string is used.
To avoid confusions, the integer must be 0 or -1 in case the string is used for indexing.
Examples

- (0, NULL) will access input 0
- (2, NULL) will access input 4
- (0, "pattern") will access input 4
- (3, "pattern") will produce an error
- (0, "pAttern") will produce an error as the string is not found

In the definition of the functions in this file, we use the term "index" for the numeric indices of a nn_t type and the term "arg_index" for the indices of the corresponding non-linear operator.


Moreover, the nn_t type can store an initializer for each input and the types of inputs/outputs used for optimization.

*/

nn_t nn_from_nlop(const struct nlop_s* op)
{
	PTR_ALLOC(struct nn_s, nn);

	int NO = nlop_get_nr_out_args(op);
	int NI = nlop_get_nr_in_args(op);

	PTR_ALLOC(const char*[NO], out_names);
	PTR_ALLOC(const char*[NI], in_names);

	PTR_ALLOC(const struct initializer_s*[NI], initializers);
	PTR_ALLOC(const struct operator_p_s*[NI], prox_ops);
	PTR_ALLOC(bool[NI], dup);
	PTR_ALLOC(enum IN_TYPE[NI], in_types);
	PTR_ALLOC(enum OUT_TYPE[NO], out_types);

	for (int i = 0; i < NI; i++) {

		(*prox_ops)[i] = NULL;
		(*in_names)[i] = NULL;
		(*initializers)[i] = NULL;
		(*in_types)[i] = IN_UNDEFINED;
		(*dup)[i] = true;
	}

	for (int o = 0; o < NO; o++) {

		(*out_names)[o] = NULL;
		(*out_types)[o] = OUT_UNDEFINED;
	}

	nn->in_names = *PTR_PASS(in_names);
	nn->out_names = *PTR_PASS(out_names);

	nn->initializers = *PTR_PASS(initializers);
	nn->prox_ops = *PTR_PASS(prox_ops);
	nn->dup = *PTR_PASS(dup);
	nn->in_types = *PTR_PASS(in_types);
	nn->out_types = *PTR_PASS(out_types);

	nn->nlop = nlop_clone(op);

	return PTR_PASS(nn);
}

void nn_free(nn_t op)
{
	int II = nn_get_nr_in_args(op);
	int OO = nn_get_nr_out_args(op);

	for (int i = 0; i < II; i++){

		xfree(op->in_names[i]);
		initializer_free(op->initializers[i]);
		operator_p_free(op->prox_ops[i]);
	}
	for (int o = 0; o < OO; o++)
		xfree(op->out_names[o]);

	xfree(op->in_names);
	xfree(op->out_names);

	xfree(op->initializers);
	xfree(op->prox_ops);
	xfree(op->dup);
	xfree(op->in_types);
	xfree(op->out_types);

	nlop_free(op->nlop);

	xfree(op);
}

nn_t nn_from_nlop_F(const struct nlop_s* op)
{
	auto result = nn_from_nlop(op);
	nlop_free(op);
	return result;
}

const struct nlop_s* nn_get_nlop(nn_t op)
{
	return op->nlop;
}

void nn_clone_arg_i_from_i(nn_t nn1, int i1, nn_t nn2, int i2)
{
	if (NULL != nn1->in_names[i1])
		xfree(nn1->in_names[i1]);

	if (NULL != nn2->in_names[i2])
		nn1->in_names[i1] = strdup(nn2->in_names[i2]);
	else
		nn1->in_names[i1] = NULL;

	initializer_free(nn1->initializers[i1]);
	nn1->initializers[i1] = initializer_clone(nn2->initializers[i2]);
	nn1->in_types[i1] = nn2->in_types[i2];

	operator_p_free(nn1->prox_ops[i1]);
	nn1->prox_ops[i1] = operator_p_ref(nn2->prox_ops[i2]);

	nn1->dup[i1] = nn2->dup[i2];
}

void nn_clone_arg_o_from_o(nn_t nn1, int o1, nn_t nn2, int o2)
{
	if (NULL != nn1->out_names[o1])
		xfree(nn1->out_names[o1]);

	if (NULL != nn2->out_names[o2])
		nn1->out_names[o1] = strdup(nn2->out_names[o2]);
	else
		nn1->out_names[o1] = NULL;

	nn1->out_types[o1] = nn2->out_types[o2];
}

nn_t nn_clone(nn_t op)
{
	auto result = nn_from_nlop(op->nlop);

	for (int i = 0; i < nn_get_nr_in_args(result); i++)
		nn_clone_arg_i_from_i(result, i, op, i);
	for (int i = 0; i < nn_get_nr_out_args(result); i++)
		nn_clone_arg_o_from_o(result, i, op, i);

	return result;
}

void nn_clone_args(nn_t dst, nn_t src)
{
	auto result = dst;

	int II = MIN(nn_get_nr_in_args(dst), nn_get_nr_in_args(src));
	int OO = MIN(nn_get_nr_out_args(dst), nn_get_nr_out_args(src));

	for (int i = 0; i < II; i++)
		nn_clone_arg_i_from_i(result, i, src, i);
	for (int i = 0; i < OO; i++)
		nn_clone_arg_o_from_o(result, i, src, i);
}


static int get_index_from_name(int N, const char* names[N], const char* name)
{
	for (int i = 0; i < N; i++)
		if ((names[i] != NULL) && (names[i] != NULL) && ((0 == strcmp(name, names[i]))))
			return i;
	return -1;
}

int nn_get_nr_in_args(nn_t op)
{
	return nlop_get_nr_in_args(op->nlop);
}

int nn_get_nr_out_args(nn_t op)
{
	return nlop_get_nr_out_args(op->nlop);
}

int nn_get_nr_named_in_args(nn_t op)
{
	int result = 0;

	for (int i = 0; i < nlop_get_nr_in_args(op->nlop); i++)
		if (NULL != op->in_names[i])
			result++;
	return result;
}

int nn_get_nr_named_out_args(nn_t op)
{
	int result = 0;

	for (int i = 0; i < nlop_get_nr_out_args(op->nlop); i++)
		if (NULL != op->out_names[i])
			result++;
	return result;
}

int nn_get_nr_unnamed_in_args(nn_t op)
{
	return nn_get_nr_in_args(op) - nn_get_nr_named_in_args(op);
}

int nn_get_nr_unnamed_out_args(nn_t op)
{
	return nn_get_nr_out_args(op) - nn_get_nr_named_out_args(op);
}

bool nn_is_num_in_index(nn_t op, int i)
{
	return i < (int)nn_get_nr_unnamed_in_args(op);
}
bool nn_is_num_out_index(nn_t op, int o)
{
	return o < (int)nn_get_nr_unnamed_out_args(op);
}

int nn_get_out_arg_index(nn_t op, int o, const char* oname)
{
	if (NULL != oname) {

		assert((-1 == o) || (0 == o)); // index is ignored anyway
		o = get_index_from_name(nlop_get_nr_out_args(op->nlop), op->out_names, oname);
		if (-1 == o)
			error("Name %s not found!", oname);
	} else {
		assert(o >= -(int)nn_get_nr_unnamed_out_args(op));
		o = o + ((o < 0) ? (int)nn_get_nr_unnamed_out_args(op) : 0);

		assert(o < (int)nn_get_nr_unnamed_out_args(op));
		for (int i = 0; i <= o; i++)
			if (NULL != op->out_names[i])
				o++;
	}

	return o;
}

int nn_get_in_arg_index(nn_t op, int i, const char* iname)
{
	if (NULL != iname) {

		assert((-1 == i) || (0 == i)); // index is ignored anyway
		i = get_index_from_name(nlop_get_nr_in_args(op->nlop), op->in_names, iname);
		if (-1 == i)
			error("Name %s not found!", iname);
	} else {
		assert(i >= -(int)nn_get_nr_unnamed_in_args(op));
		i = i + ((i < 0) ? (int)nn_get_nr_unnamed_in_args(op) : 0);

		assert(i < (int)nn_get_nr_unnamed_in_args(op));
		for (int ii = 0; ii <= i; ii++)
			if (NULL != op->in_names[ii])
				i++;
	}

	return i;
}


const char* nn_get_in_name_from_arg_index(nn_t op, int i, bool clone)
{
	assert(i < nlop_get_nr_in_args(op->nlop));
	if (NULL == op->in_names[i])
		return NULL;
	return clone ? strdup(op->in_names[i]) : op->in_names[i];
}

const char* nn_get_out_name_from_arg_index(nn_t op, int o, bool clone)
{
	assert(o < nlop_get_nr_out_args(op->nlop));
	if (NULL == op->out_names[o])
		return NULL;
	return clone ? strdup(op->out_names[o]) : op->out_names[o];
}

int nn_get_in_index_from_arg_index(nn_t op, int i)
{
	if (NULL != op->in_names[i])
		return 0;

	int result = 0;

	while (i > 0)
		if (NULL == op->in_names[--i])
			result ++;

	return result;
}

int nn_get_out_index_from_arg_index(nn_t op, int o)
{
	if (NULL != op->out_names[o])
		return 0;

	int result = 0;

	while (o > 0)
		if (NULL == op->out_names[--o])
			result ++;

	return result;
}

static bool is_name_in_list(int N, const char* names[N], const char* name)
{
	bool result = false;
	for (int i = 0; i < N; i++)
		result |= (NULL == names[i]) ? false : (0 == strcmp(names[i], name));
	return result;
}

bool nn_is_name_in_in_args(nn_t op, const char* name)
{
	if (0 == nn_get_nr_named_in_args(op))
		return false;

	return is_name_in_list(nn_get_nr_in_args(op), op->in_names, name);
}

bool nn_is_name_in_out_args(nn_t op, const char* name)
{
	if (0 == nn_get_nr_named_out_args(op))
		return false;

	return is_name_in_list(nn_get_nr_out_args(op), op->out_names, name);
}


static int find_first_free_in_name_index(nn_t op, const char* prefix)
{
	int result = -1;
	bool valid = false;
	while (!valid) {

		result++;
		char tmp_name[strlen(prefix) + 10];
		sprintf(tmp_name, "%s%d", prefix, result);
		valid = !nn_is_name_in_in_args(op, tmp_name);
	}

	return result;
}

static int find_first_free_out_name_index(nn_t op, const char* prefix)
{
	int result = -1;
	bool valid = false;
	while (!valid) {

		result++;
		char tmp_name[strlen(prefix) + 10];
		sprintf(tmp_name, "%s%d", prefix, result);
		valid = !nn_is_name_in_out_args(op, tmp_name);
	}

	return result;
}


static nn_t nn_set_input_name(nn_t op, int i, const char* name)
{
	char tmp_name[strlen(name) + 10];
	if ('_' == name[strlen(name) - 1]) {

		int index = find_first_free_in_name_index(op, name);
		sprintf(tmp_name, "%s%d", name, index);
	} else {

		sprintf(tmp_name, "%s", name);
	}

	auto result = nn_clone(op);

	i = nn_get_in_arg_index(result, i, NULL);

	PTR_ALLOC(char[strlen(tmp_name) + 1], nname);
	strcpy(*nname, tmp_name);
	result->in_names[i] = *PTR_PASS(nname);

	return result;
}

static nn_t nn_set_output_name(nn_t op, int o, const char* name)
{
	char tmp_name[strlen(name) + 10];
	if ('_' == name[strlen(name) - 1]) {

		int index = find_first_free_out_name_index(op, name);
		sprintf(tmp_name, "%s%d", name, index);
	} else {

		sprintf(tmp_name, "%s", name);
	}

	auto result = nn_clone(op);

	o = nn_get_out_arg_index(result, o, NULL);

	PTR_ALLOC(char[strlen(tmp_name) + 1], nname);
	strcpy(*nname, tmp_name);
	result->out_names[o] = *PTR_PASS(nname);

	return result;
}

nn_t nn_set_input_name_F(nn_t op, int i, const char* name)
{
	auto result = nn_set_input_name(op, i, name);
	nn_free(op);
	return result;
}

nn_t nn_set_output_name_F(nn_t op, int o, const char* name)
{
	auto result = nn_set_output_name(op, o, name);
	nn_free(op);
	return result;
}

nn_t nn_unset_input_name_F(nn_t op, const char* name)
{
	int i = nn_get_in_arg_index(op, 0, name);
	auto result = nn_clone(op);

	xfree(result->in_names[i]);
	result->in_names[i] = NULL;
	result = nn_shift_input_index_F(result, nn_get_nr_in_args(op) - 1, i);

	nn_free(op);
	return result;
}

nn_t nn_unset_output_name_F(nn_t op, const char* name)
{
	int i = nn_get_out_arg_index(op, 0, name);
	auto result = nn_clone(op);

	xfree(result->out_names[i]);
	result->out_names[i] = NULL;
	result = nn_shift_output_index_F(result, nn_get_nr_out_args(op) - 1, i);

	nn_free(op);
	return result;
}

nn_t nn_rename_input_F(nn_t op, const char* nname, const char* oname)
{
	int i = nn_get_in_arg_index(op, 0, oname);

	auto result = nn_clone(op);

	xfree(result->in_names[i]);
	PTR_ALLOC(char[strlen(nname) + 1], nnname);
	strcpy(*nnname, nname);
	result->in_names[i] = *PTR_PASS(nnname);

	nn_free(op);

	return result;
}

nn_t nn_rename_output_F(nn_t op, const char* nname, const char* oname)
{
	int o = nn_get_out_arg_index(op, 0, oname);

	auto result = nn_clone(op);

	xfree(result->out_names[o]);
	PTR_ALLOC(char[strlen(nname) + 1], nnname);
	strcpy(*nnname, nname);
	result->out_names[o] = *PTR_PASS(nnname);

	nn_free(op);

	return result;
}

void nn_get_in_names_copy(int N, const char* names[N], nn_t op)
{
	assert(nn_get_nr_named_in_args(op) == N);

	for (int i = 0, i_name= 0; i_name < N; i++)
		if (NULL != op->in_names[i])
			names[i_name++] = ptr_printf("%s", op->in_names[i]);

}

void nn_get_out_names_copy(int N, const char* names[N], nn_t op)
{
	assert(nn_get_nr_named_out_args(op) == N);

	for (int i = 0, i_name= 0; i_name < N; i++)
		if (NULL != op->out_names[i])
			names[i_name++] = ptr_printf("%s", op->out_names[i]);

}

void nn_get_in_args_names(nn_t op, int II, const char* names[II], _Bool copy)
{
	assert(II == nn_get_nr_in_args(op));
	for (int i = 0; i < II; i++)
		names[i] = (NULL != op->in_names[i]) && copy ? strdup(op->in_names[i]) : op->in_names[i];
}

void nn_get_out_args_names(nn_t op, int OO, const char* names[OO], _Bool copy)
{
	assert(OO == nn_get_nr_out_args(op));
	for (int i = 0; i < OO; i++)
		names[i] = (NULL != op->out_names[i]) && copy ? strdup(op->out_names[i]) : op->out_names[i];
}


nn_t nn_set_initializer_F(nn_t op, int i, const char* iname, const struct initializer_s* ini)
{
	auto result = nn_clone(op);
	i = nn_get_in_arg_index(result, i, iname);
	if (NULL != result->initializers[i])
		initializer_free(result->initializers[i]);
	result->initializers[i] = ini;
	nn_free(op);
	return result;
}

nn_t nn_set_prox_op_F(nn_t op, int i, const char* iname, const struct operator_p_s* opp)
{
	auto result = nn_clone(op);
	i = nn_get_in_arg_index(result, i, iname);
	if (NULL != result->prox_ops[i])
		operator_p_free(result->prox_ops[i]);
	auto iov = operator_p_domain(opp);
	assert(iovec_check(nlop_generic_domain(op->nlop, i), iov->N, iov->dims, iov->strs));
	result->prox_ops[i] = opp;
	nn_free(op);
	return result;
}

const struct operator_p_s* nn_get_prox_op(nn_t op, int i, const char* iname)
{
	i = nn_get_in_arg_index(op, i, iname);
	return nn_get_prox_op_arg_index(op, i);

}
const struct operator_p_s* nn_get_prox_op_arg_index(nn_t op, int i)
{
	return op->prox_ops[i];
}

void nn_get_prox_ops(nn_t op, int N, const struct operator_p_s* prox_ops[N])
{
	assert(N == nn_get_nr_in_args(op));
	for (int i = 0; i < N; i++)
		prox_ops[i] = op->prox_ops[i];
}

nn_t nn_set_dup_F(nn_t op, int i, const char* iname, bool dup)
{
	i = nn_get_in_arg_index(op, i, iname);
	op->dup[i] = dup;
	return op;
}

bool nn_get_dup(nn_t op, int i, const char* iname)
{
	i = nn_get_in_arg_index(op, i, iname);
	return op->dup[i];
}


nn_t nn_set_in_type_F(nn_t op, int i, const char* iname, enum IN_TYPE in_type)
{
	auto result = nn_clone(op);
	i = nn_get_in_arg_index(result, i, iname);
	result->in_types[i] = in_type;
	nn_free(op);
	return result;
}

nn_t nn_set_out_type_F(nn_t op, int o, const char* oname, enum OUT_TYPE out_type)
{
	auto result = nn_clone(op);
	o = nn_get_out_arg_index(result, o, oname);
	result->out_types[o] = out_type;
	nn_free(op);
	return result;
}

const char** nn_get_out_names(nn_t op) {

	return op->out_names;
}

const char** nn_get_in_names(nn_t op) {

	return op->in_names;
}

const struct iovec_s* nn_generic_domain(nn_t op, int i, const char* iname)
{
	i = nn_get_in_arg_index(op, i, iname);
	return nlop_generic_domain(op->nlop, i);
}

const struct iovec_s* nn_generic_codomain(nn_t op, int o, const char* oname)
{
	o = nn_get_out_arg_index(op, o, oname);
	return nlop_generic_codomain(op->nlop, o);
}

void nn_debug(enum debug_levels dl, nn_t x)
{
	int II = nn_get_nr_in_args(x);

	debug_printf(dl, "NN\ninputs: %d\n", II);

	for (int i = 0, index = 0; i < II; i++) {

		auto io = nlop_generic_domain(x->nlop, i);
		char index_name[17];
		sprintf(index_name, "INDEX %d", index);
		debug_printf(dl, "%-15s", (NULL == x->in_names[i]) ? index_name : x->in_names[i]);
		debug_print_dims(dl, io->N, io->dims);

		if (NULL == x->in_names[i])
			index++;
	}

	int OO = nn_get_nr_out_args(x);

	debug_printf(dl, "outputs: %d\n", OO);

	for (int o = 0, index = 0; o < OO; o++) {

		auto io = nlop_generic_codomain(x->nlop, o);

		char index_name[17];
		sprintf(index_name, "INDEX %d", index);
		debug_printf(dl, "%-15s", (NULL == x->out_names[o]) ? index_name : x->out_names[o]);

		debug_print_dims(dl, io->N, io->dims);

		if (NULL == x->out_names[o])
			index++;
	}
}

int nn_get_nr_weights(nn_t op)
{
	int result = 0;
	for (int i = 0; i < nn_get_nr_in_args(op); i++){

		if (NULL == op->initializers[i])
			assert((IN_OPTIMIZE != op->in_types[i]) && (IN_BATCHNORM != op->in_types[i]));
		else
			result++;
	}
	return result;
}

void nn_get_in_types(nn_t op, int N, enum IN_TYPE in_types[N])
{
	assert(N == nn_get_nr_in_args(op));

	for (int i = 0; i < N; i++)
		in_types[i] = op->in_types[i];
}

void nn_get_out_types(nn_t op, int N, enum OUT_TYPE out_types[N])
{
	assert(N == nn_get_nr_out_args(op));

	for (int i = 0; i < N; i++)
		out_types[i] = op->out_types[i];
}

nn_t nn_checkpoint_F(nn_t op, bool der_once, bool clear_mem)
{
	auto result = nn_from_nlop_F(nlop_checkpoint_create(op->nlop, der_once, clear_mem));

	for (int i = 0; i < nn_get_nr_in_args(result); i++)
		nn_clone_arg_i_from_i(result, i, op, i);
	for (int i = 0; i < nn_get_nr_out_args(result); i++)
		nn_clone_arg_o_from_o(result, i, op, i);

	nn_free(op);

	return result;
}

void nn_export_graph(const char* filename, nn_t op)
{
	int II = nlop_get_nr_in_args(op->nlop);
	int OO = nlop_get_nr_out_args(op->nlop);

	const char* arg_nodes[II + OO];

	graph_t graph = operator_get_graph(op->nlop->op);
	const char* str = print_internl_graph(graph, true, II + OO, arg_nodes);
	graph_free(graph);

	FILE *fp;
	fp = fopen(filename, "w+");

	assert(0 != fp);

	fprintf(fp, "digraph {\nnewrank=true;\n");
	fprintf(fp, "{\n%s}\n", str);

	int counter_input = 0;
	int counter_weight = 0;

	for (int i = 0; i < II; i++)
		if ((IN_OPTIMIZE == op->in_types[i]) || (IN_BATCHNORM == op->in_types[i]))
			counter_weight++;
		else
			counter_input++;

	fprintf(fp, "{\nrank=same\n");

	int index = 0;
	if (0 < counter_input) {

		fprintf(fp, "subgraph cluster_inputs{\nlabel = \"Inputs\";\nrank=source;\n");
		for (int i = 0; i < counter_input; i++, index ++) {

			while ((IN_OPTIMIZE == op->in_types[index]) || (IN_BATCHNORM == op->in_types[index]))
				index++;

			auto iov = nlop_generic_domain(nn_get_nlop(op), index);
			const char* tmp = ptr_print_dims(iov->N, iov->dims);
			const char* str_dims = ptr_printf("\\n%s", tmp);
			xfree(tmp);

			if (NULL != op->in_names[index])
				fprintf(fp, "%s [shape = diamond, label = \"%s%s\"];\n", arg_nodes[index + OO], op->in_names[index], str_dims);
			else
				fprintf(fp, "%s [shape = diamond, label = \"Input_%d%s\"];\n", arg_nodes[index + OO], i, str_dims);

			xfree(str_dims);
		}

		fprintf(fp, "}\n");
	}

	index = 0;
	if (0 < counter_weight) {

		fprintf(fp, "subgraph cluster_weights{\n label = \"Weights\";\n rank=source;\n");

		for (int i = 0; i < counter_weight; i++, index ++) {

			while (!((IN_OPTIMIZE == op->in_types[index]) || (IN_BATCHNORM == op->in_types[index])))
				index++;

			auto iov = nlop_generic_domain(nn_get_nlop(op), index);
			const char* tmp = ptr_print_dims(iov->N, iov->dims);
			const char* str_dims = ptr_printf("\\n%s", tmp);
			xfree(tmp);

			if (NULL != op->in_names[index])
				fprintf(fp, "%s [shape = diamond, label = \"%s%s\"];\n", arg_nodes[index + OO], op->in_names[index], str_dims);
			else
				fprintf(fp, "%s [shape = diamond, label = \"Weight_%d%s\"];\n", arg_nodes[index + OO], i, str_dims);

			xfree(str_dims);
		}
		fprintf(fp, "}\n");
	}

	if (1 < II) {

		fprintf(fp, "{\nedge[ style=invis];\n%s", arg_nodes[OO]);
		for (int i = 1; i < counter_weight; i++)
				fprintf(fp, " -> %s", arg_nodes[i + OO]);
		fprintf(fp, "\n}\n");
	}
	fprintf(fp, "}\n");


	if (0 < OO) {

		fprintf(fp, "subgraph cluster_outputs{\nlabel = \"Outputs\";\nrank=sink;\n");
		for (int i = 0; i < OO; i++) {

			auto iov = nlop_generic_codomain(nn_get_nlop(op), i);
			const char* tmp = ptr_print_dims(iov->N, iov->dims);
			const char* str_dims = ptr_printf("\\n%s", tmp);
			xfree(tmp);

			if (NULL != op->out_names[i])
				fprintf(fp, "%s [shape = diamond, label = \"%s%s\"];\n", arg_nodes[i], op->out_names[i], str_dims);
			else
				fprintf(fp, "%s [shape = diamond, label = \"Output_%d%s\"];\n", arg_nodes[i], i, str_dims);

			xfree(str_dims);
		}
		fprintf(fp, "}\n");
	}
	fprintf(fp, "}\n");

	fclose(fp);
}