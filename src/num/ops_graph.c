/* Copyright 2021. Uecker Lab, University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * */

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"
#include "misc/graph.h"
#include "misc/list.h"

#include "num/multind.h"
#include "num/ops.h"
#include "num/iovec.h"

#include "ops_graph.h"


struct node_operator_s {

	struct node_s super;
	const struct operator_s* op;
};

static DEF_TYPEID(node_operator_s);

static void node_operator_del(const struct node_s* _node)
{
	auto node = CAST_DOWN(node_operator_s, _node);
	operator_free(node->op);
}

static node_t node_operator_create(const struct operator_s* op, const char* name);

static struct node_s* node_operator_clone(const struct node_s* node)
{
	auto x = node_operator_create(CAST_DOWN(node_operator_s, node)->op, node->name);

	x->subgraph = graph_clone(node->subgraph);

	return x;
}

static bool node_operator_is_output(const struct node_s* _node, int idx)
{
	auto node = CAST_DOWN(node_operator_s, _node);
	return operator_get_io_flags(node->op)[idx];
	
}

static node_t node_operator_create(const struct operator_s* op, const char* name)
{
	PTR_ALLOC(struct node_operator_s, node);
	SET_TYPEID(node_operator_s, node);

	node_init(&(node->super), operator_nr_args(op), node_operator_is_output, name, false, NULL);

	node->op = operator_ref(op);

	node->super.node_del = node_operator_del;
	node->super.node_clone = node_operator_clone;

	return CAST_UP(PTR_PASS(node));
}


static node_t node_operator_container_create(const struct operator_s* op, const char* name, graph_t subgraph)
{
	PTR_ALLOC(struct node_operator_s, node);
	SET_TYPEID(node_operator_s, node);

	node_init(&(node->super), operator_nr_args(op), node_operator_is_output, name, false, subgraph);

	node->op = operator_ref(op);

	node->super.node_del = node_operator_del;
	node->super.node_clone = node_operator_clone;

	return CAST_UP(PTR_PASS(node));
}

const struct operator_s* get_operator_from_node(const struct node_s* _node)
{
	auto node = CAST_MAYBE(node_operator_s , _node);
	if (NULL == node)
		return NULL;

	return node->op;
}


struct node_arg_s {

	struct node_s super;

	const struct iovec_s* iov;
	bool output;
};

static DEF_TYPEID(node_arg_s);

static void node_arg_del(const struct node_s* _node)
{
	auto node = CAST_DOWN(node_arg_s, _node);
	iovec_free(node->iov);
}

static const char* print_node_arg(const struct node_s* _node)
{
	auto node = CAST_DOWN(node_arg_s, _node);
	auto iov = node->iov;

	const char* name = ptr_printf("%s\\n[", node->output ? "Output" : "Input");

	for (int i = 0; i < iov->N; i++) {

		auto tmp = name;

		name = ptr_printf("%s %ld", tmp, iov->dims[i]);

		xfree(tmp);
	}

	const char* name2 = ptr_printf("%s ]", name);
	xfree(name);

	const char* ret = ptr_printf("node_%p [label=\"%s\" shape=diamond];\n", node, name2);
	xfree(name2);

	return ret;
}

static node_t node_arg_create(bool output, const struct iovec_s* iov);

static struct node_s* node_arg_clone(const struct node_s* node)
{
	auto n = CAST_DOWN(node_arg_s, node);

	return node_arg_create(n->output, n->iov);
}

static bool node_arg_is_output(const struct node_s* _node, int idx)
{
	auto node = CAST_DOWN(node_arg_s, _node);
	assert(0 == idx);
	return !(node->output);
	
}

static node_t node_arg_create(bool output, const struct iovec_s* iov)
{
	PTR_ALLOC(struct node_arg_s, node);
	SET_TYPEID(node_arg_s, node);

	node_init(&(node->super), 1, node_arg_is_output, NULL, true, NULL);


	node->super.node_print = print_node_arg;

	node->output = output;
	node->iov = iovec_create2(iov->N, iov->dims, iov->strs, iov->size);

	node->super.node_del = node_arg_del;
	node->super.node_clone = node_arg_clone;

	return CAST_UP(PTR_PASS(node));
}

static graph_t create_operator_graph_from_node(node_t node)
{
	auto op = get_operator_from_node(node);
	int N = operator_nr_args(op);

	auto result = graph_create();

	graph_add_node(result, node);

	for (int i = 0; i < N; i++) {

		bool output = operator_get_io_flags(op)[i];

		node_t node_arg = node_arg_create(output, operator_arg_domain(op, i));
		graph_add_node(result, node_arg);

		struct vertex_s ver_node = {.node = node, .idx = i};
		struct vertex_s ver_node_arg = {.node = node_arg, .idx = 0};

		if (output)
			graph_add_edge(ver_node, ver_node_arg);
		else
			graph_add_edge(ver_node_arg, ver_node);
	}

	return result;
}

graph_t create_graph_operator(const struct operator_s* op, const char* name)
{
	node_t node = node_operator_create(op, name);

	return create_operator_graph_from_node(node);
}

graph_t create_graph_container(const struct operator_s* op, const char* name, graph_t subgraph)
{
	node_t node = node_operator_container_create(op, name, subgraph);

	return create_operator_graph_from_node(node);
}

graph_t operator_graph_combine_F(int N_ops, graph_t ops[N_ops])
{
	return combine_graphs_F(N_ops, ops);
}

graph_t operator_graph_chain_F(int N_ops, graph_t ops[N_ops])
{
	graph_t ops_perm[N_ops];
	for (int i = 0; i < N_ops; i++)
		ops_perm[i] = ops[N_ops - 1 - i];

	auto result = combine_graphs_F(N_ops, ops_perm);

	for (int i = 1; i < N_ops; i++)
		result = link_graphs_F(result, 2, 1);

	return result;
}

graph_t operator_graph_dup_F(graph_t op, int a, int b)
{
	return dup_graphs_F(op, a, b);
}

graph_t operator_graph_link_F(graph_t op, int oo, int ii)
{
	return link_graphs_F(op, oo, ii);;
}

graph_t operator_graph_permute_F(graph_t op, int N, const int perm[N])
{
	return perm_ext_graphs_F(op, N, perm);
}

graph_t operator_graph_reshape_F(graph_t op, int i, int N, const long dims[N])
{
	auto node = CAST_DOWN(node_arg_s, (node_t)list_get_item(op->ext_nodes, i));
	size_t size = node->iov->size;
	iovec_free(node->iov);
	node->iov = iovec_create(N, dims, size);

	return op;
}

void operator_export_graph_dot(const char* filename, const struct operator_s* op)
{
	graph_t graph = operator_get_graph(op);

	export_graph_dot(filename, graph);

	graph_free(graph);
}

static const struct iovec_s* get_iovec_from_node(node_t _node, int idx)
{
	if (NULL != CAST_MAYBE(node_operator_s , _node))
		return operator_arg_domain(get_operator_from_node(_node), idx);

	if (NULL != CAST_MAYBE(node_arg_s , _node)) {

		assert(0 == idx);
		auto node = CAST_DOWN(node_arg_s , _node);
		return node->iov;
	}

	assert(0);
	return NULL;
}





static inline void reduce_index(int a, int N, int index[N])
{
	for (int j = a + 1; j < N; j++)
		index[j]--;
}





static bool cmp_identity_node(const void* data, const void* _ref) {

	const struct node_s* _node = data;
	auto node = CAST_DOWN(node_operator_s, _node);

	assert(NULL == _ref);

	return check_simple_copy(node->op);
}

static bool cmp_end_node(const void* data, const void* /*_ref*/)
{
	const struct node_s* node = data;
	
	bool end = true;
	for (int i = 0; i < node->N_vertices; i++)
		end &= !(node->is_output(node, i));

	return end;
}

//remove identity operator from graph
graph_t operator_graph_optimize_identity_F(graph_t graph)
{
	list_t nodes = list_get_sublist(graph->nodes, NULL, cmp_identity_node);

	node_t node = list_pop(nodes);

	while (NULL != node){

		graph_bridge_node(graph, node);
		node = list_pop(nodes);
	}

	list_free(nodes);

	nodes = list_get_sublist(graph->nodes, NULL, cmp_end_node);

	node = list_pop(nodes);

	while (NULL != node){

		graph_remove_end_node(graph, node);
		node = list_pop(nodes);
	}

	list_free(nodes);

	return graph;
}

static void edge_separator_node(node_t ext_nodes[2], struct vertex_s vertex)
{
	auto iov = get_iovec_from_node(vertex.node, vertex.idx);
	bool output = vertex.node->is_output(vertex.node, vertex.idx);

	ext_nodes[0] = node_arg_create(!output, iov);
	ext_nodes[1] = node_arg_create(output, iov);
}



static enum node_identic node_cmp_operator(const struct node_s* _a, const struct node_s* _b) {

	auto a = CAST_MAYBE(node_operator_s, _a);
	auto b = CAST_MAYBE(node_operator_s, _b);

	if (NULL == a)
		return NODE_NOT_IDENTICAL;

	if (NULL == b)
		return NODE_NOT_IDENTICAL;

	if (operator_nr_args(a->op) != operator_nr_args(a->op))
		return NODE_NOT_IDENTICAL;


	auto iova = operator_arg_domain(a->op, 0);
	auto iovb = operator_arg_domain(b->op, 0);

	if (operator_is_zadd(a->op) && operator_is_zadd(b->op) && iovec_check(iova, iovb->N, iovb->dims, iovb->strs))
		return NODE_IDENTICAL_SYMMETRIC;

	if (a->op == b->op)
		return NODE_IDENTICAL;

	return NODE_NOT_IDENTICAL;
}

graph_t operator_graph_optimize_identify_F(graph_t graph)
{
	return graph_identify_nodes_F(graph, node_cmp_operator);
}


static bool node_is_sum(const struct node_s* node)
{
	if (NULL == get_operator_from_node(node))
		return false;
	return operator_is_zadd(get_operator_from_node(node));
}

// optimizes Ax + Ay to A(x+y)
static graph_t operator_graph_optimize_linops_F_internal(graph_t graph, node_cmp_t linop_identify)
{
	list_t linop_sum = graph_get_linop_sum(graph, linop_identify, node_is_sum, SUM_NODES_AND_TWO_IDENTICAL_LINOPS);

	while (0 < list_count(linop_sum)) {

		list_t sum_nodes = list_pop(linop_sum);

		assert(3 <= list_count(sum_nodes));

		node_t node_linop_1 = list_get_item(sum_nodes, list_count(sum_nodes) - 2);
		node_t node_linop_2 = list_get_item(sum_nodes, list_count(sum_nodes) - 1);

		graph_t subgraph = graph_cluster_nodes_F(graph, sum_nodes, edge_separator_node);

		int additional_sums = list_count(subgraph->ext_nodes) - 3;
		assert(0 <= additional_sums);

		int out_index = -1; //output of sum
		int in1_index = -1; //input of linop
		int in2_index = -1; //input of linop
		//other inputs are simply added to the result

		for (int i = 0; i < list_count(subgraph->ext_nodes); i++) {

			node_t ext_node = list_get_item(subgraph->ext_nodes, i);

			if (!ext_node->is_output(ext_node, 0)) {

				assert(-1 == out_index);
				out_index = i;
			} else {

				vertex_t vertex = list_get_item(ext_node->edges[0], 0);

				if ((vertex->node == node_linop_1) || (vertex->node == node_linop_2)) {

					if (-1 == in1_index)
						in1_index = i;
					else
						in2_index = i;
				}
			}
		}

		assert(-1 != out_index);
		assert(-1 != in1_index);
		assert(-1 != in2_index);

		const struct operator_s* op_linop = get_operator_from_node(node_linop_1);

		auto dom = operator_domain(op_linop);
		auto cod = operator_codomain(op_linop);

		auto op_sum = operator_zadd_create(2, dom->N, dom->dims);
		auto op_combi = operator_combi_create(2, (const struct operator_s*[2]){op_linop, op_sum});
		auto op_chain = operator_link_create(op_combi, 2, 1);

		operator_free(op_sum);
		operator_free(op_combi);

		if (0 < additional_sums) {

			op_sum = operator_zadd_create(1 +  additional_sums, cod->N, cod->dims);
			op_combi = operator_combi_create(2, (const struct operator_s*[2]){op_sum, op_chain});

			operator_free(op_chain);
			operator_free(op_sum);

			op_chain = operator_link_create(op_combi, 2 + additional_sums, 1 + additional_sums);

			operator_free(op_combi);
		}

		int perm[3 + additional_sums];
		for (int i = 0, ip = 0; i < 3 + additional_sums; i++) {

			if (i == out_index) {

				perm[i] = 0;
				continue;
			}

			if (i == in1_index) {

				perm[i] = 1 + additional_sums;
				continue;
			}

			if (i == in2_index) {

				perm[i] = 2 + additional_sums;
				continue;
			}

			perm[i] = 1 + (ip++);
		}

		auto op_result = operator_permute(op_chain, 3 + additional_sums, perm);
		operator_free(op_chain);

		graph_free(subgraph);
		subgraph = operator_get_graph(op_result);
		operator_free(op_result);

		graph = graph_reinsert_subgraph_FF(graph, subgraph);

		if (0 == list_count(linop_sum)) {

			list_free(linop_sum);
			linop_sum = graph_get_linop_sum(graph, linop_identify, node_is_sum, SUM_NODES_AND_TWO_IDENTICAL_LINOPS);
		}
	}

	list_free(linop_sum);


	return graph;
}

static graph_t create_sum_graph(bool multi_sum, int II, int out_index, int N, const long dims[N])
{
	const struct operator_s* sum_op = NULL;

	if (multi_sum)
		sum_op = operator_zadd_create(II, N, dims);
	else {
		sum_op = operator_zadd_create(2, N, dims);
		for (int i = 2; i < II; i++) {

			auto op_sum_tmp = operator_zadd_create(2, N, dims);
			auto op_combi = operator_combi_create(2, (const struct operator_s*[2]){op_sum_tmp, sum_op});

			operator_free(op_sum_tmp);
			operator_free(sum_op);

			sum_op = operator_link_create(op_combi, 3, 2);

			operator_free(op_combi);
		}
	}

	int perm[II + 1];
	for (int i = 0, ip = 0; i < II + 1; i++)
		perm[i] = (i == out_index) ? 0 : 1 + ip++;

	auto op_result = operator_permute(sum_op, 1 + II, perm);
	auto result = operator_get_graph(op_result);

	operator_free(sum_op);
	operator_free(op_result);

	return result;
}

static enum node_identic node_cmp_false(const struct node_s* /*_a*/, const struct node_s* /*_b*/)
{
	return NODE_NOT_IDENTICAL;
}

// replace sum chained into a sum by multi sum (and inverse)
graph_t operator_graph_sum_to_multi_sum_F(graph_t graph, bool inverse)
{
	list_t linop_sum = graph_get_linop_sum(graph, node_cmp_false, node_is_sum, inverse ? MULTI_SUM_NODES_ONLY : SUM_NODES_ONLY);

	while (0 < list_count(linop_sum)) {

		list_t sum_nodes = list_pop(linop_sum);

		graph_t subgraph = graph_cluster_nodes_F(graph, sum_nodes, edge_separator_node);

		int out_index = -1;
		for (int i = 0; i < list_count(subgraph->ext_nodes); i++) {

			node_t ext_node = list_get_item(subgraph->ext_nodes, i);

			if (!ext_node->is_output(ext_node, 0)) {

				assert(-1 == out_index);
				out_index = i;
			}
		}

		auto iov = get_iovec_from_node(list_get_item(subgraph->nodes, 0), 0);
		auto new_subgraph = create_sum_graph(!inverse, list_count(subgraph->ext_nodes) - 1, out_index, iov->N, iov->dims);

		graph_free(subgraph);

		graph = graph_reinsert_subgraph_FF(graph, new_subgraph);
	}

	list_free(linop_sum);

	return graph;
}

graph_t operator_graph_optimize_linops_F(graph_t graph, node_cmp_t linop_identify)
{
	list_t linop_sum = graph_get_linop_sum(graph, linop_identify, node_is_sum, SUM_OPS_AND_OPS);

	while (0 < list_count(linop_sum)) {

		list_t sum_nodes = list_pop(linop_sum);

		graph_t subgraph = graph_cluster_nodes_F(graph, sum_nodes, edge_separator_node);

		subgraph = operator_graph_optimize_linops_F_internal(subgraph, linop_identify);

		graph = graph_reinsert_subgraph_FF(graph, subgraph);

		if (0 == list_count(linop_sum)) {

			list_free(linop_sum);
			linop_sum = graph_get_linop_sum(graph, linop_identify, node_is_sum, SUM_OPS_AND_OPS);
		}
	}

	list_free(linop_sum);

	return graph;
}

struct operator_graph_s {

	operator_data_t super;
	graph_t graph;
};

DEF_TYPEID(operator_graph_s);

static void graph_apply(const operator_data_t* _data, int _N, void* _args[_N])
{
	auto d = CAST_DOWN(operator_graph_s, _data);
	int N = list_count(d->graph->nodes);

	assert(list_count(d->graph->ext_nodes) == _N);


	void** arg_lists[N];
	int* ref_counts[N];
	memset(arg_lists, 0, sizeof arg_lists);		// -fanalyzer uninitialized
	memset(ref_counts, 0, sizeof ref_counts);	// -fanalyzer uninitialized

	for (int i = 0; i < N; i++) {

		node_t node = list_get_item(d->graph->nodes, i);
		int Nv = node->N_vertices;

		void* args[Nv];
		int ref_count[Nv];
		memset(args, 0, sizeof args);		// -fanalyzer uninitialized
		memset(ref_count, 0, sizeof ref_count);	// -fanalyzer uninitialized

		for (int i = 0; i < Nv; i++) {

			if (((vertex_t)list_get_item(node->edges[i], 0))->node->external) {

				int idx = list_get_first_index(d->graph->ext_nodes, ((vertex_t)list_get_item(node->edges[i], 0))->node, NULL);
				args[i] = _args[idx];
				ref_count[i] = -1;
				continue;
			}

			if (node->is_output(node, i)) {

				auto iov = get_iovec_from_node(node, i);
				args[i] = md_alloc_sameplace(iov->N, iov->dims, iov->size, _args[0]);
				ref_count[i] = list_count(node->edges[i]);
			} else {

				int j = 0;
				vertex_t ver = list_get_item(node->edges[i], 0);

				while (list_get_item(d->graph->nodes, j) != ver->node)
					j++;
				
				args[i] = arg_lists[j][ver->idx];
				ref_count[i] = --ref_counts[j][ver->idx];
			}
		}

		arg_lists[i] = ARR_CLONE(void*[Nv], args);
		ref_counts[i] = ARR_CLONE(int[Nv], ref_count);

		const struct operator_s* op = CAST_DOWN(node_operator_s, node)->op;
		operator_generic_apply_unchecked(op, Nv, args);

		for (int i = 0; i < Nv; i++)
			if (0 == ref_count[i])
				md_free(args[i]);
	}

	for (int i = 0; i < N; i++) {

		xfree(arg_lists[i]);
		xfree(ref_counts[i]);
	}
}

static void op_graph_free(const operator_data_t* _data)
{
	auto data = CAST_DOWN(operator_graph_s, _data);

	graph_free(data->graph);
	xfree(data);
}

static const struct graph_s* operator_graph_get_graph(const struct operator_s* op)
{
	const auto d = CAST_DOWN(operator_graph_s, operator_get_data(op));
	return graph_clone(d->graph);
}



const struct operator_s* operator_graph_createF(graph_t graph)
{
	PTR_ALLOC(struct operator_graph_s, c);
	SET_TYPEID(operator_graph_s, c);

	graph = operator_graph_optimize_identity_F(graph);
	graph = operator_graph_optimize_identify_F(graph);

	c->graph = graph_topological_sort_F(graph);

	int N = list_count(graph->ext_nodes);

	bool ioflags[N];
	int D[N];
	const long* dims[N];
	const long* strs[N];

	for (int i = 0; i < N; i++) {

		struct node_arg_s* node = CAST_DOWN(node_arg_s, (node_t)list_get_item(graph->ext_nodes, i));
		
		ioflags[i] = node->output;
		D[i] = node->iov->N;
		dims[i] = node->iov->dims;
		strs[i] = node->iov->strs;
	}


	return operator_generic_create2(N, ioflags, D, dims, strs, CAST_UP(PTR_PASS(c)), graph_apply, op_graph_free, operator_graph_get_graph);
}

list_t operator_graph_get_list(const struct operator_s* op)
{
	const auto d = CAST_MAYBE(operator_graph_s, operator_get_data(op));

	if (NULL == d)
		return NULL;

	list_t result = list_create();

	for (int i = 0; i < list_count(d->graph->nodes); i++)
		list_merge(result, operator_get_list(CAST_DOWN(node_operator_s, (node_t)(list_get_item(d->graph->nodes, i)))->op), true);

	return result;
}

const struct operator_s* graph_to_operator_F(graph_t graph)
{
	return operator_graph_createF(graph);
}
