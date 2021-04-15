#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"
#include "misc/shrdptr.h"
#include "misc/nested.h"
#include "misc/list.h"
#include "misc/graph.h"
#include "misc/list.h"


#include "num/ops.h"
#include "num/iovec.h"

#include "num/ops_graph.h"


struct node_operator_s {

	INTERFACE(struct node_s);
	const struct operator_s* op;
};

static DEF_TYPEID(node_operator_s);

static void node_operator_del(const struct node_s* _node)
{
	auto node = CAST_DOWN(node_operator_s, _node);
	operator_free(node->op);
}

static node_t node_operator_create(const struct operator_s* op, const char* name)
{
	PTR_ALLOC(struct node_operator_s, node);
	SET_TYPEID(node_operator_s, node);

	node_init(&(node->INTERFACE), operator_nr_args(op), operator_get_io_flags(op), name, false, NULL);

	node->op = operator_ref(op);

	node->INTERFACE.node_del = node_operator_del;

	return CAST_UP(PTR_PASS(node));
}


static node_t node_operator_container_create(const struct operator_s* op, const char* name, graph_t subgraph)
{
	PTR_ALLOC(struct node_operator_s, node);
	SET_TYPEID(node_operator_s, node);

	node_init(&(node->INTERFACE), operator_nr_args(op), operator_get_io_flags(op), name, false, subgraph);

	node->op = operator_ref(op);

	node->INTERFACE.node_del = node_operator_del;

	return CAST_UP(PTR_PASS(node));
}

static const struct operator_s* get_operator_from_node(node_t _node)
{
	auto node = CAST_DOWN(node_operator_s , _node);
	return node->op;
}


struct node_arg_s {

	INTERFACE(struct node_s);

	const struct iovec_s* iov;
	bool output;
};

static DEF_TYPEID(node_arg_s);

static void node_arg_del(const struct node_s* _node)
{
	auto node = CAST_DOWN(node_arg_s, _node);
	iovec_free(node->iov);
}

static const char* print_node_arg(const struct node_s* node) {

	return ptr_printf("node_%p [label=\"%s\" shape=diamond];\n", node, (NULL == node->name) ? node->TYPEID->name : node->name);
}

static node_t node_arg_create(bool output, const struct iovec_s* iov)
{
	PTR_ALLOC(struct node_arg_s, node);
	SET_TYPEID(node_arg_s, node);

	const char* name = ptr_printf("%s\\n[", output ? "Output" : "Input");
	for (unsigned int i = 0; i < iov->N; i++) {
		auto tmp = name;
		name = ptr_printf("%s %ld", tmp, iov->dims[i]);
		xfree(tmp);
	}
	auto tmp = name;
	name = ptr_printf("%s ]", tmp);
	xfree(tmp);

	bool io_flags[1] = { !output };

	node_init(&(node->INTERFACE), 1, io_flags, name, true, NULL);
	xfree(name);

	node->INTERFACE.node_print = print_node_arg;

	node->output = output;
	node->iov = iovec_create2(iov->N, iov->dims, iov->strs, iov->size);

	node->INTERFACE.node_del = node_arg_del;

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
