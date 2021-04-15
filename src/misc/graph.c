
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"
#include "misc/shrdptr.h"
#include "misc/nested.h"
#include "misc/list.h"

#include "graph.h"

typedef list_t nodes_t;
typedef list_t vertices_t;

static node_t nodes_get(nodes_t nodes, int index)
{
	return list_get_item(nodes, index);
}

static int nodes_index(nodes_t nodes, node_t node)
{
	return list_get_first_index(nodes, node, NULL);
}

static vertex_t vertices_get(vertices_t vertices, int index)
{
	return list_get_item(vertices, index);
}

void node_free(node_t x)
{
	for (int i = 0; i < x->N_vertices; i++)
		if (0 < list_count(x->edges[i]))
			error("Adjacency list of node is not empty!");
		else
			list_free(x->edges[i]);

	xfree(x->edges);
	xfree(x->io_flags);

	if (NULL != x->node_del)
		x->node_del(x);
	if (NULL != x->name)
		xfree(x->name);
	if (NULL != x->subgraph)
		graph_free(x->subgraph);

	xfree(x);
}

void node_init(struct node_s* x, int N_vertices, const bool io_flags[N_vertices], const char* name, bool external, graph_t subgraph)
{
	x->name = (NULL != name) ? ptr_printf("%s", name) : NULL;

	x->N_vertices = N_vertices;
	x->edges = *TYPE_ALLOC(list_t[N_vertices]);
	for (int i = 0; i < N_vertices; i++)
		x->edges[i] = list_create();

	x->io_flags = *TYPE_ALLOC(bool[N_vertices]);
	for (int i = 0; i < N_vertices; i++)
		x->io_flags[i] = io_flags[i];

	x->count = 0;

	x->subgraph = subgraph;

	x->node_print = NULL;
	x->node_del = NULL;

	x->external = external;
}


graph_t graph_create(void)
{
	PTR_ALLOC(struct graph_s, result);

	result->nodes = list_create();
	result->ext_nodes = list_create();

	return PTR_PASS(result);
}

void graph_free(graph_t x)
{
	while (0 < list_count(x->nodes))
		graph_remove_node(x, nodes_get(x->nodes, 0));
	while (0 < list_count(x->ext_nodes))
		graph_remove_node(x, nodes_get(x->ext_nodes, 0));
	list_free(x->nodes);
	list_free(x->ext_nodes);

	xfree(x);
}

void graph_add_node(graph_t graph, node_t node)
{
	list_append(node->external ? graph->ext_nodes : graph->nodes, node);
}

void graph_remove_node(graph_t graph, node_t node)
{
	// remove all edges form/to node
	for (int i = 0; i < node->N_vertices; i++) {

		struct vertex_s vertex = {.node = node, . idx = i};
		vertices_t vertices = node->edges[i];

		while (0 < list_count(vertices)) {

			if (node->io_flags[i])
				graph_remove_edge(vertex, *vertices_get(vertices, 0));
			else
				graph_remove_edge(*vertices_get(vertices, 0), vertex);
		}
	}

	if (node->external) {

		int i = nodes_index(graph->ext_nodes, node);
		node_free(list_remove_item(graph->ext_nodes, i));
	} else {

		int i = nodes_index(graph->nodes, node);
		node_free(list_remove_item(graph->nodes, i));
	}
}

void graph_add_edge(struct vertex_s _a, struct vertex_s _b)
{
	vertex_t a = TYPE_ALLOC(struct vertex_s);
	vertex_t b = TYPE_ALLOC(struct vertex_s);

	*a = _a;
	*b = _b;

	assert(a->node->io_flags[a->idx]);
	assert(!(b->node->io_flags[b->idx]));

	list_append(a->node->edges[a->idx], b);
	list_append(b->node->edges[b->idx], a);

	//each vertex can only have one in edge
	assert(1 == list_count(b->node->edges[b->idx]));
}

static bool cmp_vertex(const void* _data, const void* _ref)
{
	const struct vertex_s* data = _data;
	const struct vertex_s* ref = _ref;

	return ((data->idx == ref->idx) && (data->node == ref->node));
}

void graph_remove_edge(struct vertex_s a, struct vertex_s b)
{
	int ia = list_get_first_index(a.node->edges[a.idx], &b, cmp_vertex);
	xfree(list_remove_item(a.node->edges[a.idx], ia));

	int ib = list_get_first_index(b.node->edges[b.idx], &a, cmp_vertex);
	xfree(list_remove_item(b.node->edges[b.idx], ib));
}

// redirect all edges connected to "old" to "new"
void graph_redirect_edge(struct vertex_s new, struct vertex_s old)
{
	list_t vertices = old.node->edges[old.idx];
	while (0 != list_count(vertices)) {

		struct vertex_s va = *vertices_get(vertices, 0);

		if (old.node->io_flags[old.idx]) {

			graph_remove_edge(old, va);
			graph_add_edge(new, va);
		} else {

			graph_remove_edge(va, old);
			graph_add_edge(va, new);
		}
	}
}


graph_t combine_graphs_F(int N, graph_t graphs[N])
{
	for (int i = 1; i < N; i++) {

		list_merge(graphs[0]->nodes, graphs[i]->nodes, false);
		list_merge(graphs[0]->ext_nodes, graphs[i]->ext_nodes, false);

		graph_free(graphs[i]);
	}

	return graphs[0];
}

graph_t link_graphs_F(graph_t graph, int oo, int ii)
{
	node_t onode = list_remove_item(graph->ext_nodes, oo);
	node_t inode = list_remove_item(graph->ext_nodes, (ii < oo) ? ii : ii - 1);

	assert(inode->io_flags[0]);
	assert(!onode->io_flags[0]);

	struct vertex_s ov = {.node = onode, .idx = 0};
	struct vertex_s iv = {.node = inode, .idx = 0};

	assert(1 == list_count(onode->edges[0]));

	int I = list_count(inode->edges[0]);
	int O = list_count(onode->edges[0]);

	struct vertex_s ivs[I];
	struct vertex_s ovs[O];

	for (int i = 0; i < I; i++) {

		ivs[i] = *(vertices_get(inode->edges[0], 0));
		graph_remove_edge(iv, ivs[i]);
	}

	for (int o = 0; o < O; o++) {

		ovs[o] = *(vertices_get(onode->edges[0], 0));
		graph_remove_edge(ovs[o], ov);
	}

	for (int i = 0; i < I; i++)
		for (int o = 0; o < O; o++)
			graph_add_edge(ovs[o], ivs[i]);

	node_free(onode);
	node_free(inode);

	return graph;
}

graph_t perm_ext_graphs_F(graph_t graph, int N, const int perm[N])
{
	auto result = graph;

	list_t new_ext_nodes = list_create();

	for (int i = 0; i < N; i++)
		list_append(new_ext_nodes, nodes_get(result->ext_nodes, perm[i]));

	while (0 < list_count(result->ext_nodes))
		list_pop(result->ext_nodes);

	while (0 < list_count(new_ext_nodes))
		list_append(result->ext_nodes, list_pop(new_ext_nodes));

	list_free(new_ext_nodes);

	return result;
}

graph_t dup_graphs_F(graph_t graph, int a, int b)
{
	assert (a < b);

	node_t anode = nodes_get(graph->ext_nodes, a);
	node_t bnode = list_remove_item(graph->ext_nodes, b);

	assert(anode->io_flags[0]);
	assert(bnode->io_flags[0]);

	while (0 < list_count(bnode->edges[0])) {

		struct vertex_s a = {.node = anode, .idx = 0};
		struct vertex_s b = {.node = bnode, .idx = 0};

		struct vertex_s n = *(vertices_get(bnode->edges[0], 0));

		graph_remove_edge(b, n);
		graph_add_edge(a, n);
	}

	node_free(bnode);

	return graph;
}

const char* print_node(const struct node_s* node) {

	if (NULL != node->node_print)
		return node->node_print(node);

	if (NULL == node->subgraph)
		return ptr_printf("node_%p [label=\"%s\"];\n", node, (NULL == node->name) ? node->TYPEID->name : node->name);

	const char* result = print_internl_graph(node->subgraph, false, 0, NULL);
	auto tmp = result;
	result = ptr_printf("subgraph cluster_subgraph_%p{\nlabel=\"%s\"\n%s}\n", node->subgraph, node->name, tmp);
	xfree(tmp);

	return result;
}

const char* print_vertex(node_t node, int idx)
{
	if (NULL == node->subgraph)
		return ptr_printf("node_%p", node);
	else
		return print_vertex(nodes_get(node->subgraph->ext_nodes, idx), 0);
}

static const char* print_edge(struct vertex_s a, struct vertex_s b)
{
	auto str_a = print_vertex(a.node, a.idx);
	auto str_b = print_vertex(b.node, b.idx);

	auto result = ptr_printf("%s -> %s [label=\"%d->%d\"];\n", str_a, str_b, a.idx, b.idx);

	xfree(str_a);
	xfree(str_b);

	return result;
}

const char* print_internl_graph(graph_t graph, bool get_ext_nodes, int N, const char* ext_nodes[N])
{
	auto result = ptr_printf("");

	for (int i = 0; i < list_count(graph->nodes); i++) {

		auto node = (node_t)nodes_get(graph->nodes, i);

		auto str_node = print_node(node);
		auto tmp = result;
		result = ptr_printf("%s%s", tmp, str_node);
		xfree(tmp);
		xfree(str_node);
	}

	if (get_ext_nodes) {

		assert (N == list_count(graph->ext_nodes));
		for (int i = 0; i < N; i++)
			ext_nodes[i] = print_vertex(nodes_get(graph->ext_nodes, i), 0);
	} else {

		auto tmp = result;
		result = ptr_printf("%ssubgraph {\nrank=same\n", tmp);
		xfree(tmp);

		const char* prev_node = NULL;

		for (int i = 0; i < list_count(graph->ext_nodes); i++) {

			node_t node = nodes_get(graph->ext_nodes, i);

			if (node->io_flags[0])
				continue;

			auto str_node = print_node(node);
			auto tmp = result;
			result = ptr_printf("%s%s", tmp, str_node);
			xfree(tmp);
			xfree(str_node);

			if (NULL != prev_node) {

				auto tmp_node_id = print_vertex(node, 0);
				auto tmp = result;
				result = ptr_printf("%s%s -> %s [style=invis];\n", tmp, prev_node, tmp_node_id);
				xfree(tmp);
				xfree(prev_node);
				prev_node = tmp_node_id;
			} else {
				prev_node = print_vertex(node, 0);
			}
		}

		xfree(prev_node);
		prev_node = NULL;

		tmp = result;
		result = ptr_printf("%s}\nsubgraph {\nrank=same\n", tmp);
		xfree(tmp);

		for (int i = 0; i < list_count(graph->ext_nodes); i++) {

			node_t node = nodes_get(graph->ext_nodes, i);

			if (!node->io_flags[0])
				continue;

			auto str_node = print_node(node);
			auto tmp = result;
			result = ptr_printf("%s%s", tmp, str_node);
			xfree(tmp);
			xfree(str_node);

			if (NULL != prev_node) {

				auto tmp_node_id = print_vertex(node, 0);
				auto tmp = result;
				result = ptr_printf("%s%s -> %s [style=invis];\n", tmp, prev_node, tmp_node_id);
				xfree(tmp);
				xfree(prev_node);
				prev_node = tmp_node_id;
			} else {
				prev_node = print_vertex(node, 0);
			}
		}

		xfree(prev_node);
		prev_node = NULL;

		tmp = result;
		result = ptr_printf("%s}\n", tmp);
		xfree(tmp);
	}

	for (int i = 0; i < list_count(graph->nodes); i++) {

		auto node = (node_t)nodes_get(graph->nodes, i);
		struct vertex_s a = {.node = node, .idx = 0};

		for (int i = 0; i < node->N_vertices; i++) {

			if (!node->io_flags[i])
				continue;

			a.idx = i;

			list_t overtices = node->edges[i];
			for (int j = 0; j < list_count(overtices); j++) {

				vertex_t b = vertices_get(overtices, j);
				auto str_edge = print_edge(a, *b);

				auto tmp = result;
				result = ptr_printf("%s%s", tmp, str_edge);
				xfree(tmp);
				xfree(str_edge);
			}
		}
	}

	for (int i = 0; i < list_count(graph->ext_nodes); i++) {

		auto node = (node_t)nodes_get(graph->ext_nodes, i);
		struct vertex_s a = {.node = node, .idx = 0};

		for (int i = 0; i < node->N_vertices; i++) {

			if (!node->io_flags[i])
				continue;

			a.idx = i;

			list_t overtices = node->edges[i];
			for (int j = 0; j < list_count(overtices); j++) {

				vertex_t b = vertices_get(overtices, j);
				auto str_edge = print_edge(a, *b);

				auto tmp = result;
				result = ptr_printf("%s%s", tmp, str_edge);
				xfree(tmp);
				xfree(str_edge);
			}
		}
	}

	return result;
}

void export_graph_dot(const char* filename, graph_t graph)
{
	int N = list_count(graph->ext_nodes);
	const char* ext_nodes[N];

	FILE *fp;
	fp = fopen(filename, "w+");
	assert(0 != fp);

	const char* tmp = print_internl_graph(graph, false, N, ext_nodes);
	fprintf(fp, "digraph {\n%s", tmp);
	xfree(tmp);

	for (int i = 0; i < N; i++) {

		tmp = print_node(nodes_get(graph->ext_nodes, i));
		fprintf(fp, "%s", tmp);
		xfree(tmp);
	}

	fprintf(fp, "}\n");

	fclose(fp);
}
