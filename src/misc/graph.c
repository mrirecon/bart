/* Copyright 2021. Uecker Lab, University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * */

#define _GNU_SOURCE
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"
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
	x->name = (NULL != name) ? strdup(name) : NULL;

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
	assert(a < b);

	node_t anode = nodes_get(graph->ext_nodes, a);
	node_t bnode = list_remove_item(graph->ext_nodes, b);

	assert(anode->io_flags[0]);
	assert(bnode->io_flags[0]);

	while (0 < list_count(bnode->edges[0])) {

		struct vertex_s a = { .node = anode, .idx = 0 };
		struct vertex_s b = { .node = bnode, .idx = 0 };

		struct vertex_s n = *(vertices_get(bnode->edges[0], 0));

		graph_remove_edge(b, n);
		graph_add_edge(a, n);
	}

	node_free(bnode);

	return graph;
}

const char* print_node(const struct node_s* node)
{
	if (NULL != node->node_print)
		return node->node_print(node);

	if (NULL == node->subgraph)
		return ptr_printf("node_%p [label=\"%s\"];\n", node, (NULL == node->name) ? node->TYPEID->name : node->name);

	const char* result = print_internl_graph(node->subgraph, false, 0, NULL);

	const char* str = ptr_printf("subgraph cluster_subgraph_%p{\nlabel=\"%s\"\n%s}\n", node->subgraph, node->name, result);

	xfree(result);

	return str;
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
		struct vertex_s a = { .node = node, .idx = 0 };

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



static bool cmp_no_in_edge(const void* _data, const void* _ref)
{
	UNUSED(_ref);
	const struct node_s* node = _data;

	for (int j = 0; j < node->N_vertices; j++)
		if ((!node->io_flags[j]) && (0 < list_count(node->edges[j])))
			return false;

	return true;
}

static bool cmp_no_out_edge(const void* _data, const void* _ref)
{
	UNUSED(_ref);
	const struct node_s* node = _data;

	for (int j = 0; j < node->N_vertices; j++)
		if ((node->io_flags[j]) && (0 < list_count(node->edges[j])))
			return false;

	return true;
}

static int count_in_edges(node_t node)
{
	int result = 0;
	for (int j = 0; j < node->N_vertices; j++)
		result += (node->io_flags[j]) ? 0 : list_count(node->edges[j]);
	return result;
}

static int count_out_edges(node_t node)
{
	int result = 0;
	for (int j = 0; j < node->N_vertices; j++)
		result += (!node->io_flags[j]) ? 0 : list_count(node->edges[j]);
	return result;
}

static void graph_reset_count(graph_t graph)
{
	for (int i = 0; i < list_count(graph->nodes); i++)
		nodes_get(graph->nodes, i)->count = 0;
	for (int i = 0; i < list_count(graph->ext_nodes); i++)
		nodes_get(graph->ext_nodes, i)->count = 0;
}

static void sort_nodes(list_t sorted_nodes, list_t out_vertices)
{
	int N = list_count(out_vertices);
	for (int i = 0; i < N; i++) {

		node_t node = vertices_get(out_vertices, i)->node;
		if (node->external)
			continue;

		int N_in = count_in_edges(node);
		node->count++;

		if (node->count == N_in) {

			list_append(sorted_nodes, node);
			node->count = 0;

			for (int j = 0; j < node->N_vertices; j++)
				if (node->io_flags[j])
					sort_nodes(sorted_nodes, node->edges[j]);
		}
	}
}

graph_t graph_topological_sort_F(graph_t graph)
{
	graph_reset_count(graph);

	list_t sorted_nodes = list_pop_sublist(graph->nodes, NULL, cmp_no_in_edge);
	int N_no_in_edge = list_count(sorted_nodes);

	for (int i = 0; i < N_no_in_edge; i++) {

		node_t node = nodes_get(sorted_nodes, i);
		for (int j = 0; j < node->N_vertices; j++)
			if (node->io_flags[j])
				sort_nodes(sorted_nodes, node->edges[j]);
	}

	for (int i = 0; i < list_count(graph->ext_nodes); i++) {

		node_t node = nodes_get(graph->ext_nodes, i);
		for (int j = 0; j < node->N_vertices; j++)
			if (node->io_flags[j])
				sort_nodes(sorted_nodes, node->edges[j]);
	}

	while (0 < list_count(graph->nodes))
		list_pop(graph->nodes);
	list_merge(graph->nodes, sorted_nodes, true);

	return graph;
}

//trys to remove a one-in-one-out (identity) node from a graph
//and connects in edges with oout edges
//returns new graph if successful, NULL else
//nodes cannot be removed if they map to output
graph_t graph_bridge_node(graph_t graph, node_t node)
{
	int N = node->N_vertices;
	if ((2 != N) || (!node->io_flags[0]) || (node->io_flags[1]))
		return NULL;

	for (int i = 0; i < list_count(node->edges[0]); i++)
		if ((vertices_get(node->edges[0], i))->node->external)
			return false;

	list_t in = node->edges[1];
	list_t out = node->edges[0];

	struct vertex_s ovt = {.node = node, .idx = 0};
	struct vertex_s ivt = {.node = node, .idx = 1};

	int I = list_count(in);
	int O = list_count(out);

	struct vertex_s ivs[I];
	struct vertex_s ovs[O];

	for (int i = 0; i < I; i++) {

		ivs[i] = *(vertices_get(in, 0));
		graph_remove_edge(ivs[i], ivt);
	}

	for (int o = 0; o < O; o++) {

		ovs[o] = *(vertices_get(out, 0));
		graph_remove_edge(ovt, ovs[o]);
	}

	for (int i = 0; i < I; i++)
		for (int o = 0; o < O; o++)
			graph_add_edge(ivs[i], ovs[o]);

	graph_remove_node(graph, node);

	return graph;
}

//trys to remove a end-node (only inputs) from a graph
//returns new graph if successful, NULL else
//nodes cannot be removed if they map to output
graph_t graph_remove_end_node(graph_t graph, node_t node)
{
	int N = node->N_vertices;

	for (int i = 0; i < N; i++) {

		struct vertex_s ver = *vertices_get(node->edges[i], 0);
		bool found = false;

		for (int i = 0; i < list_count(ver.node->edges[ver.idx]); i++)
			found = found || (vertices_get(ver.node->edges[ver.idx], i)->node != node);
		
		if (!found)
			return NULL;
	}

	graph_remove_node(graph, node);

	return graph;
}


static void identify_node_internal(graph_t graph, node_t a, node_t b)
{
	for (int i = 0; i < b->N_vertices; i++) {

		if (!a->io_flags[i])
			continue;

		struct vertex_s vb = {.node = b, .idx = i};
		struct vertex_s va = {.node = a, .idx = i};

		graph_redirect_edge(va, vb);
	}

	graph_remove_node(graph, b);
}

static bool node_cmp_wrap(node_t a, node_t b, node_cmp_t cmp)
{
	enum node_identic test = cmp(a, b);
	assert(test == NODE_NOT_IDENTICAL || test == NODE_IDENTICAL || test == NODE_IDENTICAL_SYMMETRIC);
	return test != NODE_NOT_IDENTICAL;
}

//nodes can be identified when
// 1.) they represent the same node (operator)
// 2.) they do not map to an external node
// 3.) they have the same inputs
static bool nodes_identifyable_internal(node_t a, node_t b, node_cmp_t cmp)
{
	if ( (a->N_vertices != b->N_vertices) || (!node_cmp_wrap(a, b, cmp)) )
		return false;

	for (int i = 0; i < a->N_vertices; i++) {

		if (a->io_flags[i] != b->io_flags[i])
			return false;

		if (a->io_flags[i]) {

			for (int j = 0; j < list_count(a->edges[i]); j++)
				if (vertices_get(a->edges[i], j)->node->external)
					return false;

			for (int j = 0; j < list_count(b->edges[i]); j++)
				if (vertices_get(b->edges[i], j)->node->external)
					return false;

		} else {

			vertex_t va = vertices_get(a->edges[i], 0);

			if (NODE_IDENTICAL == cmp(a, b)) {

				vertex_t vb = vertices_get(b->edges[i], 0);
				if ((va->idx != vb->idx) || (va->node != vb->node))
					return false;
			} else {

				bool found = false;

				for (int j = 0; j < a->N_vertices; j++){

					if (b->io_flags[j])
						continue;

					vertex_t vb = vertices_get(b->edges[j], 0);

					if ((va->idx == vb->idx) && (va->node == vb->node))
						found = true;
				}

				if (!found)
					return false;
			}
		}
	}

	return true;
}

static void identify_nodes_in_list(graph_t graph, list_t nodes, node_cmp_t cmp)
{
	for (int i = 0; i < list_count(nodes); i++)
		for (int j = i + 1; j < list_count(nodes); j++) {

			node_t a = nodes_get(nodes, i);
			node_t b = nodes_get(nodes, j);

			if (nodes_identifyable_internal(a, b, cmp)) {

				identify_node_internal(graph, a, b);
				list_remove_item(nodes, j);
				j--;
			}
		}
}

static void vertex_list_identify_nodes(graph_t graph, list_t vertices, node_cmp_t cmp)
{
	list_t nodes = list_create();

	// get nodes from vertices which are "visited" once from each in edge
	for (int i = 0; i < list_count(vertices); i++) {

		node_t node = (vertices_get(vertices, i))->node;
		if (!(node->external) && (++(node->count) == count_in_edges(node)))
			list_append(nodes, node);
	}

	identify_nodes_in_list(graph, nodes, cmp);

	for (int i = 0; i < list_count(nodes); i++) {

		node_t node = nodes_get(nodes, i);
		for (int j = 0; j < node->N_vertices; j++)
			if (node->io_flags[j])
				vertex_list_identify_nodes(graph, node->edges[j], cmp);
	}

	list_free(nodes);
}

static bool cmp_node_count(const void* _data, const void* _ref)
{
	UNUSED(_ref);
	const struct node_s* node = _data;
	return (-1 == node->count);
}


// trys to identfy node whith each other
// i.e. nodes representing the same node are replaced by one node
graph_t graph_identify_nodes_F(graph_t graph, node_cmp_t cmp)
{
	graph_reset_count(graph);

	list_t nodes_no_in_edge = list_get_sublist(graph->nodes, NULL, cmp_no_in_edge);

	identify_nodes_in_list(graph, nodes_no_in_edge, cmp);
	for (int i = 0; i < list_count(nodes_no_in_edge); i++) {

		node_t node = nodes_get(nodes_no_in_edge, i);
		for (int j = 0; j < node->N_vertices; j++)
			if (node->io_flags[j])
				vertex_list_identify_nodes(graph, node->edges[j], cmp);
	}

	list_free(nodes_no_in_edge);

	for (int i = 0; i < list_count(graph->ext_nodes); i++) {

		node_t node = nodes_get(graph->ext_nodes, i);
		for (int j = 0; j < node->N_vertices; j++)
			if (node->io_flags[j])
				vertex_list_identify_nodes(graph, node->edges[j], cmp);
	}

	return graph;
}


/**
 * Extract subgraph from graph
 *
 * Each vertex of a node in the nodes list must either
 * - map only to nodes in the new subgraph
 * - map only to nodes not in the new subgraph
 *
 * For each vertex which maps to nodes not in the new subgraph, the function get_separator_nodes is used
 * to create a pair of external nodes which bridge between the new subgraph and the old graph.
 *
 * By combining and linking the new subgraph can be reinserted (propably after modification) in the old graph.
 *
 * @param graph
 * @param nodes list of nodes which form the new graph
 * @param get_separator_nodes
 */
graph_t graph_cluster_nodes_F(graph_t graph, list_t nodes, edge_separator_node_f get_separator_nodes)
{
	graph_t result = graph_create();

	//remove nodes in node list from graph
	graph_reset_count(graph);
	for (int i = 0; i < list_count(nodes); i++)
		(nodes_get(nodes, i))->count = -1;
	list_free(list_pop_sublist(graph->nodes, NULL, cmp_node_count));
	for (int i = 0; i < list_count(nodes); i++)
		(nodes_get(nodes, i))->count = 0;

	for (int i = 0; i < list_count(nodes); i++){

		node_t node = nodes_get(nodes, i);
		for (int j = 0; j < node->N_vertices; j++) {

			if (0 == list_count(node->edges[j]))
				continue;

			bool internal = (-1 < nodes_index(nodes, vertices_get(node->edges[j], 0)->node));

			for (int k = 1; k < list_count(node->edges[j]); k++)
				assert(internal == (-1 < nodes_index(nodes, vertices_get(node->edges[j], k)->node)));

			if (internal)
				continue;

			struct vertex_s vertex = {.node = node, .idx = j};

			node_t node_pair[2];
			get_separator_nodes(node_pair, vertex);

			struct vertex_s ext_old = {.node = node_pair[0], .idx = 0};
			struct vertex_s ext_new = {.node = node_pair[1], .idx = 0};

			if (node->io_flags[j]) {

				graph_redirect_edge(ext_old, vertex);
				graph_add_edge(vertex, ext_new);

				graph_add_node(graph, node_pair[0]);
				graph_add_node(result, node_pair[1]);
			} else {

				assert(1 == list_count(node->edges[j]));
				struct vertex_s parent = *(vertices_get(node->edges[j], 0));

				graph_remove_edge(parent, vertex);

				list_t vertices = parent.node->edges[parent.idx];
				bool found = false;

				for (int k = 0; !found && (k < list_count(vertices)); k++) {

					int idx = nodes_index(graph->ext_nodes, vertices_get(vertices, k)->node);

					if (-1 < idx) {
						found = true;

						int idx_subgraph = idx - list_count(graph->ext_nodes) + list_count(result->ext_nodes);
						node_t new_ext_node = nodes_get(result->ext_nodes, idx_subgraph);
						graph_add_edge((struct vertex_s){.node = new_ext_node, .idx = 0}, vertex);
					}
				}

				if (found) {

					node_free(node_pair[0]);
					node_free(node_pair[1]);
				} else {

					graph_add_edge(parent, ext_old);
					graph_add_edge(ext_new, vertex);
					graph_add_node(graph, node_pair[0]);
					graph_add_node(result, node_pair[1]);
				}
			}
		}
	}

	while (0 < list_count(nodes))
		graph_add_node(result, list_pop(nodes));
	list_free(nodes);

	//permute external nodes such that outputs come before inputs in new subgraph
	int perm_old[list_count(graph->ext_nodes)];
	int perm_new[list_count(result->ext_nodes)];

	int N_old_ext = list_count(graph->ext_nodes) - list_count(result->ext_nodes);
	int N_new_ext = list_count(result->ext_nodes);

	for (int i = 0; i < N_old_ext; i++)
		perm_old[i] = i;

	int o = 0;
	for (int i = 0; i < N_new_ext; i++)
		if ((nodes_get(result->ext_nodes, i))->io_flags[0]) { //io_falgs of ext_node == ! io_flags of node

			perm_new[N_new_ext - 1 - i + o] = i;
			perm_old[N_new_ext + N_old_ext - 1 - i + o] = i + N_old_ext;
		} else {

			perm_new[o] = i;
			perm_old[o + N_old_ext] = i + N_old_ext;
			o++;
		}

	perm_ext_graphs_F(graph, N_old_ext + N_new_ext, perm_old);
	perm_ext_graphs_F(result, N_new_ext, perm_new);

	return result;
}

/**
 * Reinsert a subgraph as extracted by graph_cluster_nodes_F into a graph by combining and linking
 *
 * @param graph
 * @param subgraph
 */
graph_t graph_reinsert_subgraph_FF(graph_t graph, graph_t subgraph)
{
	int N = list_count(graph->ext_nodes);
	int Ns = list_count(subgraph->ext_nodes);
	graph = combine_graphs_F(2, (graph_t[2]){graph, subgraph});

	for (int i = 0; i < Ns; i++) {

		int ii = N - 1 - i;
		int oo = list_count(graph->ext_nodes) - 1;

		node_t node_ii = list_get_item(graph->ext_nodes, ii);

		if (!node_ii->io_flags[0])
			SWAP(ii, oo);

		graph = link_graphs_F(graph, oo, ii);
	}

	return graph;
}



void debug_nodes(enum debug_levels dl, list_t nodes)
{
	debug_printf(dl, "%d nodes:\n", list_count(nodes));
	for (int i = 0; i < list_count(nodes); i++) {

		auto tmp = print_node(nodes_get(nodes, i));
		debug_printf(dl, "%s", tmp);
		xfree(tmp);
	}
}

void debug_edges(enum debug_levels dl, list_t nodes)
{
	debug_printf(dl, "edges:\n");
	while(0 < list_count(nodes)) {

		node_t node = list_pop(nodes);
		for (int i = 0; i < node->N_vertices; i++) {

			if (!node->io_flags[i])
				continue;

			list_t vertices = list_copy(node->edges[i]);
			while(0 < list_count(vertices)) {

				struct vertex_s a = {.node = node, .idx = i};
				vertex_t b = list_pop(vertices);

				auto tmp = print_edge(a, *b);
				debug_printf(dl, "%s\n", tmp);
				xfree(tmp);
			}
			list_free(vertices);
		}
	}

	list_free(nodes);
}


static inline bool node_chainable(node_t node)
{
	return (2 == node->N_vertices) && (node->io_flags[0]) && (!node->io_flags[1]);
}

static void graph_chain_append(list_t result, list_t chain, node_t node)
{
	node->count++;

	if (node_chainable(node)) {

		list_append(chain, node);
		if (1 == list_count(node->edges[0])) {

			node_t nnode = (vertices_get(node->edges[0], 0))->node;
			graph_chain_append(result, chain, nnode);
			return;
		}
	}

	if (1 < list_count(chain))
		list_append(result, chain);
	else
		list_free(chain);

	if (count_in_edges(node) != node->count)
		return;

	for (int i = 0; i < node->N_vertices; i++)
		if (node->io_flags[i]) {

			vertices_t vertices = node->edges[i];
			for (int j = 0; j < list_count(vertices); j++)
				graph_chain_append(result, list_create(), vertices_get(vertices, j)->node);
		}
}

// returns a list of lists of nodes which are a simple chain
list_t graph_get_chains(graph_t graph)
{
	list_t result = list_create();
	graph_reset_count(graph);

	list_t starts = list_get_sublist(graph->nodes, NULL, cmp_no_in_edge);
	list_merge(starts, list_copy(graph->ext_nodes), true);

	while (0 < list_count(starts)) {

		node_t node = list_pop(starts);
		for (int i = 0; i < node->N_vertices; i++)
			if (node->io_flags[i]) {

				vertices_t vertices = node->edges[i];
				for (int j = 0; j < list_count(vertices); j++)
					graph_chain_append(result, list_create(), vertices_get(vertices, j)->node);
			}
	}

	list_free(starts);

	return result;
}


static node_t access_node(node_t node)
{
	return (count_out_edges(node) <= ++(node->count)) ? node : NULL;
}

// returns true if a node has only one following node
// there might be multiple edges to different vertices of the node
static inline bool node_one_child(node_t node, bool simple_only)
{
	if (node->external)
		return false;

	if (simple_only)
		if (!((2 == node->N_vertices) && (node->io_flags[0]) && (!node->io_flags[1])))
			return false;

	node_t follow = NULL;
	for (int i = 0; i < node->N_vertices; i++) {

		if (!node->io_flags[i])
			continue;

		for (int j = 0; j < list_count(node->edges[i]); j++) {

			if (NULL == follow)
				follow = (vertices_get(node->edges[i], j))->node;

			if (follow != (vertices_get(node->edges[i], j))->node)
				return false;
		}
	}

	return true;
}

static void graph_start_cluster(list_t result, node_t node, bool simple_only)
{
	node = access_node(node);
	if (NULL == node)
		return;

	list_t cluster = list_create();

	for (int i = 0; i < node->N_vertices; i++) {

		if (node->io_flags[i])
			continue;

		assert(1 == list_count(node->edges[i]));
		node_t tmp_node = access_node((vertices_get(node->edges[i], 0))->node);
		if (NULL == tmp_node)
			continue;

		if (node_one_child(tmp_node, simple_only)) {

			list_append(cluster, tmp_node);

			for (int j = 0; j < tmp_node->N_vertices; j++) {

				if (tmp_node->io_flags[j])
					continue;

				assert(1 == list_count(tmp_node->edges[j]));

				node_t tmp_node2 = access_node((vertices_get(tmp_node->edges[j], 0))->node);
				if (NULL != tmp_node2)
					graph_start_cluster(result, tmp_node2, simple_only);
			}
		} else {

			graph_start_cluster(result, tmp_node, simple_only);
		}
	}

	if(0 < list_count(cluster)) {

		list_append(cluster, node);
		list_append(result, cluster);
	} else {

		list_free(cluster);
	}
}

// find clusters, i.e. a node and its parents where the parents only have edges to this node
list_t graph_get_clusters(graph_t graph, bool simple_only)
{
	list_t result = list_create();
	graph_reset_count(graph);

	list_t starts = list_get_sublist(graph->nodes, NULL, cmp_no_out_edge);
	while ( 0 < list_count(starts))
		graph_start_cluster(result, list_pop(starts), simple_only);


	list_merge(starts, list_copy(graph->ext_nodes), true);
	while (0 < list_count(starts)) {

		node_t node = list_pop(starts);
		for (int i = 0; i < node->N_vertices; i++) {

			if (node->io_flags[i])
				continue;

			assert(1 == list_count(node->edges[i]));
			node_t tmp_node = access_node((vertices_get(node->edges[i], 0))->node);

			if (NULL != tmp_node)
				graph_start_cluster(result, tmp_node, simple_only);
		}
	}

	list_free(starts);

	return result;
}

// extract nodes representing a sum (of sums) and operators having an edge to this sum
static void get_sum_parents(node_t node_sum, list_t nodes_sum, list_t nodes_linops, node_is_t node_is_sum)
{
	node_sum->count = 1;
	list_append(nodes_sum, node_sum);

	for (int i = 0; i < node_sum->N_vertices; i++) {

		if (node_sum->io_flags[i])
			continue;

		node_t node_in = ((vertex_t)list_get_item(node_sum->edges[i], 0))->node;

		if (1 == list_count(node_in->edges[0])) {

			if (node_is_sum(node_in))
				get_sum_parents(node_in, nodes_sum, nodes_linops, node_is_sum);

			if ((2 == node_in->N_vertices) && !node_in->io_flags[1])
				list_append(nodes_linops, node_in);
		}
	}
}


//find clusters where the final output is a sum of linops
list_t graph_get_linop_sum(graph_t graph, node_cmp_t linop_identify, node_is_t node_is_sum, enum SUM_GRAPH_TYPE sum_graph_type)
{
	list_t result = list_create();

	graph_reset_count(graph);

	for (int i = 0; i < list_count(graph->nodes); i++) {

		node_t node_sum = list_get_item(graph->nodes, i);

		if ((0 != node_sum->count) || !node_is_sum(node_sum))
			continue;

		while ((1 == list_count(node_sum->edges[0])) && node_is_sum(((vertex_t)list_get_item(node_sum->edges[0], 0))->node))
			node_sum = ((vertex_t)list_get_item(node_sum->edges[0], 0))->node;

		list_t nodes_sum = list_create();
		list_t nodes_linops = list_create();

		get_sum_parents(node_sum, nodes_sum, nodes_linops, node_is_sum);

		bool found = false;

		switch (sum_graph_type) {

		case SUM_NODES_ONLY:

			list_free(nodes_linops);

			if (1 >= list_count(nodes_sum))
				list_free(nodes_sum);
			else
				list_append(result, nodes_sum);

			break;

		case MULTI_SUM_NODES_ONLY:

			list_free(nodes_linops);

			bool multi_sum = false;
			for (int j = 0; j < list_count(nodes_sum); j++)
				multi_sum = multi_sum || (3 < nodes_get(nodes_sum, j)->N_vertices);

			if (!multi_sum)
				list_free(nodes_sum);
			else
				list_append(result, nodes_sum);

			break;


		case SUM_NODES_AND_TWO_IDENTICAL_LINOPS:
		case SUM_OPS_AND_OPS:

			for (int j = 0; !found && j < list_count(nodes_linops); j++) {

				node_t node_linop_1 = list_get_item(nodes_linops, j);

				for (int k = j + 1; !found && k < list_count(nodes_linops); k++) {

					node_t node_linop_2 = list_get_item(nodes_linops, k);

					if (!linop_identify(node_linop_1, node_linop_2))
						continue;

					found = true;

					if (sum_graph_type == SUM_OPS_AND_OPS) {

						list_merge(nodes_sum, nodes_linops, false);
					} else {

						list_append(nodes_sum, node_linop_1);
						list_append(nodes_sum, node_linop_2);
					}
				}
			}

			list_free(nodes_linops);

			if (found)
				list_append(result, nodes_sum);
			else
				list_free(nodes_sum);

		}
	}

	return result;
}
