/* Copyright 2026. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2026 Moritz Blumenthal
 *
 */

#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>

#include "num/multind.h"

#include "misc/misc.h"
#include "misc/list.h"

#include "egraph.h"

struct enode_s {

	list_t iedges;
	list_t oedges;

	_Bool active;
	unsigned long flags;

	// for path finding
	long count;
	enode_t prev;

	const char* name;
	const void* data;
};

void enode_free(enode_t x)
{
	while (0 < list_count(x->iedges)) {

		enode_t y = list_pop(x->iedges);
		list_get_first_item(y->oedges, x, NULL, true);
	}

	list_free(x->iedges);

	while (0 < list_count(x->oedges)) {

		enode_t y = list_pop(x->oedges);
		list_get_first_item(y->iedges, x, NULL, true);
	}

	list_free(x->oedges);

	if (NULL != x->name)
		xfree(x->name);

	xfree(x);
}

enode_t enode_create(const char* name, const void* data)
{
	PTR_ALLOC(struct enode_s, x);

	x->name = name;
	x->data = data;
	x->iedges = list_create();
	x->oedges = list_create();
	x->active = true;
	x->flags = 0UL;
	x->count = 0;
	x->prev = NULL;

	return PTR_PASS(x);
}

bool enode_is_active(enode_t node)
{
	return (NULL != node) && node->active;
}

void* enode_get_data(enode_t node)
{
	return (NULL != node) ? (void*)node->data : NULL;
}

long enode_get_count(enode_t node)
{
	assert(NULL != node);
	return node->count;
}

list_t enode_get_iedges(enode_t node)
{
	assert(NULL != node);
	return node->iedges;
}

list_t enode_get_oedges(enode_t node)
{
	assert(NULL != node);
	return node->oedges;
}


//b depends on a
void enode_add_dependency(enode_t a, enode_t b)
{
	assert(a != b);

	if (-1 == list_get_first_index(a->oedges, b, NULL)) {

		list_append(a->oedges, b);
		list_append(b->iedges, a);
	}
}

egraph_t egraph_create(void)
{
	return list_create();
}

void egraph_free(egraph_t graph)
{
	while ( 0 < list_count(graph))
		enode_free(list_pop(graph));

	list_free(graph);
}

void egraph_add_node(egraph_t graph, enode_t node)
{
	list_append(graph, node);
}

enode_t egraph_get_node(egraph_t graph, int idx)
{
	return list_get_item(graph, idx);
}

enode_t egraph_remove_node(egraph_t graph, int idx)
{
	return list_remove_item(graph, idx);
}

void egraph_dijkstra(egraph_t graph, enode_t src, bool reverse)
{
	assert(NULL != src);

	for (int i = 0; i < list_count(graph); i++) {

		enode_t node = list_get_item(graph, i);
		assert(NULL != node);

		node->prev = NULL;
		node->active = false;
		node->count = -1;
	}

	src->count = 0;
	src->prev = NULL;

	enode_t current = src;

	do {
		current->active = true;

		for (int i = 0; i < list_count(reverse ? current->iedges : current->oedges); i++) {

			enode_t neighbor = list_get_item(reverse ? current->iedges : current->oedges, i);

			if (neighbor->active)
				continue;

			if ((-1 == neighbor->count) || (current->count + 1 < neighbor->count)) {

				neighbor->count = current->count + 1;
				neighbor->prev = current;
			}
		}

		long count = LONG_MAX;
		current = NULL;

		for (int i = 0; i < list_count(graph); i++) {

			enode_t node = list_get_item(graph, i);

			if (node->active || (-1 == node->count))
				continue;

			if (node->count < count) {

				count = node->count;
				current = node;
			}
		}

	} while (NULL != current);
}

void egraph_bfs(egraph_t graph, enode_t src, bool reverse)
{
	assert(NULL != src);

	for (int i = 0; i < list_count(graph); i++) {

		enode_t node = list_get_item(graph, i);

		node->prev = NULL;
		node->active = false;
		node->count = -1;
	}

	src->count = 0;
	src->prev = NULL;
	src->active = true;

	list_t queue = list_create();
	list_append(queue, src);

	while (0 < list_count(queue)) {

		enode_t current = list_remove_item(queue, 0);

		for (int i = 0; i < list_count(reverse ? current->iedges : current->oedges); i++) {

			enode_t neighbor = list_get_item(reverse ? current->iedges : current->oedges, i);

			if (neighbor->active)
				continue;

			neighbor->count = current->count + 1;
			neighbor->prev = current;
			neighbor->active = true;

			list_append(queue, neighbor);
		}
	}

	list_free(queue);
}


list_t egraph_shortest_path(egraph_t graph, enode_t dst, enode_t src)
{
	assert(NULL != src);
	assert(NULL != dst);

	egraph_dijkstra(graph, src, false);

	list_t ret = list_create();

	if (dst->count == -1)
		return ret;

	enode_t current = dst;
	while (current != NULL) {

		list_push(ret, current);
		current = current->prev;
	}

	return ret;
}

long egraph_longest_distance(enode_t* dst, enode_t* src, egraph_t graph, list_t nodes)
{
	assert((NULL == dst) == (NULL == src));

	if (NULL != dst) {

		*dst = NULL;
		*src = NULL;
	}

	long dist = -1;

	for (int i = 0; i < list_count(nodes); i++) {

		egraph_bfs(graph, list_get_item(nodes, i), false);

		for (int j = 0; j < list_count(nodes); j++) {

			enode_t tmp = list_get_item(nodes, j);

			if (dist < tmp->count) {

				if (NULL != dst) {

					*dst = tmp;
					*src = list_get_item(nodes, i);
				}

				dist = tmp->count;
			}
		}
	}

	return dist;
}

long egraph_diameter(egraph_t graph)
{
	return egraph_longest_distance(NULL, NULL, graph, graph);
}

long egraph_depth_first_search(egraph_t graph, enode_t src, long count, bool reverse)
{
	if (NULL != graph) {

		for (long i = 0; i < list_count(graph); i++) {

			enode_t node = list_get_item(graph, i);
			node->active = false;
		}
	}

	if (src->active)
		return count;

	src->count = count++;
	src->active = true;
	list_t edges = reverse ? src->iedges : src->oedges;

	for (long i = 0; i < list_count(edges); i++) {

		enode_t node = list_get_item(edges, i);
		count = egraph_depth_first_search(NULL, node, count, reverse);
	}

	return count;
}

list_t egraph_split_connected_components(egraph_t graph)
{
	list_t ret = list_create();

	while (0 < list_count(graph)) {

		enode_t node = list_get_item(graph, 0);
		egraph_depth_first_search(graph, node, 0, false);

		egraph_t component = egraph_create();

		for (long i = 0; i < list_count(graph); i++) {

			enode_t tmp = list_get_item(graph, i);

			if (tmp->active)
				list_append(component, list_remove_item(graph, i--));
		}

		list_append(ret, component);
	}

	egraph_free(graph);

	return ret;
}

extern enode_t egraph_find_most_distant(egraph_t graph, enode_t src)
{
	egraph_bfs(graph, src, false);

	enode_t most_distant = src;

	for (long i = 1; i < list_count(graph); i++)
		if (most_distant->count < ((enode_t)list_get_item(graph, i))->count)
			most_distant = list_get_item(graph, i);

	return most_distant;
}






static void sort_nodes(list_t sorted_nodes, list_t out_vertices)
{
	int N = list_count(out_vertices);

	for (int i = 0; i < N; i++) {

		enode_t node = list_get_item(out_vertices, i);

		int N_in = list_count(node->iedges);
		node->count++;

		if (node->count == N_in) {

			list_append(sorted_nodes, node);
			node->count = 0;

			sort_nodes(sorted_nodes, node->oedges);
		}
	}
}

void egraph_topological_sort_F(egraph_t graph)
{
	list_t sorted_nodes = list_create();

	for (int i = 0; i < list_count(graph); i++) {

		enode_t node = list_get_item(graph, i);
		node->count = 0;

		if (0 == list_count(node->iedges))
			list_append(sorted_nodes, node);
	}

	int N_no_in_edge = list_count(sorted_nodes);

	for (int i = 0; i < N_no_in_edge; i++) {

		enode_t node = list_get_item(sorted_nodes, i);

		sort_nodes(sorted_nodes, node->oedges);
	}

	while (0 < list_count(graph))
		list_pop(graph);

	list_merge(graph, sorted_nodes, true);
}

static bool _dnode_depends_on(enode_t a, enode_t b)
{
	if (a == b)
		return true;

	if (1 == b->count)
		return false;

	for (int j = 0; j < list_count(b->iedges); j++) {

		enode_t node = list_get_item(b->iedges, j);

		if (_dnode_depends_on(a, node))
			return true;
	}

	b->count = 1;
	return false;
}


//check if b depends on a
bool enode_depends_on(list_t graph, enode_t a, enode_t b)
{
	for (int i = 0; i < list_count(graph); i++) {

		enode_t node = list_get_item(graph, i);
		node->count = 0;
	}

	return _dnode_depends_on(a, b);
}


void egraph_set_ancestors(enode_t b)
{
	for (int i = 0; i < list_count(b->iedges); i++) {

		enode_t node = list_get_item(b->iedges, i);

		if (!MD_IS_SET(node->flags, 0))
			egraph_set_ancestors(node);
	}

	b->flags = MD_SET(b->flags, 0);
}

void egraph_set_descendants(enode_t a)
{
	for (int i = 0; i < list_count(a->oedges); i++) {

		enode_t node = list_get_item(a->oedges, i);

		if (!MD_IS_SET(node->flags, 1))
			egraph_set_descendants(node);
	}

	a->flags = MD_SET(a->flags, 1);
}

bool enode_is_ancestors(enode_t b)
{
	return MD_IS_SET(b->flags, 0);
}

bool enode_is_descendants(enode_t a)
{
	return MD_IS_SET(a->flags, 1);
}

void egraph_reset_between(list_t graph)
{
	for (int i = 0; i < list_count(graph); i++) {

		enode_t node = list_get_item(graph, i);
		node->count = 0;
		node->active = false;
		node->flags = 0;
	}
}

list_t egraph_sort_between(list_t graph)
{
	list_t before = list_create();
	list_t between = list_create();
	list_t after = list_create();

	list_t default_list = before;

	while (0 < list_count(graph)) {

		enode_t node = list_pop(graph);

		if (MD_IS_SET(node->flags, 1))
			default_list = after;

		if (MD_IS_SET(node->flags, 0) && MD_IS_SET(node->flags, 1))
			list_append(between, node);

		if (!MD_IS_SET(node->flags, 0) && MD_IS_SET(node->flags, 1))
			list_append(after, node);

		if (MD_IS_SET(node->flags, 0) && !MD_IS_SET(node->flags, 1))
			list_append(before, node);

		if (!MD_IS_SET(node->flags, 0) && !MD_IS_SET(node->flags, 1))
			list_append(default_list, node);
	}

	for (int i = 0; i < list_count(between); i++) {

		enode_t node = list_get_item(between, i);
		node->active = true;
	}

	while (0 < list_count(before))
		list_append(graph, list_pop(before));

	for (int i = 0; i < list_count(between); i++)
		list_append(graph, list_get_item(between, i));

	while (0 < list_count(after))
		list_append(graph, list_pop(after));

	list_free(before);
	list_free(after);

	return between;
}

void egraph_set_active(list_t graph)
{
	for (int i = 0; i < list_count(graph); i++) {

		enode_t node = list_get_item(graph, i);
		node->active = true;
	}
}

bool enode_is_between(enode_t node)
{
	return MD_IS_SET(node->flags, 0) && MD_IS_SET(node->flags, 1);
}

void egraph_unset_active(list_t graph)
{
	for (int i = 0; i < list_count(graph); i++) {

		enode_t node = list_get_item(graph, i);
		node->active = false;
	}
}


void export_egraph_dot(const char* filename, list_t graph)
{
	const char* nodes = strdup("");
	const char* edges = strdup("");
	bool sort = true;

	const char* last_node = NULL;

	for (int i = 0; i < list_count(graph); i++) {

		enode_t node = list_get_item(graph, i);
		if (!node->active)
			continue;

		ptr_append_printf(&nodes, "\nnode_%p", node);

		if (NULL != node->name)
			ptr_append_printf(&nodes, " [label=\"%s\"]", node->name);

		if (sort && NULL != last_node)
			ptr_append_printf(&edges, "\n%s -> node_%p [style=invis]", last_node, node);

		if (NULL != last_node)
			xfree(last_node);

		last_node = ptr_printf("node_%p", node);

		for (int j = 0; j < list_count(node->oedges); j++)
			ptr_append_printf(&edges, "\nnode_%p -> node_%p", node, list_get_item(node->oedges, j));

	}

	FILE *fp = fopen(filename, "w+");

	if (NULL == fp)
		error("Opening file\n");

	fprintf(fp, "digraph {\n%s\n%s\n}", nodes, edges);
	fclose(fp);

	free((char*)nodes);
	free((char*)edges);
}



