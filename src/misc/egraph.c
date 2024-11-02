#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/list.h"
#include "num/multind.h"

#include "egraph.h"

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

	return PTR_PASS(x);
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

void egraph_topological_sort_F(list_t graph)
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

void egraph_reset_between(list_t graph)
{
	for (int i = 0; i < list_count(graph); i++) {

		enode_t node = list_get_item(graph, i);
		node->count = 0;
		node->active = false;
		node->flags = 0;
	}
}

void egraph_sort_between(list_t graph)
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

	while (0 < list_count(between))
		list_append(graph, list_pop(between));

	while (0 < list_count(after))
		list_append(graph, list_pop(after));

	list_free(before);
	list_free(between);
	list_free(after);
}


//check if b depends on a
void subgraph_between(list_t graph, enode_t a, enode_t b)
{

	egraph_reset_between(graph);

	egraph_set_ancestors(b);
	egraph_set_descendants(a);

	egraph_sort_between(graph);
}

void egraph_set_active(list_t graph)
{
	for (int i = 0; i < list_count(graph); i++) {

		enode_t node = list_get_item(graph, i);
		node->active = true;
	}
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
	xfree(nodes);
	xfree(edges);
	fclose(fp);
}



