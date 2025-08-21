/* Copyright 2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/list.h"
#include "misc/egraph.h"

#include "misc/misc.h"
#include "utest.h"

static egraph_t get_example_graph(int val[8])
{
/*
	0   -   1   -   2 - 3 - 4
	  \           /
	    5 - 6 - 7
*/
	for (int i = 0; i < 8; i++)
		val[i] = i;

	egraph_t graph = egraph_create();
	for (int i = 0; i < 8; i++)
		list_append(graph, enode_create(NULL, val + i));

	enode_add_dependency(list_get_item(graph, 0), list_get_item(graph, 1));
	enode_add_dependency(list_get_item(graph, 1), list_get_item(graph, 2));
	enode_add_dependency(list_get_item(graph, 2), list_get_item(graph, 3));
	enode_add_dependency(list_get_item(graph, 3), list_get_item(graph, 4));
	enode_add_dependency(list_get_item(graph, 0), list_get_item(graph, 5));
	enode_add_dependency(list_get_item(graph, 5), list_get_item(graph, 6));
	enode_add_dependency(list_get_item(graph, 6), list_get_item(graph, 7));
	enode_add_dependency(list_get_item(graph, 7), list_get_item(graph, 2));

	enode_add_dependency(list_get_item(graph, 1), list_get_item(graph, 0));
	enode_add_dependency(list_get_item(graph, 2), list_get_item(graph, 1));
	enode_add_dependency(list_get_item(graph, 3), list_get_item(graph, 2));
	enode_add_dependency(list_get_item(graph, 4), list_get_item(graph, 3));
	enode_add_dependency(list_get_item(graph, 5), list_get_item(graph, 0));
	enode_add_dependency(list_get_item(graph, 6), list_get_item(graph, 5));
	enode_add_dependency(list_get_item(graph, 7), list_get_item(graph, 6));
	enode_add_dependency(list_get_item(graph, 2), list_get_item(graph, 7));

	return graph;
}

static bool test_dijkstra(void)
{
	int val[8];
	egraph_t graph = get_example_graph(val);

	egraph_dijkstra(graph, list_get_item(graph, 0), false);

	for (int i = 0; i < 5; i++)
		UT_RETURN_ON_FAILURE(i == enode_get_count(egraph_get_node(graph, i)));

	for (int i = 5; i < 8; i++)
		UT_RETURN_ON_FAILURE(i - 4 == enode_get_count(egraph_get_node(graph, i)));

	egraph_free(graph);

	return true;
}

UT_REGISTER_TEST(test_dijkstra);


static bool test_bfs(void)
{
	int val[8];
	egraph_t graph = get_example_graph(val);

	egraph_bfs(graph, list_get_item(graph, 0), false);

	for (int i = 0; i < 5; i++)
		UT_RETURN_ON_FAILURE(i == enode_get_count(egraph_get_node(graph, i)));

	for (int i = 5; i < 8; i++)
		UT_RETURN_ON_FAILURE(i - 4 == enode_get_count(egraph_get_node(graph, i)));

	egraph_free(graph);

	return true;
}

UT_REGISTER_TEST(test_bfs);


static bool test_diameter(void)
{
	int val[8];
	egraph_t graph = get_example_graph(val);

	long diameter = egraph_diameter(graph);
	egraph_free(graph);

	UT_RETURN_ASSERT(5 == diameter);
}

UT_REGISTER_TEST(test_diameter);


