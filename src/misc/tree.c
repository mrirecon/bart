/* Copyright 2023. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <assert.h>
#include <stdbool.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/types.h"

#include "tree.h"

struct node_s {

	struct tree_s* tree;
	struct node_s* parent;
	struct node_s* leafa;
	struct node_s* leafb;

	long height;

	void* item;
};

struct tree_s {

	int N;
	struct node_s* root;

	tree_rel_f relation;

#ifdef _OPENMP
	omp_lock_t lock;
#endif
};

#ifdef _OPENMP
static void tree_set_lock(tree_t tree) { omp_set_lock(&(tree->lock)); }
static void tree_unset_lock(tree_t tree) { omp_unset_lock(&(tree->lock)); }
#else
static void tree_set_lock(tree_t tree) { UNUSED(tree); }
static void tree_unset_lock(tree_t tree) { UNUSED(tree); }
#endif

typedef struct node_s* node_t;
typedef struct tree_s* tree_t;


static node_t create_node(void* item)
{
	PTR_ALLOC(struct node_s, result);

	result->item = item;
	result->parent = NULL;
	result->leafa = NULL;
	result->leafb = NULL;
	result->height = 1;

	return PTR_PASS(result);
}

/**
 * Create an empty tree
 */
tree_t tree_create(tree_rel_f rel)
{
	PTR_ALLOC(struct tree_s, result);
	
	result->N = 0;
	result->root = NULL;
	result->relation = rel;

#ifdef _OPENMP
	omp_init_lock(&(result->lock));
#endif

	return PTR_PASS(result);
}

/**
 * Create an empty tree
 */
void tree_free(tree_t tree)
{
	while (NULL != tree_get_min(tree, true));

#ifdef _OPENMP
	omp_destroy_lock(&(tree->lock));
#endif
	xfree(tree);
}


static long height(node_t node)
{
	if (NULL == node)
		return 0;
	else
		return node->height;
}

static void update_height(node_t node)
{
	if (NULL == node)
		return;
	
	node->height = 1 + MAX(height(node->leafa), height(node->leafb));
	update_height(node->parent);
}

static int balance(node_t node)
{
	if (NULL == node)
		return 0;

	return height(node->leafb) - height(node->leafa);
}



static void r_rotate(node_t node)
{
	assert(NULL != node->leafa);

	node_t root = node->parent;
	node_t leafa = node->leafa;
	node_t subtree = leafa->leafb;

	// Update edge to root
	if (NULL != root) {

		if (root->leafa == node)
			root->leafa = leafa;
		else
			root->leafb = leafa;
	} else {

		leafa->tree->root = leafa;
	}

	leafa->parent = node->parent;

	leafa->leafb = node;
	node->parent = leafa;

	node->leafa = subtree;
	if (NULL != subtree)
		subtree->parent = node;

	update_height(node);
}


static void l_rotate(node_t node)
{
	assert(NULL != node->leafb);

	node_t root = node->parent;
	node_t leafb = node->leafb;
	node_t subtree = leafb->leafa;

	// Update edge to root
	if (NULL != root) {

		if (root->leafb == node)
			root->leafb = leafb;
		else
			root->leafa = leafb;
	} else {

		leafb->tree->root = leafb;
	}

	leafb->parent = node->parent;

	leafb->leafa = node;
	node->parent = leafb;

	node->leafb = subtree;
	if (NULL != subtree)
		subtree->parent = node;

	update_height(node);
}

static node_t rebalance(node_t node) {

	if (NULL == node)
		return NULL;

	int bfp = balance(node);
	if (abs(bfp) < 2)
		return node;

	assert(2 == abs(bfp));

	if (2 == bfp) {

		if (-1 == balance(node->leafb))
			r_rotate(node->leafb);

		node_t ret = node->leafb;
		
		l_rotate(node);
		return ret;
	} else {

		if (1 == balance(node->leafa))
			l_rotate(node->leafa);

		node_t ret = node->leafa;
		
		r_rotate(node);
		return ret;
	}
}



void tree_insert(tree_t tree, void *item)
{
	tree_set_lock(tree);
	tree->N++;

	node_t node = create_node(item);
	node->tree = tree;

	if (NULL == tree->root) {

		tree->root = node;
		tree_unset_lock(tree);
		return;
	}

	node_t parent = tree->root;
	
	while (true) {

		if (0 > tree->relation(item, parent->item)) {

			if (NULL != parent->leafa)
				parent = parent->leafa;
			else {
				node->parent = parent;
				parent->leafa = node;
				break;
			}
		} else {

			if (NULL != parent->leafb)
				parent = parent->leafb;
			else {
				node->parent = parent;
				parent->leafb = node;
				break;
			}
		}
	}

	update_height(node);

	while (NULL != node->parent)
		node = rebalance(node->parent);

	tree_unset_lock(tree);
}


static node_t node_get_min(node_t node)
{
	if (NULL == node)
		return NULL;

	if (NULL == node->leafa)
		return node;
	
	return node_get_min(node->leafa);
}

static node_t node_get_max(node_t node)
{
	if (NULL == node)
		return NULL;

	if (NULL == node->leafb)
		return node;
	
	return node_get_max(node->leafb);
}


static void remove_node(node_t node)
{

	if ((NULL != node->leafa) && (NULL != node->leafb)) {

		node_t tmp = node_get_min(node->leafb);
		SWAP(node->item, tmp->item);
		node = tmp;
	}

	node_t parent = node->parent;
	node_t child = node->leafa ?: node->leafb;
	
	if (NULL == parent) {

		node->tree->root = child;
	} else {

		parent->leafa = (node == parent->leafa) ? child : parent->leafa;
		parent->leafb = (node == parent->leafb) ? child : parent->leafb;
	}

	if (NULL != child)
		child->parent = parent;
	
	node->tree->N--;

	xfree(node);

	update_height(parent);
	while (NULL != parent)
		parent = rebalance(parent)->parent;
}






static node_t node_find_min(node_t node, const void* ref, tree_rel_f rel)
{
	if (NULL == node)
		return NULL;

	if (0 <= rel(node->item, ref)) {

		node_t ret = node_find_min(node->leafa, ref, rel);
		if (NULL != ret)
			return ret;
		
		if (0 == rel(node->item, ref))
		 	return node;
		else
			return NULL;
	} else 
		return node_find_min(node->leafb, ref, rel);
}

static node_t node_find_max(node_t node, const void* ref, tree_rel_f rel)
{
	if (NULL == node)
		return NULL;

	if (0 >= rel(node->item, ref)) {

		node_t ret = node_find_max(node->leafb, ref, rel);
		if (NULL != ret)
			return ret;
		
		if (0 == rel(node->item, ref))
		 	return node;
		else
			return NULL;
	} else 
		return node_find_max(node->leafa, ref, rel);
}

static node_t node_find(node_t node, const void* ref, tree_rel_f rel)
{
	if ((NULL == node) || (0 == rel(node->item, ref)))
		return node;

	if (0 < rel(node->item, ref))
		return node_find(node->leafa, ref, rel);
	else
		return node_find(node->leafb, ref, rel);	
}




static void* node_return(node_t node, bool remove)
{
	void* item = node ? node->item : NULL;

	if ((remove) && (NULL != node))
		remove_node(node);

	return item;
}



void* tree_find_min(tree_t tree, const void* ref, tree_rel_f rel, bool remove)
{
	tree_set_lock(tree);
	void* item = node_return(node_find_min(tree->root, ref, rel), remove);
	tree_unset_lock(tree);

	return item;	
}

void* tree_find_max(tree_t tree, const void* ref, tree_rel_f rel, bool remove)
{
	tree_set_lock(tree);
	void* item = node_return(node_find_max(tree->root, ref, rel), remove);
	tree_unset_lock(tree);

	return item;
}

void* tree_find(tree_t tree, const void* ref, tree_rel_f rel, bool remove)
{
	tree_set_lock(tree);
	void* item = node_return(node_find(tree->root, ref, rel), remove);
	tree_unset_lock(tree);

	return item;
}

void* tree_get_min(tree_t tree, bool remove)
{
	tree_set_lock(tree);
	void* item = node_return(node_get_min(tree->root), remove);
	tree_unset_lock(tree);

	return item;
}

void* tree_get_max(tree_t tree, bool remove)
{
	tree_set_lock(tree);
	void* item = node_return(node_get_max(tree->root), remove);
	tree_unset_lock(tree);

	return item;
}


static void** write_node(node_t node, void** arr)
{
	if (NULL == node)
		return arr;
	
	arr = write_node(node->leafa, arr);
	arr[0] = node->item;
	arr = write_node(node->leafb, arr + 1);

	return arr;
}

void tree_to_array(tree_t tree, long N, void* arr[N])
{
	tree_set_lock(tree);
	assert(N == tree->N);
	write_node(tree->root, arr);
	tree_unset_lock(tree);
}

long tree_count(tree_t tree)
{
	return tree->N;
}

