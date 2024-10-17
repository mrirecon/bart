/* Copyright 2023. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <assert.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "misc/misc.h"

#include "tree.h"

struct node_s {

	bool alloc;

	struct tree_s* tree;
	struct node_s* parent;
	struct node_s* leafa;
	struct node_s* leafb;

	int height;

	const void* key;
	size_t len;

	void* item;
};

#define TBITS 8

struct tree_s {

	int N;
	struct node_s* root;

	size_t node_alloc;
	long search_pos;
	struct node_s* node_mem;

	struct node_s* cache[1 << TBITS];
	int cache_count[1 << TBITS];

	tree_rel_f relation;

#ifdef _OPENMP
	omp_lock_t lock;
#endif
};

#ifdef _OPENMP
static void tree_set_lock(tree_t tree) { omp_set_lock(&(tree->lock)); }
static void tree_unset_lock(tree_t tree) { omp_unset_lock(&(tree->lock)); }
#else
static void tree_set_lock(tree_t /*tree*/) { }
static void tree_unset_lock(tree_t /*tree*/) { }
#endif

typedef struct node_s* node_t;
typedef struct tree_s* tree_t;


void* tree_tag_ptr(tree_t tree, void* ptr)
{
	uint64_t tag = 1;
	int min = tree->cache_count[tag];

	for (int j = 2; j < (1 << TBITS); j++) {

		if (0 == min)
			break;

		if (tree->cache_count[j] < min) {
			min = tree->cache_count[j];
			tag = (uint64_t)j;
		}
	}

	return (void*) ((uint64_t)ptr | (tag << (64 - TBITS)));
}

void* tree_untag_ptr(tree_t /*tree*/, void* ptr)
{
	uint64_t msk = 1;
	msk = (msk << (64 - TBITS)) - 1;
	
	return (void*) ((uint64_t)ptr & msk);
}

int tree_get_tag(tree_t /*tree*/, const void* ptr)
{
	uint64_t msk = 1;
	msk = (msk << (64 - TBITS)) - 1;
	
	return (int) (((uint64_t)ptr & ~msk) >> (64 - TBITS));
}


static struct node_s create_node(void* item)
{
	struct node_s result;

	result.alloc = true;
	result.item = item;
	result.parent = NULL;
	result.leafa = NULL;
	result.leafb = NULL;
	result.height = 1;

	return result;
}

static inline int ptr_cmp(const void* a, const void* b)
{
	if (a == b)
		return 0;
	
	return (a > b) ? 1 : -1;
}

static inline int inside_p(node_t node, const void* ptr)
{
	if ((ptr >= node->key) && (ptr < node->key + node->len))
		return 0;
	
	return (node->key > ptr) ? 1 : -1;
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

	result->node_alloc = 1024;
	result->search_pos = 0;
	result->node_mem = xmalloc(result->node_alloc * sizeof(struct node_s));

	for (size_t i = 0; i < result->node_alloc; i++)
		result->node_mem[i].alloc = false; //mark as free

	for (int i = 0; i < (1 << TBITS); i++) {

		result->cache[i] = NULL;
		result->cache_count[i] = 0;
	}

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
	xfree(tree->node_mem);

#ifdef _OPENMP
	omp_destroy_lock(&(tree->lock));
#endif
	xfree(tree);
}


static int height(node_t node)
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

static node_t rebalance(node_t node)
{
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



static void tree_extend(tree_t tree)
{
	node_t new_mem = xmalloc(2 * tree->node_alloc * sizeof(struct node_s));
		
	for (size_t i = 0; i < tree->node_alloc; i++) {

		new_mem[i] = tree->node_mem[i];

		if (!new_mem[i].alloc)
			continue;

		new_mem[i].parent = (NULL != new_mem[i].parent) ? new_mem + (new_mem[i].parent - tree->node_mem) : NULL;
		new_mem[i].leafa = (NULL != new_mem[i].leafa) ? new_mem + (new_mem[i].leafa - tree->node_mem) : NULL;
		new_mem[i].leafb = (NULL != new_mem[i].leafb) ? new_mem + (new_mem[i].leafb - tree->node_mem) : NULL;
	}

	for (int i = 0; i < (1 << TBITS); i++)
		if (NULL != tree->cache[i])
			tree->cache[i] = new_mem + (tree->cache[i] - tree->node_mem);

	for (size_t  i = 0; i < tree->node_alloc; i++) 
		new_mem[i + tree->node_alloc].alloc = false;

	tree->root = new_mem + (tree->root - tree->node_mem);

	xfree(tree->node_mem);

	tree->node_alloc *= 2;
	tree->node_mem = new_mem;
}



void tree_insert(tree_t tree, void *item)
{
	tree_set_lock(tree);
	tree->N++;

	assert(NULL != tree->relation);

	if ((size_t)tree->N > tree->node_alloc)
		tree_extend(tree);

	while (tree->node_mem[tree->search_pos].alloc)
		tree->search_pos++;

	tree->node_mem[tree->search_pos] = create_node(item);

	node_t node = &(tree->node_mem[tree->search_pos]);
	node->tree = tree;

	if (NULL == tree->root) {

		tree->root = node;
		tree_unset_lock(tree);
		return;
	}

	node_t parent = tree->root;
	
	while (true) {

		if (0 > tree->relation(item, parent->item)) {

			if (NULL == parent->leafa) {

				node->parent = parent;
				parent->leafa = node;
				break;
			}

			parent = parent->leafa;

		} else {

			if (NULL == parent->leafb) {

				node->parent = parent;
				parent->leafb = node;
				break;
			}

			parent = parent->leafb;
		}
	}

	update_height(node);

	while (NULL != node->parent)
		node = rebalance(node->parent);

	tree_unset_lock(tree);
}

void ptr_tree_insert(tree_t tree, void *item, const void* key, size_t len)
{
	tree_set_lock(tree);
	tree->N++;

	assert(NULL == tree->relation);

	if ((size_t)tree->N > tree->node_alloc)
		tree_extend(tree);

	while (tree->node_mem[tree->search_pos].alloc)
		tree->search_pos++;

	tree->node_mem[tree->search_pos] = create_node(item);

	node_t node = &(tree->node_mem[tree->search_pos]);
	node->tree = tree;
	node->key = key;
	node->len = len;

	int tag = tree_get_tag(tree, (void*)key);
	tree->cache[tag] = node;
	tree->cache_count[tag]++;

	if (NULL == tree->root) {

		tree->root = node;
		tree_unset_lock(tree);
		return;
	}

	node_t parent = tree->root;
	
	while (true) {

		if (0 > ptr_cmp(key, parent->key)) {

			if (NULL == parent->leafa) {

				node->parent = parent;
				parent->leafa = node;
				break;
			}

			parent = parent->leafa;

		} else {

			if (NULL == parent->leafb) {

				node->parent = parent;
				parent->leafb = node;
				break;
			}

			parent = parent->leafb;
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
		SWAP(node->key, tmp->key);
		SWAP(node->len, tmp->len);
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

	node->tree->search_pos = MIN(node->tree->search_pos, node - node->tree->node_mem);
	node->alloc = false;
	if (NULL == node->tree->relation)
		node->tree->cache_count[tree_get_tag(node->tree, (void*)node->key)]--;

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

		return NULL;
	}

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

		return NULL;
	}

	return node_find_max(node->leafa, ref, rel);
}

static node_t node_find(node_t node, const void* ref, tree_rel_f rel)
{
	while (!((NULL == node) || (0 == rel(node->item, ref)))) {

		if (0 < rel(node->item, ref))
			node = node->leafa;
		else
			node = node->leafb;	
	}

	return node;
}

static node_t ptr_node_find(node_t node, const void* ref)
{
	while (!(((NULL == node) || (0 == inside_p(node, ref))))) {

		if (0 < inside_p(node, ref))
			node = node->leafa;
		else
			node = node->leafb;	
	}

	return node;
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
	void* item = NULL;
	
	if (NULL == rel) {

		assert(NULL == tree->relation);

		int tag = tree_get_tag(tree, (void*)ref);

		if (NULL == tree->cache[tag])
			return NULL;

		tree_set_lock(tree);

		node_t node = tree->cache[tag];
		if (!node->alloc || (0 != inside_p(node, ref)))
			node = ptr_node_find(tree->root, ref);

		if (!remove && NULL != node)
			tree->cache[tag] = node;

		item = node_return(node, remove);
	} else {

		tree_set_lock(tree);
		item = node_return(node_find(tree->root, ref, rel), remove);
	}

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

void tree_to_array(tree_t tree, int N, void* arr[N])
{
	tree_set_lock(tree);
	assert(N == tree->N);
	write_node(tree->root, arr);
	tree_unset_lock(tree);
}

int tree_count(tree_t tree)
{
	return tree->N;
}

