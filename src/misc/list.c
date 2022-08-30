/* Copyright 2021. Uecker Lab, University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * */

#include <assert.h>
#include <stdbool.h>

#include "misc/misc.h"
#include "misc/types.h"

#include "list.h"


struct node_s {

	struct node_s* next;
	struct node_s* prev;
	void* item;
};

struct list_s {

	int N;
	struct node_s* head;
	struct node_s* tail;

	int current_index;
	struct node_s* current;
};

typedef struct node_s* node_t;
typedef struct list_s* list_t;


static node_t create_node(void* item)
{
	PTR_ALLOC(struct node_s, result);

	result->item = item;
	result->next = NULL;
	result->prev = NULL;

	return PTR_PASS(result);
}

/**
 * Create an empty list
 */
list_t list_create(void)
{
	PTR_ALLOC(struct list_s, result);
	result->head = NULL;
	result->tail = NULL;
	result->N = 0;

	result->current = NULL; // for fast looping over list items
	result->current_index = -1;

	return PTR_PASS(result);
}

static void free_node(node_t node)
{
	xfree(node);
}


/**
 * Free a list and all its nodes (item in nodes must be freed manually)
 *
 * @param list
 */
void list_free(list_t list)
{
	while (0 < list->N)
		list_pop(list);
	xfree(list);
}

/**
 * Insert new item at index 0
 *
 * @param list
 * @param item pointer to item;
 */
void list_push(list_t list, void* item)
{
	assert(NULL != item);

	auto node = create_node(item);

	if (0 < list->N)
		list->head->prev=node;
	else
		list->tail = node;

	node->next = list->head;
	list->head = node;

	if (-1 < list->current_index)
		list->current_index++;

	list->N++;
}

/**
 * Insert new item at last index
 *
 * @param list
 * @param item pointer to item;
 */
void list_append(list_t list, void* item)
{
	assert(NULL != item);

	if (0 == list->N) {

		list_push(list, item);
		return;
	}

	auto node = create_node(item);

	node->prev = list->tail;
	list->tail->next = node;

	list->tail = node;
	list->N++;
}

static void list_remove_node(list_t list, node_t node)
{
	if (NULL == node->next)
		list->tail = node->prev;
	else
		node->next->prev = node->prev;

	if (NULL == node->prev)
		list->head = node->next;
	else
		node->prev->next = node->next;

	list->N--;
}

/**
 * Remove and return first item of list
 * Returns NULL if list is empty
 *
 * @param list
 */
void* list_pop(list_t list)
{
	return list_remove_item(list, 0);
}

static node_t list_get_node(list_t list, int index)
{
	assert(index < list->N);
	assert(index >= 0);


	int dist_head = index;
	int dist_tail = list->N + 1 - index;
	int dist_curr = abs(index - list->current_index);

	if (abs(dist_head) < abs(dist_curr)) {

		list->current_index = 0;
		list->current = list->head;
	}

	if (abs(dist_tail) < abs(dist_curr)) {

		list->current_index = list->N - 1;
		list->current = list->tail;
	}

	while (index > list->current_index) {

		list->current_index++;
		list->current = list->current->next;
	}

	while (index < list->current_index) {

		list->current_index--;
		list->current = list->current->prev;
	}

	return list->current;
}

/**
 * Insert new item at specified index
 *
 * @param list
 * @param item pointer to item;
 * @param index index where item is inserted
 */
void list_insert(list_t list, void* item, int index)
{
	assert(NULL != item);
	assert(0 <= index);
	assert(list->N >= index);

	if (0 == index) {

		list_push(list, item);
		return;
	}

	if (index == list->N) {

		list_append(list, item);
		return;
	}

	auto node = create_node(item);
	auto prev = list_get_node(list, index - 1);
	auto next = list_get_node(list, index);
	

	node->prev = prev;
	node->next = next;

	next->prev = node;
	prev->next = node;

	list->N++;

	list->current_index = index;
	list->current = node;
}

/**
 * Remove and return item at index
 *
 * @param list
 * @param index of item to be removed
 */
void* list_remove_item(list_t list, int index)
{
	if (0 == list->N)
		return NULL;

	if (0 > index)
		index += list->N;

	auto node = list_get_node(list, index);

	if (-1 < list->current_index) {

		if (index < list->current_index) {

			list->current_index--;
		} else {

			list->current = list->current->prev;
			list->current_index--;
		}
	}

	list_remove_node(list, node);

	auto result = node->item;
	free_node(node);

	return result;
}

/**
 * Return item at index
 *
 * @param list
 * @param index of item
 */
void* list_get_item(list_t list, int index)
{
	return list_get_node(list, index)->item;
}


static inline bool cmp_wrappper(list_cmp_t cmp, const void* item, const void* ref)
{
	if (NULL == cmp)
		return (item == ref);
	else
		return cmp(item, ref);
}

/**
 * Find first index for which cmp(item, ref) evaluates true
 * If (cmp == NULL) the first index with item==ref is retruned
 * If no index is found, -1 is returned
 *
 * @param list
 * @param ref reference pointer
 * @param cmp function to compare list entry with reference
 */
int list_get_first_index(list_t list, const void* ref, list_cmp_t cmp)
{
	node_t node = list->head;

	for (int result = 0; result < list->N; result++) {

		if (cmp_wrappper(cmp, node->item, ref)) {

			list->current_index = result;
			list->current = node;
			return result;
		}

		node = node->next;
	}

	return -1;
}

/**
 * Returns first item for which cmp(item, ref) evaluates true
 *
 * @param list
 * @param ref reference pointer
 * @param cmp function to compare list entry with reference
 * @param remove remove item from list
 */
void* list_get_first_item(list_t list, const void* ref, list_cmp_t cmp, bool remove)
{

	int idx = list_get_first_index(list, ref, cmp);

	void* result = NULL;
	
	if (0 <= idx)
		result = (remove ? list_remove_item : list_get_item)(list, idx);
	
	return result;
}

/**
 * Return number of items in list
 *
 * @param list
 */
int list_count(list_t list)
{
	return list->N;
}

/**
 * Return number of items in list for which cmp evaluates true
 * If (cmp == NULL) item==ref is evaluated
 *
 * @param list
 * @param ref reference pointer
 * @param cmp function to compare list entry with reference
 */
int list_count_cmp(list_t list, const void* ref, list_cmp_t cmp)
{
	node_t node = list->head;
	int result = 0;

	while (NULL != node) {

		if (cmp_wrappper(cmp, node->item, ref))
			result++;

		node = node->next;
	}

	return result;
}

/**
 * Copy items of list to array
 *
 * @param N number of items expected
 * @param items output array
 * @param list
 */
void list_to_array(int N, void* items[N], list_t list)
{
	assert(N == list->N);

	auto node = list->head;

	for (int i = 0; i < N; i++) {

		items[i] = node->item;
		node = node->next;
	}
}

/**
 * Copy items from array to list
 *
 * @param N number of items
 * @param item output array
 */
list_t array_to_list(int N, void* items[N])
{
	auto result = list_create();

	for (int i = N - 1; i >= 0; i--)
		list_push(result, items[i]);

	return result;
}

/**
 * Return list with all items for which cmp(item, ref) evaluates true
 * If (cmp == NULL) the first index with item==ref is retruned
 *
 * @param list
 * @param ref reference pointer
 * @param cmp function to compare list entry with reference
 */
list_t list_get_sublist(list_t list, const void* ref, list_cmp_t cmp)
{
	auto result = list_create();

	auto node = list->head;
	while (NULL != node) {

		if (cmp_wrappper(cmp, node->item, ref))
			list_append(result, node->item);

		node = node->next;
	}

	return result;
}

/**
 * Return list with all items for which cmp(item, ref) evaluates true and remove items from list
 * If (cmp == NULL) the first index with item==ref is retruned
 *
 * @param list
 * @param ref reference pointer
 * @param cmp function to compare list entry with reference
 */
list_t list_pop_sublist(list_t list, const void* ref, list_cmp_t cmp)
{
	auto result = list_create();

	list->current = NULL;
	list->current_index = -1;

	auto node = list->head;

	while (NULL != node) {

		if (cmp_wrappper(cmp, node->item, ref)) {

			auto item = node->item;
			list_remove_node(list, node);

			auto tmp = node->next;
			free_node(node);
			node = tmp;

			list_append(result, item);

		} else {

			node = node->next;
		}
	}

	return result;
}

/**
 * Append all items of list b to list a (and frees b)
 *
 * @param a
 * @param b
 * @param free
 */
void list_merge(list_t a, list_t b, bool free) {

	if ((0 < a->N) && (0 < b->N)) {

		a->tail->next = b->head;
		b->head->prev = a->tail;
		a->tail = b->tail;
		b->head = NULL;
		b->tail = NULL;
		a->N += b->N;
		b->N = 0;
	}

	if (0 == a->N) {

		a->head = b->head;
		a->tail = b->tail;
		b->head = NULL;
		b->tail = NULL;
		a->N += b->N;
		b->N = 0;
	}

	if (free)
		list_free(b);
}

/**
 * Copy all items of a list to a new list
 *
 * @param list
 */
list_t list_copy(list_t list) {

	list_t result = list_create();

	auto node = list->head;

	while (NULL != node) {

		list_append(result, node->item);
		node = node->next;
	}

	return result;
}
