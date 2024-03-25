/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * Copyright 2023. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Martin Uecker, Moritz Blumenthal
 *
*/

#include <stdbool.h>
#include <assert.h>
#include <stdint.h>

#include "misc/tree.h"
#include "misc/misc.h"
#include "misc/debug.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "mem.h"

#ifndef USE_CUDA
#define CUDA_MAX_STREAMS 0

static int cuda_get_stream_id(void)
{
	return 0;
}
#endif

bool memcache = true;

void memcache_off(void)
{
	memcache = false;
}

static bool mem_init = false;

static long unused_memory[CUDA_MAX_STREAMS + 1] = { 0 };
static long used_memory[CUDA_MAX_STREAMS + 1] = { 0 };

static tree_t mem_allocs[CUDA_MAX_STREAMS + 1] = { NULL };
static tree_t mem_cache[CUDA_MAX_STREAMS + 1] = { NULL };

static const void* min_ptr[CUDA_MAX_STREAMS + 1] = { NULL };
static const void* max_ptr[CUDA_MAX_STREAMS + 1] = { NULL };

struct mem_s {

	const void* ptr;
	size_t len;
	size_t len_used;

	const char* backtrace;
};


static int ptr_cmp(const void* _a, const void* _b)
{
	const struct mem_s* a = _a;
	const struct mem_s* b = _b;

	if (a->ptr == b->ptr)
		return 0;
	
	return (a->ptr > b->ptr) ? 1 : -1;
}

static int size_cmp(const void* _a, const void* _b)
{
	const struct mem_s* a = _a;
	const struct mem_s* b = _b;

	if (a->len == b->len)
		return 0;
	
	return (a->len > b->len) ? 1 : -1;
}

void memcache_init(void)
{
	for (int i = 0; i < CUDA_MAX_STREAMS + 1; i++) {

		assert(NULL == mem_cache[i]);
		assert(NULL == mem_allocs[i]);

		mem_cache[i] = tree_create(size_cmp);
		mem_allocs[i] = tree_create(ptr_cmp);
	}

	mem_init = true;
}

bool memcache_is_empty(void)
{
	if (!mem_init)
		return true;

	for (int i = 0; i < CUDA_MAX_STREAMS + 1; i++) {

		if (NULL != mem_cache[i] && 0 != tree_count(mem_cache[i]))
			return false;
	
		if (NULL != mem_allocs[i] && 0 != tree_count(mem_allocs[i]))
			return false;
	}

	return true;
}

void memcache_destroy(void)
{
	if (!mem_init)
		return;

	assert(memcache_is_empty());

	for (int i = 0; i < CUDA_MAX_STREAMS + 1; i++) {

		tree_free(mem_cache[i]);
		tree_free(mem_allocs[i]);

		mem_cache[i] = NULL;
		mem_allocs[i] = NULL;
	}
}

static void print_mem_tree(int dl, tree_t tree)
{
	int N = tree_count(tree);
	
	struct mem_s* m[N];
	tree_to_array(tree, N, (void**)m);

	for (int j = 0; j < N; j++) {

		debug_printf(dl, "ptr: %p, len: %zd\n", m[j]->ptr, m[j]->len);

		if (NULL != m[j]->backtrace)
			debug_printf(dl, "%s", m[j]->backtrace);
	}
}

void debug_print_memcache(int dl)
{
	if (!mem_init)
		return;

	for (int i = 0; i < CUDA_MAX_STREAMS + 1; i++) {
		
		if (NULL == mem_allocs[i])
			return;

		debug_printf(dl, "%ld allocated for stream %i (%ld used / %ld unused)\n", unused_memory[i] + used_memory[i], i, used_memory[i], unused_memory[i]);

		print_mem_tree(dl, mem_cache[i]);
		print_mem_tree(dl, mem_allocs[i]);
	}
}

static int inside_p(const void* _rptr, const void* ptr)
{
	const struct mem_s* rptr = _rptr;

	if ((ptr >= rptr->ptr) && (ptr < rptr->ptr + rptr->len))
		return 0;
	
	return (rptr->ptr > ptr) ? 1 : -1;
}


static struct mem_s* search(const void* ptr, bool remove, int i)
{
	if (NULL == mem_allocs[i])
		return NULL;

	if (NULL == min_ptr[i] || ptr < min_ptr[i])
		return NULL;

	if (NULL == max_ptr[i] || ptr > max_ptr[i])
		return NULL;

	return tree_find(mem_allocs[i], ptr, inside_p, remove);
}



static int find_free_p(const void* _rptr, const void* _cmp)
{
	const struct mem_s* rptr = _rptr;
	const long* cmp = _cmp;

	size_t min = cmp[0];
	size_t max = cmp[1];

	if ((rptr->len >= min) && (rptr->len <= max))
		return 0;
	
	return (rptr->len > max) ? 1 : -1;
}


static struct mem_s* find_free(size_t size, int i)
{
	if (NULL == mem_cache[i])
		return NULL;

	size_t cmp[2] = { size, (0 == size) ? UINT64_MAX : 4 * size };

	return tree_find_min(mem_cache[i], &cmp, find_free_p, true);
}


void memcache_clear(void (*device_free)(const void*x))
{
	if (!mem_init)
		return;

	struct mem_s* nptr = find_free(0, cuda_get_stream_id());
	
	while (NULL != nptr) {

		debug_printf(DP_DEBUG3, "Freeing %ld bytes.\n", nptr->len);

		device_free(nptr->ptr);
		xfree(nptr);

		nptr = find_free(0, cuda_get_stream_id());
	}
}


bool mem_ondevice(const void* ptr)
{
	if (NULL == ptr)
		return false;

	if (!mem_init)
		return false;

	int stream = cuda_get_stream_id();
	
	if (stream != CUDA_MAX_STREAMS)
		return (NULL != search(ptr, false, cuda_get_stream_id())) || (NULL != search(ptr, false, CUDA_MAX_STREAMS));

	if (NULL != search(ptr, false, CUDA_MAX_STREAMS))
		return true;

	for (int i = 0; i < CUDA_MAX_STREAMS; i++)
		if (NULL != search(ptr, false, i))
			return true;

	return false; 
}

void mem_device_free(void* ptr, void (*device_free)(const void* ptr))
{
	for (int i = 0; i < CUDA_MAX_STREAMS + 1; i++) {

		if (i != cuda_get_stream_id() && CUDA_MAX_STREAMS != cuda_get_stream_id())
			continue;

		struct mem_s* nptr = search(ptr, true, i);

		if (NULL == nptr)
			continue;

		if (NULL != nptr->backtrace)
			xfree(nptr->backtrace);

		nptr->backtrace = NULL;

		assert(NULL != nptr);
		assert(nptr->ptr == ptr);

		if (memcache) {
		
			tree_insert(mem_cache[i], nptr);

#pragma			omp atomic
			unused_memory[i] += nptr->len_used;

#pragma			omp atomic
			used_memory[i] -= nptr->len_used;

		} else {

#pragma 		omp atomic
			used_memory[i] -= nptr->len_used;

			device_free(ptr);
			xfree(nptr);
		}

		return;
	}

	assert(0);
}



void* mem_device_malloc(long size, void* (*device_alloc)(size_t))
{
	int stream = cuda_get_stream_id();

#pragma omp atomic
	used_memory[stream] += size;

	struct mem_s* nptr = find_free(size, stream);

	if (NULL != nptr) {

#pragma		omp atomic
		unused_memory[stream] -= size;

		nptr->len_used = size;

	} else {

		void* ptr = device_alloc(size);

#pragma 	omp critical
		{
			min_ptr[stream] = min_ptr[stream] ? MIN(min_ptr[stream], ptr) : ptr;
			max_ptr[stream] = max_ptr[stream] ? MAX(max_ptr[stream], ptr + size) : ptr + size;
		}

		PTR_ALLOC(struct mem_s, _nptr);
		_nptr->ptr = ptr;
		_nptr->len = size;
		_nptr->len_used = size;
		_nptr->backtrace = NULL;

		nptr = PTR_PASS(_nptr);
	}

	//use for debugging memcache
	//nptr->backtrace = debug_good_backtrace_string(2);

	tree_insert(mem_allocs[stream], nptr);

	return (void*)nptr->ptr;
}





