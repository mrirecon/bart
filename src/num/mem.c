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

#ifdef _OPENMP
#include <omp.h>
#endif

#include "misc/tree.h"
#include "misc/misc.h"
#include "misc/debug.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#else
#define MAX_CUDA_DEVICES 1
#define MAX_CUDA_STREAMS 1
#endif


#include "mem.h"

bool memcache = true;

void memcache_off(void)
{
	memcache = false;
}

struct mem_s {

	const void* ptr;
	size_t len;
	size_t len_used;

	int device_id;
	int stream_id;
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

static long unused_memory = 0;
static long used_memory = 0;

static bool mem_init = false;

static tree_t mem_pool = NULL;
static tree_t mem_cache[MAX_CUDA_DEVICES][MAX_CUDA_STREAMS];

void memcache_init(void)
{
	if (mem_init)
		return;
	
	#pragma omp critical(bart_memcache)
	{
		if (!mem_init) {

			for (int i = 0; i < MAX_CUDA_DEVICES; i++)
				for (int j = 0; j < MAX_CUDA_STREAMS; j++)
					mem_cache[i][j] = tree_create(size_cmp);
			
			mem_pool = tree_create(ptr_cmp); 
		}
	
		mem_init = true;
	}
}

bool memcache_is_empty(void)
{
	if (!mem_init)
		return true;

	for (int i = 0; i < MAX_CUDA_DEVICES; i++)
		for (int j = 0; j < MAX_CUDA_STREAMS; j++)
			if (0 != tree_count(mem_cache[i][j]))
				return false;
	
	return (0 == tree_count(mem_pool));
}

void memcache_destroy(void)
{
	if (!mem_init)
		return;

	#pragma omp critical(bart_memcache)
	{
		assert(memcache_is_empty());

		for (int i = 0; i < MAX_CUDA_DEVICES; i++)
			for (int j = 0; j < MAX_CUDA_STREAMS; j++)
				tree_free(mem_cache[i][j]);
		
		tree_free(mem_pool);

		mem_init = false;
	}
}


static void print_mem_tree(int dl, tree_t tree)
{
	int N = tree_count(tree);
	
	struct mem_s* m[N];
	tree_to_array(tree, N, (void**)m);

	for (int j = 0; j < N; j++) 
		debug_printf(dl, "ptr: %p, len: %zd, device_id: %d\n", m[j]->ptr, m[j]->len, m[j]->device_id);
}

void debug_print_memcache(int dl)
{
	debug_printf(dl, "%ld allocated on gpu (%ld used / %ld unused)\n", unused_memory + used_memory, used_memory, unused_memory);

	#pragma omp critical(bart_memcache)
	{
		for (int i = 0; i < MAX_CUDA_DEVICES; i++)
			for (int j = 0; j < MAX_CUDA_STREAMS; j++)
				print_mem_tree(dl, mem_cache[i][j]);
		
		print_mem_tree(dl, mem_pool);
	}
}

static int inside_p(const void* _rptr, const void* ptr)
{
	const struct mem_s* rptr = _rptr;
	if ((ptr >= rptr->ptr) && (ptr < rptr->ptr + rptr->len))
		return 0;
	
	return (rptr->ptr > ptr) ? 1 : -1;
}


static struct mem_s* search(const void* ptr, bool remove)
{
	if (!mem_init)
		return NULL;

	return tree_find(mem_pool, ptr, inside_p, remove);
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


static struct mem_s* find_free(size_t size, int dev, int stream)
{
	if (!mem_init)
		return NULL;

	size_t cmp[2] = { size, (0 == size) ? UINT64_MAX : 4 * size };

	return tree_find_min(mem_cache[dev][stream], &cmp, find_free_p, true);
}


void memcache_clear(int dev, int stream, void (*device_free)(const void*x))
{
	if (!memcache)
		return;

	struct mem_s* nptr = find_free(0, dev, stream);
	
	while (NULL != nptr) {

		debug_printf(DP_DEBUG3, "Freeing %ld bytes. (DID: %d)\n\n", nptr->len, nptr->device_id);

		device_free(nptr->ptr);
		xfree(nptr);

		nptr = find_free(0, dev, stream);
	}
}

int mem_device_num(const void* ptr)
{
	if (NULL == ptr)
		return -1;

	struct mem_s* p = search(ptr, false);
	return (NULL == p) ? -1 : p->device_id;
}


bool mem_ondevice(const void* ptr)
{
	return 0 <= mem_device_num(ptr);
}

void mem_device_free(void* ptr, void (*device_free)(const void* ptr))
{
	assert(mem_init);

	struct mem_s* nptr = search(ptr, true);

	assert(NULL != nptr);
	assert(nptr->ptr == ptr);

	if (memcache) {
		
		tree_insert(mem_cache[nptr->device_id][nptr->stream_id], nptr);

		#pragma omp atomic
		unused_memory += nptr->len_used;
		#pragma omp atomic
		used_memory -= nptr->len_used;

	} else {
		#pragma omp atomic
		used_memory -= nptr->len_used;

		device_free(ptr);
		xfree(nptr);
	}
}



void* mem_device_malloc(int device, int stream, long size, void* (*device_alloc)(size_t))
{
	assert(mem_init);

	#pragma omp atomic
	used_memory += size;

	struct mem_s* nptr = find_free(size, device, stream);

	if (NULL != nptr) {

		#pragma omp atomic
		unused_memory -= size;

		nptr->len_used = size;

	} else {

		void* ptr = device_alloc(size);

		PTR_ALLOC(struct mem_s, _nptr);
		_nptr->ptr = ptr;
		_nptr->len = size;
		_nptr->len_used = size;
		_nptr->device_id = device;
		_nptr->stream_id = stream;

		nptr = PTR_PASS(_nptr);
	}

	tree_insert(mem_pool, nptr);
	return (void*)(nptr->ptr);
}





