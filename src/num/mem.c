/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2016	Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 
*/

#include <stdbool.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "misc/misc.h"
#include "misc/debug.h"

#include "mem.h"





bool memcache = true;

void memcache_off(void)
{
	memcache = false;
}

struct mem_s {

	const void* ptr;
	size_t len;
	bool device;
	bool free;
	int device_id;
	int thread_id;
	struct mem_s* next;
};

static struct mem_s* mem_list = NULL;


static bool inside_p(const struct mem_s* rptr, const void* ptr)
{
	return (ptr >= rptr->ptr) && (ptr < rptr->ptr + rptr->len);
}

static struct mem_s* search(const void* ptr, bool remove)
{
	struct mem_s* rptr = NULL;

	#pragma omp critical
	{
		struct mem_s** nptr = &mem_list;

		while (true) {

			rptr = *nptr;

			if (NULL == rptr)
				break;

			if (inside_p(rptr, ptr)) {

				if (remove)
					*nptr = rptr->next;

				break;
			}

			nptr = &(rptr->next);
		}
	}

	return rptr;
}

static bool free_check_p(const struct mem_s* rptr, size_t size, int dev, int tid)
{
	return (rptr->free && (rptr->device_id == dev) && (rptr->len >= size)
			&& ((-1 == tid) || (rptr->thread_id == tid)));
}

static struct mem_s** find_free_unsafe(size_t size, int dev, int tid)
{
	struct mem_s* rptr = NULL;
	struct mem_s** nptr = &mem_list;

	while (true) {

		rptr = *nptr;

		if (NULL == rptr)
			break;

		if (free_check_p(rptr, size, dev, tid))
			break;

		nptr = &(rptr->next);
	}

	return nptr;
}

static struct mem_s* find_free(size_t size, int dev)
{
	struct mem_s* rptr = NULL;

	#pragma omp critical
	{
		rptr = *find_free_unsafe(size, dev, -1);

		if (NULL != rptr)
			rptr->free = false;
	}

	return rptr;
}

static void insert(const void* ptr, size_t len, bool device, int dev)
{
	PTR_ALLOC(struct mem_s, nptr);
	nptr->ptr = ptr;
	nptr->len = len;
	nptr->device = device;
	nptr->device_id = dev;
#ifdef _OPENMP
	nptr->thread_id = omp_get_thread_num();
#else
	nptr->thread_id = -1;
#endif
	nptr->free = false;

	#pragma omp critical
	{
		nptr->next = mem_list;
		mem_list = PTR_PASS(nptr);
	}
}

void memcache_clear(int dev, void (*device_free)(const void*x))
{
	struct mem_s* nptr = NULL;

	if (!memcache)
		return;

	do {
		#pragma omp critical
		{
#ifdef _OPENMP
			int tid = omp_get_thread_num();
#else
			int tid = -1;
#endif
			struct mem_s** rptr = find_free_unsafe(0, dev, tid);
			nptr = *rptr;

			// remove from list

			if (NULL != nptr)
				*rptr = nptr->next;
		}

		if (NULL != nptr) {

			assert(nptr->device);

			debug_printf(DP_DEBUG3, "Freeing %ld bytes. (DID: %d TID: %d)\n\n",
					nptr->len, nptr->device_id, nptr->thread_id);

			device_free(nptr->ptr);
			xfree(nptr);
		}

	} while (NULL != nptr);
}


bool mem_ondevice(const void* ptr)
{
	if (NULL == ptr)
		return false;

	struct mem_s* p = search(ptr, false);
	bool r = ((NULL != p) && p->device);

	return r;
}

bool mem_device_accessible(const void* ptr)
{
	struct mem_s* p = search(ptr, false);	
	return (NULL != p);
}



void mem_device_free(void* ptr, void (*device_free)(const void* ptr))
{
	struct mem_s* nptr = search(ptr, !memcache);

	assert(NULL != nptr);
	assert(nptr->ptr == ptr);
	assert(nptr->device);

	if (memcache) {

		assert(!nptr->free);
		nptr->free = true;

	} else {

		device_free(ptr);
		xfree(nptr);
	}
}


void* mem_device_malloc(int device, long size, void* (*device_alloc)(size_t))
{
	if (memcache) {

		struct mem_s* nptr = find_free(size, device);

		if (NULL != nptr) {

			assert(nptr->device);
			assert(!nptr->free);
#ifdef _OPENMP
			nptr->thread_id = omp_get_thread_num();
#else
			nptr->thread_id = -1;
#endif

			return (void*)(nptr->ptr);
		}
	}

	void* ptr = device_alloc(size);

	insert(ptr, size, true, device);

	return ptr;
}





