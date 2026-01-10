/* Copyright 2021-2023. Martin Uecker and BART Developers.
 * Copyright 2023-2024. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <stdbool.h>
#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "misc/misc.h"

#include "memcfl.h"

struct memcfl {

	const char* name;

	int D;
	const long* dims;
	complex float* data;

	int refcount;
	bool managed;

	struct memcfl* next;
};

static struct memcfl* memcfl_list = NULL;

void memcfl_register(const char* name, int D, const long dims[D], complex float* data, bool managed)
{
	PTR_ALLOC(struct memcfl, mem);

	mem->name = strdup(name);
	mem->D = D;
	mem->next = memcfl_list;

	long* ndims = *TYPE_ALLOC(long[D]);

	for (int i = 0; i < D; i++)
		ndims[i] = dims[i];

	mem->dims = ndims;
	mem->data = data;
	mem->refcount = 1;
	mem->managed = managed;

	memcfl_list = PTR_PASS(mem);
}

complex float* memcfl_create(const char* name, int D, const long dims[D])
{
	complex float* data = xmalloc((size_t)io_calc_size(D, dims, sizeof(complex float)));
	memcfl_register(name, D, dims, data, true);
	return data;
}


bool memcfl_exists(const char* name)
{
	struct memcfl* mem = memcfl_list;

	while (NULL != mem) {

		if (0 == strcmp(mem->name, name))
			return true;

		mem = mem->next;
	}

	return false;
}

const char** memcfl_list_all(void)
{
	struct memcfl* mem = memcfl_list;
	int count = 0;

	while (NULL != mem) {

		mem = mem->next;
		count++;
	}

	const char* (*list)[count + 1] = xmalloc(sizeof *list);

	(*list)[0] = (char*)(uintptr_t)count;	// !

	mem = memcfl_list;

	for(int i = 1; i < count + 1; i++) {

		(*list)[i] = NULL;

		if (mem != NULL)
			(*list)[i] = mem->name;

		mem = mem->next;
	}

	return *list;
}


complex float* memcfl_load(const char* name, int D, long dims[D])
{
	struct memcfl* mem = memcfl_list;

	while (NULL != mem) {

		if (0 == strcmp(mem->name, name))
			break;

		mem = mem->next;
	}

	if (NULL == mem)
		error("Error loading mem cfl %s\n", name);

	if (D < mem->D)
		error("Error loading mem cfl %s\n", name);

	for (int i = 0; i < D; i++)
		dims[i] = (i < mem->D) ? mem->dims[i] : 1;

	mem->refcount++;

	return mem->data;
}


bool memcfl_unmap(const complex float* p)
{
	struct memcfl* mem = memcfl_list;

	while (NULL != mem) {

		if (mem->data == p)
			break;

		mem = mem->next;
	}

	if (NULL == mem)
		return false;

	if (0 >= mem->refcount)
		error("Error unmapping mem cfl\n");

	mem->refcount--;

	return true;
}

void memcfl_unlink(const char* name)
{
	struct memcfl** mem = &memcfl_list;

	while (NULL != *mem) {

		if (0 == strcmp((*mem)->name, name))
			break;

		mem = &(*mem)->next;
	}

	struct memcfl* o = *mem;

	*mem = (*mem)->next;

	// for regular files this is not a problem

	if (0 < o->refcount)
		error("Error unlinking mem cfl\n");

	xfree(o->name);
	xfree(o->dims);
	if (o->managed)
		xfree(o->data);
	xfree(o);
}


