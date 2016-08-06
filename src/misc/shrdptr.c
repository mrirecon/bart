/* Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */
#include <stdlib.h>
#include <assert.h>

#include "misc/types.h"

#include "shrdptr.h"


void shared_obj_init(struct shared_obj_s* obj, void (*del)(const struct shared_obj_s* s))
{
	obj->refcount = 1;
	obj->del = del;
}

void shared_obj_ref(const struct shared_obj_s* obj)
{
	((struct shared_obj_s*)obj)->refcount++;
}

void shared_obj_destroy(const struct shared_obj_s* x)
{
	if (1 > --(((struct shared_obj_s*)x)->refcount))
		if (NULL != x->del)
			x->del(x);
}


void shared_ptr_init(struct shared_ptr_s* dst, void (*del)(const struct shared_ptr_s*))
{
	dst->next = dst->prev = dst;
	dst->del = del;
}


void shared_ptr_copy(struct shared_ptr_s* dst, struct shared_ptr_s* src)
{
	dst->next = src;
	dst->prev = src->prev;
	src->prev->next = dst;
	src->prev = dst;
	dst->del = src->del;
}

static void shared_unlink(struct shared_ptr_s* data)
{
	data->next->prev = data->prev;
	data->prev->next = data->next;
}

void shared_ptr_destroy(const struct shared_ptr_s* ptr)
{
	if (ptr->next == ptr) {

		assert(ptr == ptr->prev);
		ptr->del(ptr);

	} else {

		shared_unlink(CAST_CONST(struct shared_ptr_s*, ptr));
	}
}


