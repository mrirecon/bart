/* Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */


// to be included in pointed-to object

struct shared_obj_s {

	void (*del)(const struct shared_obj_s* x);
	int refcount;
};

extern void shared_obj_init(struct shared_obj_s* obj, void (*del)(const struct shared_obj_s* s));
extern void shared_obj_destroy(const struct shared_obj_s* x);
extern void shared_obj_ref(const struct shared_obj_s*);



// alternative: to be included in object with pointer

struct shared_ptr_s {

	struct shared_ptr_s* next;
	struct shared_ptr_s* prev;

	void (*del)(const struct shared_ptr_s*);
};


extern void shared_ptr_init(struct shared_ptr_s* dst, void (*del)(const struct shared_ptr_s* p));
extern void shared_ptr_copy(struct shared_ptr_s* dst, struct shared_ptr_s* src);
extern void shared_ptr_destroy(const struct shared_ptr_s* ptr);


