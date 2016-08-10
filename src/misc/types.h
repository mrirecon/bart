/* Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */


#ifndef __TYPES_H
#define __TYPES_H

#define TYPE_CHECK(T, x)	(1 ? (x) : (T)0)
#define CONTAINER_OF(x, T, member)	\
	((T*)((char*)TYPE_CHECK(__typeof__(&((T*)0)->member), x) - offsetof(T, member)))


#define CAST_CONST(T, x)  ((T)TYPE_CHECK(const T, x))
#define CAST_DOWN(T, x)	({ \
	__typeof__(x) __tmp = (x); \
	extern __typeof__(*__tmp->TYPEID) T ## _TYPEID; \
	if (__tmp->TYPEID != &T ## _TYPEID) \
		error("run-time type check failed: %s\n", #T); \
	CONTAINER_OF(__tmp, struct T, INTERFACE);	\
})
#define CAST_UP(x) (&(x)->INTERFACE)

#define INTERFACE(X) X INTERFACE

typedef const struct typeid_s { int:0; } TYPEID;

#define TYPEID(T) T ## _TYPEID
#define DEF_TYPEID(T) TYPEID TYPEID(T)
#define SET_TYPEID(T, x) (TYPE_CHECK(struct T*, x)->INTERFACE.TYPEID = &TYPEID(T))

// redefine auto - needs newer compilers
#define auto __auto_type

#endif

