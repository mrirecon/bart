/* Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */


#ifndef __TYPES_H
#define __TYPES_H

#include <stddef.h>
#include <stdnoreturn.h>

#ifndef __cplusplus
extern noreturn void error(const char* str, ...);
#else
extern __attribute__((noreturn)) void error(const char* str, ...);
#endif



#define TYPE_CHECK(T, x)	(1 ? (x) : (T)0)
#define CONTAINER_OF(x, T, member)	\
	((T*)((char*)TYPE_CHECK(__typeof__(&((T*)0)->member), x) - offsetof(T, member)))

#define CAST_CONST(T, x)  ((T)TYPE_CHECK(const T, x))
#define CAST_MAYBE(T, x)	({ \
	__typeof__(x) __tmp = (x); \
	extern __typeof__(*__tmp->TYPEID) T ## _TYPEID; \
	(__tmp->TYPEID == &T ## _TYPEID) ?		\
		CONTAINER_OF(__tmp, struct T, INTERFACE)\
		: NULL;	\
})
#define CAST_DOWN(T, x)	({ \
	__typeof__(x) __tmp = (x); \
	extern __typeof__(*__tmp->TYPEID) T ## _TYPEID; \
	if (__tmp->TYPEID != &T ## _TYPEID) \
		error("%s:%d run-time type check failed: %s\n", __FILE__, __LINE__, #T); \
	CONTAINER_OF(__tmp, struct T, INTERFACE);	\
})
#define CAST_UP(x) (&(x)->INTERFACE)

#define INTERFACE(X) X INTERFACE

typedef const struct typeid_s { int size; } TYPEID;

#define TYPEID2(T) (T ## _TYPEID)
#define TYPEID(T) (*({ extern TYPEID T ## _TYPEID; &T ## _TYPEID; }))
#define DEF_TYPEID(T) TYPEID T ## _TYPEID = { sizeof(struct T) };
#define SET_TYPEID(T, x) (TYPE_CHECK(struct T*, x)->INTERFACE.TYPEID = &TYPEID(T))

#define SIZEOF(x) ((x)->TYPEID->size)

// redefine auto - needs newer compilers
#define auto __auto_type

#endif

