/* Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef _TYPES_H
#define _TYPES_H

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
#define CONTAINER_OF_CONST(x, T, member)	\
	((T*)((const char*)TYPE_CHECK(__typeof__(&((T*)0)->member), x) - offsetof(T, member)))


#define CAST_CONST(T, x)  ((T)TYPE_CHECK(const T, x))
#define CAST_MAYBE(T, x)	({ \
	__typeof__(x) __tmp = (x); \
	extern __typeof__(*__tmp->TYPEID) T ## _TYPEID; \
	((NULL != __tmp) && (__tmp->TYPEID == &T ## _TYPEID)) ? \
		CONTAINER_OF(__tmp, struct T, super)\
		: NULL;	\
})
#define CAST_DOWN(T, x)	({ \
	__typeof__(x) __tmp = (x); \
	extern __typeof__(*__tmp->TYPEID) T ## _TYPEID; \
	if (__tmp->TYPEID != &T ## _TYPEID) \
		error("%s:%d run-time type check failed: %s\n", __FILE__, __LINE__, #T); \
	CONTAINER_OF(__tmp, struct T, super);	\
})
#define INTERFACE(X) X super
#define CAST_UP(x) (&(x)->super)

typedef const struct typeid_s { int size; const char* name; } TYPEID;

#define TYPEID2(T) (T ## _TYPEID)
#define TYPEID(T) (* __extension__ ({ extern TYPEID T ## _TYPEID; &T ## _TYPEID; }))
#define DEF_TYPEID(T) TYPEID T ## _TYPEID = { .size = sizeof(struct T), .name = "" #T "" }
#define SET_TYPEID(T, x) (TYPE_CHECK(struct T*, x)->super.TYPEID = &TYPEID(T))

#define SIZEOF(x) (size_t)((x)->TYPEID->size)

// redefine auto - needs newer compilers
#define auto __auto_type

#endif

