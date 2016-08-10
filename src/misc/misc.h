/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef __MISC_H
#define __MISC_H

#include <stdlib.h>
#include <stddef.h>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

#define MIN(x, y) ({ __typeof(x) __x = (x); __typeof(y) __y = (y); (__x < __y) ? __x : __y; })
#define MAX(x, y) ({ __typeof(x) __x = (x); __typeof(y) __y = (y); (__x > __y) ? __x : __y; })

#define UNUSED(x) (void)(x)


#define MAKE_ARRAY(x, ...) ((__typeof__(x)[]){ x, __VA_ARGS__ })
#define ARRAY_SIZE(x)	(sizeof(x) / sizeof(x[0]))


#include "misc/cppwrap.h"

extern void* xmalloc(size_t s);
extern void xfree(const void*);
extern void warn_nonnull_ptr(void*);

#define XMALLOC(x)	(x = xmalloc(sizeof(*x)))
#define XFREE(x)	(xfree(x), x = NULL)

#define _TYPE_ALLOC(T)		((T*)xmalloc(sizeof(T)))
#define TYPE_ALLOC(T)		_TYPE_ALLOC(__typeof__(T))
// #define TYPE_CHECK(T, x)	({ T* _ptr1 = 0; __typeof(x)* _ptr2 = _ptr1; (void)_ptr2; (x);  })

#define _PTR_ALLOC(T, x)										\
	T* x __attribute__((cleanup(warn_nonnull_ptr))) = xmalloc(sizeof(T))


#define PTR_ALLOC(T, x)		_PTR_ALLOC(__typeof__(T), x)
#define PTR_FREE(x)		XFREE(x)
#define PTR_PASS(x)		({ __typeof__(x) __tmp = (x); (x) = NULL; __tmp; })


extern int parse_cfl(_Complex float res[1], const char* str);
extern void error(const char* str, ...);

extern void print_dims(int D, const long dims[__VLA(D)]);
extern void debug_print_dims(int dblevel, int D, const long dims[__VLA(D)]);

typedef int (*quicksort_cmp_t)(const void* data, unsigned int a, unsigned int b);

extern void quicksort(unsigned int N, unsigned int ord[__VLA(N)], const void* data, quicksort_cmp_t cmp);

extern void mini_cmdline(int argc, char* argv[], int expected_args, const char* usage_str, const char* help_str);
extern _Bool mini_cmdline_bool(int argc, char* argv[], char flag_char, int expected_args, const char* usage_str, const char* help_str);

extern void print_long(unsigned int D, const long arr[__VLA(D)]);
extern void print_float(unsigned int D, const float arr[__VLA(D)]);
extern void print_int(unsigned int D, const int arr[__VLA(D)]);
extern void print_complex(unsigned int D, const _Complex float arr[__VLA(D)]);

extern unsigned int bitcount(unsigned int flags);

extern const char* command_line;
extern void save_command_line(int argc, char* argv[__VLA(argc)]);

#include "misc/cppwrap.h"

#endif // __MISC_H

