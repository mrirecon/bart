/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

#define MIN(x, y) ({ __typeof(x) __x = (x); __typeof(y) __y = (y); (__x < __y) ? __x : __y; })
#define MAX(x, y) ({ __typeof(x) __x = (x); __typeof(y) __y = (y); (__x > __y) ? __x : __y; })

#define UNUSED(x) (void)(x)

#define MAKE_ARRAY(x, ...) ((__typeof__(x)[]){ x, __VA_ARGS__ })

#ifdef __cplusplus
extern "C" {
#ifndef __VLA
#define __VLA(x) 
#endif
#else
#ifndef __VLA
#define __VLA(x) static x
#endif
#endif

extern void* xmalloc(size_t s);

#define XMALLOC(x) (x = xmalloc(sizeof(*x)))

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

#ifdef __cplusplus
}
#endif
#undef __VLA



