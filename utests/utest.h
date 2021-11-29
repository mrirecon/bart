/* Copyright 2016. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */


#ifndef _UTEST_H
#define _UTEST_H

#include <stdbool.h>

#include "misc/debug.h"
#include "misc/misc.h"


#define UT_ASSERT(test)	\
	return ((test) || (debug_printf(DP_ERROR, "%s:%d assertion `%s` failed.\n", __func__, __LINE__, #test), false))

#define UT_ASSERT2(test)	\
	if (!(test)) return (debug_printf(DP_ERROR, "%s:%d assertion `%s` failed.\n", __func__, __LINE__, #test), false)

#define UT_TOL 1E-6


typedef bool ut_test_f(void);

#if 0
#define UT_REGISTER_TEST(x) \
	ut_test_f* ptr_ ## x __attribute__((section(".utest"))) = &x;
#else
#define UT_REGISTER_TEST(x) 				\
	extern bool call_ ## x(void);			\
	extern bool call_ ## x(void) { return x(); };
#endif

#define UT_GPU_REGISTER_TEST(x) UT_REGISTER_TEST(x)

#define UT_UNUSED_TEST(x) 				\
	extern void unused_ ## x(void);			\
	extern void unused_ ## x(void) { x(); };


#endif
