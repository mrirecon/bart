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


#define UT_RETURN_ASSERT(test)	\
	return ((test) || (debug_printf(DP_ERROR, "%s:%d assertion `%s` failed.\n", __func__, __LINE__, #test), false))

#define UT_RETURN_ON_FAILURE(test)	\
	if (!(test)) return (debug_printf(DP_ERROR, "%s:%d assertion `%s` failed.\n", __func__, __LINE__, #test), false)

#define UT_RETURN_ASSERT_TOL(err, tol)	\
	return ((err <= tol) || (debug_printf(DP_ERROR, "%s:%d assertion `%.2e=%s<=%.2e=%s` failed.\n", __func__, __LINE__, err, #err, tol, #tol), false))

#define UT_RETURN_ON_FAILURE_TOL(err, tol)	\
	if (!(err <= tol)) return (debug_printf(DP_ERROR, "%s:%d assertion `%.2e=%s<=%.2e=%s` failed.\n", __func__, __LINE__, err, #err, tol, #tol), false)

#define UT_TOL 1E-6


typedef bool ut_test_f(void);

extern void abort_or_print(const char* testname);

#if 0
#define UT_REGISTER_TEST(x) \
	ut_test_f* ptr_ ## x __attribute__((section(".utest"))) = &x;
#else
#define UT_REGISTER_TEST(x) 				\
	extern bool call_ ## x(void);			\
	extern bool call_ ## x(void) { bool r = x(); if (!r) abort_or_print(#x); return r; };
#endif

#define UT_GPU_REGISTER_TEST(x) UT_REGISTER_TEST(x)

#define UT_UNUSED_TEST(x) 				\
	extern void unused_ ## x(void);			\
	extern void unused_ ## x(void) { x(); };


#endif
