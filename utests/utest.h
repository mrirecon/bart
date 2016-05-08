
#ifndef _UTEST_H
#define _UTEST_H

#include <stdbool.h>

#include "misc/debug.h"
#include "misc/misc.h"


#define UT_ASSERT(test)	\
	return ((test) || (debug_printf(DP_ERROR, "%s:%d assertion `%s` failed.\n", __func__, __LINE__, #test), false))

#define UT_TOL 1E-6


typedef bool ut_test_f(void);

#define UT_REGISTER_TEST(x) \
	ut_test_f* ptr_ ## x __attribute__((section(".utest"))) = &x;

#endif
