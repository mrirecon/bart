
#ifndef _UTEST_H
#define _UTEST_H

#include <stdbool.h>

#include "misc/debug.h"


#define UT_ASSERT(test)	\
	return ((test) || (debug_printf(DP_ERROR, "%s:%d assertion `%s` failed.\n", __func__, __LINE__, #test), false))

#define UT_TOL 1E-6

typedef bool ut_test_f(void);

extern void ut_run_tests(unsigned int N, ut_test_f* tests[static N]);

#endif
