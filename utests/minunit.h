
#ifndef _MINUNIT_H
#define _MINUNIT_H

#include <stdbool.h>

#include "misc/debug.h"


#define MU_ASSERT(test)	\
	return ((test) || (debug_printf(DP_ERROR, "%s:%d assertion `%s` failed.\n", __func__, __LINE__, #test), false))

#define MU_RUN_TEST(test) (num_tests_run++, (test()) || num_tests_failed++)

#define TOL 1E-6

int num_tests_run = 0;
int num_tests_failed = 0;

#endif
