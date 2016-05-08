
#include <stdlib.h>

#include "misc/debug.h"

#include "utest.h"


void ut_run_tests(unsigned int N, ut_test_f* tests[static N])
{
	int num_tests_run = 0;
	int num_tests_failed = 0;

	for (unsigned int i = 0; i < N; i++)
		(num_tests_run++, tests[i]()) || num_tests_failed++;

	debug_printf(DP_INFO, "%d/%d failed.\n", num_tests_failed, num_tests_run);

	exit((0 == num_tests_failed) ? 0 : 1);
}


