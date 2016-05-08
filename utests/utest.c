
#include <stdlib.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "utest.h"

extern ut_test_f* _utests_begin;
extern ut_test_f* _utests_end;


int main()
{
	int num_tests_run = 0;
	int num_tests_failed = 0;

	for (ut_test_f** ptr = &_utests_begin; ptr != &_utests_end; ptr++)
		UNUSED((num_tests_run++, (**ptr)()) || num_tests_failed++);

	debug_printf(DP_INFO, "%d/%d failed.\n", num_tests_failed, num_tests_run);

	exit((0 == num_tests_failed) ? 0 : 1);
}


