
#include <stdlib.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "utest.h"

extern ut_test_f* _utests_begin;
extern ut_test_f* _utests_end;


int main(int argc, char* argv[])
{
	UNUSED(argc);

	int num_tests_run = 0;
	int num_tests_pass = 0;

	for (ut_test_f** ptr = &_utests_begin; ptr != &_utests_end; ptr++)
		UNUSED((num_tests_run++, (**ptr)()) && num_tests_pass++);

	bool good = (num_tests_pass == num_tests_run);

	debug_printf(good ? DP_INFO : DP_ERROR, "%s: \t%d/%d passed.\n", argv[0], num_tests_pass, num_tests_run);

	exit(good ? 0 : 1);
}


