/* Copyright 2016. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdlib.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "utest.h"


#if 0
/* A linker script is used to assemble a list of all
 * registered unit tests and to set begin and end.
 */
extern ut_test_f* _utests_begin;
extern ut_test_f* _utests_end;
#else
/* A shell script called by make is used to create
 * the list of registered unit tests in UTESTS.
 * This also works on MacOS X.
 */
extern ut_test_f
UTESTS
dummy;

ut_test_f* ut_tests[] = {
UTESTS
};

#define _utests_begin	(ut_tests[0])
#define _utests_end	(ut_tests[ARRAY_SIZE(ut_tests)])
#endif


int main(int argc, char* argv[])
{
	UNUSED(argc);

	int num_tests_run = 0;
	int num_tests_pass = 0;

	for (ut_test_f** ptr = &_utests_begin; ptr != &_utests_end; ptr++)
		UNUSED((num_tests_run++, (**ptr)()) && num_tests_pass++);

	bool good = (num_tests_pass == num_tests_run);

	debug_printf(good ? DP_INFO : DP_ERROR, "%20s: %2d/%2d passed.\n", argv[0], num_tests_pass, num_tests_run);

	exit(good ? 0 : 1);
}


