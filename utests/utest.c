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
#include <errno.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "num/init.h"
#include "num/mpi_ops.h"
#include "num/rand.h"

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


int abort_on_error = -1;

void abort_or_print(const char* testname)
{
	if (-1 == abort_on_error) {

		char* str = getenv("BART_UTEST_ABORT");

		if (NULL != str) {

			errno = 0;
			int r = (int)strtol(str, NULL, 10);

			if ((errno == 0) && (0 <= r) && (r <= 1))
				abort_on_error = r;

			errno = 0;
		}
	}

	if (1 == abort_on_error)
		error("%s failed\n", testname);
	else
		debug_printf(DP_ERROR, "%s failed\n", testname);
}


int main(int argc, char* argv[])
{
#ifdef USE_MPI
	init_mpi(&argc, &argv);
#else
	(void)argc;
#endif

	int num_tests_run = 0;
	int num_tests_pass = 0;

#ifdef UTEST_GPU
	bart_use_gpu = true;
	num_init_gpu_support();
#else
	num_init();
#endif

	num_rand_init(0ULL);

	for (ut_test_f** ptr = &_utests_begin; ptr != &_utests_end; ptr++) {

		if ((**ptr)())
			num_tests_pass++;
		else
			debug_printf(DP_ERROR, "Test %d failed.\n", num_tests_run);

		num_tests_run++;
	}

	bool good = (num_tests_pass == num_tests_run);

	debug_printf(good ? DP_INFO : DP_ERROR, "%20s: %2d/%2d passed.\n", argv[0], num_tests_pass, num_tests_run);

	deinit_mpi();

	exit(good ? 0 : 1);
}

