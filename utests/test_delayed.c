/* Copyright 2024. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#include "misc/debug.h"
#include "misc/list.h"

#include "num/vptr.h"
#include "num/delayed.h"
#include "num/rand.h"
#include "num/init.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "utest.h"


static bool test_unnecessary_copy(void)
{
	bart_delayed_computations = true;

	enum { N = 3 };
	long dims[N] = { 3, 5, 7 };

	complex float* a = vptr_alloc(N, dims, CFL_SIZE, NULL);
	complex float* b = vptr_alloc(N, dims, CFL_SIZE, NULL);

	md_gaussian_rand(N, dims, a);
	md_copy(N, dims, b, a, CFL_SIZE);
	md_zexp(N, dims, a, a);

	md_free(b);

	struct queue_s* queue = get_global_queue();

	queue_set_compute(queue, true);
	list_t ops_queue = get_delayed_op_list(queue);
	delayed_optimize_queue(ops_queue);
	queue_set_compute(queue, false);

	UT_RETURN_ON_FAILURE(1 == list_count(ops_queue)); // only exp should stay

	release_global_queue(queue);

	md_free(a);

	delayed_compute();

	return true;
}

UT_REGISTER_TEST(test_unnecessary_copy);


static bool test_unnecessary_add(void)
{
	bart_delayed_computations = true;

	enum { N = 3 };
	long dims[N] = { 3, 5, 7 };

	complex float* a = vptr_alloc(N, dims, CFL_SIZE, NULL);
	complex float* b = vptr_alloc(N, dims, CFL_SIZE, NULL);
	complex float* c = vptr_alloc(N, dims, CFL_SIZE, NULL);

	md_gaussian_rand(N, dims, a);
	md_clear(N, dims, c, CFL_SIZE);
	md_zadd(N, dims, b, a, c);
	md_free(c);

	struct queue_s* queue = get_global_queue();

	queue_set_compute(queue, true);
	list_t ops_queue = get_delayed_op_list(queue);
	delayed_optimize_queue(ops_queue);
	queue_set_compute(queue, false);

	UT_RETURN_ON_FAILURE(1 == list_count(ops_queue));
	UT_RETURN_ON_FAILURE(delayed_op_is_copy(list_get_item(ops_queue, 0)));

	release_global_queue(queue);

	float err = md_znrmse(N, dims, b, a);

	md_free(a);
	md_free(b);

	delayed_compute();

	UT_RETURN_ASSERT(0 == err);
}

UT_REGISTER_TEST(test_unnecessary_add);

