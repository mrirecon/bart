/* Copyright 2026. Institute of Biomedical Imaging. TU Graz.
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

	delayed_compute(NULL);

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

	delayed_compute(NULL);

	UT_RETURN_ASSERT(0 == err);
}

UT_REGISTER_TEST(test_unnecessary_add);

typedef bool (*chain_fun_t)(int N, int OO, const long odims[OO][N], complex float* dst[OO],
				   int II, const long idims[II][N], const complex float* src[II]);

static bool compute_chain_wrap(float tol, unsigned long vflags, chain_fun_t fun, int N,
	int OO, const long odims[OO][N],
	int II, const long idims[II][N])
{
	struct vptr_hint_s* hint = hint_delayed_create(vflags);

	for(int i = 0; i < 16; i++)
		bart_delayed_loop_dims[i] = i;

	num_init_delayed();

	complex float* riargs[II];
	complex float* viargs[II];

	complex float* voargs[OO];
	complex float* roargs[OO];

	for (int i = 0; i < II; i++) {

		riargs[i] = md_alloc(N, idims[i], CFL_SIZE);
		md_gaussian_rand(N, idims[i], riargs[i]);

		viargs[i] = vptr_alloc(N, idims[i], CFL_SIZE, hint);
		md_copy(N, idims[i], viargs[i], riargs[i], CFL_SIZE);
	}

	for (int o = 0; o < OO; o++) {

		voargs[o] = vptr_alloc(N, odims[o], CFL_SIZE, hint);
		roargs[o] = md_alloc(N, odims[o], CFL_SIZE);
	}

	fun(N, OO, odims, voargs, II, idims, (const _Complex float **)viargs);
	fun(N, OO, odims, roargs, II, idims, (const _Complex float **)riargs);

	for (int i = 0; i < II; i++) {

		md_free(viargs[i]);
		md_free(riargs[i]);
	}

	for (int o = 0; o < OO; o++) {

		complex float* tmp_cpu = md_alloc(N, odims[o], CFL_SIZE);
		md_copy(N, odims[o], tmp_cpu, voargs[o], CFL_SIZE);
		md_free(voargs[o]);

		UT_RETURN_ON_FAILURE(md_znrmse(N, odims[o], tmp_cpu, roargs[o]) < tol);
		md_free(roargs[o]);
		md_free(tmp_cpu);
	}

	vptr_hint_free(hint);

	return true;
}

static bool chain_redu(int N, int OO, const long odims[OO][N], complex float* dst[OO],
			      int II, const long idims[II][N], const complex float* src[II])
{
	assert(1 == II);
	assert(1 == OO);

	long rdims[N];
	md_select_dims(N, ~MD_BIT(2), rdims, idims[0]);

	complex float* tmp1 = md_alloc_sameplace(N, idims[0], CFL_SIZE, src[0]);
	complex float* tmp2 = md_alloc_sameplace(N, rdims, CFL_SIZE, src[0]);

	md_zsmul(N, idims[0], tmp1, src[0], 2.0f +1.I);
	md_ztenmul(N, rdims, tmp2, idims[0], tmp1, idims[0], tmp1);
	md_zmul2(N, odims[0], MD_STRIDES(N, odims[0], CFL_SIZE), tmp1,
			MD_STRIDES(N, rdims, CFL_SIZE), tmp2,
			MD_STRIDES(N, odims[0], CFL_SIZE), tmp1);

	md_copy(N, odims[0], dst[0], tmp1, CFL_SIZE);
	md_free(tmp1);
	md_free(tmp2);

	return true;
}




static bool test_redu(void)
{
	bart_delayed_computations = true;

	enum { N = 4 };
	long idims[1][N] = { { 32, 64, 32, 2 } };
	long odims[1][N] = { { 32, 64, 32, 2 } };

	return compute_chain_wrap(1e-6, ~0UL, chain_redu, N,
		1, odims,
		1, idims);
}

UT_REGISTER_TEST(test_redu);

