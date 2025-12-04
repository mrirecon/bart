/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <stdbool.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/types.h"
#include "misc/list.h"
#include "misc/shrdptr.h"

#include "num/iovec.h"
#include "num/multind.h"

#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/const.h"
#include "nlops/stack.h"

#include "snlop.h"

static void arg_del(const struct shared_obj_s* sptr)
{
	const struct nlop_arg_s* x = CONTAINER_OF(sptr, const struct nlop_arg_s, sptr);

	if (NULL != x->name)
		xfree(x->name);

	if (NULL != x->del)
		x->del(x);

	xfree(x);
}

const struct nlop_arg_s* arg_unref(const struct nlop_arg_s* x)
{
	if (NULL == x)
		return NULL;

	shared_obj_destroy(&x->sptr);

	return x;
}

struct nlop_arg_s* arg_ref(struct nlop_arg_s* x)
{
	if (NULL == x)
		return NULL;

	shared_obj_ref(&x->sptr);

	return x;
}

int arg_ref_count(const struct nlop_arg_s* x)
{
	if (NULL == x)
		return -1;

	return x->sptr.refcount;
}

void arg_init(arg_t arg)
{
	shared_obj_init(&arg->sptr, arg_del);
	arg->TYPEID = NULL;

	arg->reshape = NULL;
	arg->dup = NULL;
	arg->stack = NULL;
	arg->del = NULL;
	arg->name = NULL;
}

static struct nlop_arg_s* arg_create(struct snlop_s* x)
{
	PTR_ALLOC(struct nlop_arg_s, arg);

	arg_init(arg);
	arg->x = x;

	return PTR_PASS(arg);
}



struct reg_arg_name {

	const char* name;
	int counter;
};

static list_t reg_arg_names = NULL;

static bool cmp_names(const void* _reg_name, const void* _sea_name)
{
	const struct reg_arg_name* reg_name = _reg_name;
	const char* sea_name = _sea_name;

	return 0 == strcmp(reg_name->name, sea_name);
}

static int get_arg_name_count(const char* name)
{
	if (NULL == reg_arg_names)
		reg_arg_names = list_create();

	struct reg_arg_name* x = list_get_first_item(reg_arg_names, name,  cmp_names, false);

	if (NULL == x) {

		PTR_ALLOC(struct reg_arg_name , n);
		n->counter = 0;
		n->name = strdup(name);
		list_append(reg_arg_names, PTR_PASS(n));
		x = list_get_item(reg_arg_names, list_count(reg_arg_names) - 1);
	}

	return (x->counter)++;
}

void arg_set_name(arg_t arg, const char* name)
{
	if ('_' == name[strlen(name) - 1]) {

		arg_set_name_F(arg, ptr_printf("%s%d", name, get_arg_name_count(name)));
		return;
	}

	if (NULL != arg->name)
		xfree(arg->name);

	arg->name = strdup(name);
}

void arg_set_name_F(arg_t arg, const char* name)
{
	arg_set_name(arg, name);
	xfree(name);
}



struct snlop_s {

	struct list_s* oargs;
	struct list_s* iargs;

	struct list_s* targs;


	bool user;

	const struct nlop_s* x;
};

static struct snlop_s* snlop_create(void)
{
	PTR_ALLOC(struct snlop_s, x);

	x->iargs = list_create();
	x->oargs = list_create();

	x->targs = list_create();

	x->x = NULL;

	x->user = true;

	return PTR_PASS(x);
}

void snlop_free(const struct snlop_s* x)
{
	arg_t arg = (arg_t)list_pop(x->iargs);

	while (NULL != arg) {

		arg->x = NULL;
		arg_unref(arg);
		arg = (arg_t)list_pop(x->iargs);
	}

	arg = (arg_t)list_pop(x->oargs);

	while (NULL != arg) {

		arg->x = NULL;
		arg_unref(arg);
		arg = (arg_t)list_pop(x->oargs);
	}

	arg = (arg_t)list_pop(x->targs);

	while (NULL != arg) {

		arg_unref(arg);
		arg = (arg_t)list_pop(x->targs);
	}

	list_free(x->iargs);
	list_free(x->oargs);
	list_free(x->targs);

	if (NULL != x->x)
		nlop_free(x->x);

	xfree(x);
}

bool snlop_check(snlop_t snlop)
{
	if (NULL == snlop)
		return true;

	if (NULL == snlop->x)
		return ((0 == list_count(snlop->oargs)) && (0 == list_count(snlop->oargs)));

	int II = nlop_get_nr_in_args(snlop->x);
	int OO = nlop_get_nr_out_args(snlop->x);

	if (II != list_count(snlop->iargs))
		return false;

	if (OO != list_count(snlop->oargs))
		return false;

	for (int i = 0; i < II; i++)
		if (snlop_get_iarg(snlop, i)->x != snlop)
			return false;

	for (int i = 0; i < OO; i++)
		if (snlop_get_oarg(snlop, i)->x != snlop)
			return false;

	for (int i = 0; i < list_count(snlop->targs); i++)
		if (((arg_t)list_get_item(snlop->targs, i))->x != snlop)
			return false;

	return true;
}

bool arg_check(arg_t arg)
{
	return snlop_check(arg->x);
}


static int snlop_get_idx(arg_t arg, bool out)
{
	if (out)
		return list_get_first_index(arg->x->oargs, arg, NULL);
	else
		return list_get_first_index(arg->x->iargs, arg, NULL);
}



arg_t snlop_get_iarg(snlop_t snlop, int i)
{
	return list_get_item(snlop->iargs, i);
}

arg_t snlop_get_oarg(snlop_t snlop, int i)
{
	return list_get_item(snlop->oargs, i);
}

arg_t snlop_get_targ(snlop_t snlop, int i)
{
	return list_get_item(snlop->targs, i);
}

int snlop_nr_iargs(snlop_t snlop)
{
	return list_count(snlop->iargs);
}

int snlop_nr_oargs(snlop_t snlop)
{
	return list_count(snlop->oargs);
}

int snlop_nr_targs(snlop_t snlop)
{
	return list_count(snlop->targs);
}




void add_to_targs(arg_t arg)
{
	assert(-1 != snlop_get_idx(arg, false));
	list_append(arg->x->targs, arg_ref(arg));
}

static void snlop_link(arg_t oarg, arg_t iarg, bool keep)
{
	assert(oarg->x == iarg->x);

	auto snlop = oarg->x;

	assert(snlop_check(snlop));

	int i = snlop_get_idx(iarg, false);
	int o = snlop_get_idx(oarg, true);

	if (keep) {

		auto cod = nlop_generic_codomain(snlop->x, o);
		auto nlop = nlop_from_linop_F(linop_identity_create(cod->N, cod->dims));
		nlop = nlop_combine_FF(nlop, snlop->x);
		nlop = nlop_dup_F(nlop, 0, i + 1);
		nlop = nlop_link_F(nlop, o + 1, 0);

		snlop->x = nlop;

		arg_t iarg = list_remove_item(snlop->iargs, i);
		if (!arg_is_output(iarg))
			iarg->x = NULL;
		arg_unref(iarg);

		list_push(snlop->oargs, list_remove_item(snlop->oargs, o));

		assert(snlop_check(snlop));
	} else {

		snlop->x = nlop_link_F(snlop->x, o, i);

		arg_t iarg = list_remove_item(snlop->iargs, i);
		if (!arg_is_output(iarg))
			iarg->x = NULL;
		arg_unref(iarg);

		arg_t oarg = list_remove_item(snlop->oargs, o);
		if (!arg_is_input(oarg))
			oarg->x = NULL;
		arg_unref(oarg);

		assert(snlop_check(snlop));
	}
}



static void snlop_merge_args(struct snlop_s* a, struct snlop_s* b)
{
	struct nlop_arg_s* arg = list_pop(b->iargs);

	while (NULL != arg) {

		arg->x = a;
		list_append(a->iargs, arg);
		arg = list_pop(b->iargs);
	}

	arg = list_pop(b->oargs);

	while (NULL != arg) {

		arg->x = a;
		list_append(a->oargs, arg);
		arg = list_pop(b->oargs);
	}

	arg = list_pop(b->targs);

	while (NULL != arg) {

		list_append(a->targs, arg);
		arg = list_pop(b->targs);
	}
}

static void snlop_combine(struct snlop_s* a, struct snlop_s* b)
{
	if (a == b)
		return;

	assert(snlop_check(a));
	assert(snlop_check(b));

	if ((NULL == b) || (NULL == b->x)) {

		snlop_free(b);
		return;
	}

	a->x = (NULL == a->x) ? b->x : nlop_combine_FF(a->x, b->x);
	b->x = NULL;

	snlop_merge_args(a, b);

	assert(snlop_check(a));

	if (b->user && !a->user) {

		snlop_merge_args(b, a);
		b->x = a->x;
		a->x = NULL;

		assert(snlop_check(a));
		assert(snlop_check(b));

		snlop_free(a);
	} else {

		snlop_free(b);
	}
}


void snlop_chain(int N, arg_t oargs[N], arg_t iargs[N], bool keep)
{
	assert(0 < N);

	for (int i = 0; i < N; i++) {

		assert(snlop_check(oargs[i]->x));
		assert(snlop_check(iargs[i]->x));
	}

	for (int i = 0; i < N; i++)
		snlop_combine(iargs[0] -> x, iargs[i]->x);

	for (int i = 0; i < N; i++)
		if (arg_is_input(oargs[0]) && !arg_is_input(oargs[i]))
			snlop_combine(oargs[i]->x, oargs[0]->x);
		else
			snlop_combine(oargs[0]->x, oargs[i]->x);


	assert(iargs[0] -> x != oargs[0] -> x);
	snlop_combine(iargs[0] -> x, oargs[0]->x);

	struct snlop_s* snlop = iargs[0] -> x;
	assert(snlop_check(snlop));

	for (int i = 0; i < N; i++)
		snlop_link(oargs[i], iargs[i], keep);

	assert(snlop_check(snlop));
}



arg_t snlop_input(int N, const long dims[N], const char* name)
{

	struct snlop_s* snlop = snlop_create();
	snlop->user = false;
	struct nlop_arg_s* arg = arg_create(snlop);

	list_append(snlop->iargs, arg_ref(arg));
	list_append(snlop->oargs, arg);

	snlop->x = nlop_from_linop_F(linop_identity_create(N, dims));
	arg->name = strdup(name);
	assert(snlop_check(snlop));

	return arg;
}

arg_t snlop_const(int N, const long dims[N], const _Complex float* data, const char* /*name*/)
{

	struct snlop_s* snlop = snlop_create();
	snlop->user = false;
	struct nlop_arg_s* arg = arg_create(snlop);
	list_append(snlop->oargs, arg);

	snlop->x = nlop_const_create(N, dims, true, data);

	assert(snlop_check(snlop));

	return arg;
}

arg_t snlop_scalar(complex float val)
{
	return snlop_const(1, MD_DIMS(1), &val, NULL);
}

snlop_t snlop_from_nlop_F(const struct nlop_s* nlop)
{
	snlop_t ret = snlop_create();
	ret->user = false;

	ret->x = nlop;

	for (int i = 0; i < nlop_get_nr_in_args(nlop); i++) {

		arg_t arg = arg_create(ret);
		list_append(ret->iargs, arg);
	}

	for (int i = 0; i < nlop_get_nr_out_args(nlop); i++) {

		arg_t arg = arg_create(ret);
		list_append(ret->oargs, arg);
	}

	assert(snlop_check(ret));

	return ret;
}

arg_t snlop_append_nlop_generic_F(int N, arg_t oargs[N], const struct nlop_s* nlop, bool keep)
{
	assert(N == nlop_get_nr_in_args(nlop));
	assert(1 == nlop_get_nr_out_args(nlop));

	for (int i = 0; i < N; i++)
		assert(snlop_check(oargs[i]->x));

	snlop_t snlop = snlop_from_nlop_F(nlop);
	snlop->user = false;

	arg_t iargs[N];
	for (int i = 0; i < N; i++)
		iargs[i] = snlop_get_iarg(snlop, i);

	arg_t ret = snlop_get_oarg(snlop, 0);

	assert(snlop_check(snlop));

	snlop_chain(N, oargs, iargs, keep);

	return ret;
}

arg_t snlop_append_nlop_F(arg_t oarg, const struct nlop_s* nlop, bool keep)
{
	assert(snlop_check(oarg->x));
	arg_t ret = snlop_append_nlop_generic_F(1, (arg_t[1]){ oarg }, nlop, keep);
	assert(snlop_check(ret->x));
	return ret;
}

arg_t snlop_prepend_nlop_generic_F(int N, arg_t iargs[N], const struct nlop_s* nlop)
{
	assert(1 == nlop_get_nr_in_args(nlop));
	assert(N == nlop_get_nr_out_args(nlop));

	for (int i = 0; i < N; i++)
		assert(snlop_check(iargs[i]->x));

	snlop_t snlop = snlop_from_nlop_F(nlop);
	snlop->user = false;

	arg_t oargs[N];
	for (int i = 0; i < N; i++)
		oargs[i] = snlop_get_oarg(snlop, i);

	arg_t ret = snlop_get_iarg(snlop, 0);

	assert(snlop_check(snlop));

	snlop_chain(N, oargs, iargs, false);

	return ret;
}

arg_t snlop_prepend_nlop_F(arg_t iarg, const struct nlop_s* nlop)
{
	assert(snlop_check(iarg->x));
	arg_t ret = snlop_prepend_nlop_generic_F(1, (arg_t[1]){ iarg }, nlop);
	assert(snlop_check(ret->x));
	return ret;
}


bool arg_is_input(arg_t arg)
{
	return (-1 != snlop_get_idx(arg, false));
}

bool arg_is_output(arg_t arg)
{
	return (-1 != snlop_get_idx(arg, true));
}

const struct iovec_s* arg_get_iov_in(arg_t arg)
{
	return nlop_generic_domain(arg->x->x, snlop_get_idx(arg, false));
}

const struct iovec_s* arg_get_iov_out(arg_t arg)
{
	return nlop_generic_codomain(arg->x->x, snlop_get_idx(arg, true));
}


const struct iovec_s* arg_get_iov(arg_t arg)
{
	if (arg_is_output(arg) && arg_is_input(arg)) {

		const struct iovec_s* a = arg_get_iov_in(arg);
		const struct iovec_s* b = arg_get_iov_out(arg);

		if (!iovec_check(a, b->N, b->dims, b->strs))
			error("Argument is input and output with different shapes!\n Use specific function!\n");

		return a;
	}

	if (arg_is_output(arg))
		return arg_get_iov_out(arg);
	else
		return arg_get_iov_in(arg);
}


const struct nlop_s* nlop_from_snlop_F(snlop_t snlop, int OO, arg_t oargs[OO], int II, arg_t iargs[II])
{
	assert(II == nlop_get_nr_in_args(snlop->x));
	assert(OO <= nlop_get_nr_out_args(snlop->x));

	int _OO = nlop_get_nr_out_args(snlop->x);
	int _II = nlop_get_nr_in_args(snlop->x);

	int iperm[_II];
	int operm[_OO];

	for (int i = 0; i < II; i++) {

		iperm[i] = snlop_get_idx(iargs[i], false);
		if (-1 == iperm[i])
			error("Argument %d is not an input of snlop!\n", i);
	}

	for (int i = 0; i < OO; i++) {

		operm[i] = snlop_get_idx(oargs[i], true);
		if (-1 == operm[i])
			error("Argument %d is not an output of snlop!\n", i);
	}

	for (int i = OO; i < _OO; i++) {

		int idx = 0;

		for (int j = 0; j < i; j++) {

			if (idx == operm[j]) {

				idx ++;
				j = -1;
			}
		}

		operm[i] = idx;
	}

	const struct nlop_s* ret = nlop_permute_inputs(snlop->x, _II, iperm);
	ret = nlop_permute_outputs_F(ret, _OO, operm);

	while (OO < nlop_get_nr_out_args(ret))
		ret = nlop_del_out_F(ret, OO);

	ret = nlop_optimize_graph_F(ret);

	snlop_free(snlop);

	return ret;
}

void snlop_del_arg(arg_t arg)
{
	snlop_t snlop = arg->x;
	int o = snlop_get_idx(arg, true);

	assert(0 <= o);

	snlop->x = nlop_optimize_graph(nlop_del_out_F(snlop->x, o));
	arg_unref(list_remove_item(snlop->oargs, o));
}

arg_t snlop_clone_arg(arg_t arg)
{
	const struct iovec_s* iov = arg_get_iov(arg);
	return snlop_append_nlop_F(arg, nlop_from_linop_F(linop_identity_create(iov->N, iov->dims)), true);
}

snlop_t snlop_from_arg(arg_t arg)
{
	arg->x->x = nlop_optimize_graph_F(arg->x->x);
	arg->x->user = true;
	return arg->x;
}

void snlop_replace_iarg(arg_t narg, arg_t oarg)
{
	snlop_t x = oarg->x;
	int i = snlop_get_idx(oarg, false);

	arg_unref(list_remove_item(x->iargs, i));
	list_insert(x->iargs, narg, i);
	narg->x = x;
}

void snlop_replace_oarg(arg_t narg, arg_t oarg)
{
	snlop_t x = oarg->x;
	int i = snlop_get_idx(oarg, true);

	arg_unref(list_remove_item(x->oargs, i));
	list_insert(x->oargs, narg, i);
	narg->x = x;
}

arg_t arg_reshape_out(arg_t arg, int N, const long dims[N])
{
	arg_t ret = snlop_clone_arg(arg);

	int o = snlop_get_idx(ret, true);
	arg->x->x = nlop_reshape_out_F(arg->x->x, o, N, dims);

	return ret;
}

arg_t arg_reshape_in(arg_t arg, int N, const long dims[N])
{
	arg_t ret;

	if (NULL != arg->reshape)
		ret = arg->reshape(arg, N, dims);
	else
		ret = arg_create(arg->x);

	int i = snlop_get_idx(arg, false);
	arg->x->x = nlop_reshape_in_F(arg->x->x, i, N, dims);

	snlop_replace_iarg(ret, arg);

	return ret;
}

arg_t arg_reshape(arg_t arg, int N, const long dims[N])
{
	if (arg_is_input(arg) && arg_is_output(arg))
		error("Argument is input and output!\n Use specific function!\n");

	if (arg_is_input(arg))
		return arg_reshape_in(arg, N, dims);
	else
		return arg_reshape_out(arg, N, dims);
}


arg_t snlop_stack(arg_t a, arg_t b, int stack_dim)
{
	const struct iovec_s* iova = arg_get_iov(a);
	const struct iovec_s* iovb = arg_get_iov(b);

	arg_t ret = NULL;
	if (NULL != a->stack)
		ret = a->stack(a, b, stack_dim, false);

	if ((NULL == ret) && (NULL != b->stack))
		ret = b->stack(a, b, stack_dim, false);

	assert(0 <= stack_dim);

	int N = MAX(iova->N, iovb->N);
	N = MAX(N, stack_dim + 1);

	long adims[N];
	long bdims[N];
	long odims[N];

	md_singleton_dims(N, adims);
	md_singleton_dims(N, bdims);

	md_copy_dims(iova->N, adims, iova->dims);
	md_copy_dims(iovb->N, bdims, iovb->dims);

	for (int i = 0; i < N; i++) {

		odims[i] = adims[i];

		if (i == stack_dim)
			odims[i] += bdims[i];
		else
			assert(adims[i] == bdims[i]);
	}

	const struct nlop_s* nlop = nlop_stack_create(N, odims, adims, bdims, stack_dim);

	nlop = nlop_reshape_in_F(nlop, 0, iova->N, iova->dims);
	nlop = nlop_reshape_in_F(nlop, 1, iovb->N, iovb->dims);

   arg_t arg =  snlop_append_nlop_generic_F(2, (arg_t[2]){ a, b }, nlop, true);

	if (NULL != ret) {

		snlop_replace_oarg(ret, arg);
		return ret;
	}

	return arg;
}

arg_t snlop_stack_F(arg_t a, arg_t b, int stack_dim)
{
	const struct iovec_s* iova = arg_get_iov(a);
	const struct iovec_s* iovb = arg_get_iov(b);

	arg_t ret = NULL;
	if (NULL != a->stack)
		ret = a->stack(a, b, stack_dim, false);

	if ((NULL == ret) && (NULL != b->stack))
		ret = b->stack(a, b, stack_dim, false);

	assert(0 <= stack_dim);

	int N = MAX(iova->N, iovb->N);
	N = MAX(N, stack_dim + 1);

	long adims[N];
	long bdims[N];
	long odims[N];

	md_singleton_dims(N, adims);
	md_singleton_dims(N, bdims);

	md_copy_dims(iova->N, adims, iova->dims);
	md_copy_dims(iovb->N, bdims, iovb->dims);

	for (int i = 0; i < N; i++) {

		odims[i] = adims[i];

		if (i == stack_dim)
			odims[i] += bdims[i];
		else
			assert(adims[i] == bdims[i]);
	}

	const struct nlop_s* nlop = nlop_stack_create(N, odims, adims, bdims, stack_dim);

	nlop = nlop_reshape_in_F(nlop, 0, iova->N, iova->dims);
	nlop = nlop_reshape_in_F(nlop, 1, iovb->N, iovb->dims);

	arg_t arg =  snlop_append_nlop_generic_F(2, (arg_t[2]){ a, b }, nlop, false);

	if (NULL != ret) {

		snlop_replace_oarg(ret, arg);
		return ret;
	}

	return arg;
}

arg_t snlop_stack_in(arg_t a, arg_t b, int stack_dim)
{
	const struct iovec_s* iova = arg_get_iov_in(a);
	const struct iovec_s* iovb = arg_get_iov_in(b);

	arg_t ret = NULL;
	if (NULL != a->stack)
		ret = a->stack(a, b, stack_dim, false);

	if ((NULL == ret) && (NULL != b->stack))
		ret = b->stack(a, b, stack_dim, false);

	assert(0 <= stack_dim);

	int N = MAX(iova->N, iovb->N);
	N = MAX(N, stack_dim + 1);

	long adims[N];
	long bdims[N];
	long odims[N];

	md_singleton_dims(N, adims);
	md_singleton_dims(N, bdims);

	md_copy_dims(iova->N, adims, iova->dims);
	md_copy_dims(iovb->N, bdims, iovb->dims);

	for (int i = 0; i < N; i++) {

		odims[i] = adims[i];

		if (i == stack_dim)
			odims[i] += bdims[i];
		else
			assert(adims[i] == bdims[i]);
	}

	const struct nlop_s* nlop = nlop_destack_create(N, adims, bdims, odims, stack_dim);

	nlop = nlop_reshape_out_F(nlop, 0, iova->N, iova->dims);
	nlop = nlop_reshape_out_F(nlop, 1, iovb->N, iovb->dims);

	arg_t arg = snlop_prepend_nlop_generic_F(2, (arg_t[2]){ a, b }, nlop);

	if (NULL != ret) {

		snlop_replace_iarg(ret, arg);
		return ret;
	}

	return arg;
}

arg_t snlop_dup(arg_t a, arg_t b)
{
	if (snlop_get_idx(a, false) > snlop_get_idx(b, false))
		SWAP(a, b);

	snlop_t x = a->x;

	int ia = snlop_get_idx(a, false);
	int ib = snlop_get_idx(b, false);

	x->x = nlop_dup_F(x->x, ia, ib);

	arg_t ret = NULL;

	if (NULL != a->stack)
		ret = a->dup(a, b);

	if ((NULL == ret) && (NULL != b->dup))
		ret = b->dup(a, b);

	if (NULL == ret)
		ret = arg_create(x);

	arg_unref(list_remove_item(x->iargs, ib));
	snlop_replace_iarg(ret, a);

	return ret;
}


void snlop_debug(int dl, struct snlop_s* x)
{
	int II = nlop_get_nr_in_args(x->x);

	debug_printf(dl, "sNLOP\ninputs: %d\n", II);

	for (int i = 0; i < II; i++) {

		arg_t arg = snlop_get_iarg(x, i);
		debug_printf(dl, "%s: ", arg->name ?: "ARG");

		auto io = nlop_generic_domain(x->x, i);
		debug_print_dims(dl, io->N, io->dims);
	}

	int OO = nlop_get_nr_out_args(x->x);

	debug_printf(dl, "outputs: %d\n", OO);

	for (int o = 0; o < OO; o++) {

		arg_t arg = snlop_get_oarg(x, o);
		debug_printf(dl, "%s: ", arg->name ?: "ARG");

		auto io = nlop_generic_codomain(x->x, o);
		debug_print_dims(dl, io->N, io->dims);
	}

	int TN = list_count(x->targs);
	debug_printf(dl, "attached args: %d\n", TN);

	for (int i = 0; i < TN; i++) {

		arg_t arg = list_get_item(x->targs, i);
		debug_printf(dl, "%s: ", arg->name ?: "ARG");

		auto io = arg_get_iov_in(arg);
		debug_print_dims(dl, io->N, io->dims);
	}
}

list_t snlop_get_oargs(int N, arg_t args[N])
{
	list_t ret = list_create();

	for (int i = 0; i < N; i++) {

		snlop_t snlop = args[i]->x;

		for (int j = 0; j < list_count(snlop->oargs); j++)
			list_append(ret, list_get_item(snlop->oargs, j));
	}

	return ret;
}

void snlop_prune_oargs(arg_t oarg, list_t keep_args)
{
	snlop_t snlop = oarg->x;
	list_append(keep_args, oarg);

	for (int i = 0; i < list_count(snlop->oargs); i++) {

		arg_t arg = list_get_item(snlop->oargs, i);

		if (-1 == list_get_first_index(keep_args, arg, NULL)) {

			snlop_del_arg(arg);
			i--;
		}
	}

	list_free(keep_args);
}


void snlop_export_graph(snlop_t snlop, const char* path)
{
	nlop_export_graph(path, snlop->x);
}