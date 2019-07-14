/* Copyright 2014. The Regents of the University of California.
 * Copyright 2016-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014 Frank Ong <frankong@berkeley.edu>
 */

#include <complex.h>
#include <stdbool.h>
#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/ops_p.h"
#include "num/ops.h"

#include "misc/misc.h"
#include "misc/shrdptr.h"
#include "misc/debug.h"
#include "misc/shrdptr.h"

#include "linop.h"





struct shared_data_s {

	INTERFACE(operator_data_t);

	linop_data_t* data;
	del_fun_t del;

	struct shared_ptr_s sptr;

	union {

		lop_fun_t apply;
		lop_p_fun_t apply_p;
	} u;
};

static DEF_TYPEID(shared_data_s);



static void shared_del(const operator_data_t* _data)
{
	auto data = CAST_DOWN(shared_data_s, _data);

	shared_ptr_destroy(&data->sptr);
	
	xfree(data);
}

static void shared_apply(const operator_data_t* _data, unsigned int N, void* args[N])
{
	auto data = CAST_DOWN(shared_data_s, _data);

	assert(2 == N);
	debug_trace("ENTER %p\n", data->u.apply);
	data->u.apply(data->data, args[0], args[1]);
	debug_trace("LEAVE %p\n", data->u.apply);
}

static void shared_apply_p(const operator_data_t* _data, float lambda, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(shared_data_s, _data);

	debug_trace("ENTER %p\n", data->u.apply_p);
	data->u.apply_p(data->data, lambda, dst, src);
	debug_trace("LEAVE %p\n", data->u.apply_p);
}


static void sptr_del(const struct shared_ptr_s* p)
{
	auto data = CONTAINER_OF(p, struct shared_data_s, sptr);

	data->del(data->data);
}


/**
 * Create a linear operator (with strides)
 */
struct linop_s* linop_create2(unsigned int ON, const long odims[ON], const long ostrs[ON],
				unsigned int IN, const long idims[IN], const long istrs[IN],
				linop_data_t* data, lop_fun_t forward, lop_fun_t adjoint, lop_fun_t normal,
				lop_p_fun_t norm_inv, del_fun_t del)
{
	PTR_ALLOC(struct linop_s, lo);

	struct shared_data_s* shared_data[4];

	for (unsigned int i = 0; i < 4; i++) {

		shared_data[i] = TYPE_ALLOC(struct shared_data_s);
		SET_TYPEID(shared_data_s, shared_data[i]);
	}

	for (unsigned int i = 0; i < 4; i++) {

		shared_data[i]->data = data;
		shared_data[i]->del = del;

		if (0 == i)
			shared_ptr_init(&shared_data[i]->sptr, sptr_del);
		else
			shared_ptr_copy(&shared_data[i]->sptr, &shared_data[0]->sptr);
	}

	shared_data[0]->u.apply = forward;
	shared_data[1]->u.apply = adjoint;
	shared_data[2]->u.apply = normal;
	shared_data[3]->u.apply_p = norm_inv;

	assert((NULL != forward));
	assert((NULL != adjoint));

	lo->forward = operator_create2(ON, odims, ostrs, IN, idims, istrs, CAST_UP(shared_data[0]), shared_apply, shared_del);
	lo->adjoint = operator_create2(IN, idims, istrs, ON, odims, ostrs, CAST_UP(shared_data[1]), shared_apply, shared_del);

	if (NULL != normal) {

		lo->normal = operator_create2(IN, idims, istrs, IN, idims, istrs, CAST_UP(shared_data[2]), shared_apply, shared_del);

	} else {

		shared_ptr_destroy(&shared_data[2]->sptr);
		xfree(shared_data[2]);
#if 0
		lo->normal = NULL;
#else
		lo->normal = operator_chain(lo->forward, lo->adjoint);
#endif
	}

	if (NULL != norm_inv) {

		lo->norm_inv = operator_p_create2(IN, idims, istrs, IN, idims, istrs, CAST_UP(shared_data[3]), shared_apply_p, shared_del);
	
	} else {

		shared_ptr_destroy(&shared_data[3]->sptr);
		xfree(shared_data[3]);
		lo->norm_inv = NULL;
	}

	return PTR_PASS(lo);
}


/**
 * Create a linear operator (without strides)
 *
 * @param N number of dimensions
 * @param odims dimensions of output (codomain)
 * @param idims dimensions of input (domain)
 * @param data data for applying the operator
 * @param forward function for applying the forward operation, A
 * @param adjoint function for applying the adjoint operation, A^H
 * @param normal function for applying the normal equations operation, A^H A
 * @param norm_inv function for applying the pseudo-inverse operation, (A^H A + mu I)^-1
 * @param del function for freeing the data
 */
struct linop_s* linop_create(unsigned int ON, const long odims[ON], unsigned int IN, const long idims[IN], linop_data_t* data,
				lop_fun_t forward, lop_fun_t adjoint, lop_fun_t normal, lop_p_fun_t norm_inv, del_fun_t del)
{
	long ostrs[ON];
	long istrs[IN];
	md_calc_strides(ON, ostrs, odims, CFL_SIZE);
	md_calc_strides(IN, istrs, idims, CFL_SIZE);

	return linop_create2(ON, odims, ostrs, IN, idims, istrs, data, forward, adjoint, normal, norm_inv, del);
}

/**
 * Return the data associated with the linear operator
 * 
 * @param ptr linear operator
 */
const linop_data_t* linop_get_data(const struct linop_s* ptr)
{
	auto sdata = CAST_MAYBE(shared_data_s, operator_get_data(ptr->forward));
	return sdata == NULL ? NULL : sdata->data;
}


/**
 * Make a copy of a linear operator
 * @param x linear operator
 */
extern const struct linop_s* linop_clone(const struct linop_s* x)
{
	PTR_ALLOC(struct linop_s, lo);

	lo->forward = operator_ref(x->forward);
	lo->adjoint = operator_ref(x->adjoint);
	lo->normal = operator_ref(x->normal);
	lo->norm_inv = operator_p_ref(x->norm_inv);

	return PTR_PASS(lo);
}


/**
 * Apply the forward operation of a linear operator: y = A x
 * Checks that dimensions are consistent for the linear operator
 *
 * @param op linear operator
 * @param DN number of destination dimensions
 * @param ddims dimensions of the output (codomain)
 * @param dst output data
 * @param SN number of source dimensions
 * @param sdims dimensions of the input (domain)
 * @param src input data
 */
void linop_forward(const struct linop_s* op, unsigned int DN, const long ddims[DN], complex float* dst, 
			unsigned int SN, const long sdims[SN], const complex float* src)
{
	assert(op->forward);
	operator_apply(op->forward, DN, ddims, dst, SN, sdims, src);
}


/**
 * Apply the adjoint operation of a linear operator: y = A^H x
 * Checks that dimensions are consistent for the linear operator
 *
 * @param op linear operator
 * @param DN number of destination dimensions
 * @param ddims dimensions of the output (domain)
 * @param dst output data
 * @param SN number of source dimensions
 * @param sdims dimensions of the input (codomain)
 * @param src input data
 */
void linop_adjoint(const struct linop_s* op, unsigned int DN, const long ddims[DN], complex float* dst,
			unsigned int SN, const long sdims[SN], const complex float* src)
{
	assert(op->adjoint);
	operator_apply(op->adjoint, DN, ddims, dst, SN, sdims, src);
}


/**
 * Apply the pseudo-inverse operation of a linear operator: x = (A^H A + lambda I)^-1 A^H y
 * Checks that dimensions are consistent for the linear operator
 *
 * @param op linear operator
 * @param lambda regularization parameter
 * @param DN number of destination dimensions
 * @param ddims dimensions of the output (domain)
 * @param dst output data
 * @param SN number of source dimensions
 * @param sdims dimensions of the input (codomain)
 * @param src input data
 */
void linop_pseudo_inv(const struct linop_s* op, float lambda,
			unsigned int DN, const long ddims[DN], complex float* dst,
			unsigned int SN, const long sdims[SN], const complex float* src)
{
	complex float* adj = md_alloc_sameplace(DN, ddims, CFL_SIZE, dst);
	linop_adjoint(op, DN, ddims, adj, SN, sdims, src);

	assert(op->norm_inv);
	operator_p_apply(op->norm_inv, lambda, DN, ddims, dst, DN, ddims, adj);
	md_free(adj);
}


/**
 * Apply the normal equations operation of a linear operator: y = A^H A x
 * Checks that dimensions are consistent for the linear operator
 *
 * @param op linear operator
 * @param N number of dimensions
 * @param dims dimensions
 * @param dst output data
 * @param src input data
 */
void linop_normal(const struct linop_s* op, unsigned int N, const long dims[N], complex float* dst, const complex float* src)
{
	assert(op->normal);
	operator_apply(op->normal, N, dims, dst, N, dims, src);
}


/**
 * Apply the forward operation of a linear operator: y = A x
 * Does not check that the dimensions are consistent for the linear operator
 *
 * @param op linear operator
 * @param dst output data
 * @param src input data
 */
void linop_forward_unchecked(const struct linop_s* op, complex float* dst, const complex float* src)
{
	assert(op->forward);
	operator_apply_unchecked(op->forward, dst, src);
}


/**
 * Apply the adjoint operation of a linear operator: y = A^H x
 * Does not check that the dimensions are consistent for the linear operator
 *
 * @param op linear operator
 * @param dst output data
 * @param src input data
 */
void linop_adjoint_unchecked(const struct linop_s* op, complex float* dst, const complex float* src)
{
	assert(op->adjoint);
	operator_apply_unchecked(op->adjoint, dst, src);
}


/**
 * Apply the normal equations operation of a linear operator: y = A^H A x
 * Does not check that the dimensions are consistent for the linear operator
 *
 * @param op linear operator
 * @param dst output data
 * @param src input data
 */
void linop_normal_unchecked(const struct linop_s* op, complex float* dst, const complex float* src)
{
	assert(op->normal);
	operator_apply_unchecked(op->normal, dst, src);
}


/**
 * Apply the pseudo-inverse operation of a linear operator: y = (A^H A + lambda I)^-1 x
 * Does not check that the dimensions are consistent for the linear operator
 *
 * @param op linear operator
 * @param lambda regularization parameter
 * @param dst output data
 * @param src input data
 */
void linop_norm_inv_unchecked(const struct linop_s* op, float lambda, complex float* dst, const complex float* src)
{
	operator_p_apply_unchecked(op->norm_inv, lambda, dst, src);
}


/**
 * Return the dimensions and strides of the domain of a linear operator
 *
 * @param op linear operator
 */
const struct iovec_s* linop_domain(const struct linop_s* op)
{
	return operator_domain(op->forward);
}


/**
 * Return the dimensions and strides of the codomain of a linear operator
 *
 * @param op linear operator
 */
const struct iovec_s* linop_codomain(const struct linop_s* op)
{
	return operator_codomain(op->forward);
}





struct linop_s* linop_null_create2(unsigned int N, const long odims[N], const long ostrs[N], const long idims[N], const long istrs[N])
{
	PTR_ALLOC(struct linop_s, c);

	const struct operator_s* nudo = operator_null_create2(N, idims, istrs);
	const struct operator_s* zedo = operator_zero_create2(N, idims, istrs);
	const struct operator_s* nuco = operator_null_create2(N, odims, ostrs);
	const struct operator_s* zeco = operator_zero_create2(N, odims, ostrs);

	c->forward = operator_combi_create(2, MAKE_ARRAY(zeco, nudo));
	c->adjoint = operator_combi_create(2, MAKE_ARRAY(zedo, nuco));
	c->normal = operator_combi_create(2, MAKE_ARRAY(zedo, nudo));
	c->norm_inv = NULL;

	operator_free(nudo);
	operator_free(zedo);
	operator_free(nuco);
	operator_free(zeco);

	return PTR_PASS(c);
}



struct linop_s* linop_null_create(unsigned int N, const long odims[N], const long idims[N])
{
	return linop_null_create2(N, odims, MD_STRIDES(N, odims, CFL_SIZE),
					idims, MD_STRIDES(N, idims, CFL_SIZE));
}


/**
 * Create chain of linear operators.
 * C = B A 
 * C^H = A^H B^H
 * C^H C = A^H B^H B A
 */
struct linop_s* linop_chain(const struct linop_s* a, const struct linop_s* b)
{
	PTR_ALLOC(struct linop_s, c);

	c->forward = operator_chain(a->forward, b->forward);
	c->adjoint = operator_chain(b->adjoint, a->adjoint);

	if (NULL == b->normal) {

		c->normal = operator_chain(c->forward, c->adjoint);

	} else {

		const struct operator_s* top = operator_chain(b->normal, a->adjoint);
		c->normal = operator_chain(a->forward, top);
		operator_free(top);
	}

	c->norm_inv = NULL;

	return PTR_PASS(c);
}


struct linop_s* linop_chain_FF(const struct linop_s* a, const struct linop_s* b)
{
	struct linop_s* x = linop_chain(a, b);

	linop_free(a);
	linop_free(b);

	return x;
}


struct linop_s* linop_chainN(unsigned int N, struct linop_s* a[N])
{
	assert(N > 0);

	if (1 == N)
		return a[0];

	return linop_chain(a[0], linop_chainN(N - 1, a + 1));	// FIXME: free intermed.
}





struct linop_s* linop_stack(int D, int E, const struct linop_s* a, const struct linop_s* b)
{
	PTR_ALLOC(struct linop_s, c);

	c->forward = operator_stack(D, E, a->forward, b->forward);
	c->adjoint = operator_stack(E, D, b->adjoint, a->adjoint);

	const struct operator_s* an = a->normal;

	if (NULL == an)
		an = operator_chain(a->forward, a->adjoint);

	const struct operator_s* bn = b->normal;

	if (NULL == bn)
		bn = operator_chain(b->forward, b->adjoint);

	c->normal = operator_stack(D, D, an, bn);

	c->norm_inv = NULL;

	return PTR_PASS(c);
}







struct linop_s* linop_loop(unsigned int D, const long dims[D], struct linop_s* op)
{
	PTR_ALLOC(struct linop_s, op2);
	op2->forward = operator_loop(D, dims, op->forward);
	op2->adjoint = operator_loop(D, dims, op->adjoint);
	op2->normal = (NULL == op->normal) ? NULL : operator_loop(D, dims, op->normal);
	op2->norm_inv = NULL; // FIXME
	return op2;
}


/**
 * Free the linear operator and associated data,
 * Note: only frees the data if its reference count is zero
 *
 * @param op linear operator
 */
void linop_free(const struct linop_s* op)
{
	if (NULL == op)
		return;
	operator_free(op->forward);
	operator_free(op->adjoint);
	operator_free(op->normal);
	operator_p_free(op->norm_inv);
	xfree(op);
}




struct plus_data_s {

	INTERFACE(linop_data_t);

	const struct linop_s* a;
	const struct linop_s* b;
};

static DEF_TYPEID(plus_data_s);

static void plus_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(plus_data_s, _data);
	auto iov = linop_codomain(data->a);

	complex float* tmp = md_alloc_sameplace(iov->N, iov->dims, iov->size, dst);

	linop_forward_unchecked(data->a, dst, src);
	linop_forward_unchecked(data->b, tmp, src);

	md_zadd(iov->N, iov->dims, dst, dst, tmp);
	md_free(tmp);
}

static void plus_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(plus_data_s, _data);
	auto iov = linop_domain(data->a);

	complex float* tmp = md_alloc_sameplace(iov->N, iov->dims, iov->size, dst);

	linop_adjoint_unchecked(data->a, dst, src);
	linop_adjoint_unchecked(data->b, tmp, src);

	md_zadd(iov->N, iov->dims, dst, dst, tmp);
	md_free(tmp);
}


static void plus_free(const linop_data_t* _data)
{
	auto data = CAST_DOWN(plus_data_s, _data);

	linop_free(data->a);
	linop_free(data->b);

	xfree(data);
}

struct linop_s* linop_plus(const struct linop_s* a, const struct linop_s* b)
{
#if 1
	// detect null operations and just clone

	if (operator_zero_or_null_p(a->forward))
		return (struct linop_s*)linop_clone(b);

	if (operator_zero_or_null_p(b->forward))
		return (struct linop_s*)linop_clone(a);
#endif

	auto bdo = linop_domain(b);
	assert(CFL_SIZE == bdo->size);
	iovec_check(linop_domain(a), bdo->N, bdo->dims, bdo->strs);

	auto bco = linop_codomain(b);
	assert(CFL_SIZE == bco->size);
	iovec_check(linop_codomain(a), bco->N, bco->dims, bco->strs);

	PTR_ALLOC(struct plus_data_s, data);
	SET_TYPEID(plus_data_s, data);

	data->a = linop_clone(a);
	data->b = linop_clone(b);

	return linop_create(bco->N, bco->dims, bdo->N, bdo->dims, CAST_UP(PTR_PASS(data)), plus_apply, plus_adjoint, NULL, NULL, plus_free);
}


