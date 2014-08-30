/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2014 Frank Ong <frankong@berkeley.edu>
 *
 */

#include <complex.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "ops.h"

struct operator_s {

	const struct iovec_s* domain;
	const struct iovec_s* codomain;

	void* data;
	int refcount;

	void (*apply)(const void* data, _Complex float* dst, const _Complex float* src);
	void (*del)(const void* data);
};


/**
 * operator with 1 parameter
 */
struct operator_p_s {

	const struct iovec_s* domain;
	const struct iovec_s* codomain;

	void* data;

	int refcount;

	void (*apply)(const void* data, float mu, _Complex float* dst, const _Complex float* src);
	void (*del)(const void* data);
};

/**
 * Create an operator (with strides)
 */
const struct operator_s* operator_create2(unsigned int N, const long out_dims[N], const long out_strs[N],
			const long in_dims[N], const long in_strs[N],
			void* data, operator_fun_t apply, operator_del_t del)
{
	struct operator_s* op = xmalloc(sizeof(struct operator_s));

	op->domain = iovec_create2(N, in_dims, in_strs);
	op->codomain = iovec_create2(N, out_dims, out_strs);

	op->data = data;
	op->apply = apply;

	op->refcount = 1;
	op->del = del;

	return op;
}

/**
 * Create an operator (without strides)
 *
 * @param N number of dimensions
 * @param out_dims dimensions of output
 * @param in_dims dimensions of input
 * @param data data for applying the operation
 * @param apply function that applies the operation
 * @param del function that frees the data
 */
const struct operator_s* operator_create(unsigned int N, const long out_dims[N], const long in_dims[N],
		void* data, operator_fun_t apply, operator_del_t del)
{
	long out_strs[N];
	long in_strs[N];
	md_calc_strides(N, out_strs, out_dims, CFL_SIZE);
	md_calc_strides(N, in_strs, in_dims, CFL_SIZE);

	return operator_create2(N, out_dims, out_strs, in_dims, in_strs, data, apply, del);
}


/**
 * Increment the reference count of an operator
 * 
 * @param x operator
 */
const struct operator_s* operator_ref(const struct operator_s* x)
{
	if (NULL != x)
		((struct operator_s*)x)->refcount++;

	return x;
}

/**
 * Return the data of the associated operator
 *
 * @param x operator
 */
void* operator_get_data(const struct operator_s* x)
{
	return x->data;
}

/**
 * Return the data of the associated operator_p
 *
 * @param x operator_p
 */
void* operator_p_get_data(const struct operator_p_s* x)
{
	return x->data;
}

/**
 * Free the operator struct
 * Note: also frees the data if the operator's reference count is zero
 *
 * @param x operator
 */
void operator_free(const struct operator_s* x)
{
	if (NULL == x) 
		return;

	if (1 > --(((struct operator_s*)x)->refcount)) {

		if (NULL != x->del)
			x->del(x->data);

		iovec_free(x->domain);
		iovec_free(x->codomain);
		free((void*)x);
	}
}


/**
 * Free the operator_p struct
 * Note: also frees the data if the operator's reference count is zero
 *
 * @param x operator_p
 */
void operator_p_free(const struct operator_p_s* x)
{
	if (NULL == x)
		return;

	if (1 > --(((struct operator_p_s*)x)->refcount)) {

		if (NULL != x->del)
			x->del(x->data);

		iovec_free(x->domain);
		iovec_free(x->codomain);
		free((void*)x);
	}
}

/**
 * Increment the reference count of an operator_p
 * 
 * @param x operator_p
 */
const struct operator_p_s* operator_p_ref(const struct operator_p_s* x)
{
	if (NULL != x)
		((struct operator_p_s*)x)->refcount++;

	return x;
}


/**
 * Return the dimensions and strides of the domain of an operator
 *
 * @param op operator
 */
const struct iovec_s* operator_domain(const struct operator_s* op)
{
	return op->domain;
}


/**
 * Return the dimensions and strides of the codomain of an operator
 *
 * @param op operator
 */
const struct iovec_s* operator_codomain(const struct operator_s* op)
{
	return op->codomain;
}

/**
 * Return the dimensions and strides of the domain of an operator_p
 *
 * @param op operator_p
 */
const struct iovec_s* operator_p_domain(const struct operator_p_s* op)
{
	return op->domain;
}


/**
 * Return the dimensions and strides of the codomain of an operator_p
 *
 * @param op operator_p
 */
const struct iovec_s* operator_p_codomain(const struct operator_p_s* op)
{
	return op->codomain;
}


/**
 * Create an operator with one parameter (without strides)
 */
const struct operator_p_s* operator_p_create2(unsigned int N, const long out_dims[N], const long out_strs[N], const long in_dims[N], const long in_strs[N], void* data, operator_p_fun_t apply, operator_del_t del)
{
	struct operator_p_s* o = xmalloc(sizeof(struct operator_p_s));

	o->domain = iovec_create2(N, in_dims, in_strs);
	o->codomain = iovec_create2(N, out_dims, out_strs);

	o->data = data;
	o->apply = apply;

	o->refcount = 1;
	o->del = del;

	if (NULL == del)
		debug_printf(DP_WARN, "Warning: no delete function specified for operator_p_create! Possible memory leak.\n");

	return o;
}

/**
 * Create an operator with one parameter (without strides)
 *
 * @param N number of dimensions
 * @param out_dims dimensions of output
 * @param in_dims dimensions of input
 * @param data data for applying the operation
 * @param apply function that applies the operation
 * @param del function that frees the data
 */
const struct operator_p_s* operator_p_create(unsigned int N, const long out_dims[N], const long in_dims[N], void* data, operator_p_fun_t apply, operator_del_t del)
{
	long out_strs[N];
	long in_strs[N];
	md_calc_strides(N, out_strs, out_dims, CFL_SIZE);
	md_calc_strides(N, in_strs, in_dims, CFL_SIZE);

	return operator_p_create2(N, out_dims, out_strs, in_dims, in_strs, data, apply, del);
}


struct operator_chain_s {

	const struct operator_s* a;
	const struct operator_s* b;
};


static void chain_apply(const void* _data, complex float* dst, const complex float* src)
{
	const struct operator_chain_s* data = _data;

	const struct iovec_s* iovec = data->a->codomain;
	complex float* tmp = md_alloc_sameplace(iovec->N, iovec->dims, CFL_SIZE, dst);

	operator_apply_unchecked(data->a, tmp, src);
	operator_apply_unchecked(data->b, dst, tmp);

	md_free(tmp);
}

/*
 * Free data associated with chained operator
 */
static void chain_free(const void* _data)
{
	const struct operator_chain_s* data = _data;

	operator_free(data->a);
	operator_free(data->b);

	free((void*)data);
}



/**
 * Create a new operator that first applies a, then applies b:
 * c(x) = b(a(x))
 */
const struct operator_s* operator_chain(const struct operator_s* a, const struct operator_s* b)
{
	struct operator_chain_s* c = xmalloc(sizeof(struct operator_chain_s));
	
	// check compatibility
	assert(a->codomain->N == b->domain->N);
	assert(md_calc_size(a->codomain->N, a->codomain->dims) == md_calc_size(b->domain->N, b->domain->dims));

	// check whether intermediate storage can be simple

	assert(a->codomain->N == md_calc_blockdim(a->codomain->N, a->codomain->dims, a->codomain->strs, CFL_SIZE));
	assert(b->domain->N == md_calc_blockdim(b->domain->N, b->domain->dims, b->domain->strs, CFL_SIZE));

	c->a = operator_ref(a);
	c->b = operator_ref(b);

	const struct iovec_s* dom = a->domain;
	const struct iovec_s* cod = b->codomain;
	return operator_create2(dom->N, cod->dims, cod->strs, dom->dims, dom->strs, c, chain_apply, chain_free);
}



extern void operator_apply_unchecked(const struct operator_s* op, complex float* dst, const complex float* src)
{
	op->apply((void*)op->data, dst, src);
}

extern void operator_apply2(const struct operator_s* op, unsigned int IN, const long idims[IN], const long istrs[IN], complex float* dst, const long ON, const long odims[ON], const long ostrs[ON], const complex float* src)
{
	assert(iovec_check(op->domain, IN, idims, istrs));
	assert(iovec_check(op->codomain, ON, odims, ostrs));

	operator_apply_unchecked((void*)op->data, dst, src);
}


extern void operator_apply(const struct operator_s* op, unsigned int IN, const long idims[IN], complex float* dst, const long ON, const long odims[ON], const complex float* src)
{
	long ostrs[ON];
	long istrs[IN];
	md_calc_strides(ON, ostrs, odims, CFL_SIZE);
	md_calc_strides(IN, istrs, idims, CFL_SIZE);
	operator_apply2(op, ON, odims, ostrs, dst, IN, idims, istrs, src);
}


extern void operator_p_apply2(const struct operator_p_s* op, float mu, unsigned int IN, const long idims[IN], const long istrs[IN], complex float* dst, const long ON, const long odims[ON], const long ostrs[ON], const complex float* src)
{
	assert(iovec_check(op->domain, IN, idims, istrs));
	assert(iovec_check(op->codomain, ON, odims, ostrs));

	op->apply((void*)op->data, mu, dst, src);
}


extern void operator_p_apply(const struct operator_p_s* op, float mu, unsigned int IN, const long idims[IN], complex float* dst, const long ON, const long odims[ON], const complex float* src)
{
	long ostrs[ON];
	long istrs[IN];
	md_calc_strides(ON, ostrs, odims, CFL_SIZE);
	md_calc_strides(IN, istrs, idims, CFL_SIZE);
	operator_p_apply2(op, mu, ON, odims, ostrs, dst, IN, idims, istrs, src);
}


extern void operator_p_apply_unchecked(const struct operator_p_s* op, float mu, complex float* dst, const complex float* src)
{
	op->apply((void*)op->data, mu, dst, src);
}


extern void operator_iter(void* _o, float* _dst, const float* _src )
{
	struct operator_s* o = _o;
	complex float* dst = (complex float*) _dst;
	const complex float* src = (complex float*) _src;

	operator_apply_unchecked(o, dst, src);
}

extern void operator_p_iter( void* _o, float lambda, float* _dst, const float* _src )
{
	struct operator_p_s* o = _o;
	complex float* dst = (complex float*)_dst;
	const complex float* src = (complex float*)_src;

	operator_p_apply_unchecked(o, lambda, dst, src);
}

