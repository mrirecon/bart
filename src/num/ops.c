/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013-2014 Martin Uecker <uecker@eecs.berkeley.edu>
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

	unsigned int N;
	const struct iovec_s** domain;

	void* data;
	int refcount;

	void (*apply)(const void* data, unsigned int N, void* args[N]);
	void (*del)(const void* data);
};



/**
 * Create an operator (with strides)
 */
const struct operator_s* operator_create2(unsigned int ON, const long out_dims[ON], const long out_strs[ON],
			unsigned int IN, const long in_dims[IN], const long in_strs[IN],
			void* data, operator_fun_t apply, operator_del_t del)
{
	struct operator_s* op = xmalloc(sizeof(struct operator_s));

	const struct iovec_s** dom = xmalloc(2 * sizeof(struct iovec_s*));

	dom[1] = iovec_create2(IN, in_dims, in_strs, CFL_SIZE);
	dom[0] = iovec_create2(ON, out_dims, out_strs, CFL_SIZE);

	op->N = 2;
	op->domain = dom;
	op->data = data;
	op->apply = apply;

	op->refcount = 1;
	op->del = del;

	return op;
}

/**
 * Create an operator (without strides)
 *
 * @param ON number of output dimensions
 * @param out_dims dimensions of output
 * @param IN number of input dimensions
 * @param in_dims dimensions of input
 * @param data data for applying the operation
 * @param apply function that applies the operation
 * @param del function that frees the data
 */
const struct operator_s* operator_create(unsigned int ON, const long out_dims[ON], 
		unsigned int IN, const long in_dims[IN],
		void* data, operator_fun_t apply, operator_del_t del)
{
	long out_strs[ON];
	long in_strs[IN];
	md_calc_strides(ON, out_strs, out_dims, CFL_SIZE);
	md_calc_strides(IN, in_strs, in_dims, CFL_SIZE);

	return operator_create2(ON, out_dims, out_strs, IN, in_dims, in_strs, data, apply, del);
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

		for (unsigned int i = 0; i < x->N; i++)
			iovec_free(x->domain[i]);

		free(x->domain);
		free((void*)x);
	}
}







/**
 * Return the dimensions and strides of the domain of an operator
 *
 * @param op operator
 */
const struct iovec_s* operator_domain(const struct operator_s* op)
{
	assert(2 == op->N);
	return op->domain[1];
}


/**
 * Return the dimensions and strides of the codomain of an operator
 *
 * @param op operator
 */
const struct iovec_s* operator_codomain(const struct operator_s* op)
{
	assert(2 == op->N);
	return op->domain[0];
}




struct operator_p_s {

	struct operator_s op;
};

const struct operator_p_s* operator_p_ref(const struct operator_p_s* x)
{
	operator_ref(&x->op);
	return x;
}

/**
 * Return the dimensions and strides of the domain of an operator_p
 *
 * @param op operator_p
 */
const struct iovec_s* operator_p_domain(const struct operator_p_s* op)
{
	assert(3 == op->op.N);
	return op->op.domain[2];
}


/**
 * Return the dimensions and strides of the codomain of an operator_p
 *
 * @param op operator_p
 */
const struct iovec_s* operator_p_codomain(const struct operator_p_s* op)
{
	assert(3 == op->op.N);
	return op->op.domain[1];
}



void operator_p_free(const struct operator_p_s* x)
{
	operator_free(&x->op);
}


struct op_p_data_s {

	void* data;
	operator_p_fun_t apply;
	operator_del_t del;
};

static void op_p_apply(const void* _data, unsigned int N, void* args[N])
{
	const struct op_p_data_s* data = _data;
	assert(3 == N);
	data->apply(data->data, *((float*)args[0]), args[1], args[2]);
}

static void op_p_del(const void* _data)
{
	const struct op_p_data_s* data = _data;
	data->del(data->data);
	free((void*)data);
}

void* operator_p_get_data(const struct operator_p_s* x)
{
	struct op_p_data_s* data = operator_get_data(&x->op);
	return data->data;
}

/**
 * Create an operator with one parameter (without strides)
 */
const struct operator_p_s* operator_p_create2(unsigned int ON, const long out_dims[ON], const long out_strs[ON], 
		unsigned int IN, const long in_dims[IN], const long in_strs[IN],
		void* data, operator_p_fun_t apply, operator_del_t del)
{
	struct operator_p_s* o = xmalloc(sizeof(struct operator_p_s));
	struct op_p_data_s* op = xmalloc(sizeof(struct op_p_data_s));

	op->data = data;
	op->apply = apply;
	op->del = del;

	const struct iovec_s** dom = xmalloc(3 * sizeof(struct iovec_s*));

	dom[0] = iovec_create2(1, MD_DIMS(1), MD_DIMS(0), FL_SIZE);
	dom[1] = iovec_create2(ON, out_dims, out_strs, CFL_SIZE);
	dom[2] = iovec_create2(IN, in_dims, in_strs, CFL_SIZE);

	o->op.N = 3;
	o->op.domain = dom;
	o->op.data = op;
	o->op.apply = op_p_apply;

	o->op.refcount = 1;
	o->op.del = op_p_del;

	if (NULL == del)
		debug_printf(DP_WARN, "Warning: no delete function specified for operator_p_create! Possible memory leak.\n");

	return o;
}


/**
 * Create an operator with one parameter (without strides)
 *
 * @param ON number of output dimensions
 * @param out_dims dimensions of output
 * @param IN number of input dimensions
 * @param in_dims dimensions of input
 * @param data data for applying the operation
 * @param apply function that applies the operation
 * @param del function that frees the data
 */
const struct operator_p_s* operator_p_create(unsigned int ON, const long out_dims[ON], 
		unsigned int IN, const long in_dims[IN], 
		void* data, operator_p_fun_t apply, operator_del_t del)
{
	long out_strs[ON];
	long in_strs[IN];
	md_calc_strides(ON, out_strs, out_dims, CFL_SIZE);
	md_calc_strides(IN, in_strs, in_dims, CFL_SIZE);

	return operator_p_create2(ON, out_dims, out_strs, IN, in_dims, in_strs, data, apply, del);
}



static void identity_apply(const void* _data, unsigned int N, void* args[N])
{
        const struct iovec_s* domain = _data;
	assert(2 == N);
        md_copy2(domain->N, domain->dims, domain->strs, args[0], domain->strs, args[1], domain->size);
}


static void identity_free(const void* data)
{
        iovec_free((const struct iovec_s*)data);
}


/**
 * Create an Identity operator: I x
 * @param N number of dimensions
 * @param dims dimensions of input (domain)
 */
const struct operator_s* operator_identity_create(unsigned int N, const long dims[N])
{
        const struct iovec_s* domain = iovec_create(N, dims, CFL_SIZE);

        return operator_create(N, dims, N, dims, (void*)domain, identity_apply, identity_free);
}




struct operator_chain_s {

	const struct operator_s* a;
	const struct operator_s* b;
};


static void chain_apply(const void* _data, unsigned int N, void* args[N])
{
	const struct operator_chain_s* data = _data;

	assert(2 == N);
	assert(2 == data->a->N);
	assert(2 == data->b->N);

	const struct iovec_s* iovec = data->a->domain[0];
	complex float* tmp = md_alloc_sameplace(iovec->N, iovec->dims, iovec->size, args[0]);

	operator_apply_unchecked(data->a, tmp, args[1]);
	operator_apply_unchecked(data->b, args[0], tmp);

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

	assert((2 == a->N) && (2 == b->N));
	assert(a->domain[0]->N == b->domain[1]->N);
	assert(md_calc_size(a->domain[0]->N, a->domain[0]->dims) == md_calc_size(b->domain[1]->N, b->domain[1]->dims));

	// check whether intermediate storage can be simple

	assert(a->domain[0]->N == md_calc_blockdim(a->domain[0]->N, a->domain[0]->dims, a->domain[0]->strs, a->domain[0]->size));
	assert(b->domain[1]->N == md_calc_blockdim(b->domain[1]->N, b->domain[1]->dims, b->domain[1]->strs, a->domain[1]->size));

	c->a = operator_ref(a);
	c->b = operator_ref(b);

	const struct iovec_s* dom = a->domain[1];
	const struct iovec_s* cod = b->domain[0];
	return operator_create2(cod->N, cod->dims, cod->strs, dom->N, dom->dims, dom->strs, c, chain_apply, chain_free);
}



const struct operator_s* operator_chainN(unsigned int N, const struct operator_s* ops[N])
{
	assert(N > 0);

	const struct operator_s* s = operator_identity_create(ops[0]->domain[0]->N, ops[0]->domain[1]->dims);

	for (unsigned int i = 0; i < N; i++)
		s = operator_chain(s, ops[i]);

	return s;
}



struct operator_stack_s {

	const struct operator_s* a;
	const struct operator_s* b;

	long dst_offset;
	long src_offset;
};


static void stack_apply(const void* _data, unsigned int N, void* args[N])
{
	const struct operator_stack_s* data = _data;
	assert(2 == N);

	operator_apply_unchecked(data->a, args[0], args[1]);
	operator_apply_unchecked(data->b, args[0] + data->dst_offset, args[1] + data->src_offset);
}

static void stack_free(const void* _data)
{
	const struct operator_stack_s* data = _data;

	operator_free(data->a);
	operator_free(data->b);

	free((void*)data);
}

static bool stack_compatible(unsigned int D, const struct iovec_s* a, const struct iovec_s* b)
{
	if (a->N != b->N)
		return false;

	unsigned int N = a->N;

	for (unsigned int i = 0; i < N; i++)
		if ((D != i) && ((a->dims[i] != b->dims[i] || (a->strs[i] != b->strs[i]))))
			return false;

	if ((1 != a->dims[D]) || (1 != b->dims[D]))
		return false;

	return true;
}

static void stack_dims(unsigned int N, long dims[N], long strs[N], unsigned int D, const struct iovec_s* a, const struct iovec_s* b)
{
	md_copy_dims(N, dims, a->dims);
	md_copy_strides(N, strs, a->strs);

	UNUSED( b );

	strs[D] = md_calc_size(N, a->dims) * CFL_SIZE;	// FIXME
	dims[D] = 2;
}

/**
 * Create a new operator that stacks a and b along dimension D
 */
const struct operator_s* operator_stack(unsigned int D, unsigned int E, const struct operator_s* a, const struct operator_s* b)
{
	struct operator_stack_s* c = xmalloc(sizeof(struct operator_stack_s));

	assert(stack_compatible(D, a->domain[0], b->domain[0]));
	assert(stack_compatible(E, a->domain[1], b->domain[1]));

	c->a = operator_ref(a);
	c->b = operator_ref(b);

	unsigned int cod_N = a->domain[0]->N;
	long cod_dims[cod_N];
	long cod_strs[cod_N];
	stack_dims(cod_N, cod_dims, cod_strs, D, a->domain[0], b->domain[0]);

	unsigned int dom_N = a->domain[1]->N;
	long dom_dims[dom_N];
	long dom_strs[dom_N];
	stack_dims(dom_N, dom_dims, dom_strs, E, a->domain[1], b->domain[1]);

	assert(dom_N == cod_N);

	c->dst_offset = cod_strs[D];
	c->src_offset = dom_strs[D];

	return operator_create2(cod_N, cod_dims, cod_strs, dom_N, dom_dims, dom_strs, c, stack_apply, stack_free);
}


extern void operator_apply_unchecked(const struct operator_s* op, complex float* dst, const complex float* src)
{
	op->apply((void*)op->data, 2, (void*[2]){ (void*)dst, (void*)src });
}

extern void operator_apply2(const struct operator_s* op, unsigned int IN, const long idims[IN], const long istrs[IN], complex float* dst, const long ON, const long odims[ON], const long ostrs[ON], const complex float* src)
{
	assert(2 == op->N);
	assert(iovec_check(op->domain[1], IN, idims, istrs));
	assert(iovec_check(op->domain[0], ON, odims, ostrs));

	operator_apply_unchecked(op, dst, src);
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
	assert(3 == op->op.N);
	assert(iovec_check(op->op.domain[2], IN, idims, istrs));
	assert(iovec_check(op->op.domain[1], ON, odims, ostrs));

	operator_p_apply_unchecked(op, mu, dst, src);
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
	op->op.apply(op->op.data, 3, (void*[3]){ &mu, (void*)dst, (void*)src });
}


extern void operator_iter(void* o, float* _dst, const float* _src)
{
	complex float* dst = (complex float*)_dst;
	const complex float* src = (complex float*)_src;

	operator_apply_unchecked(o, dst, src);
}

extern void operator_p_iter(void* o, float lambda, float* _dst, const float* _src)
{
	complex float* dst = (complex float*)_dst;
	const complex float* src = (complex float*)_src;

	operator_p_apply_unchecked(o, lambda, dst, src);
}

