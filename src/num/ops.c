/* Copyright 2015. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2014 Frank Ong <frankong@berkeley.edu>
 *
 * operator expressions working on multi-dimensional arrays 
 */

#include <complex.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <alloca.h>

#include "num/multind.h"
#include "num/iovec.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "ops.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif
#ifndef FL_SIZE
#define FL_SIZE sizeof(float)
#endif

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
const struct operator_s* operator_generic_create2(unsigned int N, const unsigned int D[N],
			const long* dims[N], const long* strs[N],
			void* data, operator_fun_t apply, operator_del_t del)
{
	PTR_ALLOC(struct operator_s, op);
	PTR_ALLOC(const struct iovec_s*[N], dom);

	for (unsigned int i = 0; i < N; i++)
		(*dom)[i] = iovec_create2(D[i], dims[i], strs[i], CFL_SIZE);

	op->N = N;
	op->domain = *dom;
	op->data = data;
	op->apply = apply;

	op->refcount = 1;
	op->del = del;

	return op;
}



/**
 * Create an operator (without strides)
 */
const struct operator_s* operator_generic_create(unsigned int N, const unsigned int D[N],
			const long* dims[N], void* data, operator_fun_t apply, operator_del_t del)
{
	const long* strs[N];

	for (unsigned int i = 0; i < N; i++)
		strs[i] = MD_STRIDES(D[i], dims[i], CFL_SIZE);

	return operator_generic_create2(N, D, dims, strs, data, apply, del);
}



/**
 * Create an operator (with strides)
 */
const struct operator_s* operator_create2(unsigned int ON, const long out_dims[ON], const long out_strs[ON],
			unsigned int IN, const long in_dims[IN], const long in_strs[IN],
			void* data, operator_fun_t apply, operator_del_t del)
{
	return operator_generic_create2(2, (unsigned int[2]){ ON, IN }, (const long* [2]){ out_dims, in_dims },
			(const long* [2]){ out_strs, in_strs }, data, apply, del);
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
 * Return the number of args
 *
 * @param op operator
 */
unsigned int operator_nr_args(const struct operator_s* op)
{
	return op->N;
}
 

/**
 * Return the iovec of arg n
 *
 * @param op operator
 * @param n  arg number
 */
const struct iovec_s* operator_arg_domain(const struct operator_s* op, unsigned int n)
{
	assert(n < op->N);
	return op->domain[n];
}


/**
 * Return the dimensions and strides of the domain of an operator
 *
 * @param op operator
 */
const struct iovec_s* operator_domain(const struct operator_s* op)
{
	return operator_arg_domain(op, 1);
}


/**
 * Return the dimensions and strides of the codomain of an operator
 *
 * @param op operator
 */
const struct iovec_s* operator_codomain(const struct operator_s* op)
{
	return operator_arg_domain(op, 0);
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
	PTR_ALLOC(struct operator_p_s, o);
	PTR_ALLOC(struct op_p_data_s, op);

	op->data = data;
	op->apply = apply;
	op->del = del;

	PTR_ALLOC(const struct iovec_s*[3], dom);

	(*dom)[0] = iovec_create2(1, MD_DIMS(1), MD_DIMS(0), FL_SIZE);
	(*dom)[1] = iovec_create2(ON, out_dims, out_strs, CFL_SIZE);
	(*dom)[2] = iovec_create2(IN, in_dims, in_strs, CFL_SIZE);

	o->op.N = 3;
	o->op.domain = *dom;
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


struct identity_s {

	const struct iovec_s* domain;
	const struct iovec_s* codomain;
};

static void identity_apply(const void* _data, unsigned int N, void* args[N])
{
        const struct identity_s* d = _data;
	assert(2 == N);
        md_copy2(d->domain->N, d->domain->dims, d->codomain->strs, args[0], d->domain->strs, args[1], d->domain->size);
}


static void identity_free(const void* _data)
{
        const struct identity_s* d = _data;
        iovec_free(d->domain);
        iovec_free(d->codomain);
	free((void*)d);
}


const struct operator_s* operator_identity_create2(unsigned int N, const long dims[N],
					const long ostrs[N], const long istrs[N])
{
	PTR_ALLOC(struct identity_s, data);

        data->domain = iovec_create2(N, dims, istrs, CFL_SIZE);
        data->codomain = iovec_create2(N, dims, ostrs, CFL_SIZE);

        return operator_create2(N, dims, ostrs, N, dims, istrs, data, identity_apply, identity_free);
}

/**
 * Create an Identity operator: I x
 * @param N number of dimensions
 * @param dims dimensions of input (domain)
 */
const struct operator_s* operator_identity_create(unsigned int N, const long dims[N])
{
	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);
        return operator_identity_create2(N, dims, strs, strs);
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
	PTR_ALLOC(struct operator_chain_s, c);
	
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
	PTR_ALLOC(struct operator_stack_s, c);

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



void operator_generic_apply_unchecked(const struct operator_s* op, unsigned int N, void* args[N])
{
	op->apply((void*)op->data, N, args);
}


void operator_apply_unchecked(const struct operator_s* op, complex float* dst, const complex float* src)
{
	operator_generic_apply_unchecked(op, 2, (void*[2]){ (void*)dst, (void*)src });
}

void operator_apply2(const struct operator_s* op, unsigned int IN, const long idims[IN], const long istrs[IN], complex float* dst, const long ON, const long odims[ON], const long ostrs[ON], const complex float* src)
{
	assert(2 == op->N);
	assert(iovec_check(op->domain[1], IN, idims, istrs));
	assert(iovec_check(op->domain[0], ON, odims, ostrs));

	operator_apply_unchecked(op, dst, src);
}

void operator_apply(const struct operator_s* op, unsigned int IN, const long idims[IN], complex float* dst, const long ON, const long odims[ON], const complex float* src)
{
	long ostrs[ON];
	long istrs[IN];
	md_calc_strides(ON, ostrs, odims, CFL_SIZE);
	md_calc_strides(IN, istrs, idims, CFL_SIZE);
	operator_apply2(op, ON, odims, ostrs, dst, IN, idims, istrs, src);
}


void operator_p_apply2(const struct operator_p_s* op, float mu, unsigned int IN, const long idims[IN], const long istrs[IN], complex float* dst, const long ON, const long odims[ON], const long ostrs[ON], const complex float* src)
{
	assert(3 == op->op.N);
	assert(iovec_check(op->op.domain[2], IN, idims, istrs));
	assert(iovec_check(op->op.domain[1], ON, odims, ostrs));

	operator_p_apply_unchecked(op, mu, dst, src);
}


void operator_p_apply(const struct operator_p_s* op, float mu, unsigned int IN, const long idims[IN], complex float* dst, const long ON, const long odims[ON], const complex float* src)
{
	long ostrs[ON];
	long istrs[IN];
	md_calc_strides(ON, ostrs, odims, CFL_SIZE);
	md_calc_strides(IN, istrs, idims, CFL_SIZE);
	operator_p_apply2(op, mu, ON, odims, ostrs, dst, IN, idims, istrs, src);
}


void operator_p_apply_unchecked(const struct operator_p_s* op, float mu, complex float* dst, const complex float* src)
{
	op->op.apply(op->op.data, 3, (void*[3]){ &mu, (void*)dst, (void*)src });
}


void operator_iter(void* o, float* _dst, const float* _src)
{
	complex float* dst = (complex float*)_dst;
	const complex float* src = (complex float*)_src;

	operator_apply_unchecked(o, dst, src);
}

void operator_p_iter(void* o, float lambda, float* _dst, const float* _src)
{
	complex float* dst = (complex float*)_dst;
	const complex float* src = (complex float*)_src;

	operator_p_apply_unchecked(o, lambda, dst, src);
}



struct op_loop_s {

	unsigned int N;
	unsigned int D;
	const long** strs;
	const long** dims;
	const long* dims0;
	const struct operator_s* op;
};

static void op_loop_del(const void* _data)
{
	const struct op_loop_s* data = _data;
	operator_free(data->op);

	for (unsigned int i = 0; i < data->N; i++) {

		free((void*)data->dims[i]);
		free((void*)data->strs[i]);
	}

	free((void*)data->strs);
	free((void*)data->dims);
	free((void*)data->dims0);
	free((void*)data);
}

static void op_loop_nary(void* _data, void* ptr[])
{
	const struct op_loop_s* data = _data;
	operator_generic_apply_unchecked(data->op, data->N, ptr);
}

static void op_loop_fun(const void* _data, unsigned int N, void* args[N])
{
	const struct op_loop_s* data = _data;
	assert(N == data->N);
	md_nary(N, data->D, data->dims0, data->strs, args, (void*)data, op_loop_nary);
}

static void merge_dims(unsigned int D, long odims[D], const long idims1[D], const long idims2[D])
{
	md_copy_dims(D, odims, idims1);

	for (unsigned int i = 0; i < D; i++) {

		assert((1 == odims[i]) | (1 == idims2[i]));

		if (1 == odims[i])
			odims[i] = idims2[i];
	}
}

const struct operator_s* (operator_loop2)(unsigned int N, const unsigned int D,
				const long dims[D], const long (*strs)[D],
				const struct operator_s* op)
{
	assert(N == operator_nr_args(op));

	unsigned int D2[N];
	PTR_ALLOC(long[D], dims0);
	md_copy_dims(D, *dims0, dims);

	PTR_ALLOC(const long*[N], dims2);
	PTR_ALLOC(const long*[N], strs2);


	// TODO: we should have a flag and ignore args with flag

	for (unsigned int i = 0; i < N; i++) {

		const struct iovec_s* io = operator_arg_domain(op, i);

		assert(D == io->N);

		for (unsigned int j = 0; j < D; j++) {

			assert((0 == io->strs[j]) || (io->strs[j] == strs[i][j]));
			assert((1 == io->dims[j]) == (0 == io->strs[j]));
		}

		D2[i] = D;

		PTR_ALLOC(long[D], tdims);
		merge_dims(D, *tdims, dims, io->dims);

		PTR_ALLOC(long[D], tstrs);
		md_copy_strides(D, *tstrs, strs[i]);

		(*dims2)[i] = *tdims;
		(*strs2)[i] = *tstrs;
	}

	PTR_ALLOC(struct op_loop_s, data);
	data->N = N;
	data->D = D;
	data->op = op;

	data->dims0 = *dims0;
	data->dims = *dims2;
	data->strs = *strs2;

	return operator_generic_create2(N, D2, *dims2, *strs2, data, op_loop_fun, op_loop_del);
}

const struct operator_s* operator_loop(unsigned int D, const long dims[D], const struct operator_s* op)
{
	unsigned int N = operator_nr_args(op);
	long strs[N][D];

	for (unsigned int i = 0; i < N; i++) {

		long tdims[D];
		merge_dims(D, tdims, dims, operator_arg_domain(op, i)->dims);
		md_calc_strides(D, strs[i], tdims, operator_arg_domain(op, i)->size);
	}

	return operator_loop2(N, D, dims, strs, op);
}


#ifdef USE_CUDA
static void gpuwrp_fun(const void* _data, unsigned int N, void* args[N])
{
	const struct operator_s* op = _data;
	void* gpu_ptr[N];

	assert(N == operator_nr_args(op));

	for (unsigned int i = 0; i < N; i++) {

		const struct iovec_s* io = operator_arg_domain(op, i);
		gpu_ptr[i] = md_gpu_move(io->N, io->dims, args[i], io->size);
	}

	operator_generic_apply_unchecked(op, N, gpu_ptr);
	
	for (unsigned int i = 0; i < N; i++) {

		const struct iovec_s* io = operator_arg_domain(op, i);
		md_copy(io->N, io->dims, args[i], gpu_ptr[i], io->size);
		md_free(gpu_ptr[i]);
	}
}

static void gpuwrp_del(const void* _data)
{
	const struct operator_s* op = _data;
	operator_free(op);
}

const struct operator_s* operator_gpu_wrapper(const struct operator_s* op)
{
	unsigned int N = operator_nr_args(op);

	unsigned int D[N];
	const long* dims[N];
	const long* strs[N];

	for (unsigned int i = 0; i < N; i++) {

		const struct iovec_s* io = operator_arg_domain(op, i);

		D[i] = io->N;
		dims[i] = io->dims;
		strs[i] = io->strs;
	}

	// op = operator_ref(op);

	return operator_generic_create2(N, D, dims, strs, (void*)op, gpuwrp_fun, gpuwrp_del);
}
#endif

