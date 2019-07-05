/* Copyright 2015. The Regents of the University of California.
 * Copyright 2016-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2014 Frank Ong <frankong@berkeley.edu>
 *
 * operator expressions working on multi-dimensional arrays 
 */

#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <alloca.h>

#include "num/multind.h"
#include "num/iovec.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"
#include "misc/shrdptr.h"
#include "misc/nested.h"

#ifdef USE_CUDA
#ifdef _OPENMP
#include <omp.h>
#endif
#include "num/gpuops.h"
#endif

#include "ops.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif
#ifndef FL_SIZE
#define FL_SIZE sizeof(float)
#endif

struct operator_s {

	unsigned int N;
	unsigned int io_flags;
	const struct iovec_s** domain;

	operator_data_t* data;
	void (*apply)(const operator_data_t* data, unsigned int N, void* args[N]);
	void (*del)(const operator_data_t* data);

	struct shared_obj_s sptr;
};


static void operator_del(const struct shared_obj_s* sptr)
{
	const struct operator_s* x = CONTAINER_OF(sptr, const struct operator_s, sptr);

	if (NULL != x->del)
		x->del(x->data);

	for (unsigned int i = 0; i < x->N; i++)
		iovec_free(x->domain[i]);

	xfree(x->domain);
	xfree(x);
}



/**
 * Create an operator (with strides)
 */
const struct operator_s* operator_generic_create2(unsigned int N, unsigned int io_flags,
			const unsigned int D[N], const long* dims[N], const long* strs[N],
			operator_data_t* data, operator_fun_t apply, operator_del_t del)
{
	PTR_ALLOC(struct operator_s, op);
	PTR_ALLOC(const struct iovec_s*[N], dom);

	for (unsigned int i = 0; i < N; i++)
		(*dom)[i] = iovec_create2(D[i], dims[i], strs[i], CFL_SIZE);

	op->N = N;
	op->io_flags = io_flags;
	op->domain = *PTR_PASS(dom);
	op->data = data;
	op->apply = apply;
	op->del = del;

	shared_obj_init(&op->sptr, operator_del);

	return PTR_PASS(op);
}



/**
 * Create an operator (without strides)
 */
const struct operator_s* operator_generic_create(unsigned int N, unsigned int io_flags,
			const unsigned int D[N], const long* dims[N],
			operator_data_t* data, operator_fun_t apply, operator_del_t del)
{
	const long* strs[N];

	for (unsigned int i = 0; i < N; i++)
		strs[i] = MD_STRIDES(D[i], dims[i], CFL_SIZE);

	return operator_generic_create2(N, io_flags, D, dims, strs, data, apply, del);
}



/**
 * Create an operator (with strides)
 */
const struct operator_s* operator_create2(unsigned int ON, const long out_dims[ON], const long out_strs[ON],
			unsigned int IN, const long in_dims[IN], const long in_strs[IN],
			operator_data_t* data, operator_fun_t apply, operator_del_t del)
{
	return operator_generic_create2(2, MD_BIT(0), (unsigned int[2]){ ON, IN },
				(const long* [2]){ out_dims, in_dims }, (const long* [2]){ out_strs, in_strs },
				data, apply, del);
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
		operator_data_t* data, operator_fun_t apply, operator_del_t del)
{
	return operator_create2(ON, out_dims, MD_STRIDES(ON, out_dims, CFL_SIZE),
				IN, in_dims, MD_STRIDES(IN, in_dims, CFL_SIZE),
				data, apply, del);
}


/**
 * Increment the reference count of an operator
 * 
 * @param x operator
 */
const struct operator_s* operator_ref(const struct operator_s* x)
{
	if (NULL != x)
		shared_obj_ref(&x->sptr);

	return x;
}



/**
 * Decrement the reference count of an operator
 *
 * @param x operator
 */
const struct operator_s* operator_unref(const struct operator_s* x)
{
	if (NULL != x)
		shared_obj_unref(&x->sptr);

	return x;
}




/**
 * Return the data of the associated operator
 *
 * @param x operator
 */
operator_data_t* operator_get_data(const struct operator_s* x)
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

	shared_obj_destroy(&x->sptr);
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
 * Return the number of input args
 *
 * @param op operator
 */
unsigned int operator_nr_in_args(const struct operator_s* op)
{
	return operator_nr_args(op) - operator_nr_out_args(op);
}


/**
 * Return the number of input args
 *
 * @param op operator
 */
unsigned int operator_nr_out_args(const struct operator_s* op)
{
	unsigned int N = operator_nr_args(op);
	unsigned int O = 0;

	for (unsigned int i = 0; i < N; i++)
		if (MD_IS_SET(op->io_flags, i))
			O++;

	return O;
}


unsigned int operator_ioflags(const struct operator_s* op)
{
	return op->io_flags;
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
	assert(2 == op->N);
	assert(1u == op->io_flags);
	return operator_arg_domain(op, 1);
}


/**
 * Return the dimensions and strides of the codomain of an operator
 *
 * @param op operator
 */
const struct iovec_s* operator_codomain(const struct operator_s* op)
{
	assert(2 == op->N);
	assert(1u == op->io_flags);
	return operator_arg_domain(op, 0);
}







struct identity_s {

	INTERFACE(operator_data_t);

	const struct iovec_s* domain;
	const struct iovec_s* codomain;
};

static DEF_TYPEID(identity_s);

static void identity_apply(const operator_data_t* _data, unsigned int N, void* args[N])
{
        const auto d = CAST_DOWN(identity_s, _data);
	assert(2 == N);
        md_copy2(d->domain->N, d->domain->dims, d->codomain->strs, args[0], d->domain->strs, args[1], d->domain->size);
}


static void identity_free(const operator_data_t* _data)
{
        const auto d = CAST_DOWN(identity_s, _data);
        iovec_free(d->domain);
        iovec_free(d->codomain);
	xfree(d);
}


const struct operator_s* operator_identity_create2(unsigned int N, const long dims[N],
					const long ostrs[N], const long istrs[N])
{
	PTR_ALLOC(struct identity_s, data);
	SET_TYPEID(identity_s, data);

        data->domain = iovec_create2(N, dims, istrs, CFL_SIZE);
        data->codomain = iovec_create2(N, dims, ostrs, CFL_SIZE);

        return operator_create2(N, dims, ostrs, N, dims, istrs, CAST_UP(PTR_PASS(data)), identity_apply, identity_free);
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




struct zero_s {

	INTERFACE(operator_data_t);

	const struct iovec_s* codomain;
};

static DEF_TYPEID(zero_s);

static void zero_apply(const operator_data_t* _data, unsigned int N, void* args[N])
{
        auto d = CAST_DOWN(zero_s, _data);
	assert(1 == N);
        md_clear2(d->codomain->N, d->codomain->dims, d->codomain->strs, args[0], d->codomain->size);
}

static void zero_free(const operator_data_t* _data)
{
        auto d = CAST_DOWN(zero_s, _data);
        iovec_free(d->codomain);
	xfree(d);
}

const struct operator_s* operator_zero_create2(unsigned int N, const long dims[N], const long strs[N])
{

	PTR_ALLOC(struct zero_s, data);
	SET_TYPEID(zero_s, data);

        data->codomain = iovec_create2(N, dims, strs, CFL_SIZE);

        return operator_generic_create2(1, 1u, (unsigned int[1]){ N },
			(const long*[1]){ dims },
			(const long*[2]){ strs }, CAST_UP(PTR_PASS(data)), zero_apply, zero_free);
}

const struct operator_s* operator_zero_create(unsigned int N, const long dims[N])
{
        return operator_zero_create2(N, dims, MD_STRIDES(N, dims, CFL_SIZE));
}



struct null_s {

	INTERFACE(operator_data_t);
};

static DEF_TYPEID(null_s);

static void null_apply(const operator_data_t* _data, unsigned int N, void* args[N])
{
	UNUSED(_data);
	assert(1 == N);
	UNUSED(args[0]);
}

static void null_free(const operator_data_t* _data)
{
        xfree(CAST_DOWN(null_s, _data));
}

const struct operator_s* operator_null_create2(unsigned int N, const long dims[N], const long strs[N])
{
	PTR_ALLOC(struct null_s, data);
	SET_TYPEID(null_s, data);

        return operator_generic_create2(1, 0u, (unsigned int[1]){ N },
			(const long*[1]){ dims },
			(const long*[2]){ strs }, CAST_UP(PTR_PASS(data)), null_apply, null_free);
}

const struct operator_s* operator_null_create(unsigned int N, const long dims[N])
{
        return operator_null_create2(N, dims, MD_STRIDES(N, dims, CFL_SIZE));
}




/**
 * Create a new operator that first applies a, then applies b:
 * c(x) = b(a(x))
 */
const struct operator_s* operator_chain(const struct operator_s* a, const struct operator_s* b)
{
	assert((2 == a->N) && (2 == b->N));

	// check compatibility

	debug_printf(DP_DEBUG4, "operator chain:\n");
	debug_print_dims(DP_DEBUG4, a->domain[0]->N, a->domain[0]->dims);
	debug_print_dims(DP_DEBUG4, b->domain[1]->N, b->domain[1]->dims);
	debug_printf(DP_DEBUG4, "IO Flags: %d %d\n", a->io_flags, b->io_flags);


	assert((MD_BIT(0) == a->io_flags) && (MD_BIT(0) == b->io_flags));
	assert(a->domain[0]->N == b->domain[1]->N);
	assert(md_calc_size(a->domain[0]->N, a->domain[0]->dims) == md_calc_size(b->domain[1]->N, b->domain[1]->dims));

	// check whether intermediate storage can be simple

	assert(a->domain[0]->N == md_calc_blockdim(a->domain[0]->N, a->domain[0]->dims, a->domain[0]->strs, a->domain[0]->size));
	assert(b->domain[1]->N == md_calc_blockdim(b->domain[1]->N, b->domain[1]->dims, b->domain[1]->strs, b->domain[1]->size));



	auto op = operator_combi_create(2, MAKE_ARRAY(b, a));

	assert((MD_BIT(0) | MD_BIT(2)) == op->io_flags);

	auto op2 = operator_link_create(op, 2, 1);
	operator_free(op);
	return op2;
}




const struct operator_s* operator_chainN(unsigned int N, const struct operator_s* ops[N])
{
	assert(N > 0);

	const struct operator_s* s = operator_identity_create(ops[0]->domain[0]->N, ops[0]->domain[1]->dims);

	for (unsigned int i = 0; i < N; i++)
		s = operator_chain(OP_PASS(s), ops[i]);

	return s;
}





void operator_generic_apply_unchecked(const struct operator_s* op, unsigned int N, void* args[N])
{
	assert(op->N == N);
	debug_trace("ENTER %p\n", op->apply);
	op->apply(op->data, N, args);
	debug_trace("LEAVE %p\n", op->apply);
}


void operator_apply_unchecked(const struct operator_s* op, complex float* dst, const complex float* src)
{
	assert(1u == op->io_flags);
	operator_generic_apply_unchecked(op, 2, (void*[2]){ (void*)dst, (void*)src });
}

void operator_apply2(const struct operator_s* op, unsigned int ON, const long odims[ON], const long ostrs[ON], complex float* dst, const long IN, const long idims[IN], const long istrs[ON], const complex float* src)
{
	assert(2 == op->N);
	assert(iovec_check(op->domain[1], IN, idims, istrs));
	assert(iovec_check(op->domain[0], ON, odims, ostrs));

	operator_apply_unchecked(op, dst, src);
}

void operator_apply(const struct operator_s* op, unsigned int ON, const long odims[ON], complex float* dst, const long IN, const long idims[IN], const complex float* src)
{
	operator_apply2(op, ON, odims, MD_STRIDES(ON, odims, CFL_SIZE), dst,
			    IN, idims, MD_STRIDES(IN, idims, CFL_SIZE), src);
}



struct attach_data_s {

	INTERFACE(operator_data_t);

	const struct operator_s* op;

	void* ptr;
	void (*del)(const void* ptr);
};

static DEF_TYPEID(attach_data_s);


static void attach_fun(const operator_data_t* _data, unsigned int N, void* args[N])
{
	const auto op = CAST_DOWN(attach_data_s, _data)->op;
	operator_generic_apply_unchecked(op, N, args);
}

static void attach_del(const operator_data_t* _data)
{
	const auto data = CAST_DOWN(attach_data_s, _data);

	operator_free(data->op);

	data->del(data->ptr);

	xfree(data);
}

const struct operator_s* operator_attach(const struct operator_s* op, void* ptr, void (*del)(const void* ptr))
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
	PTR_ALLOC(struct attach_data_s, data);
	SET_TYPEID(attach_data_s, data);
	data->op = op;
	data->ptr = ptr;
	data->del = del;

	return operator_generic_create2(N, op->io_flags, D, dims, strs, CAST_UP(PTR_PASS(data)), attach_fun, attach_del);
}


struct op_bind_s {

	INTERFACE(operator_data_t);

	unsigned int D;
	unsigned int arg;

	const struct operator_s* op;
	void* ptr;
};

static DEF_TYPEID(op_bind_s);

static void op_bind_apply(const operator_data_t* _data, unsigned int N, void* args[N])
{
	const auto data = CAST_DOWN(op_bind_s, _data);
	assert(data->D == N + 1);

	void* n_args[N + 1];

	for (unsigned int i = 0, j = 0; i < N; i++, j++) {

		// insert bound argument

		if (data->arg == i)
			n_args[j++] = data->ptr;

		n_args[j] = args[i];
	}

	operator_generic_apply_unchecked(data->op, N + 1, n_args);
}

static void op_bind_del(const operator_data_t* _data)
{
	const auto data = CAST_DOWN(op_bind_s, _data);

	operator_free(data->op);

	xfree(data);
}


/**
 * Create a new operator that binds argument 'arg'.
 */
const struct operator_s* operator_bind2(const struct operator_s* op, unsigned int arg,
			unsigned int N, const long dims[N], const long strs[N], void* ptr)
{
	unsigned int D = operator_nr_args(op);
	assert(arg < D);
	assert(!MD_IS_SET(op->io_flags, arg));
	assert(iovec_check(operator_arg_domain(op, arg), N, dims, strs));

	unsigned int nn[D - 1];
	const long* ndims[D - 1];
	const long* nstrs[D - 1];

	unsigned int n_flags = 0u;

	for (unsigned int i = 0, j = 0; i < D; i++) {

		if (arg == i)
			continue;

		nn[j] = operator_arg_domain(op, i)->N;
		ndims[j] = operator_arg_domain(op, i)->dims;
		nstrs[j] = operator_arg_domain(op, i)->strs;

		if (MD_IS_SET(op->io_flags, i))
			n_flags |= MD_BIT(j);

		j++;
	}

	PTR_ALLOC(struct op_bind_s, data);
	SET_TYPEID(op_bind_s, data);

	data->D = D;
	data->arg = arg;
	data->ptr = ptr;
	data->op = op;

	return operator_generic_create2(D - 1, n_flags,
		nn, ndims, nstrs,
		CAST_UP(PTR_PASS(data)), op_bind_apply, op_bind_del);
}





struct op_loop_s {

	INTERFACE(operator_data_t);

	unsigned int N;
	unsigned int D;
	const long** strs;
	const long** dims;
	const long* dims0;
	const struct operator_s* op;

	unsigned int parallel;
	bool gpu;
};

static DEF_TYPEID(op_loop_s);

static void op_loop_del(const operator_data_t* _data)
{
	const auto data = CAST_DOWN(op_loop_s, _data);
	operator_free(data->op);

	for (unsigned int i = 0; i < data->N; i++) {

		xfree(data->dims[i]);
		xfree(data->strs[i]);
	}

	xfree(data->strs);
	xfree(data->dims);
	xfree(data->dims0);
	xfree(data);
}

static void op_loop_fun(const operator_data_t* _data, unsigned int N, void* args[N])
{
	const auto data = CAST_DOWN(op_loop_s, _data);
	assert(N == data->N);

	if (data->gpu) {
#if defined(USE_CUDA) && defined(_OPENMP)
                int nr_cuda_devices = cuda_devices();
                omp_set_num_threads(nr_cuda_devices * 2);
//              fft_set_num_threads(1);
#else
                assert(0);
#endif
	}

	extern bool num_auto_parallelize;
	bool ap_save = num_auto_parallelize;
	num_auto_parallelize = false;

	NESTED(void, op_loop_nary, (void* ptr[]))
	{
		operator_generic_apply_unchecked(data->op, data->N, ptr);
	};

	md_parallel_nary(N, data->D, data->dims0, data->parallel, data->strs, args, op_loop_nary);

	num_auto_parallelize = ap_save;
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

const struct operator_s* (operator_loop_parallel2)(unsigned int N, const unsigned int D,
				const long dims[D], const long (*strs)[D],
				const struct operator_s* op,
				unsigned int flags, bool gpu)

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

		(*dims2)[i] = *PTR_PASS(tdims);
		(*strs2)[i] = *PTR_PASS(tstrs);
	}

	PTR_ALLOC(struct op_loop_s, data);
	SET_TYPEID(op_loop_s, data);
	data->N = N;
	data->D = D;
	data->op = op;

	data->dims0 = *PTR_PASS(dims0);
	data->dims = */*PTR_PASS*/(dims2);
	data->strs = */*PTR_PASS*/(strs2);

	data->parallel = flags;
	data->gpu = gpu;

	const struct operator_s* rop = operator_generic_create2(N, op->io_flags, D2, *dims2, *strs2, CAST_UP(PTR_PASS(data)), op_loop_fun, op_loop_del);

	PTR_PASS(dims2);
	PTR_PASS(strs2);

	return rop;
}

const struct operator_s* (operator_loop2)(unsigned int N, const unsigned int D,
				const long dims[D], const long (*strs)[D],
				const struct operator_s* op)
{
	return operator_loop_parallel2(N, D, dims, strs, op, 0u, false);
}

const struct operator_s* operator_loop_parallel(unsigned int D, const long dims[D], const struct operator_s* op, unsigned int parallel, bool gpu)
{
	unsigned int N = operator_nr_args(op);
	long strs[N][D];

	for (unsigned int i = 0; i < N; i++) {

		long tdims[D];
		merge_dims(D, tdims, dims, operator_arg_domain(op, i)->dims);
		md_calc_strides(D, strs[i], tdims, operator_arg_domain(op, i)->size);
	}

	return operator_loop_parallel2(N, D, dims, strs, op, parallel, gpu);
}

const struct operator_s* operator_loop(unsigned int D, const long dims[D], const struct operator_s* op)
{
	return operator_loop_parallel(D, dims, op, 0u, false);
}

struct copy_data_s {

	INTERFACE(operator_data_t);

	const struct operator_s* op;

	unsigned int N;
	const long** strs;
};

static DEF_TYPEID(copy_data_s);

static void copy_fun(const operator_data_t* _data, unsigned int N, void* args[N])
{
	const auto data = CAST_DOWN(copy_data_s, _data);
	const struct operator_s* op = data->op;
	void* ptr[N];

	assert(N == operator_nr_args(op));

	for (unsigned int i = 0; i < N; i++) {

		const struct iovec_s* io = operator_arg_domain(op, i);

		ptr[i] = md_alloc(io->N, io->dims, io->size);

		if (!MD_IS_SET(op->io_flags, i))
			md_copy2(io->N, io->dims, io->strs, ptr[i], data->strs[i], args[i], io->size);
	}

	operator_generic_apply_unchecked(op, N, ptr);

	for (unsigned int i = 0; i < N; i++) {

		const struct iovec_s* io = operator_arg_domain(op, i);

		if (MD_IS_SET(op->io_flags, i))
			md_copy2(io->N, io->dims, data->strs[i], args[i], io->strs, ptr[i], io->size);

		md_free(ptr[i]);
	}
}

static void copy_del(const operator_data_t* _data)
{
	const auto data = CAST_DOWN(copy_data_s, _data);

	operator_free(data->op);

	for (unsigned int i = 0; i < data->N; i++)
		xfree(data->strs[i]);

	xfree(data->strs);
	xfree(data);
}

const struct operator_s* operator_copy_wrapper(unsigned int N, const long* strs[N], const struct operator_s* op)
{
	assert(N == operator_nr_args(op));

	// op = operator_ref(op);
	PTR_ALLOC(struct copy_data_s, data);
	SET_TYPEID(copy_data_s, data);
	data->op = op;

	unsigned int D[N];
	const long* dims[N];
	const long* (*strs2)[N] = TYPE_ALLOC(const long*[N]);

	for (unsigned int i = 0; i < N; i++) {

		const struct iovec_s* io = operator_arg_domain(op, i);

		D[i] = io->N;
		dims[i] = io->dims;

		long (*strsx)[io->N] = TYPE_ALLOC(long[io->N]);
		md_copy_strides(io->N, *strsx, strs[i]);
		(*strs2)[i] = *strsx;

		long tstrs[io->N];
		md_calc_strides(io->N, tstrs, io->dims, CFL_SIZE);

		for (unsigned int i = 0; i < io->N; i++)
			assert(io->strs[i] == tstrs[i]);
	}

	data->N = N;
	data->strs = *strs2;

	return operator_generic_create2(N, op->io_flags, D, dims, *strs2, CAST_UP(PTR_PASS(data)), copy_fun, copy_del);
}




struct gpu_data_s {

	INTERFACE(operator_data_t);

	long move_flags;
	const struct operator_s* op;
};

static DEF_TYPEID(gpu_data_s);


#if defined(USE_CUDA) && defined(_OPENMP)
#include <omp.h>
#define MAX_CUDA_DEVICES 16
omp_lock_t gpulock[MAX_CUDA_DEVICES];
#endif


static void gpuwrp_fun(const operator_data_t* _data, unsigned int N, void* args[N])
{
#if defined(USE_CUDA) && defined(_OPENMP)
	const auto data = CAST_DOWN(gpu_data_s, _data);
	const auto op = data->op;
	void* gpu_ptr[N];

	assert(N == operator_nr_args(op));

	debug_printf(DP_DEBUG1, "GPU start.\n");

        int nr_cuda_devices = MIN(cuda_devices(), MAX_CUDA_DEVICES);
        int gpun = omp_get_thread_num() % nr_cuda_devices;

        cuda_init(gpun);

	for (unsigned int i = 0; i < N; i++) {

		if (!MD_IS_SET(data->move_flags, i)) {

			gpu_ptr[i] = args[i];
			continue;
		}

		const struct iovec_s* io = operator_arg_domain(op, i);

		if (MD_IS_SET(op->io_flags, i))
			gpu_ptr[i] = md_alloc_gpu(io->N, io->dims, io->size);
		else
			gpu_ptr[i] = md_gpu_move(io->N, io->dims, args[i], io->size);
	}

	omp_set_lock(&gpulock[gpun]);
	operator_generic_apply_unchecked(op, N, gpu_ptr);
	omp_unset_lock(&gpulock[gpun]);
	
	for (unsigned int i = 0; i < N; i++) {

		if (!MD_IS_SET(data->move_flags, i))
			continue;

		const struct iovec_s* io = operator_arg_domain(op, i);

		if (MD_IS_SET(op->io_flags, i))
			md_copy(io->N, io->dims, args[i], gpu_ptr[i], io->size);

		md_free(gpu_ptr[i]);
	}

	debug_printf(DP_DEBUG1, "GPU end.\n");

#else
	UNUSED(_data); UNUSED(N); UNUSED(args);
	assert(0);
#endif
}

static void gpuwrp_del(const operator_data_t* _data)
{
	const auto data = CAST_DOWN(gpu_data_s, _data);

	operator_free(data->op);

	xfree(data);
}

const struct operator_s* operator_gpu_wrapper2(const struct operator_s* op, long move_flags)
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
	PTR_ALLOC(struct gpu_data_s, data);
	SET_TYPEID(gpu_data_s, data);
	data->move_flags = move_flags;
	data->op = operator_ref(op);

	return operator_generic_create2(N, op->io_flags, D, dims, strs, CAST_UP(PTR_PASS(data)), gpuwrp_fun, gpuwrp_del);
}


const struct operator_s* operator_gpu_wrapper(const struct operator_s* op)
{
	return operator_gpu_wrapper2(op, ~0L);
}


struct operator_combi_s {

	INTERFACE(operator_data_t);

	int N;
	const struct operator_s** x;
};

static DEF_TYPEID(operator_combi_s);


static void combi_apply(const operator_data_t* _data, unsigned int N, void* args[N])
{
	auto data = CAST_DOWN(operator_combi_s, _data);

	debug_printf(DP_DEBUG4, "combi apply: ops: %d args: %d\n", data->N, N);

	int off = N;

	for (int i = (int)data->N - 1; 0 <= i; i--) {

		int A = operator_nr_args(data->x[i]);

		off -= A;

		for (int a = 0; a < A; a++)
			debug_printf(DP_DEBUG4, "combi apply: op[%d].arg[%d] == %p\n", i, a, args[off + a]);

		assert(0 <= off);
		assert(off < (int)N);

		operator_generic_apply_unchecked(data->x[i], A, args + off);
	}

	assert(0 == off);
}

static void combi_free(const operator_data_t* _data)
{
	auto data = CAST_DOWN(operator_combi_s, _data);

	for (int i = 0; i < data->N; i++)
		operator_free(data->x[i]);

	xfree(data->x);
	xfree(data);
}



/**
 * Create a new operator that combines several others
 */
const struct operator_s* operator_combi_create(int N, const struct operator_s* x[N])
{
	PTR_ALLOC(struct operator_combi_s, c);
	SET_TYPEID(operator_combi_s, c);

	PTR_ALLOC(const struct operator_s*[N], xp);

	for (int i = 0; i < N; i++)
		(*xp)[i] = operator_ref(x[i]);

	c->x = *PTR_PASS(xp);
	c->N = N;

	int A = 0;

	for (int i = 0; i < N; i++)
		A += operator_nr_args(x[i]);

	unsigned int io_flags = 0;;
	unsigned int D[A];
	const long* dims[A];
	const long* strs[A];

	int a = 0;

	for (int i = 0; i < N; i++) {

		for (int j = 0; j < (int)operator_nr_args(x[i]); j++) {

			auto iov = operator_arg_domain(x[i], j);

			D[a] = iov->N;
			dims[a] = iov->dims;
			strs[a] = iov->strs;

			io_flags |= ((x[i]->io_flags >> j) & 1) << a;
			a++;
		}
	}

	return operator_generic_create2(A, io_flags, D, dims, strs, CAST_UP(PTR_PASS(c)), combi_apply, combi_free);
}




struct operator_dup_s {

	INTERFACE(operator_data_t);

	int a;
	int b;
	const struct operator_s* x;
};

static DEF_TYPEID(operator_dup_s);


static void dup_apply(const operator_data_t* _data, unsigned int N, void* args[N])
{
	auto data = CAST_DOWN(operator_dup_s, _data);

	debug_printf(DP_DEBUG4, "dup apply: duplicating %d-%d from %d (io flags: %d)\n",
				data->a, data->b, N + 1, data->x->io_flags);

	void* args2[N + 1];

	for (int i = 0, j = 0; j < (int)(N + 1); j++) {

		if (data->b == j)
			continue;

		args2[j] = args[i++];
	}

	assert(N > 0);

	auto iov = operator_arg_domain(data->x, data->a);
	auto iovb = operator_arg_domain(data->x, data->b);

	assert(iovec_check(iovb, iov->N, iov->dims, iov->strs));

	assert(N + 1 == operator_nr_args(data->x));

	args2[data->b] = args2[data->a];

	operator_generic_apply_unchecked(data->x, N + 1, args2);
}

static void dup_del(const operator_data_t* _data)
{
	auto data = CAST_DOWN(operator_dup_s, _data);

	operator_free(data->x);
	xfree(data);
}

const struct operator_s* operator_dup_create(const struct operator_s* op, unsigned int a, unsigned int b)
{
	unsigned int N = operator_nr_args(op);

	assert(a < N);
	assert(b < N);
	assert(a != b);

	unsigned int io_flags = 0u;
	unsigned int D[N - 1];
	const long* dims[N - 1];
	const long* strs[N - 1];

	debug_printf(DP_DEBUG3, "Duplicating args %d-%d of %d.\n", a, b, N);

	for (unsigned int s = 0, t = 0; s < N; s++) {

		if (s == b)
			continue;

		auto io = operator_arg_domain(op, s);

		D[t] = io->N;
		dims[t] = io->dims;
		strs[t] = io->strs;

		io_flags |= ((op->io_flags >> s) & 1) << t;

		t++;
	}

	auto ioa = operator_arg_domain(op, a);
	auto iob = operator_arg_domain(op, b);

	assert(ioa->N == md_calc_blockdim(ioa->N, ioa->dims, ioa->strs, ioa->size));
	assert(iob->N == md_calc_blockdim(iob->N, iob->dims, iob->strs, iob->size));


	PTR_ALLOC(struct operator_dup_s, data);
	SET_TYPEID(operator_dup_s, data);

	data->a = a;
	data->b = b;
	data->x = operator_ref(op);

	return operator_generic_create2(N - 1, io_flags, D, dims, strs, CAST_UP(PTR_PASS(data)), dup_apply, dup_del);
}




// FIXME: we should reimplement link in terms of dup and bind (caveat: gpu)

struct operator_link_s {

	INTERFACE(operator_data_t);

	int a;
	int b;
	const struct operator_s* x;
};

static DEF_TYPEID(operator_link_s);


static void link_apply(const operator_data_t* _data, unsigned int N, void* args[N])
{
	auto data = CAST_DOWN(operator_link_s, _data);

	debug_printf(DP_DEBUG4, "link apply: linking %d-%d of %d (io flags: %d)\n",
				data->a, data->b, N + 2, data->x->io_flags);

	void* args2[N + 2];

	for (int i = 0, j = 0; j < (int)(N + 2); j++) {


		if ((data->a == j) || (data->b == j))
			continue;

		debug_printf(DP_DEBUG4, "link apply: in arg[%d] = %p\n", i, args[i]);
		debug_printf(DP_DEBUG4, "link apply: mapping %d -> %d\n", i, j);

		args2[j] = args[i++];
	}

	assert(N > 0);

	auto iov = operator_arg_domain(data->x, data->a);
	auto iovb = operator_arg_domain(data->x, data->b);

	assert(iovec_check(iovb, iov->N, iov->dims, iov->strs));

	void* tmp = md_alloc_sameplace(iov->N, iov->dims, iov->size, args[0]);

	assert(N + 2 == operator_nr_args(data->x));

	args2[data->a] = tmp;
	args2[data->b] = tmp;

	debug_printf(DP_DEBUG4, "link apply: %d-%d == %p\n", data->a, data->b, tmp);

	for (int i = 0; i < (int)(N + 2); i++)
		debug_printf(DP_DEBUG4, "link apply: out arg[%d] = %p\n", i, args2[i]);


	operator_generic_apply_unchecked(data->x, N + 2, args2);

	md_free(tmp);
}

static void link_del(const operator_data_t* _data)
{
	auto data = CAST_DOWN(operator_link_s, _data);

	operator_free(data->x);
	xfree(data);
}

const struct operator_s* operator_link_create(const struct operator_s* op, unsigned int o, unsigned int i)
{
	unsigned int N = operator_nr_args(op);

	assert(i < N);
	assert(o < N);
	assert(i != o);

	assert( (op->io_flags & MD_BIT(o)));
	assert(~(op->io_flags & MD_BIT(i)));

	unsigned int io_flags = 0u;
	unsigned int D[N - 2];
	const long* dims[N - 2];
	const long* strs[N - 2];

	debug_printf(DP_DEBUG3, "Linking args %d-%d of %d.\n", i, o, N);

	for (unsigned int s = 0, t = 0; s < N; s++) {

		if ((s == i) || (s == o))
			continue;

		auto io = operator_arg_domain(op, s);

		D[t] = io->N;
		dims[t] = io->dims;
		strs[t] = io->strs;

		io_flags |= ((op->io_flags >> s) & 1) << t;

		t++;
	}

	auto ioi = operator_arg_domain(op, i);
	auto ioo = operator_arg_domain(op, o);

	assert(ioi->N == md_calc_blockdim(ioi->N, ioi->dims, ioi->strs, ioi->size));
	assert(ioo->N == md_calc_blockdim(ioo->N, ioo->dims, ioo->strs, ioo->size));


	PTR_ALLOC(struct operator_link_s, data);
	SET_TYPEID(operator_link_s, data);

	data->a = i;
	data->b = o;
	data->x = operator_ref(op);

	return operator_generic_create2(N - 2, io_flags, D, dims, strs, CAST_UP(PTR_PASS(data)), link_apply, link_del);
}



struct permute_data_s {

	INTERFACE(operator_data_t);

	const int* perm;
	const struct operator_s* op;
};

static DEF_TYPEID(permute_data_s);



static void permute_fun(const operator_data_t* _data, unsigned int N, void* args[N])
{
	auto data = CAST_DOWN(permute_data_s, _data);

	assert(N == operator_nr_args(data->op));

	void* ptr[N];

	for (int i = 0; i < (int)N; i++)
		ptr[data->perm[i]] = args[i];

	for (int i = 0; i < (int)N; i++)
		debug_printf(DP_DEBUG4, "permute apply: in arg[%d] = %p\n", i, args[i]);

	for (int i = 0; i < (int)N; i++)
		debug_printf(DP_DEBUG4, "permute apply: out arg[%d] = %p\n", i, ptr[i]);

	operator_generic_apply_unchecked(data->op, N, ptr);
}


static void permute_del(const operator_data_t* _data)
{
	auto data = CAST_DOWN(permute_data_s, _data);

	operator_free(data->op);

	xfree(data->perm);

	xfree(data);
}

const struct operator_s* operator_permute(const struct operator_s* op, int N, const int perm[N])
{
	assert(N == (int)operator_nr_args(op));

	unsigned long flags = 0;
	unsigned long io_flags = 0;
	unsigned int D[N];
	const long* dims[N];
	const long* strs[N];

	for (int i = 0; i < N; i++) {

		assert(perm[i] < N);
		flags |= MD_BIT(perm[i]);

		const struct iovec_s* io = operator_arg_domain(op, perm[i]);

		D[i] = io->N;
		dims[i] = io->dims;
		strs[i] = io->strs;

		if (op->io_flags & MD_BIT(perm[i]))
			io_flags |= 1 << i;
	}

	assert(MD_BIT(N) == flags + 1);

	// op = operator_ref(op);
	PTR_ALLOC(struct permute_data_s, data);
	SET_TYPEID(permute_data_s, data);
	data->op = operator_ref(op);

	int* nperm = *TYPE_ALLOC(int[N]);
	memcpy(nperm, perm, sizeof(int[N]));

	data->perm = nperm;

	return operator_generic_create2(N, io_flags, D, dims, strs, CAST_UP(PTR_PASS(data)), permute_fun, permute_del);
}


struct extract_data_s {

	INTERFACE(operator_data_t);

	int a;
	size_t off;
	const struct operator_s* op;
};

static DEF_TYPEID(extract_data_s);



static void extract_fun(const operator_data_t* _data, unsigned int N, void* args[N])
{
	auto data = CAST_DOWN(extract_data_s, _data);

	assert(N == operator_nr_args(data->op));

	void* ptr[N];

	for (int i = 0; i < (int)N; i++)
		ptr[i] = args[i] + ((data->a == i) ? data->off : 0);

	operator_generic_apply_unchecked(data->op, N, ptr);
}


static void extract_del(const operator_data_t* _data)
{
	auto data = CAST_DOWN(extract_data_s, _data);

	operator_free(data->op);

	xfree(data);
}


const struct operator_s* operator_extract_create2(const struct operator_s* op, int a, int Da, const long dimsa[Da], const long strsa[Da], const long pos[Da])
{
	int N = (int)operator_nr_args(op);

	assert(a < N);

	unsigned int D[N];
	const long* dims[N];
	const long* strs[N];

	for (int i = 0; i < N; i++) {

		const struct iovec_s* io = operator_arg_domain(op, i);

		D[i] = io->N;
		dims[i] = io->dims;
		strs[i] = io->strs;
	}

	assert(Da == (int)D[a]);

	for (int i = 0; i < Da; i++) {

		assert((0 <= pos[i]) && (pos[i] < dimsa[i]));
		assert(dims[a][i] + pos[i] <= dimsa[i]);
		assert((0 == strs[a][i]) || (strs[a][i] == strsa[i]));
	}

	dims[a] = dimsa;
	strs[a] = strsa;

	PTR_ALLOC(struct extract_data_s, data);
	SET_TYPEID(extract_data_s, data);

	data->op = operator_ref(op);
	data->a = a;
	data->off = md_calc_offset(Da, strsa, pos);

	return operator_generic_create2(N, op->io_flags, D, dims, strs, CAST_UP(PTR_PASS(data)), extract_fun, extract_del);
}


const struct operator_s* operator_extract_create(const struct operator_s* op, int a, int Da, const long dimsa[Da], const long pos[Da])
{
	return operator_extract_create2(op, a, Da, dimsa, MD_STRIDES(Da, dimsa, operator_arg_domain(op, a)->size), pos);
}

static bool stack_compatible(unsigned int D, const struct iovec_s* a, const struct iovec_s* b)
{
	if (a->N != b->N)
		return false;

	unsigned int N = a->N;

	for (unsigned int i = 0; i < N; i++)
		if ((D != i) && ((a->dims[i] != b->dims[i] || (a->strs[i] != b->strs[i]))))
			return false;

	if ((1 != a->dims[D]) && (1 != b->dims[D]))
		if (a->strs[D] != b->strs[D])
			return false;

	long dims[N];
	md_select_dims(N, ~MD_BIT(D), dims, a->dims);

	long S = md_calc_size(N, dims) * a->size;

	if ((1 != a->dims[D]) && (S != a->strs[D]))
		return false;

	if ((1 != b->dims[D]) && (S != b->strs[D]))
		return false;

	return true;
}

static void stack_dims(unsigned int N, long dims[N], long strs[N], unsigned int D, const struct iovec_s* a, const struct iovec_s* b)
{
	md_copy_dims(N, dims, a->dims);
	md_copy_strides(N, strs, a->strs);

	long dimsa[N];
	md_select_dims(N, ~MD_BIT(D), dimsa, a->dims);

	strs[D] = md_calc_size(N, dimsa) * a->size;
	dims[D] = a->dims[D] + b->dims[D];
}


const struct operator_s* operator_stack2(int M, const int arg_list[M], const int dim_list[M], const struct operator_s* a, const struct operator_s* b)
{
	a = operator_ref(a);
	b = operator_ref(b);

	for (int m = 0; m < M; m++) {

		int arg = arg_list[m];
		int dim = dim_list[m];

		auto ia = operator_arg_domain(a, arg);
		auto ib = operator_arg_domain(b, arg);

		assert(stack_compatible(dim, ia, ib));

		int D = ia->N;

		long dims[D];
		long strs[D];
		stack_dims(D, dims, strs, dim, ia, ib);

		long pos[D];

		for (int i = 0; i < D; i++)
			pos[i] = 0;

		auto aa = operator_extract_create2(a, arg, D, dims, strs, (pos[dim] = 0, pos));
		auto bb = operator_extract_create2(b, arg, D, dims, strs, (pos[dim] = 1, pos));

		operator_free(a);
		operator_free(b);

		a = aa;
		b = bb;
	}

	auto c = operator_combi_create(2, MAKE_ARRAY(a, b));

	operator_free(a);
	operator_free(b);

	int N = operator_nr_args(a);

	for (int i = N - 1; i >= 0; i--) {

		auto d = operator_dup_create(c, i, N + i);
		operator_free(c);
		c = d;
	}

	return c;
}


/**
 * Create a new operator that stacks a and b along dimension D
 */
const struct operator_s* operator_stack(unsigned int D, unsigned int E, const struct operator_s* a, const struct operator_s* b)
{
	return operator_stack2(2, (int[2]){ 0, 1 }, (int[2]){ D, E }, a, b);
}






bool operator_zero_or_null_p(const struct operator_s* op)
{
	auto opd = operator_get_data(op);

	if (   (NULL != CAST_MAYBE(zero_s, opd))
	    || (NULL != CAST_MAYBE(null_s, opd)))
		return true;

	auto p = CAST_MAYBE(permute_data_s, opd);

	if (NULL != p)
		return operator_zero_or_null_p(p->op);

	// FIXME: unwrap other types...

	auto c = CAST_MAYBE(operator_combi_s, opd);

	if (NULL != c) {

		for (int i = 0; i < c->N; i++)
			if (!operator_zero_or_null_p(c->x[i]))
				return false;

		return true;
	}

	return false;
}


