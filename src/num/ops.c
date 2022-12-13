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

#ifdef _WIN32
#include <malloc.h>
#else
#include <alloca.h>
#endif

#include "num/multind.h"
#include "num/iovec.h"
#include "num/flpmath.h"
#include "num/ops_graph.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"
#include "misc/shrdptr.h"
#include "misc/nested.h"
#include "misc/list.h"
#include "misc/graph.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
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
	const bool* io_flags;
	const struct iovec_s** domain;

	operator_data_t* data;
	void (*apply)(const operator_data_t* data, unsigned int N, void* args[N]);
	void (*del)(const operator_data_t* data);

	const struct graph_s* (*get_graph)(const struct operator_s* op);

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
	xfree(x->io_flags);
	xfree(x);
}



/**
 * Create an operator (with strides)
 */
const struct operator_s* operator_generic_create2(unsigned int N, const bool io_flags[N],
			const unsigned int D[N], const long* dims[N], const long* strs[N],
			operator_data_t* data, operator_fun_t apply, operator_del_t del, operator_get_graph_t get_graph)
{
	PTR_ALLOC(struct operator_s, op);
	PTR_ALLOC(const struct iovec_s*[N], dom);

	for (unsigned int i = 0; i < N; i++)
		(*dom)[i] = iovec_create2(D[i], dims[i], strs[i], CFL_SIZE);

	op->N = N;
	op->io_flags = ARR_CLONE(_Bool[N], io_flags);
	op->domain = *PTR_PASS(dom);
	op->data = data;
	op->apply = apply;
	op->del = del;
	op->get_graph = get_graph;

	shared_obj_init(&op->sptr, operator_del);

	return PTR_PASS(op);
}



/**
 * Create an operator (without strides)
 */
const struct operator_s* operator_generic_create(unsigned int N, const bool io_flags[N],
			const unsigned int D[N], const long* dims[N],
			operator_data_t* data, operator_fun_t apply, operator_del_t del, operator_get_graph_t get_graph)
{
	const long* strs[N];

	for (unsigned int i = 0; i < N; i++)
		strs[i] = MD_STRIDES(D[i], dims[i], CFL_SIZE);

	return operator_generic_create2(N, io_flags, D, dims, strs, data, apply, del, get_graph);
}



/**
 * Create an operator (with strides)
 */
const struct operator_s* operator_create2(unsigned int ON, const long out_dims[ON], const long out_strs[ON],
			unsigned int IN, const long in_dims[IN], const long in_strs[IN],
			operator_data_t* data, operator_fun_t apply, operator_del_t del)
{
	return operator_generic_create2(2, (bool[2]){ true, false }, (unsigned int[2]){ ON, IN },
				(const long* [2]){ out_dims, in_dims }, (const long* [2]){ out_strs, in_strs },
				data, apply, del, NULL);
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


void operator_debug(enum debug_levels dl, const struct operator_s* x)
{
	int N = operator_nr_args(x);

	debug_printf(dl, "OPERATOR\n");

	for (int i = 0; i < N; i++) {

		debug_printf(dl, "%d : ", x->io_flags[i]);
		auto io = x->domain[i];
		debug_print_dims(dl, io->N, io->dims);
	}
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

const bool* operator_get_io_flags(const struct operator_s* x)
{
	return x->io_flags;
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
		if (op->io_flags[i])
			O++;

	return O;
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
 * Return the iovec of input arg n
 *
 * @param op operator
 * @param n input arg number
 */
const struct iovec_s* operator_arg_in_domain(const struct operator_s* op, unsigned int n)
{
	assert(n < operator_nr_in_args(op));

	unsigned int count = 0;
	unsigned int index = 0;

	for (; count <= n; index++)
		if (!op->io_flags[index])
			count++;

	return operator_arg_domain(op, index - 1);
}


/**
 * Return the iovec of output arg n
 *
 * @param op operator
 * @param n output arg number
 */
const struct iovec_s* operator_arg_out_codomain(const struct operator_s* op, unsigned int n)
{
	assert(n < operator_nr_out_args(op));

	unsigned int count = 0;
	unsigned int index = 0;

	for (; count <= n; index++)
		if (op->io_flags[index])
			count++;

	return operator_arg_domain(op, index - 1);
}


/**
 * Return the dimensions and strides of the domain of an operator
 *
 * @param op operator
 */
const struct iovec_s* operator_domain(const struct operator_s* op)
{
	assert(2 == op->N);
	assert(op->io_flags[0]);
	assert(!op->io_flags[1]);

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
	assert(op->io_flags[0]);
	assert(!op->io_flags[1]);

	return operator_arg_domain(op, 0);
}







struct identity_s {

	INTERFACE(operator_data_t);

	const struct iovec_s* domain;
	const struct iovec_s* codomain;
};

static DEF_TYPEID(identity_s);

static const struct identity_s* get_identity_data(const struct operator_s* op)
{
	return CAST_MAYBE(identity_s, operator_get_data(op));
}

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

/**
 * Create a Reshape operator: I x
 * @param A number of out dimensions
 * @param out_dims dimensions of output (codomain)
 * @param B number of in dimensions
 * @param in_dims dimensions of input (domain)
 */
const struct operator_s* operator_reshape_create(unsigned int A, const long out_dims[A], int B, const long in_dims[B])
{
	auto id = operator_identity_create(A, out_dims);
	auto result = operator_reshape(id, 1, B, in_dims);
	operator_free(id);
	return result;
}


struct op_reshape_s {

	INTERFACE(operator_data_t);

	const struct operator_s* x;
};

static DEF_TYPEID(op_reshape_s );

static const struct op_reshape_s* get_reshape_data(const struct operator_s* op)
{
	return CAST_MAYBE(op_reshape_s, operator_get_data(op));
}

static void reshape_apply(const operator_data_t* _data, unsigned int N, void* args[N])
{
	const auto d = CAST_DOWN(op_reshape_s, _data);
	operator_generic_apply_unchecked(d->x, N, args);
}

static void reshape_free(const operator_data_t* _data)
{
	const auto d = CAST_DOWN(op_reshape_s, _data);
	operator_free(d->x);
	xfree(d);
}

static const struct graph_s* operator_reshape_get_graph(const struct operator_s* op)
{
	const auto d = CAST_DOWN(op_reshape_s, op->data);
	auto result = operator_get_graph(d->x);
	for (unsigned int i = 0; i < op->N; i++)
		result = operator_graph_reshape_F(result, i, op->domain[i]->N, op->domain[i]->dims);
	return result;
}

const struct operator_s* operator_reshape(const struct operator_s* op, unsigned int i, long N, const long dims[N])
{
	PTR_ALLOC(struct op_reshape_s, data);
	SET_TYPEID(op_reshape_s, data);

	assert(md_calc_size(N, dims) == md_calc_size(operator_arg_domain(op, i)->N, operator_arg_domain(op, i)->dims));

	auto opdata_test = CAST_MAYBE(op_reshape_s, op->data);
	if (NULL != opdata_test)
		data->x = operator_ref(opdata_test->x);
	else
		data->x = operator_ref(op);

	long strs[N];
	md_calc_strides(N, strs, dims, operator_arg_domain(op, i)->size);

	unsigned int A = operator_nr_args(op);
	unsigned int D[A];
	const long* op_dims[A];
	const long* op_strs[A];

	for (unsigned int j = 0; j < A; j++) {

		auto iov = operator_arg_domain(op, j);
		D[j] = iov->N;
		op_dims[j] = iov->dims;
		op_strs[j] = iov->strs;
	}

	D[i] = N;
	op_dims[i] = dims;
	op_strs[i] = strs;

	return operator_generic_create2(A, op->io_flags, D, op_dims, op_strs, CAST_UP(PTR_PASS(data)), reshape_apply, reshape_free, operator_reshape_get_graph);
}


bool check_simple_copy(const struct operator_s* op)
{
	if (NULL != get_identity_data(op))
		return true;

	if (NULL != get_reshape_data(op))
		return check_simple_copy(get_reshape_data(op)->x);

	return false;
}

const struct operator_s* get_in_reshape(const struct operator_s* op) {

	if (NULL != get_reshape_data(op))
		return get_reshape_data(op)->x;
	
	return NULL;
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

	return operator_generic_create2(1, (bool[1]){ true }, (unsigned int[1]){ N },
			(const long*[1]){ dims },
			(const long*[2]){ strs }, CAST_UP(PTR_PASS(data)), zero_apply, zero_free, NULL);
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
	UNUSED(args);

	assert(1 == N);
}

static void null_free(const operator_data_t* _data)
{
	xfree(CAST_DOWN(null_s, _data));
}

const struct operator_s* operator_null_create2(unsigned int N, const long dims[N], const long strs[N])
{
	PTR_ALLOC(struct null_s, data);
	SET_TYPEID(null_s, data);

	return operator_generic_create2(1, (bool[1]){ false }, (unsigned int[1]){ N },
			(const long*[1]){ dims },
			(const long*[2]){ strs }, CAST_UP(PTR_PASS(data)), null_apply, null_free, NULL);
}

const struct operator_s* operator_null_create(unsigned int N, const long dims[N])
{
	return operator_null_create2(N, dims, MD_STRIDES(N, dims, CFL_SIZE));
}

void operator_generic_apply_unchecked(const struct operator_s* op, unsigned int N, void* args[N])
{
	assert(op->N == N);
	debug_trace("ENTER %p\n", op->apply);
	op->apply(op->data, N, args);
	debug_trace("LEAVE %p\n", op->apply);
}

void operator_generic_apply_parallel_unchecked(int D, const struct operator_s* op[__VLA(D)], unsigned int N, void* args[__VLA(D)][N], int num_threads)
{
#ifdef _OPENMP
	if (0 < num_threads) {

		int max_threads = omp_get_max_threads();
		omp_set_num_threads(num_threads);

		struct cuda_threads_s* gpu_stat = gpu_threads_create(NULL);

		#pragma omp parallel
		{
			gpu_threads_enter(gpu_stat);

			for (int i = omp_get_thread_num(); i < D; i += omp_get_num_threads())
				operator_generic_apply_unchecked(op[i], N, args[i]);

			gpu_threads_leave(gpu_stat);
		}

		gpu_threads_free(gpu_stat);

		omp_set_num_threads(max_threads);
	
		return;
	}
#else
	UNUSED(num_threads);
#endif

	for (int i = 0; i < D; i++)
		operator_generic_apply_unchecked(op[i], N, args[i]);
}


void operator_apply_unchecked(const struct operator_s* op, complex float* dst, const complex float* src)
{
	assert(op->io_flags[0]);
	assert(!op->io_flags[1]);

	operator_generic_apply_unchecked(op, 2, (void*[2]){ (void*)dst, (void*)src });
}

void operator_apply_parallel_unchecked(int D, const struct operator_s* op[D], complex float* dst[D], const complex float* src[D], int num_threads)
{
	void* args[D][2];
	
	for (int i = 0; i < D; i++) {

		assert(op[i]->io_flags[0]);
		assert(!op[i]->io_flags[1]);

		args[i][0] = dst[i];
		args[i][1] = (void*)src[i];
	}

	operator_generic_apply_parallel_unchecked(D, op, 2, args, num_threads);
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

	data->op = operator_ref(op);
	data->ptr = ptr;
	data->del = del;

	return operator_generic_create2(N, op->io_flags, D, dims, strs, CAST_UP(PTR_PASS(data)), attach_fun, attach_del, NULL);
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
	assert(!op->io_flags[arg]);
	assert(iovec_check(operator_arg_domain(op, arg), N, dims, strs));

	unsigned int nn[D - 1];
	const long* ndims[D - 1];
	const long* nstrs[D - 1];

	bool n_flags[D + 1];

	for (unsigned int i = 0; i < D + 1; i++)
		n_flags[i] = false;

	for (unsigned int i = 0, j = 0; i < D; i++) {

		if (arg == i)
			continue;

		nn[j] = operator_arg_domain(op, i)->N;
		ndims[j] = operator_arg_domain(op, i)->dims;
		nstrs[j] = operator_arg_domain(op, i)->strs;

		if (op->io_flags[i])
			n_flags[j] = true;

		j++;
	}

	PTR_ALLOC(struct op_bind_s, data);
	SET_TYPEID(op_bind_s, data);

	data->D = D;
	data->arg = arg;
	data->ptr = ptr;
	data->op = operator_ref(op);

	return operator_generic_create2(D - 1, n_flags,
		nn, ndims, nstrs,
		CAST_UP(PTR_PASS(data)), op_bind_apply, op_bind_del, NULL);
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
		int nr_cuda_devices = cuda_num_devices();
		omp_set_num_threads(nr_cuda_devices * 2);
//		fft_set_num_threads(1);
#else
		error("Both OpenMP and CUDA are necessary for op_loop_fun. At least one was not found.\n");
#endif
	}

	extern bool num_auto_parallelize;
	bool ap_save = num_auto_parallelize;
	num_auto_parallelize = false;

	struct cuda_threads_s* gpu_stat = gpu_threads_create(NULL);

	NESTED(void, op_loop_nary, (void* ptr[]))
	{
		gpu_threads_enter(gpu_stat);
		operator_generic_apply_unchecked(data->op, data->N, ptr);
		gpu_threads_leave(gpu_stat);
	};

	md_parallel_nary(N, data->D, data->dims0, data->parallel, data->strs, args, op_loop_nary);

	gpu_threads_free(gpu_stat);

	num_auto_parallelize = ap_save;
}

static void merge_dims(unsigned int D, long odims[D], const long idims1[D], const long idims2[D])
{
	md_copy_dims(D, odims, idims1);

	for (unsigned int i = 0; i < D; i++) {

		assert((1 == odims[i]) || (1 == idims2[i]));

		if (1 == odims[i])
			odims[i] = idims2[i];
	}
}

const struct operator_s* operator_loop_parallel2(unsigned int N, unsigned int D,
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

	for (int i = 0; i < (int)N; i++) {

		const struct iovec_s* io = operator_arg_domain(op, i);

		assert((int)D == io->N);

		for (int j = 0; j < (int)D; j++) {

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
	data->op = operator_ref(op);

	data->dims0 = *PTR_PASS(dims0);
	data->dims = */*PTR_PASS*/(dims2);
	data->strs = */*PTR_PASS*/(strs2);

	data->parallel = flags;
	data->gpu = gpu;

	const struct operator_s* rop = operator_generic_create2(N, op->io_flags, D2, *dims2, *strs2, CAST_UP(PTR_PASS(data)), op_loop_fun, op_loop_del, NULL);

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

	const void* ref;

	unsigned int N;
	const long** strs;
};

static DEF_TYPEID(copy_data_s);

static void copy_fun(const operator_data_t* _data, unsigned int N, void* args[N])
{
	const auto data = CAST_DOWN(copy_data_s, _data);
	const struct operator_s* op = data->op;
	void* ptr[N];
	bool allocated[N];

	assert(N == operator_nr_args(op));

	for (unsigned int i = 0; i < N; i++) {

		const struct iovec_s* io = operator_arg_domain(op, i);

		allocated[i] = (!md_check_equal_dims(io->N, io->strs, data->strs[i], md_nontriv_dims(io->N, io->dims)) || ((NULL != data->ref) && !md_is_sameplace(data->ref, args[i])));

		if (allocated[i]) {

			ptr[i] = md_alloc_sameplace(io->N, io->dims, io->size, NULL == data->ref ? args[i] : data->ref);

			if (!op->io_flags[i])
				md_copy2(io->N, io->dims, io->strs, ptr[i], data->strs[i], args[i], io->size);
		} else {

			ptr[i] = args[i];
		}
	}

	operator_generic_apply_unchecked(op, N, ptr);

	for (unsigned int i = 0; i < N; i++) {

		const struct iovec_s* io = operator_arg_domain(op, i);

		if (op->io_flags[i] && allocated[i])
			md_copy2(io->N, io->dims, data->strs[i], args[i], io->strs, ptr[i], io->size);

		if (allocated[i])
			md_free(ptr[i]);
	}
}

static void copy_del(const operator_data_t* _data)
{
	const auto data = CAST_DOWN(copy_data_s, _data);

	operator_free(data->op);
	md_free(data->ref);

	for (unsigned int i = 0; i < data->N; i++)
		xfree(data->strs[i]);

	xfree(data->strs);
	xfree(data);
}

const struct operator_s* operator_copy_wrapper_sameplace(unsigned int N, const long* strs[N], const struct operator_s* op, const void* ref)
{
	assert(N == operator_nr_args(op));

	PTR_ALLOC(struct copy_data_s, data);
	SET_TYPEID(copy_data_s, data);
	data->op = operator_ref(op);

	unsigned int D[N];
	const long* dims[N];
	const long* (*strs2)[N] = TYPE_ALLOC(const long*[N]);

	for (int i = 0; i < (int)N; i++) {

		const struct iovec_s* io = operator_arg_domain(op, i);

		D[i] = io->N;
		dims[i] = io->dims;

		long (*strsx)[io->N] = TYPE_ALLOC(long[io->N]);
		md_copy_strides(io->N, *strsx, strs[i]);
		(*strs2)[i] = *strsx;

		// check for trivial strides

		long tstrs[io->N];
		md_calc_strides(io->N, tstrs, io->dims, CFL_SIZE);

		for (int i = 0; i < io->N; i++)
			if (1 != io->dims[i])
				assert(io->strs[i] == tstrs[i]);
	}

	data->N = N;
	data->strs = *strs2;
	data->ref = (NULL == ref) ? NULL : md_alloc_sameplace(1, MD_DIMS(1), FL_SIZE, ref);

	return operator_generic_create2(N, op->io_flags, D, dims, *strs2, CAST_UP(PTR_PASS(data)), copy_fun, copy_del, NULL);
}

const struct operator_s* operator_copy_wrapper(unsigned int N, const long* strs[N], const struct operator_s* op)
{
	return operator_copy_wrapper_sameplace(N, strs, op, NULL);
}

const struct operator_s* operator_cpu_wrapper(const struct operator_s* op)
{
	int ref = 1;
	int N = operator_nr_args(op);

	const long* strs[N];

	for (int i = 0; i < N; i++) {

		auto dom = operator_arg_domain(op, i);
		assert(md_check_equal_dims(dom->N, MD_STRIDES(dom->N, dom->dims, dom->size), dom->strs, ~0));
		strs[i] = dom->strs;
	}

	return operator_copy_wrapper_sameplace(N, strs, op, &ref);
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
omp_nest_lock_t gpulock[MAX_CUDA_DEVICES];
#endif


static void gpuwrp_fun(const operator_data_t* _data, unsigned int N, void* args[N])
{
#if defined(USE_CUDA) && defined(_OPENMP)
	const auto data = CAST_DOWN(gpu_data_s, _data);
	const auto op = data->op;
	void* gpu_ptr[N];

	assert(N == operator_nr_args(op));

	debug_printf(DP_DEBUG3, "GPU start.\n");

	int nr_cuda_devices = MIN(cuda_num_devices(), MAX_CUDA_DEVICES);
	int gpun = omp_get_thread_num() % nr_cuda_devices;

	cuda_set_device(gpun);

	for (unsigned int i = 0; i < N; i++) {

		if ((!MD_IS_SET(data->move_flags, i) || cuda_ondevice(args[i]))) {

			gpu_ptr[i] = args[i];
			continue;
		}

		const struct iovec_s* io = operator_arg_domain(op, i);

		if (op->io_flags[i])
			gpu_ptr[i] = md_alloc_gpu(io->N, io->dims, io->size);
		else
			gpu_ptr[i] = md_gpu_move(io->N, io->dims, args[i], io->size);
	}

	omp_set_nest_lock(&gpulock[gpun]);
	operator_generic_apply_unchecked(op, N, gpu_ptr);
	omp_unset_nest_lock(&gpulock[gpun]);

	for (unsigned int i = 0; i < N; i++) {

		if (gpu_ptr[i] == args[i])
			continue;

		const struct iovec_s* io = operator_arg_domain(op, i);

		if (op->io_flags[i])
			md_copy(io->N, io->dims, args[i], gpu_ptr[i], io->size);

		md_free(gpu_ptr[i]);
	}

	debug_printf(DP_DEBUG3, "GPU end.\n");

#else
	UNUSED(_data); UNUSED(N); UNUSED(args);
	error("Both OpenMP and CUDA are necessary for automatic GPU execution. At least one was not found.\n");
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
	assert(N <= 8 * sizeof(move_flags));

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

	return operator_generic_create2(N, op->io_flags, D, dims, strs, CAST_UP(PTR_PASS(data)), gpuwrp_fun, gpuwrp_del, NULL);
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

static const struct graph_s* operator_combi_get_graph(const struct operator_s* op)
{
	const auto d = CAST_DOWN(operator_combi_s, op->data);
	const struct graph_s* tmp_graphs[d->N];
	for (int i = 0; i < d->N; i++)
		tmp_graphs[i] = operator_get_graph(d->x[i]);
	return operator_graph_combine_F(d->N, tmp_graphs);
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

	bool io_flags[A];
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

			io_flags[a] = x[i]->io_flags[j];
			a++;
		}
	}

	return operator_generic_create2(A, io_flags, D, dims, strs, CAST_UP(PTR_PASS(c)), combi_apply, combi_free, operator_combi_get_graph);
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

static const struct graph_s* operator_dup_get_graph(const struct operator_s* op)
{
	const auto d = CAST_DOWN(operator_dup_s, op->data);
	return operator_graph_dup_F(operator_get_graph(d->x), d->a, d->b);
}

const struct operator_s* operator_dup_create(const struct operator_s* op, unsigned int a, unsigned int b)
{
	unsigned int N = operator_nr_args(op);

	assert(a < N);
	assert(b < N);
	assert(a != b);

	bool io_flags[N - 1];
	unsigned int D[N - 1];
	const long* dims[N - 1];
	const long* strs[N - 1];

	debug_printf(DP_DEBUG4, "Duplicating args %d-%d of %d.\n", a, b, N);

	for (unsigned int s = 0, t = 0; s < N; s++) {

		if (s == b)
			continue;

		auto io = operator_arg_domain(op, s);

		D[t] = io->N;
		dims[t] = io->dims;
		strs[t] = io->strs;

		io_flags[t] = op->io_flags[s];

		t++;
	}

	auto ioa = operator_arg_domain(op, a);
	auto iob = operator_arg_domain(op, b);

	assert((int)ioa->N == md_calc_blockdim(ioa->N, ioa->dims, ioa->strs, ioa->size));
	assert((int)iob->N == md_calc_blockdim(iob->N, iob->dims, iob->strs, iob->size));


	PTR_ALLOC(struct operator_dup_s, data);
	SET_TYPEID(operator_dup_s, data);

	data->a = a;
	data->b = b;
	data->x = operator_ref(op);

	return operator_generic_create2(N - 1, io_flags, D, dims, strs, CAST_UP(PTR_PASS(data)), dup_apply, dup_del, operator_dup_get_graph);
}




// FIXME: we should reimplement link in terms of dup and bind (caveat: gpu; io_flags)
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

#ifdef USE_CUDA
	// Allocate tmp on GPU when one argument is on the GPU.
	// The scalar parameters of op_p may be on CPU.

	bool gpu = false;

	for (unsigned int i = 0; i < N; i++)
		gpu = gpu || cuda_ondevice(args[i]);

	void* tmp = (gpu ? md_alloc_gpu : md_alloc)(iov->N, iov->dims, iov->size);
#else
	void* tmp = md_alloc(iov->N, iov->dims, iov->size);
#endif

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

static const struct graph_s* operator_link_get_graph(const struct operator_s* op)
{
	const auto d = CAST_DOWN(operator_link_s, op->data);
	return operator_graph_link_F(operator_get_graph(d->x), d->b, d->a);
}

const struct operator_s* operator_link_create(const struct operator_s* op, unsigned int o, unsigned int i)
{
	unsigned int N = operator_nr_args(op);

	assert(i < N);
	assert(o < N);
	assert(i != o);

	assert( op->io_flags[o]);
	assert(!op->io_flags[i]);

	auto iov = operator_arg_domain(op, i);
	auto iovb = operator_arg_domain(op, o);

	assert(iovec_check(iovb, iov->N, iov->dims, iov->strs));

	bool io_flags[N - 2];
	unsigned int D[N - 2];
	const long* dims[N - 2];
	const long* strs[N - 2];

	debug_printf(DP_DEBUG4, "Linking args %d-%d of %d.\n", i, o, N);

	for (unsigned int s = 0, t = 0; s < N; s++) {

		if ((s == i) || (s == o))
			continue;

		auto io = operator_arg_domain(op, s);

		D[t] = io->N;
		dims[t] = io->dims;
		strs[t] = io->strs;

		io_flags[t] = op->io_flags[s];

		t++;
	}

	auto ioi = operator_arg_domain(op, i);
	auto ioo = operator_arg_domain(op, o);

	assert((int)ioi->N == md_calc_blockdim(ioi->N, ioi->dims, ioi->strs, ioi->size));
	assert((int)ioo->N == md_calc_blockdim(ioo->N, ioo->dims, ioo->strs, ioo->size));


	PTR_ALLOC(struct operator_link_s, data);
	SET_TYPEID(operator_link_s, data);

	data->a = i;
	data->b = o;
	data->x = operator_ref(op);

	return operator_generic_create2(N - 2, io_flags, D, dims, strs, CAST_UP(PTR_PASS(data)), link_apply, link_del, operator_link_get_graph);
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

static const struct graph_s* operator_permute_get_graph(const struct operator_s* op)
{
	const auto d = CAST_DOWN(permute_data_s, op->data);
	return operator_graph_permute_F(operator_get_graph(d->op), d->op->N, d->perm);
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
	bool io_flags[N];
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

		io_flags[i] = op->io_flags[perm[i]];
	}

	assert(((int)sizeof(flags) <= N) || (MD_BIT(N) == flags + 1));

	// op = operator_ref(op);
	PTR_ALLOC(struct permute_data_s, data);
	SET_TYPEID(permute_data_s, data);

	int* nperm = *TYPE_ALLOC(int[N]);

	auto opdata_test = CAST_MAYBE(permute_data_s, op->data);
	if (NULL != opdata_test) {

		data->op = operator_ref(opdata_test->op);
		for (int i = 0; i < N; i++)
			nperm[i] = opdata_test->perm[perm[i]];

	} else {

		data->op = operator_ref(op);
		memcpy(nperm, perm, sizeof(int[N]));
	}
	data->perm = nperm;

	return operator_generic_create2(N, io_flags, D, dims, strs, CAST_UP(PTR_PASS(data)), permute_fun, permute_del, operator_permute_get_graph);
}


struct extract_data_s {

	INTERFACE(operator_data_t);

	int a;
	size_t off;
	const struct operator_s* op;

	long* strs;
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


static void extract_copy_fun(const operator_data_t* _data, unsigned int N, void* args[N])
{
	auto data = CAST_DOWN(extract_data_s, _data);

	assert(N == operator_nr_args(data->op));

	void* ptr[N];

	for (int i = 0; i < (int)N; i++)
		ptr[i] = args[i];

	auto iov = operator_arg_domain(data->op, data->a);

	ptr[data->a] = md_alloc_sameplace(iov->N, iov->dims, iov->size, args[data->a]);

	if (!(data->op->io_flags[data->a]))
		md_copy2(iov->N, iov->dims, iov->strs, ptr[data->a], data->strs, args[data->a] + data->off, iov->size);

	operator_generic_apply_unchecked(data->op, N, ptr);

	if ((data->op->io_flags[data->a]))
		md_copy2(iov->N, iov->dims, data->strs, args[data->a] + data->off, iov->strs, ptr[data->a], iov->size);

	md_free(ptr[data->a]);
}


static void extract_del(const operator_data_t* _data)
{
	auto data = CAST_DOWN(extract_data_s, _data);

	operator_free(data->op);

	xfree(data->strs);

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

	auto ioa = operator_arg_domain(op, a);
	bool trivial_strides = iovec_check(ioa, ioa->N, ioa->dims, MD_STRIDES(ioa->N, ioa->dims, ioa->size));
	bool copy_needed = false;

	for (int i = 0; i < Da; i++) {

		assert((0 <= pos[i]) && (pos[i] < dimsa[i]));
		assert(dims[a][i] + pos[i] <= dimsa[i]);
		copy_needed = copy_needed || !((0 == strs[a][i]) || (strs[a][i] == strsa[i]));
	}

	assert(trivial_strides || !copy_needed);

	dims[a] = dimsa;
	strs[a] = strsa;

	PTR_ALLOC(struct extract_data_s, data);
	SET_TYPEID(extract_data_s, data);

	data->op = operator_ref(op);
	data->a = a;
	data->off = md_calc_offset(Da, strsa, pos);

	PTR_ALLOC(long[Da], nstrs);
	md_copy_strides(Da, *nstrs, strsa);
	data->strs = *PTR_PASS(nstrs);

	return operator_generic_create2(N, op->io_flags, D, dims, strs, CAST_UP(PTR_PASS(data)), copy_needed ? extract_copy_fun : extract_fun, extract_del, NULL);
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

	for (unsigned int i = D + 1; i < N; i++)
		if (((a->dims[i] != 1) && (a->strs[i] != 0)) || ((b->dims[i] != 1) && (b->strs[i] != 0)))
			return false;

	if ((1 != a->dims[D]) && (1 != b->dims[D]))
		if (a->strs[D] != b->strs[D])
			return false;

	long dims[N];
	md_select_dims(N, ~MD_BIT(D), dims, a->dims);

	long S = md_calc_size(N, dims) * a->size;

	if ((1 != a->dims[D]) && (S != a->strs[D]))
		return false;


	return true;
}

static bool stack_compatible_copy(unsigned int D, const struct iovec_s* a, const struct iovec_s* b)
{
	if (a->N != b->N)
		return false;

	unsigned int N = a->N;

	if (!iovec_check(a, a->N, a->dims, MD_STRIDES(a->N, a->dims, a->size)))
		return false;
	if (!iovec_check(b, b->N, b->dims, MD_STRIDES(b->N, b->dims, b->size)))
		return false;

	for (unsigned int i = 0; i < N; i++)
		if ((D != i) && (a->dims[i] != b->dims[i]))
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

static void stack_dims_trivial(unsigned int N, long dims[N], long strs[N], unsigned int D, const struct iovec_s* a, const struct iovec_s* b)
{
	md_copy_dims(N, dims, a->dims);
	dims[D] = a->dims[D] + b->dims[D];
	md_calc_strides(N, strs, dims, a->size);
}

/**
 * Stacks two operators over selected arguments, the other arguments are duplex
 *
 * @param M nr of arguments to stack over
 * @param arg_list index of arguments to stack
 * @param dim_list stack dimension for respective argument
 * @param a first operator to stack
 * @param b second operator to stack
 **/
const struct operator_s* operator_stack2(int M, const int arg_list[M], const int dim_list[M], const struct operator_s* a, const struct operator_s* b)
{
	a = operator_ref(a);
	b = operator_ref(b);

	for (int m = 0; m < M; m++) {

		int arg = arg_list[m];
		int dim = dim_list[m];

		auto ia = operator_arg_domain(a, arg);
		auto ib = operator_arg_domain(b, arg);

		assert(stack_compatible(dim, ia, ib) || stack_compatible_copy(dim, ia, ib));

		int D = ia->N;

		long dims[D];
		long strs[D];

		if (stack_compatible(dim, ia, ib))
			stack_dims(D, dims, strs, dim, ia, ib);
		else
			stack_dims_trivial(D, dims, strs, dim, ia, ib);

		long pos[D];

		for (int i = 0; i < D; i++)
			pos[i] = 0;

		int X = ia->dims[dim];

		auto aa = operator_extract_create2(a, arg, D, dims, strs, (pos[dim] = 0, pos));
		auto bb = operator_extract_create2(b, arg, D, dims, strs, (pos[dim] = X, pos));

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

struct operator_sum_s {

	INTERFACE(operator_data_t);

	int II;

	const struct iovec_s* iov;
};

static DEF_TYPEID(operator_sum_s);

static void sum_apply(const operator_data_t* _data, unsigned int N, void* args[N])
{
	auto d = CAST_DOWN(operator_sum_s, _data);

	assert(1 + d->II == (int)N);
	assert(3 <= N);

	void* dst = args[0];
	void* src1 = args[1];
	void* src2 = args[2];

	md_zadd2(d->iov->N, d->iov->dims, d->iov->strs, dst, d->iov->strs, src1, d->iov->strs, src2);

	for (int i = 2; i < d->II; i++)
		md_zadd2(d->iov->N, d->iov->dims, d->iov->strs, dst, d->iov->strs, dst, d->iov->strs, args[1 + i]);
}

static void sum_free(const operator_data_t* _data)
{
	auto data = CAST_DOWN(operator_sum_s, _data);

	iovec_free(data->iov);

	xfree(data);
}

const struct operator_s* operator_zadd_create(int II, int N, const long dims[N])
{
	PTR_ALLOC(struct operator_sum_s, c);
	SET_TYPEID(operator_sum_s, c);

	c->iov = iovec_create(N, dims, CFL_SIZE);
	c->II = II;
	assert(2 <= II);

	bool io_flags[1 + II];
	unsigned int D [1 + II];
	const long* dims_op[1 + II];
	const long* strs_op[1 + II];

	io_flags[0] = true;
	D[0] = N;
	dims_op[0] = c->iov->dims;
	strs_op[0] = c->iov->strs;

	for (int i = 0; i < II; i++) {

		io_flags[i + 1] = false;
		D[i + 1] = N;
		dims_op[i + 1] = c->iov->dims;
		strs_op[i + 1] = c->iov->strs;
	}

	return operator_generic_create2(1 + II, io_flags, D, dims_op, strs_op, CAST_UP(PTR_PASS(c)), sum_apply, sum_free, NULL);
}

bool operator_is_zadd(const struct operator_s* op)
{
	return (NULL != CAST_MAYBE(operator_sum_s, op->data));
}

/**
 * For simple operators with one in and one output, we provide "flat containers".
 * These operators can be optimized (simultaneous application) easily.
 * There are two types of containers, i.e. chains and sums.
 * */

struct operator_plus_s {

	INTERFACE(operator_data_t);

	const struct operator_s* a;
	const struct operator_s* b;
};

static DEF_TYPEID(operator_plus_s);

static void plus_apply(const operator_data_t* _data, unsigned int N, void* args[N])
{
	auto data = CAST_DOWN(operator_plus_s, _data);

	assert(2 == N);

	void* src = args[1];
	void* dst = args[0];

	auto iov = operator_codomain(data->b);
	complex float* tmp = md_alloc_sameplace(iov->N, iov->dims, iov->size, src);

	operator_apply_joined_unchecked(2, MAKE_ARRAY(data->a, data->b), MAKE_ARRAY((complex float*)args[0], tmp), args[1]);

	md_zadd2(iov->N, iov->dims, iov->strs, dst,iov->strs, dst, iov->strs, tmp);
	md_free(tmp);
}

static void plus_free(const operator_data_t* _data)
{
	auto data = CAST_DOWN(operator_plus_s, _data);

	operator_free(data->a);
	operator_free(data->b);

	xfree(data);
}

static const struct graph_s* operator_plus_get_graph(const struct operator_s* op)
{
	const auto d = CAST_DOWN(operator_plus_s, op->data);

	auto graph_a = operator_get_graph(d->a);
	auto graph_b = operator_get_graph(d->b);

	auto iov = operator_codomain(d->a);
	auto op_sum = operator_zadd_create(2, iov->N, iov->dims);
	auto graph_sum = operator_get_graph(op_sum);
	operator_free(op_sum);

	auto graph_combi = operator_graph_combine_F(3, (const struct graph_s*[3]){graph_sum, graph_a, graph_b});

	graph_combi = operator_graph_link_F(graph_combi, 5, 2);
	graph_combi = operator_graph_link_F(graph_combi, 2, 1);
	graph_combi = operator_graph_dup_F(graph_combi, 1, 2);

	return graph_combi;
}

/**
 * Create a new operator that adds the output of two
 */
const struct operator_s* operator_plus_create(const struct operator_s* a, const struct operator_s* b)
{
	//check compatibility
	assert((2 == a->N) && (2 == b->N));
	assert(a->io_flags[0]);
	assert(!a->io_flags[1]);
	assert(b->io_flags[0]);
	assert(!b->io_flags[1]);

	auto doma = operator_domain(a);
	auto domb = operator_domain(b);

	assert(iovec_check(doma, domb->N, domb->dims, domb->strs));
	assert(doma->size == domb->size);

	auto codoma = operator_codomain(a);
	auto codomb = operator_codomain(b);

	assert(iovec_check(codoma, codomb->N, codomb->dims, codomb->strs));
	assert(codoma->size == codomb->size);
	//endcheck compatibility

	PTR_ALLOC(struct operator_plus_s, c);
	SET_TYPEID(operator_plus_s, c);

	c->a = operator_ref(a);
	c->b = operator_ref(b);

	bool io_flags[2] = { true, false };
	unsigned int D[] = { codoma->N, doma->N };
	const long* dims[] = { codoma->dims, doma->dims };
	const long* strs[] = { codoma->strs, doma->strs };

	return operator_generic_create2(2, io_flags, D, dims, strs, CAST_UP(PTR_PASS(c)), plus_apply, plus_free, operator_plus_get_graph);
}


static const struct operator_plus_s* get_plus_data(const struct operator_s* plus_op)
{
	return CAST_MAYBE(operator_plus_s, operator_get_data(plus_op));
}

struct operator_chain_s {

	INTERFACE(operator_data_t);

	unsigned int N;
	const struct operator_s** x;
};

static DEF_TYPEID(operator_chain_s);

static void chain_apply(const operator_data_t* _data, unsigned int N, void* args[N])
{
	auto data = CAST_DOWN(operator_chain_s, _data);

	assert(2 == N);

	complex float* src = args[1];

	for(unsigned int i = 0; i < data->N; i++) {

		auto iov = operator_codomain(data->x[i]);
		complex float* dst;

		if (i == data->N - 1)
			dst = args[0];
		else
			dst = md_alloc_sameplace(iov->N, iov->dims, iov->size, src);

		operator_apply_unchecked(data->x[i], dst, src);

		if (0 != i)
			md_free(src);

		src = dst;
	}
}

static void chain_free(const operator_data_t* _data)
{
	auto data = CAST_DOWN(operator_chain_s, _data);

	for (unsigned int i = 0; i < data->N; i++)
		operator_free(data->x[i]);

	xfree(data->x);
	xfree(data);
}

static const struct graph_s* operator_chain_get_graph(const struct operator_s* op)
{
	const auto d = CAST_DOWN(operator_chain_s, op->data);
	const struct graph_s* tmp_graphs[d->N];
	for (unsigned int i = 0; i < d->N; i++)
		tmp_graphs[i] = operator_get_graph(d->x[i]);
	return operator_graph_chain_F(d->N, tmp_graphs);
}

/**
 * Create a new operator that chaines several others
 */
const struct operator_s* operator_chainN(unsigned int N, const struct operator_s* x[N])
{
	debug_printf(DP_DEBUG4, "operator chainN N=%d:\n", N);
	for (unsigned int i = 0; i < N; i++) {

		assert(2 == x[i]->N);
		assert(x[i]->io_flags[0]);
		assert(!x[i]->io_flags[1]);

		if ((signed)i < (signed)N - 1) {

			auto a = x[i];
			auto b = x[i + 1];

			debug_printf(DP_DEBUG4, "\t[%d] in [%d]:\n\t", i, i + 1, N);
			debug_print_dims(DP_DEBUG4, a->domain[0]->N, a->domain[0]->dims);
			debug_printf(DP_DEBUG4, "\t");
			debug_print_dims(DP_DEBUG4, b->domain[1]->N, b->domain[1]->dims);

			assert(a->domain[0]->N == b->domain[1]->N);
			assert(md_calc_size(a->domain[0]->N, a->domain[0]->dims) == md_calc_size(b->domain[1]->N, b->domain[1]->dims));
			assert((int)a->domain[0]->N == md_calc_blockdim(a->domain[0]->N, a->domain[0]->dims, a->domain[0]->strs, a->domain[0]->size));
			assert((int)b->domain[1]->N == md_calc_blockdim(b->domain[1]->N, b->domain[1]->dims, b->domain[1]->strs, b->domain[1]->size));
		}
	}

	PTR_ALLOC(struct operator_chain_s, c);
	SET_TYPEID(operator_chain_s, c);

	PTR_ALLOC(const struct operator_s*[N], xp);

	for (unsigned int i = 0; i < N; i++)
		(*xp)[i] = operator_ref(x[i]);

	c->x = *PTR_PASS(xp);
	c->N = N;

	return operator_generic_create2(2, (bool[2]){true, false}, (unsigned int[2]){ operator_codomain(x[N - 1])->N, operator_domain(x[0])->N},
						(const long*[2]){ operator_codomain(x[N - 1])->dims, operator_domain(x[0])->dims},
						(const long*[2]){ operator_codomain(x[N - 1])->strs, operator_domain(x[0])->strs},
						CAST_UP(PTR_PASS(c)), chain_apply, chain_free, operator_chain_get_graph);
}

static const struct operator_chain_s* get_chain_data(const struct operator_s* chain_op)
{
	return CAST_MAYBE(operator_chain_s, operator_get_data(chain_op));
}


/**
 * Create a new operator that first applies a, then applies b:
 * c(x) = b(a(x))
 */
const struct operator_s* operator_chain(const struct operator_s* a, const struct operator_s* b)
{
#if 1
	if (NULL != get_plus_data(b)) {

		auto bd = get_plus_data(b);
		auto tmpa = operator_chain(a, bd->a);
		auto tmpb = operator_chain(a, bd->b);

		auto result = operator_plus_create(tmpa, tmpb);

		operator_free(tmpa);
		operator_free(tmpb);

		return result;
	}
#endif

	//Get array of operators in a
	unsigned int Na = 1;
	const struct operator_s** ops_a;

	if (NULL != get_chain_data(a)) {

		auto ad = get_chain_data(a);

		Na = ad->N;
		ops_a = ad->x;

	} else {

		ops_a = &a;
	}

	//Get array of operators in b
	unsigned int Nb = 1;
	const struct operator_s** ops_b;

	if (NULL != get_chain_data(b)) {

		auto bd = get_chain_data(b);

		Nb = bd->N;
		ops_b = bd->x;

	} else {

		ops_b = &b;
	}

	unsigned int N = Na + Nb;
	const struct operator_s* ops[N];

	for (unsigned int i = 0; i < Na; i++)
		ops[i] = ops_a[i];

	for (unsigned int i = 0; i < Nb; i++)
		ops[i + Na] = ops_b[i];

	return operator_chainN(N, ops);
}




//get list of all operators applied in one operator
list_t operator_get_list(const struct operator_s* op) {

	auto data_combi = CAST_MAYBE(operator_combi_s, op->data);
	auto data_link = CAST_MAYBE(operator_link_s, op->data);
	auto data_dup = CAST_MAYBE(operator_dup_s, op->data);
	auto data_reshape = CAST_MAYBE(op_reshape_s, op->data);
	auto data_chain = CAST_MAYBE(operator_chain_s, op->data);
	auto data_perm = CAST_MAYBE(permute_data_s, op->data);
	auto data_plus = CAST_MAYBE(operator_plus_s, op->data);
	auto data_copy = CAST_MAYBE(copy_data_s, op->data);
	auto data_attach = CAST_MAYBE(attach_data_s, op->data);

	if (NULL != data_combi) {

		list_t result = list_create();
		for (int i = 0; i < data_combi->N; i++)
			list_merge(result, operator_get_list(data_combi->x[i]), true);
		return result;
	}

	if (NULL != data_chain) {

		list_t result = list_create();
		for (unsigned int i = 0; i < data_chain->N; i++)
			list_merge(result, operator_get_list(data_chain->x[i]), true);
		return result;
	}

	if (NULL != data_link) {

		return operator_get_list(data_link->x);
	}

	if (NULL != data_dup) {

		return operator_get_list(data_dup->x);
	}

	if (NULL != data_reshape) {

		return operator_get_list(data_reshape->x);
	}

	if (NULL != data_perm) {

		return operator_get_list(data_perm->op);
	}

	if (NULL != data_perm) {

		return operator_get_list(data_perm->op);
	}

	if (NULL != data_plus) {

		list_t result = operator_get_list(data_plus->a);
		list_merge(result, operator_get_list(data_plus->b), true);
		return result;
	}

	if (NULL != data_copy) {

		return operator_get_list(data_copy->op);
	}

	if (NULL != data_attach) {

		return operator_get_list(data_attach->op);
	}

	list_t result = list_create();
	list_append(result, (void*)op);

	return result;
}


const struct graph_s* operator_get_graph(const struct operator_s* op)
{
	if (NULL != op->get_graph)
		return op->get_graph(op);

	return create_graph_operator(op, op->data->TYPEID->name);
}



const struct operator_s* graph_optimize_operator_F(const struct operator_s* op)
{

	int N = operator_nr_out_args(op);

	auto tmp = operator_ref(op);

	bool sorted = true;
	for (int i = 0; i < N; i++)
		sorted = sorted && op->io_flags[i];

	for (int i = 0; sorted && i < N; i++) {

		auto iov = tmp->domain[N-1];
		auto id = operator_identity_create(iov->N, iov->dims);
		auto ttmp = operator_combi_create(2, (const struct operator_s*[2]){id, tmp});
		operator_free(tmp);
		operator_free(id);
		tmp = operator_link_create(ttmp, N + 1, 1);
		operator_free(ttmp);
	}

	auto graph = operator_get_graph(tmp);
	operator_free(tmp);
	operator_free(op);

	return graph_to_operator_F(graph);
}

bool operator_identify(const struct operator_s* a, const struct operator_s* b)
{
	if (a == b)
		return true;
	if (a->data->TYPEID != b->data->TYPEID)
		return false;

	if (NULL != CAST_MAYBE(operator_sum_s, a->data)) {

		auto iova = CAST_DOWN(operator_sum_s, a->data)->iov;
		auto iovb = CAST_DOWN(operator_sum_s, b->data)->iov;

		return iovec_check(iova, iovb->N, iovb->dims, iovb->strs);
	}

	return false;
}

void operator_apply_joined_unchecked(unsigned int N, const struct operator_s* op[N], complex float* dst[N], const complex float* src)
{
	auto combi = operator_combi_create(N, op);

	int perm[2 * N];
	for (unsigned int i = 0; i < N; i++) {

		perm[i] = 2 * i;
		perm[i + N] = 2 * i + 1;
	}

	auto dup = operator_permute(combi, 2 * N, perm);
	operator_free(combi);

	for (unsigned int i = 0; i < N - 1; i++) {

		auto tmp = operator_dup_create(dup, N, N + 1);
		operator_free(dup);
		dup = tmp;
	}

	void* args[N + 1];
	for (unsigned int i = 0; i < N; i++)
		args[i] = dst[i];
	args[N] = (void*)src;


	auto op_optimized = graph_optimize_operator_F(dup);

	operator_generic_apply_unchecked(op_optimized, N + 1, args);

	operator_free(op_optimized);

}
