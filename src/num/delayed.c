/* Copyright 2024. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 *
*/

#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <complex.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "misc/list.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/shrdptr.h"
#include "misc/egraph.h"
#include "misc/types.h"
#include "misc/mmio.h"

#include "num/init.h"
#include "num/flpmath.h"
#include "num/multind.h"
#include "num/vecops.h"
#include "num/optimize.h"
#include "num/vptr.h"
#include "num/vptr_fun.h"
#include "num/mpi_ops.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "delayed.h"

#define MAX_FLAGS 512
typedef struct flag_s {	unsigned long flags[MAX_FLAGS / sizeof(unsigned long)]; } flag_t;
flag_t flag_none = { .flags = { [0 ... MAX_FLAGS / sizeof(unsigned long) - 1] = 0 } };

#define BITMASK(b) ({ int _b = (b); 1UL << (_b % (int)sizeof(unsigned long)); })
#define BITSLOT(b) ({ int _b = (b); _b / (int)sizeof(unsigned long); })
#define BITSET(a, b) ((a).flags[BITSLOT(b)] |= BITMASK(b))
#define BITCLEAR(a, b) ((a).flags[BITSLOT(b)] &= ~BITMASK(b))
#define BITTEST(a, b) ((a).flags[BITSLOT(b)] & BITMASK(b))

#define BIT(i) \
({					\
	int _i = (i);			\
	flag_t _flag = flag_none;	\
	BITSET(_flag, _i);		\
	_flag;				\
})

#define UNUSED(x) (void)x

struct queue_s {

#ifdef _OPENMP
	omp_lock_t lock;
#endif
	bool compute;
	list_t ops;
};

typedef struct queue_s* queue_t;

#define MAX_WORKER 128
static queue_t global_queue[MAX_WORKER] = { [0 ... MAX_WORKER - 1] = NULL };

static void queue_init(void)
{
	if (NULL == global_queue[cfl_loop_worker_id()]) {

#pragma omp critical
		if (NULL == global_queue[cfl_loop_worker_id()]) {

			auto tmp = TYPE_ALLOC(struct queue_s);
			tmp->ops = list_create();
			tmp->compute = false;
#ifdef _OPENMP
			omp_init_lock(&(tmp->lock));
#endif
			global_queue[cfl_loop_worker_id()] = tmp;
		}
	}
}

static void queue_set_lock(queue_t queue)
{
#ifdef _OPENMP
	omp_set_lock(&(queue->lock));
#else
	UNUSED(queue);
#endif
}

static void queue_unset_lock(queue_t queue)
{
#ifdef _OPENMP
	omp_unset_lock(&(queue->lock));
#else
	UNUSED(queue);
#endif
}

queue_t get_global_queue(void)
{
	queue_init();

	queue_t queue = global_queue[cfl_loop_worker_id()];
	queue_set_lock(queue);

	return queue;
}

void release_global_queue(queue_t queue)
{
	queue_unset_lock(queue);
}

list_t get_delayed_op_list(struct queue_s* queue)
{
	return queue->ops;
}

void queue_set_compute(queue_t queue, bool compute)
{
	queue->compute = compute;
}


int delayed_dl = DP_DEBUG4;
const char* debug_graph_path = NULL; //"/home/mblum/Bart/bart/";

struct delayed_op_s;
typedef struct delayed_op_s delayed_op_t;

typedef void (*delayed_op_fun_t)(delayed_op_t* op, unsigned long slice_flag, long index);
typedef const char* (*delayed_op_debug_t)(delayed_op_t* op, bool nested);
typedef void (*delayed_op_del_t)(const delayed_op_t* op);

struct delayed_op_arg_s {

	void* ptr;
	void* ptr_base;

	int N;
	long* adims;
	long* astrs;

	long* mdims;
	long* mstrs;
	long* mpos;

	unsigned long sflags;
	unsigned long lflags;

	bool simple;
	bool fitting;
};

static struct delayed_op_arg_s arg_create(int N, const long dims[N], const long _strs[N], const void* ptr, size_t size)
{
	struct delayed_op_arg_s arg;

	arg.ptr = (void*)ptr;
	arg.ptr_base = (void*)ptr - vptr_get_offset(ptr);

	arg.N = 1 + MAX(N, vptr_get_N(ptr));

	long strs[N];
	md_select_strides(N, md_nontriv_dims(N, dims), strs, _strs);

	long adims[arg.N];
	md_singleton_dims(arg.N, adims);
	adims[0] = (long)size;
	md_copy_dims(N, adims + 1, dims);
	arg.adims = ARR_CLONE(long[arg.N], adims);

	long astrs[arg.N];
	md_singleton_strides(arg.N, astrs);
	astrs[0] = 1 < size ? 1 : 0;
	md_select_strides(N, md_nontriv_dims(N, dims), astrs + 1, strs);
	arg.astrs = ARR_CLONE(long[arg.N], astrs);

	check_vptr_valid_access(arg.N, arg.adims, arg.astrs, ptr, 1);

	long mdims[arg.N];
	md_singleton_dims(arg.N, mdims);
	vptr_get_dims(ptr, vptr_get_N(ptr), mdims + 1);
	mdims[0] = (long)vptr_get_size(ptr);
	arg.mdims = ARR_CLONE(long[arg.N], mdims);

	long mpos[arg.N];
	md_set_dims(arg.N, mpos, 0);
	md_unravel_index(arg.N, mpos, ~0ul, arg.mdims, (arg.ptr - arg.ptr_base));
	arg.mpos = ARR_CLONE(long[arg.N], mpos);

	long mstrs[arg.N];
	md_calc_strides(arg.N, mstrs, mdims, 1);
	arg.mstrs = ARR_CLONE(long[arg.N], mstrs);

	unsigned long aflags[arg.N];
	loop_access_dims(arg.N, aflags, arg.adims, arg.astrs, arg.N, arg.mdims, (arg.ptr - arg.ptr_base));

	arg.sflags = md_nontriv_dims(arg.N, arg.mdims);
	for (int k = 0; k < arg.N; k++)
		arg.sflags &= ~aflags[k];

	arg.lflags = 0ul;
	for (int k = 0; k <arg.N; k++)
		if (aflags[k] == MD_BIT(k))
			arg.lflags |= MD_BIT(k);

	for (int k = 0; k < arg.N; k++)
		if (aflags[k] != MD_BIT(k))
			arg.lflags &= ~aflags[k];


	long tdims[arg.N];
	long tstrs[arg.N];
	md_copy_dims(arg.N, tdims, adims);
	md_copy_dims(arg.N, tstrs, astrs);

	for (int i = 0; i < arg.N; i++) {

		if ((tstrs[i] == tdims[0]) && (tdims[0] * tdims[i] <= mdims[0])) {

			tstrs[i] = 0;
			tdims[0] *= tdims[i];
			tdims[i] = 1;
		}
	}

	arg.fitting =   md_check_equal_dims(arg.N, tstrs, MD_STRIDES(arg.N, arg.mdims, 1), ~0ul)
		     && md_check_equal_dims(arg.N, tdims, mdims, md_nontriv_dims(arg.N, arg.mdims))
		     && md_check_equal_dims(arg.N, tdims, mdims, ~0ul);

	md_select_dims(arg.N, md_nontriv_strides(arg.N, tstrs), tdims, tdims);

	arg.simple =   md_check_equal_dims(arg.N, tstrs, MD_STRIDES(arg.N, arg.mdims, 1), ~0ul)
		     && md_check_equal_dims(arg.N, tdims, mdims, md_nontriv_dims(arg.N, arg.mdims))
		     && md_check_equal_dims(arg.N, tdims, mdims, ~0ul);

	return arg;
}

static void arg_free(struct delayed_op_arg_s arg)
{
	xfree(arg.adims);
	xfree(arg.astrs);
	xfree(arg.mdims);
	xfree(arg.mstrs);
	xfree(arg.mpos);
}

static struct delayed_op_arg_s arg_clone(struct delayed_op_arg_s arg)
{
	struct delayed_op_arg_s ret = arg;

	ret.adims = ARR_CLONE(long[arg.N], arg.adims);
	ret.astrs = ARR_CLONE(long[arg.N], arg.astrs);
	ret.mdims = ARR_CLONE(long[arg.N], arg.mdims);
	ret.mstrs = ARR_CLONE(long[arg.N], arg.mstrs);
	ret.mpos = ARR_CLONE(long[arg.N], arg.mpos);

	return ret;
}

static void arg_extend(int N, struct delayed_op_arg_s* arg)
{
	if (arg->N >= N + 1)
		return;

	long adims[N + 1];
	long astrs[N + 1];
	long mdims[N + 1];
	long mstrs[N + 1];
	long mpos[N + 1];

	md_singleton_dims(N + 1, adims);
	md_singleton_strides(N + 1, astrs);
	md_singleton_dims(N + 1, mdims);
	md_singleton_strides(N + 1, mstrs);
	md_set_dims(N + 1, mpos, 0);

	md_copy_dims(arg->N, adims, arg->adims);
	md_copy_dims(arg->N, astrs, arg->astrs);
	md_copy_dims(arg->N, mdims, arg->mdims);
	md_copy_dims(arg->N, mstrs, arg->mstrs);
	md_copy_dims(arg->N, mpos, arg->mpos);

	xfree(arg->adims);
	xfree(arg->astrs);
	xfree(arg->mdims);
	xfree(arg->mstrs);
	xfree(arg->mpos);

	arg->adims = ARR_CLONE(long[N + 1], adims);
	arg->astrs = ARR_CLONE(long[N + 1], astrs);
	arg->mdims = ARR_CLONE(long[N + 1], mdims);
	arg->mstrs = ARR_CLONE(long[N + 1], mstrs);
	arg->mpos = ARR_CLONE(long[N + 1], mpos);

	arg->N = N + 1;
}

static bool arg_same(struct delayed_op_arg_s arg1, struct delayed_op_arg_s arg2)
{
	if (arg1.ptr != arg2.ptr)
		return false;

	if (arg1.ptr_base != arg2.ptr_base)
		return false;

	if ((arg1.sflags != arg2.sflags) || (arg1.lflags != arg2.lflags))
		return false;

	if (   !md_check_equal_dims(MIN(arg1.N, arg2.N), arg1.adims, arg2.adims, ~0UL)
	    || (1 < md_calc_size(arg1.N - MIN(arg1.N, arg2.N), arg1.adims + MIN(arg1.N, arg2.N)))
	    || (1 < md_calc_size(arg2.N - MIN(arg1.N, arg2.N), arg2.adims + MIN(arg1.N, arg2.N))))
		return false;

	if (   !md_check_equal_dims(MIN(arg1.N, arg2.N), arg1.astrs, arg2.astrs, ~0UL)
	    || (1 < md_calc_size(arg1.N - MIN(arg1.N, arg2.N), arg1.astrs + MIN(arg1.N, arg2.N)))
	    || (1 < md_calc_size(arg2.N - MIN(arg1.N, arg2.N), arg2.astrs + MIN(arg1.N, arg2.N))))
		return false;

	if (   !md_check_equal_dims(MIN(arg1.N, arg2.N), arg1.mdims, arg2.mdims, ~0UL)
	    || (1 < md_calc_size(arg1.N - MIN(arg1.N, arg2.N), arg1.mdims + MIN(arg1.N, arg2.N)))
	    || (1 < md_calc_size(arg2.N - MIN(arg1.N, arg2.N), arg2.mdims + MIN(arg1.N, arg2.N))))
		return false;

	return true;
}




struct delayed_op_s {

	TYPEID* TYPEID;

	int D;
	long* ldims;

	int N;
	flag_t write_flags;
	flag_t read_flags;
	flag_t buffer_flags;
	struct delayed_op_arg_s* args;

	delayed_op_fun_t fun;
	delayed_op_del_t del;
	delayed_op_debug_t debug;

	long mchange;
	long mpeak;

	struct shared_obj_s sptr;
};

static void delayed_op_del(const struct shared_obj_s* sptr)
{
	const struct delayed_op_s* x = CONTAINER_OF(sptr, const struct delayed_op_s, sptr);

	if (NULL != x->del)
		(x->del)(x);

	for (int i = 0; i < x->N; i++)
		arg_free(x->args[i]);

	xfree(x->ldims);
	xfree(x->args);

	xfree(x);
}

static void delayed_op_init(delayed_op_t* op, int D, const long dim[D],
			  int N, struct delayed_op_arg_s args[N],
			  flag_t write_flags, flag_t read_flags, flag_t buffer_flags,
			  long mchange, long mpeak,
			  delayed_op_fun_t fun, delayed_op_del_t del, delayed_op_debug_t debug)
{
	assert(8 * (int)sizeof(flag_t) >= N);
	shared_obj_init(&op->sptr, delayed_op_del);

	op->N = N;

	op->D = D;
	for (int i = 0; i < N; i++)
		op->D = MAX(op->D, args[i].N - 1);

	long ldims[op->D];
	md_singleton_dims(op->D, ldims);
	md_copy_dims(D, ldims, dim);
	op->ldims = ARR_CLONE(long [op->D], ldims);

	op->args = ARR_CLONE(struct delayed_op_arg_s[N], args);
	for (int i = 0; i < N; i++)
		arg_extend(op->D, &(op->args[i]));

	op->write_flags = write_flags;
	op->read_flags = read_flags;
	op->buffer_flags = buffer_flags;

	op->mchange = mchange;
	op->mpeak = mpeak;

	op->fun = fun;
	op->debug = debug;
	op->del = del;
}

static void delayed_op_free(const delayed_op_t* x)
{
	if (NULL == x)
		return;

	shared_obj_destroy(&x->sptr);
}

static const char* _print_ptr(struct delayed_op_arg_s* arg, bool read, bool write, bool buffer)
{
	return ptr_printf("(%sL%lu)%s%s%s: %p", arg->fitting ? "F" : "", arg->lflags / 2, read ? "R": "", write ? "W" : "", buffer ? "B" : "", arg->ptr_base);
}

static const char* print_ptr(delayed_op_t* op)
{
	if (0 == op->N)
		return "";

	const char* ret = ptr_printf("(");

	for (int i = 0; i < op->N; i++) {

		const char* ptr = _print_ptr(&op->args[i], BITTEST(op->read_flags, i), BITTEST(op->write_flags, i), BITTEST(op->buffer_flags, i));
		ptr_append_printf(&ret, "%s%s", ptr, (i < op->N - 1) ? ", " : ")");
		xfree(ptr);
	}

	return ret;
}

static __thread int delayed_nested_level = 0;

static const char* print_delayed_fun_f(delayed_op_t* op, bool nested)
{
	const char* ret;

	if (NULL != op->debug) {

		ret = (op->debug)(op, nested);
	} else {

		const char* ptr = print_ptr(op);
		ret = ptr_printf("%s %s", op->TYPEID->name, ptr);
		xfree(ptr);
		ptr_append_print_dims(&ret, op->D, op->ldims);
	}

	const char* ret2 = ptr_printf("%*s%s", delayed_nested_level * 4, "", ret);
	xfree(ret);
	return ret2;
}

static void delayed_queue(delayed_op_t* x)
{
	if (debug_level > delayed_dl) {

		const char* op = print_delayed_fun_f(x, false);
		debug_printf(delayed_dl, "Queue delayed op %s\n", op);
		xfree(op);
	}

#ifdef USE_CUDA
	if (!cuda_is_stream_default())
		error("Delayed computation is incompatible with using multiple CUDA streams!\n");
#endif

	queue_t queue = global_queue[cfl_loop_worker_id()];

	queue_set_lock(queue);
	list_append(queue->ops, (void*)x);
	queue_unset_lock(queue);
}


static void delayed_op_exec(delayed_op_t* op, unsigned long slice_flags, long index)
{

	delayed_nested_level++;

#if 0
	if (debug_level >= delayed_dl) {
		const char* prefix = ptr_printf("Exec delayed op ");
		const char* op_str = print_delayed_fun_f(op, false);
		debug_printf(delayed_dl, "%s%s\n", prefix, op_str);
		xfree(op_str);
		xfree(prefix);
	}
#endif

	long pos[op->D];
	md_set_dims(op->D, pos, 0);
	md_unravel_index(op->D, pos, slice_flags, op->ldims, index);

	if (NULL != op->fun)
		(op->fun)(op, slice_flags, index);

	delayed_nested_level--;
}


bool is_delayed(const void* ptr)
{
	if (!bart_delayed_computations)
		return false;

	if (!is_vptr(ptr))
		return false;

	queue_init();
	return !global_queue[cfl_loop_worker_id()]->compute;
}


void delayed_overwrite(const void* ptr)
{
	queue_t queue = global_queue[cfl_loop_worker_id()];

	if (is_delayed(ptr)) {

		queue_set_lock(queue);

		auto ops = queue->ops;
		for (int i = 0; i < list_count(ops); i++) {

			delayed_op_t* op = list_get_item(ops, i);
			if (delayed_op_is_alloc(op)) {

				if (vptr_overlap(ptr, op->args[0].ptr))
					delayed_op_free(list_remove_item(ops, i--));

				continue;
			}

			for (int j = 0; j < op->N; j++) {

				if (vptr_overlap(ptr, op->args[j].ptr)) {

					queue_unset_lock(queue);
					delayed_compute();
					return;
				}
			}
		}

		queue_unset_lock(queue);
	}
}

static void delayed_op_exec_resolve(delayed_op_t* op, int D, long ldims[D], int N, long dims[N][D], void* ptr[N], unsigned long slice_flags, long index)
{
	assert(D == op->D);
	assert(N == op->N);

	long pos[D];
	md_set_dims(D, pos, 0);
	md_unravel_index(D, pos, slice_flags, op->ldims, index);

	if (NULL != ldims)
		md_select_dims(D, ~slice_flags, ldims, op->ldims);

	for (int i = 0; i < N; i++) {

		if (NULL != dims)
			md_select_dims(D, ~slice_flags, dims[i], op->args[i].adims + 1);

		ptr[i] = op->args[i].ptr + md_calc_offset(D, op->args[i].astrs + 1, pos);
	}
}




struct delayed_alloc_s {

	delayed_op_t super;
	bool tmp_buffer;
};

static DEF_TYPEID(delayed_alloc_s);

struct delayed_free_s {

	delayed_op_t super;
	bool tmp_buffer;
};

static DEF_TYPEID(delayed_free_s);



static void delayed_op_alloc_fun(delayed_op_t* op, unsigned long flags, long /*index*/)
{
	if (!vptr_is_mem_allocated(op->args[0].ptr_base))
		vptr_set_loop_flags(op->args[0].ptr_base, flags & md_nontriv_dims(op->D, op->ldims));
}

static const char* delayed_op_alloc_free_debug(delayed_op_t* op, bool /*nested*/)
{
	const char* ptr = print_ptr(op);
	const char* ret = ptr_printf("%s %s", op->TYPEID->name, ptr);
	xfree(ptr);
	ptr_append_print_dims(&ret, op->args[0].N - 1, op->args[0].adims + 1);

	if (CAST_MAYBE(delayed_alloc_s, op) && !CAST_MAYBE(delayed_alloc_s, op)->tmp_buffer)
		return ret;

	if (CAST_MAYBE(delayed_free_s, op) && !CAST_MAYBE(delayed_free_s, op)->tmp_buffer)
		return ret;

	ptr_append_printf(&ret, " loop: (");

	for (int i = 0; i < op->D; i++)
		if (1 < op->ldims[i])
			ptr_append_printf(&ret, "%d, ", i);

	ptr_append_printf(&ret, ")");

	return ret;

}

static struct delayed_op_s* delayed_alloc_create(const void* ptr, int N, const long dims[N], size_t size)
{
	PTR_ALLOC(struct delayed_alloc_s, op);
	SET_TYPEID(delayed_alloc_s, op);

	long strs[N];
	md_calc_strides(N, strs, dims, size);

	struct delayed_op_arg_s arg[1] = { arg_create(N, dims, strs, ptr, size) };

	delayed_op_init(CAST_UP(op), N, dims, 1, arg, BIT(0), flag_none, flag_none, md_calc_size(N, dims) * (long)size, md_calc_size(N, dims) * (long)size, delayed_op_alloc_fun, NULL, delayed_op_alloc_free_debug);

	op->tmp_buffer = false;

	return CAST_UP(PTR_PASS(op));
}

void delayed_alloc(const void* ptr, int N, const long dims[N], size_t size)
{
	delayed_queue(delayed_alloc_create(ptr, N, dims, size));
}

bool delayed_op_is_alloc(const delayed_op_t* op)
{
	return NULL != CAST_MAYBE(delayed_alloc_s, op);
}


static void delayed_op_free_fun(delayed_op_t* op, unsigned long flags, long index)
{
	UNUSED(flags);
	UNUSED(index);

	long free_dims[op->D];
	void* ptr[1];
	delayed_op_exec_resolve(op, op->D, free_dims, 1, NULL, ptr, flags, index);
	for(int i = 0; i < op->D; i++)
		if (!MD_IS_SET(flags, i))
			free_dims[i] = op->args[0].adims[i + 1];

	vptr_free_mem(op->D, free_dims, op->args[0].mstrs + 1, ptr[0], (size_t)op->args[0].mdims[0]);
}

static void delayed_op_free_del(const delayed_op_t* op)
{
	vptr_free(CAST_DOWN(delayed_free_s, op)->super.args[0].ptr);
}

static struct delayed_op_s* delayed_free_create(const void* ptr, int N, const long dims[N], size_t size)
{
	PTR_ALLOC(struct delayed_free_s, op);
	SET_TYPEID(delayed_free_s, op);

	long strs[N];
	md_calc_strides(N, strs, dims, size);

	struct delayed_op_arg_s arg[1] = { arg_create(N, dims, strs, ptr, size) };
	delayed_op_init(CAST_UP(op), N, dims, 1, arg, BIT(0), flag_none, flag_none, -md_calc_size(N, dims) * (long)size, 0, delayed_op_free_fun, delayed_op_free_del, delayed_op_alloc_free_debug);

	op->tmp_buffer = false;

	return CAST_UP(PTR_PASS(op));
}

void delayed_free(const void* ptr, int N, const long dims[N], size_t size)
{
	queue_t queue = global_queue[cfl_loop_worker_id()];

	queue_set_lock(queue);

	if (0 == list_count(queue->ops)) {

		queue->compute = true;
		vptr_free(ptr);
		queue->compute = false;

		queue_unset_lock(queue);

		return;
	} else {

		queue_unset_lock(queue);
	}

	delayed_queue(delayed_free_create(ptr, N, dims, size));
}

bool delayed_op_is_free(const delayed_op_t* op)
{
	return NULL != CAST_MAYBE(delayed_free_s, op);
}




struct delayed_copy_s {

	delayed_op_t super;
};

static DEF_TYPEID(delayed_copy_s);

static void delayed_op_copy_fun(delayed_op_t* op, unsigned long flags, long index)
{
	long dims[op->D];
	void* ptr[2];

	delayed_op_exec_resolve(op, op->D, dims, 2, NULL, ptr, flags, index);
	md_copy2(op->D, dims, op->args[0].astrs + 1, ptr[0], op->args[1].astrs + 1, ptr[1], (size_t)op->args[0].adims[0]);
}

static struct delayed_op_s* delayed_copy_create(int D, const long dim[D], const long ostr[D], void* optr, const long istr[D], const void* iptr, size_t size)
{
	long ndims[D];
	long nostr[D];
	long nistr[D];

	md_copy_dims(D, ndims, dim);
	md_copy_strides(D, nostr, ostr);
	md_copy_strides(D, nistr, istr);

	long (*nstrs[2])[D] = { &nostr, &nistr };
	int ND = optimize_dims_gpu(2, D, ndims, nstrs);

	struct delayed_op_arg_s arg[2];

	if (   1 == ND
	    && nostr[0] == (long)size
	    && nistr[0] == (long)size
	    && vptr_is_same_type(optr, iptr)
	    && (size_t)ndims[0] * size == vptr_get_len(optr)) {

		D = vptr_get_N(optr);
		long dims[D];
		vptr_get_dims(optr, D, dims);

		size = vptr_get_size(optr);

		arg[0] = arg_create(D, dims, MD_STRIDES(D, dims, size), optr, size);
		arg[1] = arg_create(D, dims, MD_STRIDES(D, dims, size), iptr, size);

		PTR_ALLOC(struct delayed_copy_s, op);
		SET_TYPEID(delayed_copy_s, op);

		delayed_op_init(CAST_UP(op), D, dims, 2, arg, BIT(0), BIT(1), flag_none, 0, 0, delayed_op_copy_fun, NULL, NULL);

		return CAST_UP(PTR_PASS(op));

	} else {

		arg[0] = arg_create(D, dim, ostr, optr, size);
		arg[1] = arg_create(D, dim, istr, iptr, size);

		PTR_ALLOC(struct delayed_copy_s, op);
		SET_TYPEID(delayed_copy_s, op);

		delayed_op_init(CAST_UP(op), D, dim, 2, arg, BIT(0), BIT(1), flag_none, 0, 0, delayed_op_copy_fun, NULL, NULL);

		return CAST_UP(PTR_PASS(op));
	}
}

bool delayed_queue_copy(int D, const long dim[D], const long ostr[D], void* optr, const long istr[D], const void* iptr, size_t size)
{
	if (!is_delayed(optr) || !is_delayed(iptr))
		return false;

	delayed_queue(delayed_copy_create(D, dim, ostr, optr, istr, iptr, size));

	return true;
}

bool delayed_op_is_copy(const delayed_op_t* op)
{
	return NULL != CAST_MAYBE(delayed_copy_s, op);
}




struct delayed_circ_shift_s {

	delayed_op_t super;

	const long* center;
};

static DEF_TYPEID(delayed_circ_shift_s);

static void delayed_circ_shift_free(const delayed_op_t* op)
{
	xfree(CAST_DOWN(delayed_circ_shift_s, op)->center);
}

static void delayed_op_circ_shift_fun(delayed_op_t* op, unsigned long flags, long index)
{
	long dims[2][op->D];
	void* ptr[2];

	delayed_op_exec_resolve(op, op->D, NULL, 2, dims, ptr, flags, index);

	auto shift_op = CAST_DOWN(delayed_circ_shift_s, op);

	md_circ_shift2(op->D, dims[0], shift_op->center, op->args[0].astrs + 1, ptr[0], op->args[1].astrs + 1, ptr[1], (size_t)op->args[0].adims[0]);
}

bool delayed_queue_circ_shift(int D, const long dimensions[D], const long center[D], const long str1[D], void* dst, const long str2[D], const void* src, size_t size)
{
	if (!is_delayed(dst) || !is_delayed(src))
		return false;

	long ldims[D];
	unsigned long lflags = ~md_nontriv_strides(D, center);
	md_select_dims(D, lflags, ldims, dimensions);

	struct delayed_op_arg_s arg[2] = {
		arg_create(D, dimensions, str1, dst, size),
		arg_create(D, dimensions, str2, src, size),
	};

	PTR_ALLOC(struct delayed_circ_shift_s, op);
	SET_TYPEID(delayed_circ_shift_s, op);

	op->center = ARR_CLONE(long[D], center);

	delayed_op_init(CAST_UP(op), D, ldims, 2, arg, BIT(0), BIT(1), flag_none, 0, 0, delayed_op_circ_shift_fun, delayed_circ_shift_free, NULL);

	delayed_queue(CAST_UP(PTR_PASS(op)));

	return true;
}


struct delayed_clear_s {

	delayed_op_t super;
};

static DEF_TYPEID(delayed_clear_s);

static void delayed_op_clear_fun(delayed_op_t* op, unsigned long flags, long index)
{
	long dims[op->D];
	void* ptr[1];

	delayed_op_exec_resolve(op, op->D, dims, 1, NULL, ptr, flags, index);

	if (!vptr_is_mem_allocated(ptr[0]))
		vptr_set_clear(ptr[0]);
	else
		md_clear2(op->D, dims, op->args[0].astrs + 1, ptr[0], (size_t)op->args[0].adims[0]);
}

static struct delayed_op_s* delayed_clear_create(int D, const long dim[D], const long str[D], void* ptr, size_t size)
{
	long dims[D];
	md_select_dims(D, md_nontriv_strides(D, str), dims, dim);

	struct delayed_op_arg_s arg[2] = { arg_create(D, dim, str, ptr, size) };

	PTR_ALLOC(struct delayed_clear_s, op);
	SET_TYPEID(delayed_clear_s, op);

	delayed_op_init(CAST_UP(op), D, dims, 1, arg, BIT(0), flag_none, flag_none, 0, 0, delayed_op_clear_fun, NULL, NULL);

	return CAST_UP(PTR_PASS(op));
}

bool delayed_queue_clear(int D, const long dim[D], const long str[D], void* ptr, size_t size)
{
	if (!is_delayed(ptr))
		return false;

	delayed_queue(delayed_clear_create(D, dim, str, ptr, size));

	return true;
}

bool delayed_op_is_clear(const delayed_op_t* op)
{
	return NULL != CAST_MAYBE(delayed_clear_s, op);
}



struct delayed_md_fun_s {

	delayed_op_t super;

	enum delayed_md_fun_type type;
	size_t offset;
	unsigned long mpi_r_flags;
};

static DEF_TYPEID(delayed_md_fun_s);

static void delayed_op_md_fun(delayed_op_t* op, unsigned long flags, long index)
{
	long dims[op->D];
	void* ptr[op->N];

	delayed_op_exec_resolve(op, op->D, dims, op->N, NULL, ptr, flags, index);

	auto md_op = CAST_DOWN(delayed_md_fun_s, op);
	size_t offset = md_op->offset;

	for (int i = 0; i < op->N; i++)
		if (MD_IS_SET(md_op->mpi_r_flags, i))
			mpi_set_reduce(ptr[i]);

	switch (md_op->type) {

		case delayed_op_type_z3op:
			make_z3op(offset, op->D, dims, op->args[0].astrs + 1, ptr[0], op->args[1].astrs + 1, ptr[1], op->args[2].astrs + 1, ptr[2]);
			break;

		case delayed_op_type_3op:
			make_3op(offset, op->D, dims, op->args[0].astrs + 1, ptr[0], op->args[1].astrs + 1, ptr[1], op->args[2].astrs + 1, ptr[2]);
			break;

		case delayed_op_type_z3opd:
			make_z3opd(offset, op->D, dims, op->args[0].astrs + 1, ptr[0], op->args[1].astrs + 1, ptr[1], op->args[2].astrs + 1, ptr[2]);
			break;

		case delayed_op_type_3opd:
			make_3opd(offset, op->D, dims, op->args[0].astrs + 1, ptr[0], op->args[1].astrs + 1, ptr[1], op->args[2].astrs + 1, ptr[2]);
			break;

		case delayed_op_type_z2op:
			make_z2op(offset, op->D, dims, op->args[0].astrs + 1, ptr[0], op->args[1].astrs + 1, ptr[1]);
			break;

		case delayed_op_type_2op:
			make_2op(offset, op->D, dims, op->args[0].astrs + 1, ptr[0], op->args[1].astrs + 1, ptr[1]);
			break;

		case delayed_op_type_z2opd:
			make_z2opd(offset, op->D, dims, op->args[0].astrs + 1, ptr[0], op->args[1].astrs + 1, ptr[1]);
			break;

		case delayed_op_type_2opd:
			make_2opd(offset, op->D, dims, op->args[0].astrs + 1, ptr[0], op->args[1].astrs + 1, ptr[1]);
			break;

		case delayed_op_type_z2opf:
			make_z2opf(offset, op->D, dims, op->args[0].astrs + 1, ptr[0], op->args[1].astrs + 1, ptr[1]);
			break;

		case delayed_op_type_2opf:
			make_2opf(offset, op->D, dims, op->args[0].astrs + 1, ptr[0], op->args[1].astrs + 1, ptr[1]);
			break;

		error("Delayed md_fun type not implemented!\n");
	};

	for (int i = 0; i < op->N; i++)
		if (MD_IS_SET(md_op->mpi_r_flags, i))
			mpi_unset_reduce(ptr[i]);
}

static const char* delayed_op_md_fun_debug(delayed_op_t* op, bool /*nested*/)
{
	const char* name = NULL;
	switch(CAST_DOWN(delayed_md_fun_s, op)->offset) {
	case offsetof(struct vec_ops, float2double) : name = "float2double"; break;
	case offsetof(struct vec_ops, double2float) : name = "double2float"; break;
	case offsetof(struct vec_ops, dot) : name = "dot"; break;
	case offsetof(struct vec_ops, asum) : name = "asum"; break;
	case offsetof(struct vec_ops, zsum) : name = "zsum"; break;
	case offsetof(struct vec_ops, zl1norm) : name = "zl1norm"; break;
	case offsetof(struct vec_ops, zdot) : name = "zdot"; break;
	case offsetof(struct vec_ops, axpy) : name = "axpy"; break;
	case offsetof(struct vec_ops, axpbz) : name = "axpbz"; break;
	case offsetof(struct vec_ops, pow) : name = "pow"; break;
	case offsetof(struct vec_ops, sqrt) : name = "sqrt"; break;
	case offsetof(struct vec_ops, zle) : name = "zle"; break;
	case offsetof(struct vec_ops, le) : name = "le"; break;
	case offsetof(struct vec_ops, add) : name = "add"; break;
	case offsetof(struct vec_ops, sub) : name = "sub"; break;
	case offsetof(struct vec_ops, mul) : name = "mul"; break;
	case offsetof(struct vec_ops, div) : name = "div"; break;
	case offsetof(struct vec_ops, fmac) : name = "fmac"; break;
	case offsetof(struct vec_ops, fmacD) : name = "fmacD"; break;
	case offsetof(struct vec_ops, smul) : name = "smul"; break;
	case offsetof(struct vec_ops, sadd) : name = "sadd"; break;
	case offsetof(struct vec_ops, zmul) : name = "zmul"; break;
	case offsetof(struct vec_ops, zdiv) : name = "zdiv"; break;
	case offsetof(struct vec_ops, zfmac) : name = "zfmac"; break;
	case offsetof(struct vec_ops, zfmacD) : name = "zfmacD"; break;
	case offsetof(struct vec_ops, zmulc) : name = "zmulc"; break;
	case offsetof(struct vec_ops, zfmacc) : name = "zfmacc"; break;
	case offsetof(struct vec_ops, zfmaccD) : name = "zfmaccD"; break;
	case offsetof(struct vec_ops, zfsq2) : name = "zfsq2"; break;
	case offsetof(struct vec_ops, zsmul) : name = "zsmul"; break;
	case offsetof(struct vec_ops, zsadd) : name = "zsadd"; break;
	case offsetof(struct vec_ops, zpow) : name = "zpow"; break;
	case offsetof(struct vec_ops, zphsr) : name = "zphsr"; break;
	case offsetof(struct vec_ops, zconj) : name = "zconj"; break;
	case offsetof(struct vec_ops, zexpj) : name = "zexpj"; break;
	case offsetof(struct vec_ops, zexp) : name = "zexp"; break;
	case offsetof(struct vec_ops, zlog) : name = "zlog"; break;
	case offsetof(struct vec_ops, zarg) : name = "zarg"; break;
	case offsetof(struct vec_ops, zabs) : name = "zabs"; break;
	case offsetof(struct vec_ops, zatanr) : name = "zatanr"; break;
	case offsetof(struct vec_ops, exp) : name = "exp"; break;
	case offsetof(struct vec_ops, log) : name = "log"; break;
	case offsetof(struct vec_ops, zsin) : name = "zsin"; break;
	case offsetof(struct vec_ops, zcos) : name = "zcos"; break;
	case offsetof(struct vec_ops, zacosr) : name = "zacosr"; break;
	case offsetof(struct vec_ops, zsinh) : name = "zsinh"; break;
	case offsetof(struct vec_ops, zcosh) : name = "zcosh"; break;
	case offsetof(struct vec_ops, zcmp) : name = "zcmp"; break;
	case offsetof(struct vec_ops, zdiv_reg) : name = "zdiv_reg"; break;
	case offsetof(struct vec_ops, zfftmod) : name = "zfftmod"; break;
	case offsetof(struct vec_ops, zmax) : name = "zmax"; break;
	case offsetof(struct vec_ops, zsmax) : name = "zsmax"; break;
	case offsetof(struct vec_ops, zsmin) : name = "zsmin"; break;
	case offsetof(struct vec_ops, smax) : name = "smax"; break;
	case offsetof(struct vec_ops, max) : name = "max"; break;
	case offsetof(struct vec_ops, min) : name = "min"; break;
	case offsetof(struct vec_ops, zsoftthresh_half) : name = "zsoftthresh_half"; break;
	case offsetof(struct vec_ops, zsoftthresh) : name = "zsoftthresh"; break;
	case offsetof(struct vec_ops, softthresh_half) : name = "softthresh_half"; break;
	case offsetof(struct vec_ops, softthresh) : name = "softthresh"; break;
//	case offsetof(struct vec_ops, swap) : name = "swap"; break;
	case offsetof(struct vec_ops, zhardthresh) : name = "zhardthresh"; break;
	case offsetof(struct vec_ops, zhardthresh_mask) : name = "zhardthresh_mask"; break;
	case offsetof(struct vec_ops, pdf_gauss) : name = "pdf_gauss"; break;
	case offsetof(struct vec_ops, real) : name = "real"; break;
	case offsetof(struct vec_ops, imag) : name = "imag"; break;
	case offsetof(struct vec_ops, zcmpl_real) : name = "zcmpl_real"; break;
	case offsetof(struct vec_ops, zcmpl_imag) : name = "zcmpl_imag"; break;
	case offsetof(struct vec_ops, zcmpl) : name = "zcmpl"; break;
	case offsetof(struct vec_ops, zfill) : name = "zfill"; break;
	default: assert(0);
	}

	const char* ptr = print_ptr(op);
	const char* ret = ptr_printf("md_function %s %s", name, ptr);
	xfree(ptr);
	ptr_append_print_dims(&ret, op->D, op->ldims);
	return ret;
}

bool delayed_queue_make_op(enum delayed_md_fun_type type, size_t offset, int D, const long dim[D], int N, const long* strs[N], const void* ptr[N], const size_t sizes[N])
{
	for (int i = 0; i < N; i++)
		if (!is_delayed(ptr[i]))
			return false;

	for (int i = 0; i < N; i++)
		if (!vptr_is_init(ptr[i])) {

			debug_vptr(DP_INFO, ptr[i]);
			error("Vptr not initialized!\n");
		}

	PTR_ALLOC(struct delayed_md_fun_s, op);
	SET_TYPEID(delayed_md_fun_s, op);

	struct delayed_op_arg_s arg[N];

	for (int i = 0; i < N; i++)
		arg[i] = arg_create(D, dim, strs[i], ptr[i], sizes[i]);

	flag_t read_flags = flag_none;
	flag_t write_flags = BIT(0);
	BITSET(read_flags, 1);
	BITSET(read_flags, 2);

	delayed_op_init(CAST_UP(op), D, dim, N, arg, write_flags, read_flags, flag_none, 0, 0, delayed_op_md_fun, NULL, delayed_op_md_fun_debug);

	op->offset = offset;
	op->type = type;
	op->mpi_r_flags = 0;

	for(int i = 0; i < N; i++)
		if (mpi_is_set_reduce(ptr[i]))
			op->mpi_r_flags |= MD_BIT(i);

	if (   (offset == offsetof(struct vec_ops, fmac))
	    || (offset == offsetof(struct vec_ops, fmacD))
	    || (offset == offsetof(struct vec_ops, zfmac))
	    || (offset == offsetof(struct vec_ops, zfmacD))
	    || (offset == offsetof(struct vec_ops, zfmacc))
	    || (offset == offsetof(struct vec_ops, zfmaccD)))
		BITSET(op->super.read_flags, 0);

	delayed_queue(CAST_UP(PTR_PASS(op)));

	return true;
}


struct delayed_vptr_fun_s {

	delayed_op_t super;

	vptr_fun_t fun;
	struct vptr_fun_data_s* data;
	bool resolve;
};

static DEF_TYPEID(delayed_vptr_fun_s);

static void delayed_op_fun_free(const delayed_op_t* op)
{
	if (NULL != CAST_DOWN(delayed_vptr_fun_s, op)->data->del)
		CAST_DOWN(delayed_vptr_fun_s, op)->data->del(CAST_DOWN(delayed_vptr_fun_s, op)->data);

	xfree(CAST_DOWN(delayed_vptr_fun_s, op)->data);
}

static void delayed_op_fun_fun(delayed_op_t* op, unsigned long flags, long index)
{
	long sdims[op->D];
	void* ptr[op->N];

	struct delayed_vptr_fun_s* fun = CAST_DOWN(delayed_vptr_fun_s, op);

	long tdims[op->N][op->D];
	long tstrs[op->N][op->D];
	size_t sizes[op->N];

	delayed_op_exec_resolve(op, op->D, sdims, op->N, tdims, ptr, flags, index);

	const long* dims[op->N];
	const long* strs[op->N];

	for (int i = 0; i < op->N; i++) {

		md_select_strides(op->D, md_nontriv_dims(op->D, tdims[i]), tstrs[i], op->args[i].astrs + 1);
		sizes[i] = (size_t)op->args[i].adims[0];

		dims[i] = tdims[i];
		strs[i] = tstrs[i];
	}

	exec_vptr_fun_internal(fun->fun, fun->data, op->N, op->D, md_nontriv_dims(op->D, sdims), dims, strs, ptr, sizes, fun->resolve);
}

static const char* delayed_op_fun_debug(delayed_op_t* op, bool /*nested*/)
{
	const char* ptr = print_ptr(op);
	const char* ret = ptr_printf("fun %s %s", CAST_DOWN(delayed_vptr_fun_s, op)->data->TYPEID->name, ptr);
	xfree(ptr);
	ptr_append_print_dims(&ret, op->D, op->ldims);
	return ret;
}


void exec_vptr_fun_delayed(vptr_fun_t fun, vptr_fun_data_t* data, int N, int D, unsigned long lflags, unsigned long wflags, unsigned long rflags, const long* dims[N], const long* strs[N], void* ptr[N], size_t sizes[N], bool resolve)
{
	long ldims[D];
	md_singleton_dims(D, ldims);

	for (int i = 0; i < N; i++) {

		long ldims1[D];
		md_select_dims(D, lflags, ldims1, dims[i]);
		md_max_dims(D, ~0UL, ldims, ldims, ldims1);

		for (int j = 0; j < N; j++) {

			long ldims2[D];
			md_select_dims(D, lflags, ldims2, dims[j]);
			assert(md_check_compat(D, ~0UL, ldims1, ldims2));
		}
	}

	struct delayed_op_arg_s arg[N];
	for (int i = 0; i < N; i++)
		arg[i] = arg_create(D, dims[i], strs[i], ptr[i], sizes[i]);

	PTR_ALLOC(struct delayed_vptr_fun_s, op);
	SET_TYPEID(delayed_vptr_fun_s, op);

	flag_t read_flags = flag_none;
	flag_t write_flags = flag_none;

	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(rflags, i))
			BITSET(read_flags, i);

		if (MD_IS_SET(wflags, i))
			BITSET(write_flags, i);
	}

	delayed_op_init(CAST_UP(op), D, ldims, N, arg, write_flags, read_flags, flag_none, 0, 0, delayed_op_fun_fun, delayed_op_fun_free, delayed_op_fun_debug);

	op->fun = fun;
	op->data = data;
	op->resolve = resolve;

	delayed_op_t* opt = CAST_UP(PTR_PASS(op));
	delayed_queue(opt);
}



struct delayed_exp_slice_s {

	delayed_op_t super;

	delayed_op_t* op;

	unsigned long flags;
	long index;
};

static DEF_TYPEID(delayed_exp_slice_s);

static void delayed_slice_fun(delayed_op_t* op, unsigned long flags, long index)
{
	long dims[op->D];
	md_select_dims(op->D, flags, dims, op->ldims);

	long pos1[op->D];
	md_set_dims(op->D, pos1, 0);
	md_unravel_index(op->D, pos1, flags, dims, index);

	long pos2[op->D];
	md_set_dims(op->D, pos2, 0);
	md_unravel_index(op->D, pos2, CAST_DOWN(delayed_exp_slice_s, op)->flags, dims, CAST_DOWN(delayed_exp_slice_s, op)->index);

	unsigned long slice_flags = (~(CAST_DOWN(delayed_exp_slice_s, op)->flags)) & flags;

	for (int i = 0; i < op->D; i++)
		if (MD_IS_SET((CAST_DOWN(delayed_exp_slice_s, op)->flags), i) && (pos1[i] != pos2[i]))
			return;

	delayed_op_exec(CAST_DOWN(delayed_exp_slice_s, op)->op, slice_flags, md_ravel_index(op->D, pos1, slice_flags, dims));
}

static void delayed_slice_free(const delayed_op_t* op)
{
	delayed_op_free(CAST_DOWN(delayed_exp_slice_s, op)->op);
}

static const char* delayed_slice_debug(delayed_op_t* op, bool nested)
{
	struct delayed_exp_slice_s* ops = CAST_DOWN(delayed_exp_slice_s, op);
	const char* ptr = print_ptr(op);
	const char* ret = ptr_printf("slice %lu %ld %s", ops->flags, ops->index, ptr);
	xfree(ptr);

	ptr_append_print_dims(&ret, op->D, op->ldims);

	if (!nested)
		return ret;

	delayed_nested_level++;
	const char* opn = print_delayed_fun_f(ops->op, true);
	delayed_nested_level--;
	ptr_append_printf(&ret, "\n%s", opn);
	xfree(opn);

	return ret;
}

static delayed_op_t* delayed_op_expand_slice_create(delayed_op_t* ops, int D, const long dims[D], const long pos[D])
{
	assert(D >= ops->D);

	unsigned long flags = md_nontriv_dims(D, dims) & (~md_nontriv_dims(ops->D, ops->ldims));

	if (0 == flags)
		return ops;

	for (int i = 0; i < D; i++)
		assert(MD_IS_SET(flags, i) || 0 == pos[i]);

	PTR_ALLOC(struct delayed_exp_slice_s, x);
	SET_TYPEID(delayed_exp_slice_s, x);

	int N = ops->N;
	struct delayed_op_arg_s arg[N];

	for (int i = 0; i < N; i++)
		arg[i] = arg_clone(ops->args[i]);

	delayed_op_init(CAST_UP(x), D, dims, N, arg, ops->write_flags, ops->read_flags, ops->buffer_flags, ops->mchange, ops->mpeak, delayed_slice_fun, delayed_slice_free, delayed_slice_debug);

	x->op = ops;
	x->flags = flags;
	x->index = md_ravel_index(D, pos, flags, dims);

	return CAST_UP(PTR_PASS(x));
}

static delayed_op_t* delayed_op_expand_slice(delayed_op_t* ops)
{
	int N = ops->N;
	int D = ops->D;

	unsigned long sflags = 0UL;

	long ldims[D];
	long pos[D];
	md_singleton_dims(D, ldims);
	md_set_dims(D, pos, 0);

	//slice dim must be singleton dim in all arguments
	for (int i = 0; i < N; i++)
		sflags |= md_nontriv_dims(ops->args[i].N - 1, ops->args[i].adims + 1);

	for (int i = 0; i < N; i++) {

		unsigned long sflag_i = ops->args[i].sflags / 2;

		for (int j = 0; j < D; j++) {

			if (!MD_IS_SET(sflag_i, j))
				continue;

			if (!MD_IS_SET(sflags, j)) {

				ldims[j] = ops->args[i].mdims[j + 1];
				pos[j] = ops->args[i].mpos[j + 1];
			} else {

				if ((ldims[j] != ops->args[i].mdims[j + 1]) || (pos[j] != ops->args[i].mpos[j + 1]))
					ldims[j] = 1;
			}
		}
	}

	md_max_dims(D, ~0UL, ldims, ops->ldims, ldims);

	return delayed_op_expand_slice_create(ops, D, ldims, pos);
}



struct delayed_chain_s {

	delayed_op_t super;

	int M;
	delayed_op_t** ops;
};

static DEF_TYPEID(delayed_chain_s);

static void delayed_chain_fun(delayed_op_t* _op, unsigned long flags, long index)
{
	struct delayed_chain_s* op = CAST_DOWN(delayed_chain_s, _op);

	for (int i = 0; i < op->M; i++)
		delayed_op_exec(op->ops[i], flags, index);
}

static void delayed_chain_free(const delayed_op_t* _op)
{
	struct delayed_chain_s* op = CAST_DOWN(delayed_chain_s, _op);

	for (int i = 0; i < op->M; i++)
		delayed_op_free(op->ops[i]);

	xfree(op->ops);
}

static const char* delayed_chain_debug(delayed_op_t* _op, bool nested)
{
	struct delayed_chain_s* op = CAST_DOWN(delayed_chain_s, _op);

	const char* ptr = print_ptr(_op);
	const char* ret = ptr_printf("chain (%d ops) (%ld/%ld) %s ", op->M, _op->mchange, _op->mpeak, ptr);
	xfree(ptr);

	ptr_append_print_dims(&ret, _op->D, _op->ldims);

	if (!nested)
		return ret;

	delayed_nested_level++;

	for (int i = 0; i < op->M; i++) {

		const char* ops = print_delayed_fun_f(op->ops[i], true);
		ptr_append_printf(&ret, "\n %s", ops);
		xfree(ops);
	}

	delayed_nested_level--;

	return ret;
}

static bool delayed_ptr_required(delayed_op_t* op, const void* ptr);

static bool subset(struct delayed_op_arg_s a, struct delayed_op_arg_s b)
{
	if (a.ptr_base != b.ptr_base)
		return false;

	if (b.fitting)
		return true;

	if (b.simple)
		return true;

	return false;
}

static void delayed_optimize_alloc(list_t ops_queue);
static void delayed_optimize_free(list_t ops_queue);

static delayed_op_t* delayed_op_chain(int M, delayed_op_t* _ops[M])
{
	if (1 == M)
		return _ops[0];

	for (int i = 1 ; i < M; i++)
		assert(   md_check_equal_dims(MIN(_ops[0]->D, _ops[i]->D), _ops[i]->ldims, _ops[0]->ldims, ~0UL)
		       && md_calc_size(_ops[0]->D, _ops[0]->ldims) == md_calc_size(_ops[i]->D, _ops[i]->ldims));

	list_t ops_list = list_create();

	for (int i = 0; i < M; i++) {

		if (CAST_MAYBE(delayed_chain_s, _ops[i])) {

			for (int j = 0; j < CAST_DOWN(delayed_chain_s, _ops[i])->M; j++)
				list_append(ops_list, CAST_DOWN(delayed_chain_s, _ops[i])->ops[j]);

			CAST_DOWN(delayed_chain_s, _ops[i])->M = 0;
			delayed_op_free(_ops[i]);

		} else {

			list_append(ops_list, _ops[i]);
		}
	}

	delayed_optimize_alloc(ops_list);
	delayed_optimize_free(ops_list);

	M = list_count(ops_list);
	delayed_op_t* ops[M];
	list_to_array(M, (void**)ops, ops_list);
	list_free(ops_list);

	int Nmax = 0;

	for (int i = 0; i < M; i++)
		Nmax += ops[i]->N;

	int N = 0;
	struct delayed_op_arg_s args[Nmax];

	flag_t oread_flags[M];

	for (int i = 0; i < M; i++)
		oread_flags[i] = ops[i]->read_flags;

	for (int i = 0; i < M; i++) {

		for (int j = 0; j < ops[i]->N; j++) {

			if (BITTEST(ops[i]->read_flags, j))
				continue;

			if (!BITTEST(ops[i]->write_flags, j))
				continue;

			bool overwrite = true;

			for (int k = 0; k <  ops[i]->N; k++)
				if (BITTEST(ops[i]->read_flags, k) && ops[i]->args[j].ptr_base == ops[i]->args[k].ptr_base)
					overwrite = false;

			if (!overwrite)
				continue;

			for (int k = i + 1; k < M; k++)
				for (int l = 0; l < ops[k]->N; l++)
					if (subset(ops[k]->args[l], ops[i]->args[j]))
						BITCLEAR(oread_flags[k], l);
		}
	}

	flag_t read_flags = flag_none;
	flag_t write_flags = flag_none;
	flag_t buffer_flags = flag_none;

	for (int i = 0; i < M; i++) {

		for (int j = 0; j < ops[i]->N; j++) {

			struct delayed_op_arg_s arg = ops[i]->args[j];

			int k = 0;

			for (; k < N; k++) {

				if (arg_same(arg, args[k]))
					break;
			}

			if (k == N)
				args[N++] = arg_clone(arg);

			assert(8 * (int)sizeof(flag_t) >= N);

			if (BITTEST(oread_flags[i], j))
				BITSET(read_flags, k);

			if (BITTEST(ops[i]->write_flags, j))
				BITSET(write_flags, k);

			if (BITTEST(ops[i]->buffer_flags, j))
				BITSET(buffer_flags, k);
		}
	}

	flag_t skip_flags = flag_none;
	assert(8 * (int)sizeof(flag_t) >= M);
	int skipcount = 0;

	for (int i = 0; i < M; i++)
		if (delayed_op_is_alloc(ops[i]))
			for (int k = 0; k < N; k++)
				if (args[k].ptr_base == ops[i]->args[0].ptr_base) {

					BITSET(skip_flags, k);
					skipcount++;
				}




	struct delayed_op_arg_s nargs[N - skipcount];

	flag_t nread_flags = flag_none;
	flag_t nwrite_flags = flag_none;
	flag_t nbuffer_flags = flag_none;

	int ip = 0;

	for (int i = 0; i < N; i++) {

		if (BITTEST(skip_flags, i)) {

			arg_free(args[i]);
		} else {

			nargs[ip] = args[i];

			if (BITTEST(read_flags, i))
				BITSET(nread_flags, ip);

			if (BITTEST(write_flags, i))
				BITSET(nwrite_flags, ip);

			if (BITTEST(buffer_flags, i))
				BITSET(nbuffer_flags, ip);

			ip++;
		}
	}

	for (int i = 0; i < ip; i++) {

		for (int j = 0; j < ip; j++) {

			if (nargs[i].ptr_base == nargs[j].ptr_base) {

				if (BITTEST(nread_flags, i))
					BITSET(nread_flags, j);

				if (BITTEST(nwrite_flags, i))
					BITSET(nwrite_flags, j);
			}
		}
	}

	long mchange = 0;
	long mpeak = 0;

	for (int i = 0; i < M; i++) {

		mpeak = MAX(mpeak, ops[i]->mpeak + mchange);
		mchange += ops[i]->mchange;
	}

	long ldims[ops[0]->D];
	md_copy_dims(ops[0]->D, ldims, ops[0]->ldims);

	//asssure possible reduction is completed before read
	for (int i = 0; i < ip; i++)
		if (BITTEST(nread_flags, i) && BITTEST(nwrite_flags, i))
			md_select_dims(ops[0]->D, (nargs[i].lflags | nargs[i].sflags) / 2, ldims, ldims);



	PTR_ALLOC(struct delayed_chain_s, x);
	SET_TYPEID(delayed_chain_s, x);

	delayed_op_init(CAST_UP(x), ops[0]->D, ldims, ip, nargs, nwrite_flags, nread_flags, nbuffer_flags, mchange, mpeak, delayed_chain_fun, delayed_chain_free, delayed_chain_debug);

	x->M = M;
	x->ops = ARR_CLONE(delayed_op_t*[M], ops);

	delayed_op_t* op = CAST_UP(PTR_PASS(x));

//	if (md_calc_size(ops[0]->D, ldims) != md_calc_size(ops[0]->D, ops[0]->ldims))
//		debug_printf(DP_INFO, "NON MAX LOOP:\n%s\n", print_delayed_fun_f(op, true));

	return op;
}


struct delayed_unloop_s {

	delayed_op_t super;

	unsigned long seq_flags;
	delayed_op_t* op;
};

static DEF_TYPEID(delayed_unloop_s);

static void delayed_unloop_fun(delayed_op_t* _op, unsigned long flags, long index)
{
	struct delayed_unloop_s* op = CAST_DOWN(delayed_unloop_s, _op);

	int D = op->op->D;

	long pos[D];
	md_set_dims(D, pos, 0);
	md_unravel_index(D, pos, flags, _op->ldims, index);

	do {
		long index = md_ravel_index(D, pos, flags | op->seq_flags, op->op->ldims);
		delayed_op_exec(op->op, flags | op->seq_flags, index);
	} while (md_next(D, op->op->ldims, op->seq_flags, pos));
}

static void delayed_unloop_free(const delayed_op_t* _op)
{
	struct delayed_unloop_s* op = CAST_DOWN(delayed_unloop_s, _op);
	delayed_op_free(op->op);
}

static const char* delayed_unloop_debug(delayed_op_t* _op, bool nested)
{
	struct delayed_unloop_s* op = CAST_DOWN(delayed_unloop_s, _op);

	const char* ptr = print_ptr(_op);
	const char* ret = ptr_printf("unloop (%ld/%ld) (%lu) %s ", _op->mchange, _op->mpeak, op->seq_flags, ptr);
	xfree(ptr);

	ptr_append_print_dims(&ret, _op->D, _op->ldims);

	if (!nested)
		return ret;

	delayed_nested_level++;
	const char* ops = print_delayed_fun_f(op->op, nested);
	delayed_nested_level--;

	ptr_append_printf(&ret, "\n%s", ops);
	xfree(ops);

	return ret;
}

static delayed_op_t* delayed_op_unloop(delayed_op_t* op, unsigned long keep_flags, unsigned long seq_flags)
{
	keep_flags &= md_nontriv_dims(op->D, op->ldims);
	seq_flags &= md_nontriv_dims(op->D, op->ldims) & ~keep_flags;

	if (keep_flags == md_nontriv_dims(op->D, op->ldims))
		return op;

	if (delayed_op_is_alloc(op) || delayed_op_is_free(op)) {

		md_select_dims(op->D, keep_flags, op->ldims, op->ldims);
		return op;
	}

	if (0 == op->mpeak)
		seq_flags = 0;

	if (NULL != CAST_MAYBE(delayed_unloop_s, op)) {

		auto op_unloop = CAST_DOWN(delayed_unloop_s, op);
		op_unloop->seq_flags |= seq_flags;
		md_select_dims(op->D, keep_flags, op->ldims, op->ldims);
		return op;
	}

	if (NULL != CAST_MAYBE(delayed_exp_slice_s, op)) {

		auto op_slice = CAST_DOWN(delayed_exp_slice_s, op);

		if (0 == (op_slice->flags & ~keep_flags)) {

			md_select_dims(op->D, keep_flags, op->ldims, op->ldims);
			op_slice->op = delayed_op_unloop(op_slice->op, keep_flags, seq_flags);

			return op;
		}
	}


	int N = op->N;
	struct delayed_op_arg_s args[N];

	for (int i = 0; i < N; i++)
		args[i] = arg_clone(op->args[i]);


	PTR_ALLOC(struct delayed_unloop_s, x);
	SET_TYPEID(delayed_unloop_s, x);

	int D = op->D;
	long ldims[D];
	md_select_dims(D, keep_flags, ldims, op->ldims);

	delayed_op_init(CAST_UP(x), D, ldims, N, args, op->write_flags, op->read_flags, op->buffer_flags, op->mchange, op->mpeak, delayed_unloop_fun, delayed_unloop_free, delayed_unloop_debug);

	x->seq_flags = seq_flags;
	x->op = op;

	return CAST_UP(PTR_PASS(x));
}

static delayed_op_t* delayed_op_unloop_unloopable(delayed_op_t* op)
{
	unsigned long keep_flags = 0UL;

	for (int i = 0; i < op->N; i++)
		keep_flags |= op->args[i].lflags / 2;

	return delayed_op_unloop(op, keep_flags, 0UL);
}

void delayed_queue_exec(queue_t queue)
{
	if (NULL == queue)
		return;

	list_t ops = queue->ops;

	if (0 == list_count(ops))
		return;

	queue->compute = true;

#ifdef USE_CUDA
	if (!cuda_is_stream_default())
		error("Delayed computation is incompatible with using multiple CUDA streams!\n");
#endif

	int N = list_count(ops);
	float mpeak1 = compute_mpeak(ops, false);
	delayed_optimize_queue(ops);
	float mpeak2 = compute_mpeak(ops, false);
	delayed_optimize_queue_blocking(ops);
	float mpeak3 = compute_mpeak(ops, false);

	debug_printf(DP_DEBUG3, "Optimized queue with %d operations %3.2fGB -> %3.2fGB -> %3.2fGB\n", N, mpeak1 / powf(1024., 3), mpeak2 / powf(1024., 3), mpeak3 / powf(1024., 3));

	if (0 < list_count(ops))
		debug_printf(delayed_dl, "Execute queue with %d operations\n", list_count(ops));

	delayed_op_t* op = list_pop(ops);

	while (NULL != op) {

		delayed_op_exec(op, 0, 0);
		delayed_op_free(op);
		op = list_pop(ops);
	}

	queue->compute = false;
}

void delayed_compute(void)
{
	queue_t queue = get_global_queue();
	delayed_queue_exec(queue);
	release_global_queue(queue);
}

void debug_delayed_queue(int dl, list_t ops_queue)
{
	if (NULL == ops_queue) {

		queue_t queue = global_queue[cfl_loop_worker_id()];

		queue_set_lock(queue);
		queue->compute = true;

		if (0 < list_count(queue->ops))
			debug_printf(dl, "Delayed queue with %d operations\n", list_count(queue->ops));

		for (int i = 0; i < list_count(queue->ops); i++) {

			const char* op = print_delayed_fun_f(list_get_item(queue->ops, i), true);
			debug_printf(dl, "%s\n", op);
			xfree(op);
		}

		queue->compute = false;
		queue_unset_lock(queue);
	} else {

		if (0 < list_count(ops_queue))
			debug_printf(dl, "Delayed queue with %d operations\n", list_count(ops_queue));

		for (int i = 0; i < list_count(ops_queue); i++) {

			const char* op = print_delayed_fun_f(list_get_item(ops_queue, i), true);
			debug_printf(dl, "%s\n", op);
			xfree(op);
		}
	}
}



static bool delayed_ptr_required(delayed_op_t* op, const void* ptr)
{
	for (int i = 0; i < op->N; i++)
		if (ptr == op->args[i].ptr_base)
			return true;

	return false;
}

static bool delayed_ptr_required_write(delayed_op_t* op, const void* ptr)
{
	for (int i = 0; i < op->N; i++)
		if (ptr == op->args[i].ptr_base && BITTEST(op->write_flags, i))
			return true;

	return false;
}

static void delayed_set_tmp_buffer(list_t ops_queue)
{
	for (int i = 0; i < list_count(ops_queue); i++) {

		delayed_op_t* op_allo = list_get_item(ops_queue, i);

		if (!delayed_op_is_alloc(op_allo))
			continue;

		int j = i + 1;

		for (; j < list_count(ops_queue); j++) {

			delayed_op_t* op_free = list_get_item(ops_queue, j);

			if (!delayed_op_is_free(op_free))
				continue;

			if (op_allo->args[0].ptr == op_free->args[0].ptr && !vptr_is_mem_allocated(op_allo->args[0].ptr)) {

				CAST_DOWN(delayed_alloc_s, op_allo)->tmp_buffer = true;
				CAST_DOWN(delayed_free_s, op_free)->tmp_buffer = true;

				break;
			}
		}
	}
}


static void delayed_independent_overwrite_buffer(list_t ops_queue)
{
	debug_printf(delayed_dl, "Delayed Optimize: Replace overwrites:\n");

	for (int i = 0; i < list_count(ops_queue) - 1; i++) {

		delayed_op_t* op_overwrite = list_get_item(ops_queue, i);

		if (delayed_op_is_alloc(op_overwrite) || delayed_op_is_free(op_overwrite))
			continue;

		for (int j = 0; j < op_overwrite->N; j++) {

			if (BITTEST(op_overwrite->read_flags, j))
				continue;

			if (!BITTEST(op_overwrite->write_flags, j))
				continue;

			if (!op_overwrite->args[j].fitting)
				continue;

			if (vptr_is_writeback(op_overwrite->args[j].ptr_base))
				continue;

			bool overwrite = true;

			for (int k = 0; k <  op_overwrite->N; k++)
				if (BITTEST(op_overwrite->read_flags, k) && op_overwrite->args[j].ptr_base == op_overwrite->args[k].ptr_base)
					overwrite = false;

			if (!overwrite)
				continue;

			if (vptr_is_mem_allocated(op_overwrite->args[j].ptr_base))
				continue;

			int k = i + 1;
			delayed_op_t* op_free = list_get_item(ops_queue, k);
			while (!delayed_op_is_free(op_free) || op_overwrite->args[j].ptr_base != op_free->args[0].ptr_base) {

				k++;
				if (k < list_count(ops_queue))
					op_free = list_get_item(ops_queue, k);
				else
					break;
			}

			if (list_count(ops_queue) == k)
				continue;

			struct delayed_op_arg_s arg = op_overwrite->args[j];

			void* optr = arg.ptr_base;
			void* nptr = vptr_alloc_sameplace(arg.N - 1, arg.mdims + 1, (size_t)arg.mdims[0], arg.ptr_base);

			debug_printf(delayed_dl, "\t%p -> %p\n", optr, nptr);

			list_insert(ops_queue, list_remove_item(ops_queue, k), i++);
			list_insert(ops_queue, delayed_alloc_create(nptr, arg.N - 1, arg.mdims + 1, (size_t)arg.mdims[0]), i++);
			list_insert(ops_queue, delayed_free_create(nptr, arg.N - 1, arg.mdims + 1, (size_t)arg.mdims[0]), k + 2);

			for (int l = i; l < k + 2; l++) {

				delayed_op_t* op = list_get_item(ops_queue, l);

				for (int j = 0; j < op->N; j++) {

					if (optr == op->args[j].ptr_base) {

						op->args[j].ptr_base = nptr;
						op->args[j].ptr = nptr + (op->args[j].ptr - optr);
					}
				}
			}
		}
	}
}


static void delayed_optimize_free(list_t ops_queue)
{
	for (int i = 0; i < list_count(ops_queue); i++) {

		delayed_op_t* op = list_get_item(ops_queue, i);

		if (delayed_op_is_free(op))
			list_remove_item(ops_queue, i);
		else
			continue;

		int j = i;

		while (0 < j && !delayed_ptr_required(list_get_item(ops_queue, j - 1), op->args[0].ptr_base))
			j--;

		if (0 == j) {

			delayed_op_free(op);
			i--;
		} else {

			list_insert(ops_queue, op, j);

			delayed_op_t* op2 = list_get_item(ops_queue, j - 1);
			bool keep = false;

			for (int k = 0; k < op2->N; k++)
				if (BITTEST(op2->write_flags, k) && op2->args[k].ptr_base != op->args[0].ptr_base)
					keep = true;

			if (!keep) {

				delayed_op_free(list_remove_item(ops_queue, j - 1));
				i = -1;
			}
		}
	}
}

static void delayed_merge_alloc_clear(list_t ops_queue)
{
	for (int i = list_count(ops_queue) - 1; i >= 0; i--) {

		delayed_op_t* op = list_get_item(ops_queue, i);

		if (delayed_op_is_alloc(op))
			list_remove_item(ops_queue, i);
		else
			continue;

		int j = i;

		while (list_count(ops_queue) > j && !delayed_ptr_required(list_get_item(ops_queue, j), op->args[0].ptr_base))
			j++;

		list_insert(ops_queue, op, j);

		if (j + 1 < list_count(ops_queue)) {

			delayed_op_t* op2 = list_get_item(ops_queue, j + 1);
			if (delayed_op_is_clear(op2)) {

				assert(!vptr_is_mem_allocated(op->args[0].ptr_base));

				assert(op2->args[0].ptr_base == op->args[0].ptr_base);
				delayed_op_free(list_remove_item(ops_queue, j + 1));
				vptr_set_clear(op->args[0].ptr_base);
				i = j + 1;
			}
		}
	}
}


static void delayed_optimize_alloc(list_t ops_queue)
{
	for (int i = list_count(ops_queue) - 1; i >= 0; i--) {

		delayed_op_t* op = list_get_item(ops_queue, i);

		if (delayed_op_is_alloc(op) && vptr_is_mem_allocated(op->args[0].ptr_base))
			delayed_op_free(list_remove_item(ops_queue, i));
	}

	delayed_merge_alloc_clear(ops_queue);

	for (int i = list_count(ops_queue) - 1; i >= 0; i--) {

		delayed_op_t* op = list_get_item(ops_queue, i);

		if (delayed_op_is_alloc(op))
			list_remove_item(ops_queue, i);
		else
			continue;

		if (vptr_is_mem_allocated(op->args[0].ptr_base)) {

			delayed_op_free(op);
			continue;
		}

		int j = i;

		while (list_count(ops_queue) > j && !delayed_ptr_required(list_get_item(ops_queue, j), op->args[0].ptr_base))
			j++;

		if (j == list_count(ops_queue)) {

			delayed_op_free(op);
			continue;
		}

		if (delayed_op_is_free(list_get_item(ops_queue, j))) {

			delayed_op_t* fop = list_remove_item(ops_queue, j);
			assert(fop->args[0].ptr_base == op->args[0].ptr_base);
			delayed_op_free(fop);
			delayed_op_free(op);
			continue;
		}

		list_insert(ops_queue, op, j);

		if (vptr_is_set_clear(op->args[0].ptr_base)) {

			vptr_unset_clear(op->args[0].ptr_base);
			list_insert(ops_queue, delayed_clear_create(op->args[0].N - 1, op->args[0].adims + 1, op->args[0].astrs + 1, op->args[0].ptr_base, (size_t)op->args[0].adims[0]), j + 1);
		}
	}
}

static void replace_inplace(list_t ops_queue, delayed_op_t* op_allo, delayed_op_t* op_free)
{
	void* nptr = op_allo->args[0].ptr;
	void* optr = op_free->args[0].ptr;

	debug_printf(delayed_dl, "\t%p -> %p\n", optr, nptr);

	for (int i = 0; i < list_count(ops_queue); i++) {

		delayed_op_t* op = list_get_item(ops_queue, i);

		if (delayed_op_is_alloc(op) && optr == op->args[0].ptr_base) {

			op->args[0].ptr = nptr;
			op->args[0].ptr_base = nptr;
			continue;
		}

		if (op == op_allo) {

			delayed_op_free(list_remove_item(ops_queue, i));
			i--;
			continue;
		}

		if (op == op_free) {

			delayed_op_free(list_remove_item(ops_queue, i));
			i--;
			continue;
		}

		for (int j = 0; j < op->N; j++) {

			assert(!CAST_MAYBE(delayed_chain_s, op));
			assert(!CAST_MAYBE(delayed_exp_slice_s, op));
			assert(!CAST_MAYBE(delayed_unloop_s, op));

			if (optr == op->args[j].ptr_base) {

				op->args[j].ptr = nptr + (op->args[j].ptr - optr);
				op->args[j].ptr_base = nptr;
			}
		}
	}
}

static bool delayed_optimize_inplace(list_t ops_queue)
{
	for (int i = 0; i < list_count(ops_queue); i++) {

		delayed_op_t* op = list_get_item(ops_queue, i);

		if (1 >= op->N)
			continue;

		if (0 == i || list_count(ops_queue) - 1 == i)
			continue;

		if (delayed_op_is_alloc(op) || delayed_op_is_free(op))
			continue;



		for (int ialloc = i - 1; ialloc >= 0; ialloc--) {

			delayed_op_t* op_allo = list_get_item(ops_queue, ialloc);
			if (!delayed_op_is_alloc(op_allo))
				break;

			for (int ifree = i + 1; ifree < list_count(ops_queue); ifree++) {

				delayed_op_t* op_free = list_get_item(ops_queue, ifree);
				if (!delayed_op_is_free(op_free))
					break;

				if (   !(CAST_DOWN(delayed_alloc_s, op_allo)->tmp_buffer)
				    || !(CAST_DOWN(delayed_free_s, op_free)->tmp_buffer)
		    || (op_allo->args[0].ptr_base == op_free->args[0].ptr_base)
		    || (!vptr_is_same_type(op_allo->args[0].ptr_base, op_free->args[0].ptr_base))
		    || vptr_is_mem_allocated(op_allo->args[0].ptr_base) || vptr_is_mem_allocated(op_free->args[0].ptr_base)
		    || !((NULL != CAST_MAYBE(delayed_md_fun_s, op)) || (NULL != CAST_MAYBE(delayed_copy_s, op)) || (NULL != CAST_MAYBE(delayed_vptr_fun_s, op))))
			continue;

		if (vptr_is_set_clear(op_allo->args[0].ptr_base)) {

			bool read = false;

			for (int i = 0; i < op->N; i++)
				if (BITTEST(op->read_flags, i) && op->args[i].ptr_base == op_allo->args[0].ptr_base)
					read = true;

			if (read)
				continue;
		}

		bool inplace = true;

		for (int j = 0; j < op->N && inplace; j++) {

			if (!(op->args[j].ptr_base == op_allo->args[0].ptr_base) && !(op->args[j].ptr_base == op_free->args[0].ptr_base))
				continue;

			if (!op->args[j].fitting)
				inplace = false;
		}

		if (!inplace)
			continue;

				replace_inplace(ops_queue, op_allo, op_free);

				if (delayed_op_is_copy(op))
					delayed_op_free(list_remove_item(ops_queue, i - 1));

				delayed_optimize_inplace(ops_queue);

				return true;
			}
		}
	}

	return false;
}

static bool delayed_optimize_clear(list_t ops_queue)
{
	for (int i = 0; i < list_count(ops_queue); i++) {

		delayed_op_t* op = list_get_item(ops_queue, i);

		if (delayed_op_is_alloc(op) && vptr_is_set_clear(op->args[0].ptr_base)) {

			vptr_unset_clear(op->args[0].ptr_base);
			op = delayed_clear_create(op->args[0].N - 1, op->args[0].adims + 1, op->args[0].astrs + 1, op->args[0].ptr_base, (size_t)op->args[0].adims[0]);
			i++;
			list_insert(ops_queue, op, i);
		}
	}

	bool changed = false;

	for (int i = list_count(ops_queue) - 1; i >= 0; i--) {

		delayed_op_t* op = list_get_item(ops_queue, i);

		if (delayed_op_is_clear(op))
			list_remove_item(ops_queue, i);
		else
			continue;

		int j = i;

		while (list_count(ops_queue) > j && !delayed_ptr_required(list_get_item(ops_queue, j), op->args[0].ptr_base))
			j++;

		if (j == list_count(ops_queue)) {

			list_insert(ops_queue, op, j);
			continue;
		}

		delayed_op_t* op2 = list_get_item(ops_queue, j);

		bool read = false;
		bool write = false;

		for (int k = 0; k < op2->N; k++) {

			if (op2->args[k].ptr_base == op->args[0].ptr_base) {

				if (BITTEST(op2->read_flags, k))
					read = true;

				if (BITTEST(op2->write_flags, k) && subset(op->args[0], op2->args[k]))
					write = true;
			}
		}

		if (!read && write) {

			changed = true;
			delayed_op_free(op);
			continue;
		}

		if (   (delayed_op_is_copy(op2) || (NULL != CAST_MAYBE(delayed_circ_shift_s, op2)))
		    && op2->args[0].ptr_base != op->args[0].ptr_base
		    && subset(op2->args[1], op->args[0])) {

			changed = true;
			list_insert(ops_queue, delayed_clear_create(op2->args[0].N - 1, op2->args[0].adims + 1, op2->args[0].astrs + 1, op2->args[0].ptr, (size_t)op2->args[0].adims[0]), j + 1);
			delayed_op_free(list_remove_item(ops_queue, j));
			list_insert(ops_queue, op, j);
			i = j + 1;
			continue;
		}

		list_insert(ops_queue, op, j);
		auto md_op = CAST_MAYBE(delayed_md_fun_s, op2);

		if (NULL != md_op)  {

			bool stop = false;

			for (int k = 0; k < op2->N; k++) {

				if (BITTEST(op2->write_flags, k) && (op->args[0].ptr_base == op2->args[k].ptr_base)) {

					stop = true;
					break;
				}
			}

			if (stop)
				continue;

			delayed_op_t* nop = NULL;

			switch(md_op->offset) {

				case offsetof(struct vec_ops, add):

					if (subset(op2->args[1], op->args[0])) {

						nop = delayed_copy_create(op2->args[1].N - 1, op2->args[0].adims + 1, op2->args[0].astrs + 1, op2->args[0].ptr, op2->args[2].astrs + 1, op2->args[2].ptr, (size_t)op2->args[0].adims[0]);
						break;
					}

					if (subset(op2->args[2], op->args[0]))
						nop = delayed_copy_create(op2->args[1].N - 1, op2->args[0].adims + 1, op2->args[0].astrs + 1, op2->args[0].ptr, op2->args[1].astrs + 1, op2->args[1].ptr, (size_t)op2->args[0].adims[0]);

					break;

				case offsetof(struct vec_ops, sub):

					if (subset(op2->args[2], op->args[0]))
						nop = delayed_copy_create(op2->args[1].N - 1, op2->args[0].adims + 1, op2->args[0].astrs + 1, op2->args[0].ptr, op2->args[1].astrs + 1, op2->args[1].ptr, (size_t)op2->args[0].adims[0]);
					break;

				case offsetof(struct vec_ops, mul):
				case offsetof(struct vec_ops, div):
				case offsetof(struct vec_ops, zmul):
				case offsetof(struct vec_ops, zdiv):
				case offsetof(struct vec_ops, zmulc):

					if ((subset(op2->args[1], op->args[0])) || (subset(op2->args[2], op->args[0])))
						nop = delayed_clear_create(op2->args[0].N - 1, op2->args[0].adims + 1, op2->args[0].astrs + 1, op2->args[0].ptr, (size_t)op2->args[0].adims[0]);
					break;

				default:
					break;
			}

			if (NULL != nop) {

				changed = true;
				delayed_op_free(list_remove_item(ops_queue, j + 1));
				list_insert(ops_queue, nop, j + 1);
				i = j + 2;
			}
		}
	}

	return changed;
}

static void delayed_optimize_copy(list_t ops_queue)
{
	for (int i = list_count(ops_queue) - 1; i >= 0; i--) {

		delayed_op_t* op = list_get_item(ops_queue, i);

		if (delayed_op_is_copy(op))
			list_remove_item(ops_queue, i);
		else
			continue;

		//remove inplace copy
		if (op->args[0].ptr == op->args[1].ptr && (op->args[0].N == op->args[1].N) && md_check_equal_dims(op->args[0].N, op->args[0].astrs, op->args[1].astrs, ~0UL)) {

			delayed_op_free(op);
			continue;
		}

		int j = i;

		while (	   list_count(ops_queue) > j
			&& !delayed_ptr_required(list_get_item(ops_queue, j), op->args[0].ptr_base)
			&& !delayed_ptr_required_write(list_get_item(ops_queue, j), op->args[1].ptr_base))
			j++;

		list_insert(ops_queue, op, j);

		if (j + 1 == list_count(ops_queue))
			continue;

		bool read = false;
		bool write = false;

		delayed_op_t* op2 = list_get_item(ops_queue, j + 1);

		if (delayed_op_is_free(op2) && op2->args[0].ptr_base == op->args[1].ptr_base) {

			int k = -1;

			for (int l = j + 2; l < list_count(ops_queue); l++) {

				delayed_op_t* op3 = list_get_item(ops_queue, l);
				if (delayed_op_is_free(op3) && op3->args[0].ptr_base == op->args[0].ptr_base)
					k = l;
			}

			if (0 < k) {

				list_insert(ops_queue, list_remove_item(ops_queue, j + 1), k);
				i = j + 1;
				continue;
			}
		}

		for (int k = 0; k < op2->N; k++) {

			if (op2->args[k].ptr_base == op->args[0].ptr_base) {

				if (BITTEST(op2->read_flags, k))
					read = true;

				if (BITTEST(op2->write_flags, k) && subset(op->args[0], op2->args[k]))
					write = true;
			}
		}

		if (!read && write) {

			delayed_op_free(list_remove_item(ops_queue, j));
			continue;
		}

		if (   op->args[0].fitting && op->args[1].fitting
		    && vptr_is_same_type(op->args[0].ptr, op->args[1].ptr)) {

			bool rep = false;

			for (int k = 0; k < op2->N; k++) {


				if (op2->args[k].ptr_base == op->args[0].ptr_base && !BITTEST(op2->write_flags, k) && BITTEST(op2->read_flags, k)) {

					op2->args[k].ptr = op->args[1].ptr_base + (op2->args[k].ptr - op2->args[k].ptr_base);
					op2->args[k].ptr_base = op->args[1].ptr_base;
					rep = true;
				}
			}

			if (rep)
				i = j + 1;
		}
	}
}


void delayed_optimize_queue(list_t ops_queue)
{
	if (debug_level >= delayed_dl) {

		if (0 < list_count(ops_queue))
			debug_printf(delayed_dl, "Optimize queue with %d operations\n", list_count(ops_queue));

		for (int i = 0; i < list_count(ops_queue); i++) {

			const char* op = print_delayed_fun_f((delayed_op_t*)list_get_item(ops_queue, i), true);
			debug_printf(delayed_dl, "  Delayed op %s\n", op);
			xfree(op);
		}
	}

	delayed_optimize_alloc(ops_queue);
	delayed_optimize_free(ops_queue);

	delayed_optimize_clear(ops_queue);
	delayed_optimize_copy(ops_queue);

	delayed_optimize_alloc(ops_queue);
	delayed_optimize_free(ops_queue);
	delayed_independent_overwrite_buffer(ops_queue);

	delayed_optimize_alloc(ops_queue);
	delayed_optimize_free(ops_queue);
	delayed_set_tmp_buffer(ops_queue);

	debug_printf(delayed_dl, "Delayed Optimize: Replace inplace:\n");

	bool repeat = true;

	while (repeat) {

		repeat = false;

		delayed_optimize_alloc(ops_queue);
		delayed_optimize_free(ops_queue);

		if (delayed_optimize_inplace(ops_queue))
			repeat = true;

		if (delayed_optimize_clear(ops_queue))
			repeat = true;

		delayed_optimize_copy(ops_queue);
	}

	delayed_optimize_alloc(ops_queue);
	delayed_optimize_free(ops_queue);

	if (debug_level >= delayed_dl) {

		if (0 < list_count(ops_queue))
			debug_printf(delayed_dl, "\nFirst step: Optimized queue with %d operations\n", list_count(ops_queue));

		for (int i = 0; i < list_count(ops_queue); i++) {

			delayed_nested_level++;
			delayed_nested_level++;

			const char* op = print_delayed_fun_f((delayed_op_t*)list_get_item(ops_queue, i), true);
			debug_printf(delayed_dl, "%s\n", op);
			xfree(op);

			delayed_nested_level--;
			delayed_nested_level--;
		}
	}
}











struct blocking_s
{
	const void* ptr;
	unsigned long lflags;
};

static void delayed_optimize_set_blocking(list_t ops_queue, unsigned long lflags)
{
	delayed_optimize_alloc(ops_queue);

	//chain clear with following operations to avoid read flags in chain
	for (int i = list_count(ops_queue) - 1; i >= 0; i--) {

		delayed_op_t* op = list_get_item(ops_queue, i);

		if (!delayed_op_is_clear(op))
			continue;

		int j = i;
		op = list_remove_item(ops_queue, j);
		while (j < list_count(ops_queue) && !delayed_ptr_required(list_get_item(ops_queue, j), op->args[0].ptr_base))
			j++;

		if (j == list_count(ops_queue)) {

			list_insert(ops_queue, op, j);
			continue;
		}

		delayed_op_t* op2 = list_get_item(ops_queue, j);

		bool read = false;

		for (int i = 0; i < op2->N; i++)
			if (BITTEST(op2->read_flags, i) && op2->args[i].ptr_base == op->args[0].ptr_base)
				read = true;

		if (!read) {

			list_insert(ops_queue, op, j);
			continue;
		}

		if (op2->D > op->D) {

			long dims[op2->D];
			long strs[op2->D];

			md_singleton_dims(op2->D, dims);
			md_singleton_strides(op2->D, strs);

			md_copy_dims(op->D, dims, op->args[0].adims + 1);
			md_copy_strides(op->D, strs, op->args[0].astrs + 1);

			delayed_op_t* top = delayed_clear_create(op2->D, dims, strs, op->args[0].ptr, (size_t)op->args[0].adims[0]);
			delayed_op_free(op);
			op = top;
		}

		if (   md_check_compat(MIN(op->D, op2->D), ~0UL, op->ldims, op2->ldims)
		    && (1 == md_calc_size(op->D - MIN(op->D, op2->D), op->ldims + MIN(op->D, op2->D)))
		    && (1 == md_calc_size(op2->D - MIN(op->D, op2->D), op2->ldims + MIN(op->D, op2->D)))
		    && (CAST_MAYBE(delayed_md_fun_s, op2) || CAST_MAYBE(delayed_copy_s, op2) || CAST_MAYBE(delayed_vptr_fun_s, op2))) {

			op2 = list_remove_item(ops_queue, j);
			op = delayed_op_unloop(op, md_nontriv_dims(op2->D, op2->ldims), lflags);

			long dims[MAX(op->D, op2->D)];
			long pos[MAX(op->D, op2->D)];
			md_singleton_dims(MAX(op->D, op2->D), dims);
			md_set_dims(MAX(op->D, op2->D), pos, 0);
			md_copy_dims(op2->D, dims, op2->ldims);

			op = delayed_op_expand_slice_create(op, MAX(op->D, op2->D), dims, pos);
			list_insert(ops_queue, delayed_op_chain(2, (delayed_op_t*[2]) { op, op2 }), j);
		} else {

			list_insert(ops_queue, op, j);
		}
	}


	for (int i = 0; i < list_count(ops_queue); i++) {

		delayed_op_t* op = list_get_item(ops_queue, i);

		if (!delayed_op_is_alloc(op) || !CAST_DOWN(delayed_alloc_s, op)->tmp_buffer)
			continue;

		unsigned long lflags = md_nontriv_strides(op->D, op->ldims) & vptr_delayed_loop_flags(op->args[0].ptr_base);

		for (int j = i + 1; j < list_count(ops_queue); j++) {

			delayed_op_t* op2 = list_get_item(ops_queue, j);

			for (int k = 0; k < op2->N; k++) {

				if (op->args[0].ptr_base == op2->args[k].ptr_base) {

					unsigned long lflags2 = (op2->args[k].sflags / 2);
					lflags2 |= (op2->args[k].lflags / 2) & md_nontriv_dims(op2->D, op2->ldims);

					lflags &= lflags2;

					BITSET(op2->buffer_flags, k);
				}
			}

			if (delayed_op_is_free(op2) && (op->args[0].ptr_base == op2->args[0].ptr_base)) {

				md_select_dims(op->D, lflags & lflags, op->ldims, op->ldims);
				md_select_dims(op2->D, lflags & lflags, op2->ldims, op2->ldims);
				break;
			}
		}
	}
}

static inline bool delayed_arg_depends_on(struct delayed_op_arg_s arg1, struct delayed_op_arg_s arg2)
{
	if (arg1.ptr_base != arg2.ptr_base)
		return false;

	int N = MIN(arg1.N, arg2.N);
	unsigned long sflags = arg1.sflags & arg2.sflags;

	for (int k = 0; k < N; k++)
		if (MD_IS_SET(sflags, k) && (arg1.mpos[k] != arg2.mpos[k]))
			return false;

	if (!md_overlap(arg1.N, arg1.adims, arg1.astrs, arg1.ptr, 1, arg2.N, arg2.adims, arg2.astrs, arg2.ptr, 1))
		return false;

	return true;
}



extern long num_chunk_size;


static list_t ops_queue_to_graph(list_t ops_queue)
{
	list_t graph = list_create();

	for (int i = 0; i < list_count(ops_queue); i++) {

		delayed_op_t* op = list_get_item(ops_queue, i);

		const char* name = (NULL != debug_graph_path  && (0 == mpi_get_rank())) ? print_delayed_fun_f(op, true) : NULL;
		list_append(graph, enode_create(name, op));

		for (int k = 0; k < op->N; k++) {

			for (int j = i - 1; j >= 0; j--) {

				delayed_op_t* op2 = list_get_item(ops_queue, j);
				bool overwritten = false;

				for (int l = 0; l < op2->N && !overwritten; l++)
					if (BITTEST(op2->write_flags, l) || BITTEST(op->write_flags, k))
						if (delayed_arg_depends_on(op->args[k], op2->args[l])) {

							unsigned long loop_flags = ~0ul;
							loop_flags &= md_nontriv_dims(op->D, op->ldims);
							loop_flags &= md_nontriv_dims(op2->D, op2->ldims);
							loop_flags &= op->args[k].lflags / 2;
							loop_flags &= op2->args[l].lflags / 2;

							auto ops = CAST_MAYBE(delayed_exp_slice_s, op);
							auto op2s = CAST_MAYBE(delayed_exp_slice_s, op2);

							if (NULL != ops && NULL != op2s && (ops->flags == op2s->flags) && (op2s->index <= ops->index))
								loop_flags |= op2s->flags;

							enode_add_dependency(list_get_item(graph, j), list_get_item(graph, i));


							if (BITTEST(op2->write_flags, l) && op2->args[l].fitting)
								overwritten = true;
						}

				if (overwritten)
					break;
			}
		}
	}

	return graph;
}

void debug_mpeak_queue(int dl, list_t ops_queue, bool node)
{
	if (0 < list_count(ops_queue))
		debug_printf(dl, "Peak memory of queueu with %d operations\n", list_count(ops_queue));

	if (dl > debug_level)
		return;

	long mchange = 0;
	long mpeak = 0;

	for (int i = 0; i < list_count(ops_queue); i++) {

		delayed_op_t* op = NULL;

		if (node)
			op = (void*)((enode_t)list_get_item(ops_queue, i))->data;
		else
			op = list_get_item(ops_queue, i);

		mpeak = MAX(mpeak, mchange + op->mpeak);

		const char* ops = print_delayed_fun_f(op, false);
		debug_printf(dl, "%d: %ld - %s\n", i, mchange + op->mpeak, ops);
		mchange += op->mchange;
		xfree(ops);
	}

	debug_printf(dl, "Total peak: %ld\n", mpeak);
}

long compute_mpeak(list_t ops_queue, bool node)
{
	long mpeak = 0;
	long mchange = 0;

	for (int i = 0; i < list_count(ops_queue); i++) {

		delayed_op_t* op = NULL;

		if (node)
			op = (void*)((enode_t)list_get_item(ops_queue, i))->data;
		else
			op = list_get_item(ops_queue, i);

		mpeak = MAX(mpeak, mchange + op->mpeak);
		mchange += op->mchange;
	}

	return mpeak;
}


static bool optimize_graph_simple(list_t ops_queue, bool multiple, bool include_loop, unsigned long lflags)
{
	list_t graph = ops_queue_to_graph(ops_queue);
	egraph_set_active(graph);

	for (int i = 0; i < list_count(graph); i++) {

		enode_t fnode = list_get_item(graph, i);
		delayed_op_t* fop = (void*)fnode->data;

		if (!(delayed_op_is_free(fop) && (CAST_DOWN(delayed_free_s, fop)->tmp_buffer) && (1 < md_calc_size(fop->D, fop->ldims))))
			continue;

		egraph_reset_between(graph);

		list_t open_free = list_create();
		list_append(open_free, fop->args[0].ptr_base);

		egraph_set_ancestors(fnode);

		for (int j = i - 1; j >= 0 && (0 < list_count(open_free)); j--) {

			enode_t node = list_get_item(graph, j);
			delayed_op_t* op = (void*)node->data;

			if (delayed_op_is_alloc(op) && (-1 != list_get_first_index(open_free, op->args[0].ptr_base, NULL))) {

				egraph_set_descendants(node);
				list_remove_item(open_free, list_get_first_index(open_free, op->args[0].ptr_base, NULL));
			}

			if (   multiple && delayed_op_is_free(op) && (CAST_DOWN(delayed_free_s, op)->tmp_buffer)
			    && md_check_equal_dims(MIN(op->D, fop->D), op->ldims, fop->ldims, ~0UL)
			    && (md_calc_size(op->D, op->ldims) == md_calc_size(fop->D, fop->ldims))) {

				list_append(open_free, op->args[0].ptr_base);
				egraph_set_ancestors(node);
			}
		}

		assert(0 == list_count(open_free));
		list_free(open_free);

		list_t tmp_graph = list_create();
		for (int i = 0; i < list_count(graph); i++)
			list_append(tmp_graph, list_get_item(graph, i));

		egraph_sort_between(tmp_graph);

		long ldims[fop->D];
		md_select_dims(fop->D, lflags, ldims, fop->ldims);

		long old_mpeak = compute_mpeak(graph, true);
		long new_mpeak = compute_mpeak(tmp_graph, true);

		if (include_loop) {

			for (int j = 0; j < list_count(tmp_graph); j++) {

				enode_t node = list_get_item(tmp_graph, j);
				delayed_op_t* op = (void*)node->data;

				if (!node->active)
					continue;

				if (delayed_op_is_free(op) || delayed_op_is_alloc(op)) {

					op->mchange /= md_calc_size(fop->D, ldims);
					op->mpeak /= md_calc_size(fop->D, ldims);
				}
			}

			new_mpeak = compute_mpeak(tmp_graph, true);

			for (int j = 0; j < list_count(tmp_graph); j++) {

				enode_t node = list_get_item(tmp_graph, j);
				delayed_op_t* op = (void*)node->data;

				if (!node->active)
					continue;

				if (delayed_op_is_free(op) || delayed_op_is_alloc(op)) {

					op->mchange *= md_calc_size(fop->D, ldims);
					op->mpeak *= md_calc_size(fop->D, ldims);
				}
			}
		}

		if (new_mpeak > old_mpeak) {

//			debug_printf(DP_INFO, "Could not loop %s due to peak memory:\n", print_delayed_fun_f(fop, false));
//			debug_mpeak_queue(DP_INFO, tmp_graph, true);

			list_free(tmp_graph);
			continue;
		}

		unsigned long compat_flags = md_nontriv_dims(fop->args[0].N - 1, fop->args[0].adims + 1) & ~md_nontriv_dims(fop->D, fop->ldims);
		bool stop = false;

		for (int j = 0; j < list_count(tmp_graph); j++) {

			enode_t node = list_get_item(tmp_graph, j);
			delayed_op_t* op = (void*)node->data;

			if (!node->active)
				continue;

			if (!md_check_compat(MIN(op->D, fop->D), compat_flags, fop->ldims, op->ldims)
			    || (1 != md_calc_size(op->D - MIN(op->D, fop->D), op->ldims + MIN(op->D, fop->D)))
			    || (1 != md_calc_size(fop->D - MIN(op->D, fop->D), fop->ldims + MIN(op->D, fop->D)))) {

				stop = true;

//				debug_printf(DP_INFO, "Could not loop %s due to dims of\n %s:\n", print_delayed_fun_f(fop, false), print_delayed_fun_f(op, false));
//				debug_mpeak_queue(DP_INFO, tmp_graph, true);

				break;
			}
		}

		if (stop) {

			list_free(tmp_graph);
			continue;
		}

		while (0 != list_count(ops_queue))
			list_pop(ops_queue);

		enode_t node = list_pop(tmp_graph);

		while (!node->active) {

			list_append(ops_queue, (void*)node->data);
			enode_free(node);
			node = list_pop(tmp_graph);
		}

		list_t chain = list_create();

		while (node && node->active) {

			list_append(chain, (void*)node->data);
			enode_free(node);
			node = list_pop(tmp_graph);
		}

		int N = list_count(chain);
		delayed_op_t* ops[N];
		list_to_array(N, (void**)ops, chain);

		for (int j = 0; j < N; j++) {

			if (delayed_op_is_free(ops[j]) || delayed_op_is_alloc(ops[j])) {

				ops[j]->mchange /= md_calc_size(fop->D, ldims);
				ops[j]->mpeak /= md_calc_size(fop->D, ldims);
			}
		}

		for (int j = 0; j < N; j++)
			ops[j] = delayed_op_unloop(ops[j], ~compat_flags, lflags);

		list_free(chain);

		delayed_op_t* nop = delayed_op_chain(N, ops);

		unsigned long loop_flags = 0;

		for (int j = 0; j < nop->N; j++)
			if (BITTEST(nop->buffer_flags, j))
				loop_flags |= (md_nontriv_dims(nop->args[j].N, nop->args[j].mdims) / 2);

		nop = delayed_op_unloop(nop, loop_flags, lflags);

		list_append(ops_queue, nop);

		while (node) {

			list_append(ops_queue, (void*)node->data);
			enode_free(node);
			node = list_pop(tmp_graph);
		}

		list_free(tmp_graph);
		list_free(graph);

		return true;
	}

	while (0 < list_count(graph)) {

		enode_t node = list_pop(graph);
		enode_free(node);
	}

	list_free(graph);
	return false;
}

void delayed_optimize_queue_blocking(list_t ops_queue)
{
	if (0 == bart_delayed_loop_flags)
		return;

	delayed_optimize_set_blocking(ops_queue, bart_delayed_loop_flags);

	for (int i = 0; i < list_count(ops_queue); i++)
		list_insert(ops_queue, delayed_op_unloop_unloopable(list_remove_item(ops_queue, i)), i);

	for (int i = 0; i < list_count(ops_queue); i++)
		list_insert(ops_queue, delayed_op_expand_slice(list_remove_item(ops_queue, i)), i);

	unsigned long loop_flags = bart_delayed_loop_flags;

	for (int i = 0; i < list_count(ops_queue); i++)
		list_insert(ops_queue, delayed_op_unloop(list_remove_item(ops_queue, i), loop_flags, bart_delayed_loop_flags), i);

	do {
		long mpeak = compute_mpeak(ops_queue, false);

		while (optimize_graph_simple(ops_queue, false, false, loop_flags));
		while (optimize_graph_simple(ops_queue, true, false, loop_flags));
		while (optimize_graph_simple(ops_queue, false, true, loop_flags));
		while (optimize_graph_simple(ops_queue, true, true, loop_flags));

		assert(mpeak >= compute_mpeak(ops_queue, false));

		int i = 15;
		while (-1 == bart_delayed_loop_dims[i] || !MD_IS_SET(loop_flags, bart_delayed_loop_dims[i]))
			i--;

		loop_flags &= ~MD_BIT(bart_delayed_loop_dims[i]);

		for (int i = 0; i < list_count(ops_queue); i++)
			list_insert(ops_queue, delayed_op_unloop(list_remove_item(ops_queue, i), loop_flags, bart_delayed_loop_flags), i);

	} while (0 != loop_flags);

	if (debug_level >= delayed_dl) {

		if (0 < list_count(ops_queue))
			debug_printf(delayed_dl, "\nSecond step: Optimized queue with %d operations\n", list_count(ops_queue));

		for (int i = 0; i < list_count(ops_queue); i++) {

			delayed_nested_level++;
			delayed_nested_level++;

			const char* op = print_delayed_fun_f((delayed_op_t*)list_get_item(ops_queue, i), true);
			debug_printf(delayed_dl, "%s\n", op);
			xfree(op);

			delayed_nested_level--;
			delayed_nested_level--;
		}
	}

	for (int i = 0; i < list_count(ops_queue); i++) {

		delayed_op_t* op = list_remove_item(ops_queue, i);
		if (0 < op->mpeak)
			op = delayed_op_unloop(op, 0, bart_delayed_loop_flags);

		list_insert(ops_queue, op, i);
	}
}

static void delayed_op_append_node(list_t graph, delayed_op_t* op)
{
	if (CAST_MAYBE(delayed_chain_s, op)) {

		for (int i = 0; i < CAST_MAYBE(delayed_chain_s, op)->M; i++)
			delayed_op_append_node(graph, CAST_MAYBE(delayed_chain_s, op)->ops[i]);

		return;
	}

	if (CAST_MAYBE(delayed_unloop_s, op)) {

		delayed_op_append_node(graph, CAST_MAYBE(delayed_unloop_s, op)->op);
		return;
	}

	if (CAST_MAYBE(delayed_exp_slice_s, op)) {

		delayed_op_append_node(graph, CAST_MAYBE(delayed_exp_slice_s, op)->op);
		return;
	}

	const char* name = print_delayed_fun_f(op, true);
	list_append(graph, enode_create(name, op));

	for (int k = 0; k < op->N; k++) {

		bool done = false;

		for (int l = 0; l < k; l++)
			if (op->args[k].ptr_base == op->args[l].ptr_base)
				done = true;

		if (done)
			continue;

		for (int j = list_count(graph) - 2; j >= 0; j--) {

			enode_t node = list_get_item(graph, j);
			delayed_op_t* op2 = (void*)node->data;

			bool found = false;
			for (int l = 0; l < op2->N; l++)
				if (op2->args[k].ptr_base == op->args[l].ptr_base)
					found = true;

			if (found) {

				enode_add_dependency(node, list_get_item(graph, list_count(graph) - 1));
				break;
			}
		}
	}
}


static void delayed_export_queue(list_t ops_queue, const char* name)
{
	list_t graph = list_create();

	for (int i = 0; i < list_count(ops_queue); i++)
		delayed_op_append_node(graph, list_get_item(ops_queue, i));

	export_egraph_dot(name, graph);

	while (0 < list_count(graph))
		enode_free(list_pop(graph));

	list_free(graph);
}


void delayed_compute_debug(const char* name)
{
	queue_t queue = get_global_queue();

	list_t ops = queue->ops;

	if (0 == list_count(ops))
		return;

	queue->compute = true;

	delayed_optimize_queue(ops);
	delayed_optimize_queue_blocking(ops);

	delayed_export_queue(ops, name);

	delayed_op_t* op = list_pop(ops);

	while (NULL != op) {

		delayed_op_exec(op, 0, 0);
		delayed_op_free(op);
		op = list_pop(ops);
	}

	queue->compute = false;

	release_global_queue(queue);
}

