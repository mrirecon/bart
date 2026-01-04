/* Copyright 2026. TU Graz. Institute of Biomedical Imaging.
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
#include <stdio.h>
#include <math.h>
#include <errno.h>

#include "misc/list.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/egraph.h"
#include "misc/types.h"
#include "misc/mmio.h"
#include "misc/lock.h"

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

#define STRING_MEM_SIZE(size)							\
({										\
	double _fsize = (size);							\
	int _idx = 0;								\
	const char* str_byte[] = { "B", "KB", "MB", "GB", "TB" } ;		\
	while (_idx + 1 < (int)ARRAY_SIZE(str_byte) && 1000 <= fabs(_fsize)) {	\
										\
		_idx++;								\
		_fsize /= 1024;							\
	}									\
	int len = snprintf(NULL, 0, "%.3g %s", _fsize, str_byte[_idx]);		\
	char* _ret = alloca(((unsigned int)len + 1) * sizeof(char));		\
	sprintf(_ret, "%.3g %s", _fsize, str_byte[_idx]);			\
	_ret;									\
})

#define MAX_DIMS 64
#define MAX_WORKER 128

struct queue_s;
struct delayed_op_s;
struct delayed_op_arg_s;
struct ldim_s;

typedef struct queue_s queue_t;
typedef struct delayed_op_s delayed_op_t;


int delayed_dl = -1;
const char* debug_graph_path = NULL;
static struct queue_s* global_queue[MAX_WORKER] = { [0 ... MAX_WORKER - 1] = NULL };
static __thread int delayed_nested_level = 0;

struct queue_s {

	bart_lock_t* lock;
	bool compute;
	list_t ops;
};

typedef void (*delayed_op_fun_t)(delayed_op_t* op, unsigned long slice_flag, long pos[MAX_DIMS]);
typedef const char* (*delayed_op_debug_t)(delayed_op_t* op, bool nested);
typedef void (*delayed_op_del_t)(const delayed_op_t* op);

struct delayed_op_arg_s {

	void* ptr;
	void* ptr_base;

	int N;
	long adims[MAX_DIMS];
	long astrs[MAX_DIMS];
	size_t asize;

	long mdims[MAX_DIMS];
	size_t msize;
	long mpos[MAX_DIMS];

	unsigned long lflags;		// memory is accessed slice wise, i.e. memory position
					// corresponds to access position and the pointer is not
					// aliased by another position
	unsigned long sflags;   	// only a slice of memory is accessed
	unsigned long non_first_pos_flags;

	bool full_access;	// all memory is accessed
	bool fitting;		// memory access fits into the underlying allocation
	bool read;
	bool write;
};

struct ldim_s {

	int D;

	// loop dim should be exposed and can become a loop dim
	unsigned long loop_flags;
	long dims[MAX_DIMS];

	// when looped, the function should only be executed in a selected slice
	unsigned long slice_flags;
	long slice_pos[MAX_DIMS];
};

struct delayed_op_s {

	TYPEID* TYPEID;

	int D;
	struct ldim_s loop;

	int N;
	struct delayed_op_arg_s* args;

	delayed_op_fun_t fun;
	delayed_op_del_t del;
	delayed_op_debug_t debug;

	long mchange;
	long mpeak;
};

static void delayed_op_exec(delayed_op_t* op, unsigned long slice_flags, long pos[MAX_DIMS]);

/*
 * Basic idea of looping with delayed operations:
 * With the function "delayed_op_exec", we can execute a slice of a delayed operation. For each
 * delayed operation, we store information about which dimensions can be sliced and, hence, looped over
 * in the struct ldim_s.
 *
 * Here, loop_flags indicates which dimensions can be sliced and dims indicate the size of the dimension,
 * i.e. the maximum number of slices in the respective dimensions. Slice_flags and position indicate
 * special operations, which themselves only access a slice of the data.
 *
 * To be slicable, a delayed operation must fulfill three conditions for this dimension:
 *  1) The operation itself must support slicing, i.e. independent execution on slices.
 *     For example, the FFT opearion can only be sliced along dimensions, which are not transformed.
 *  2) The strides are consisten with the memory layout of the accessed vptr. This means:
 *     a) strides of the access equal the strides of the vptr
 *     b) when an operation is sliced along a dimension, the position in the accessed memory equals
 *        the position (delayed_op_exec argument) in this dimension.
 *        An exception is when the vptr has a singleton dimension, in this case, any position is valid
 *        to allow for reductions.
 *  3) All vptr arguments must have compatible sizes in this dimension, i.e. the size either equals
 *     the size of the loop dimension or is a singleton dimension.
 *
 * This mapping of operations to the memory layout simplifies detection of chainable operations
 * significantly, as when two operations are slicable along the same dimension, they can be chained
 * and the chain can be looped over. An caveat are singleton dimensions, as here, a position
 * can be accessed out of loop order, i.e., position 0 in memory is accessed for all loop positions.
*/

static void delayed_queue(delayed_op_t* x);

static void delayed_op_free(const delayed_op_t* x);
static const char* print_delayed_fun_f(delayed_op_t* op, bool nested);
static void delayed_op_exec_resolve(delayed_op_t* op, int D, int N, long dims[N][D], void* ptr[N], unsigned long slice_flags, long pos[MAX_DIMS]);

static struct delayed_op_s* delayed_op_alloc_create(const void* ptr, int N, const long dims[N], size_t size);
static struct delayed_op_s* delayed_op_free_create(const void* ptr, int N, const long dims[N], size_t size);
static struct delayed_op_s* delayed_op_clear_create(int D, const long dim[D], const long str[D], void* ptr, size_t size);
static struct delayed_op_s* delayed_op_copy_create(int D, const long dim[D], const long ostr[D], void* optr, const long istr[D], const void* iptr, size_t size);
static struct delayed_op_s* delayed_op_circ_shift_create(int D, const long dimensions[D], const long center[D], const long str1[D], void* dst, const long str2[D], const void* src, size_t size);
static struct delayed_op_s* delayed_op_md_fun_create(enum delayed_md_fun_type type, size_t offset, int D, const long dim[D], int N, const long* strs[N], const void* ptr[N], const size_t sizes[N]);
static struct delayed_op_s* delayed_op_vptr_fun_create(vptr_fun_t fun, vptr_fun_data_t* data, int N, int D, unsigned long lflags, unsigned long wflags, unsigned long rflags, const long* dims[N], const long* strs[N], void* ptr[N], size_t sizes[N], bool resolve);
static struct delayed_op_s* delayed_op_chain_create(list_t ops_list, unsigned long seq_flags);

static inline bool delayed_arg_same_access(struct delayed_op_arg_s arg1, struct delayed_op_arg_s arg2);
static inline bool delayed_arg_depends_on(struct delayed_op_arg_s arg1, struct delayed_op_arg_s arg2);
static inline bool delayed_arg_subset(struct delayed_op_arg_s a, struct delayed_op_arg_s b);

static void delayed_optimize_set_tmp_buffer(list_t ops_queue);
static void delayed_optimize_alloc(list_t ops_queue);;
static void delayed_optimize_free(list_t ops_queue);;
static void delayed_optimize_new_buffer_on_overwrite(list_t ops_queue);
static bool delayed_optimize_inplace(list_t ops_queue);
static bool delayed_optimize_clear(list_t ops_queue);
static void delayed_optimize_copy(list_t ops_queue);


static unsigned long queue_compute_loop_flags(long loop_dims[MAX_DIMS], list_t ops_queue);;
static bool delayed_ptr_required(delayed_op_t* op, const void* ptr);
static bool delayed_ptr_required_write(delayed_op_t* op, const void* ptr);

static egraph_t delayed_op_queue_to_graph(list_t ops_queue);
static unsigned long delayed_tmp_buffer_compute_loop_flags(egraph_t graph);

static void delayed_optimize_queue_looping_flags(egraph_t graph, bool multiple, unsigned long lflags);
static void delayed_op_append_node(egraph_t graph, delayed_op_t* op);




/******************************************************************************
 * General functions for delayed computation and queue management
 ******************************************************************************/



static void queue_init(void)
{
	if (-1 == delayed_dl) {

		delayed_dl = DP_DEBUG4;

		char* str = getenv("BART_DELAYED_DEBUG_LEVEL");

		if (NULL != str) {

			errno = 0;
			long r = strtol(str, NULL, 10);

			if ((errno == 0) && (0 <= r) && (r < 10))
				delayed_dl = r;

			errno = 0;
		}
	}

	if (NULL == global_queue[cfl_loop_worker_id()]) {

#pragma omp critical
		if (NULL == global_queue[cfl_loop_worker_id()]) {

			struct queue_s* tmp = TYPE_ALLOC(struct queue_s);
			tmp->ops = list_create();
			tmp->compute = false;
			tmp->lock = bart_lock_create();
			global_queue[cfl_loop_worker_id()] = tmp;
		}
	}
}

struct queue_s* get_global_queue(void)
{
	queue_init();

	struct queue_s* queue = global_queue[cfl_loop_worker_id()];
	bart_lock(queue->lock);

	return queue;
}

void release_global_queue(struct queue_s* queue)
{
	bart_unlock(queue->lock);
}

list_t get_delayed_op_list(struct queue_s* queue)
{
	return queue->ops;
}

void queue_set_compute(struct queue_s* queue, bool compute)
{
	queue->compute = compute;
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

void delayed_compute(const void* /*ptr*/)
{
	struct queue_s* queue = get_global_queue();

	list_t ops = queue->ops;
	queue->compute = true;

#ifdef USE_CUDA
	if (!cuda_is_stream_default())
		error("Delayed computation is incompatible with using multiple CUDA streams!\n");
#endif

	delayed_optimize_queue(ops);

	if (0 < list_count(ops))
		debug_printf(delayed_dl, "Execute queue with %d operations\n", list_count(ops));

	long pos[MAX_DIMS] = { 0 };

	delayed_op_t* op = list_pop(ops);

	while (NULL != op) {

		delayed_op_exec(op, 0, pos);
		delayed_op_free(op);
		op = list_pop(ops);
	}

	queue->compute = false;
	release_global_queue(queue);
}

void debug_delayed_queue(int dl, list_t ops_queue, bool nested)
{
	if (-1 == debug_level)			// force initialization of environment
		debug_printf(DP_INFO, "%s", "");

	if (debug_level < dl)
		return;

	if (NULL == ops_queue) {

		struct queue_s* queue = global_queue[cfl_loop_worker_id()];

		bart_lock(queue->lock);
		queue->compute = true;

		debug_delayed_queue(dl, queue->ops, nested);

		queue->compute = false;
		bart_unlock(queue->lock);
	} else {

		delayed_nested_level++;

		for (int i = 0; i < list_count(ops_queue); i++) {

			const char* op = print_delayed_fun_f(list_get_item(ops_queue, i), nested);
			debug_printf(dl, "%s\n", op);
			xfree(op);
		}

		delayed_nested_level--;
	}
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

	struct queue_s* queue = global_queue[cfl_loop_worker_id()];

	bart_lock(queue->lock);
	list_append(queue->ops, (void*)x);
	bart_unlock(queue->lock);
}



/******************************************************************************
 * Queueing external delayed operations
 ******************************************************************************/


void delayed_alloc(const void* ptr, int N, const long dims[N], size_t size)
{
	delayed_queue(delayed_op_alloc_create(ptr, N, dims, size));
}

void delayed_free(const void* ptr, int N, const long dims[N], size_t size)
{
	struct queue_s* queue = global_queue[cfl_loop_worker_id()];

	bart_lock(queue->lock);

	if (0 == list_count(queue->ops)) {

		queue->compute = true;
		vptr_free(ptr);
		queue->compute = false;

		bart_unlock(queue->lock);
	} else {

		bart_unlock(queue->lock);
		delayed_queue(delayed_op_free_create(ptr, N, dims, size));
	}

}

bool delayed_queue_clear(int D, const long dim[D], const long str[D], void* ptr, size_t size)
{
	if (!is_delayed(ptr))
		return false;

	delayed_queue(delayed_op_clear_create(D, dim, str, ptr, size));

	return true;
}

bool delayed_queue_copy(int D, const long dim[D], const long ostr[D], void* optr, const long istr[D], const void* iptr, size_t size)
{
	if (is_delayed(optr) && is_delayed(iptr)) {

		delayed_queue(delayed_op_copy_create(D, dim, ostr, optr, istr, iptr, size));
		return true;
	}

	if (is_delayed(iptr))
		delayed_compute(iptr);

	if (is_delayed(optr)) {

		// we only need to compute if optr is already used in the current queue

		struct queue_s* queue = global_queue[cfl_loop_worker_id()];
		bart_lock(queue->lock);

		auto ops = queue->ops;
		for (int i = 0; i < list_count(ops); i++) {

			delayed_op_t* op = list_get_item(ops, i);

			if (delayed_op_is_alloc(op))
				continue;

			for (int j = 0; j < op->N; j++) {

				if (vptr_overlap(optr, op->args[j].ptr)) {

					bart_unlock(queue->lock);
					delayed_compute(optr);
					return false;
				}
			}
		}

		bart_unlock(queue->lock);
	}

	return false;
}

bool delayed_queue_circ_shift(int D, const long dimensions[D], const long center[D], const long str1[D], void* dst, const long str2[D], const void* src, size_t size)
{
	if (!is_delayed(dst) || !is_delayed(src))
		return false;

	delayed_queue(delayed_op_circ_shift_create(D, dimensions, center, str1, dst, str2, src, size));
	return true;
}

bool delayed_queue_make_op(enum delayed_md_fun_type type, size_t offset, int D, const long dim[D], int N, const long* strs[N], const void* ptr[N], const size_t sizes[N])
{
	for (int i = 0; i < N; i++)
		if (!is_delayed(ptr[i]))
			return false;

	for (int i = 0; i < N; i++)
		if (!vptr_is_init(ptr[i])) {

			vptr_debug(DP_INFO, ptr[i]);
			error("Vptr not initialized!\n");
		}

	delayed_queue(delayed_op_md_fun_create(type, offset, D, dim, N, strs, ptr, sizes));
	return true;
}

void exec_vptr_fun_delayed(vptr_fun_t fun, vptr_fun_data_t* data, int N, int D, unsigned long lflags, unsigned long wflags, unsigned long rflags, const long* dims[N], const long* strs[N], void* ptr[N], size_t sizes[N], bool resolve)
{
	delayed_queue(delayed_op_vptr_fun_create(fun, data, N, D, lflags, wflags, rflags, dims, strs, ptr, sizes, resolve));
}



/******************************************************************************
 * Definition of delayed operations
 ******************************************************************************/


static struct delayed_op_arg_s arg_create(int N, const long dims[N], const long strs[N], const void* ptr, size_t size, bool read, bool write)
{
	struct delayed_op_arg_s arg;

	arg.ptr = (void*)ptr;
	arg.ptr_base = (void*)ptr - vptr_get_offset(ptr);

	arg.read = read;
	arg.write = write;

	const struct vptr_shape_s* shape = vptr_get_shape(ptr);

	arg.N = MAX(N, shape->N);
	assert(arg.N <= MAX_DIMS);

	md_singleton_dims(MAX_DIMS, arg.adims);
	md_copy_dims(N, arg.adims, dims);
	arg.asize = size;

	md_singleton_strides(MAX_DIMS, arg.astrs);
	md_select_strides(N, md_nontriv_dims(N, dims), arg.astrs, strs);

	md_singleton_dims(MAX_DIMS, arg.mdims);
	md_copy_dims(shape->N, arg.mdims, shape->dims);
	arg.msize = shape->size;

	arg.non_first_pos_flags = ~md_nontriv_dims(arg.N, arg.mdims);

	md_set_dims(MAX_DIMS, arg.mpos, 0);
	md_unravel_index(arg.N, arg.mpos, ~0ul, arg.mdims, (arg.ptr - arg.ptr_base) / (long)arg.msize);

	long mdims[arg.N + 1];
	mdims[0] = (long)arg.msize;
	md_copy_dims(arg.N, mdims + 1, arg.mdims);

	unsigned long aflags[N];
	loop_access_dims(N, aflags, arg.adims, arg.astrs, arg.N + 1, mdims, (arg.ptr - arg.ptr_base));
	for (int k = 0; k < N; k++)
		aflags[k] /= 2;

	long mstrides[arg.N];
	md_calc_strides(arg.N, mstrides, arg.mdims, arg.msize);

	arg.sflags = ~0UL;
	arg.lflags = 0UL;

	for (int k = 0; k < N; k++)
		if ((0 == arg.mpos[k]) && (arg.astrs[k] == mstrides[k]) && ((arg.mdims[k] == arg.adims[k]) || 1 == arg.mdims[k]))
			arg.lflags |= MD_BIT(k);

	for (int k = 0; k < N; k++) {

		arg.lflags &= ~(aflags[k] & (~MD_BIT(k)));
		arg.sflags &= ~aflags[k];
	}

	arg.fitting =   md_check_equal_dims(arg.N, arg.adims, arg.mdims, ~0ul)
		     && md_check_equal_dims(arg.N, arg.astrs, MD_STRIDES(arg.N, arg.mdims, arg.msize), ~0ul)
		     && (arg.asize == arg.msize);

	arg.full_access = arg.fitting;

	if (arg.fitting)
		return arg;

	long tdims[arg.N];
	long tstrs[arg.N];
	md_copy_dims(arg.N, tdims, arg.adims);
	md_copy_dims(arg.N, tstrs, arg.astrs);
	long tsize = (long)arg.asize;

	for (int k = 0; k < arg.N; k++)
		tstrs[k] = labs(tstrs[k]);

	long (*tstrs2[1])[arg.N] = { &tstrs };
	int ND = optimize_dims_gpu(1, arg.N, tdims, tstrs2);

	for (int i = 0; i < ND; i++)
		if (tstrs[i] <= tsize)
			tsize += (tdims[i] - 1) * tstrs[i];

	assert(tsize <= md_calc_size(arg.N, arg.mdims) * (long)arg.msize);
	if (tsize == md_calc_size(arg.N, arg.mdims) * (long)arg.msize)
		arg.full_access = true;

	return arg;
}


static struct ldim_s ldim_init(int D, unsigned long lflags, int N, struct delayed_op_arg_s args[N])
{
	struct ldim_s ret;

	ret.D = D;
	for (int i = 0; i < N; i++)
		ret.D = MAX(ret.D, args[i].N);

	md_set_dims(MAX_DIMS, ret.dims, 1);
	md_set_dims(MAX_DIMS, ret.slice_pos, 0);

	// step 1: make slice flags loop flags, where possible
	ret.slice_flags = ~0UL;
	for (int i = 0; i < N; i++) {

		ret.slice_flags &= args[i].sflags;
		ret.slice_flags &= ~md_nontriv_dims(args[i].N, args[i].adims);
		md_max_dims(args[i].N, ret.slice_flags, ret.dims, ret.dims, args[i].mdims);
		md_max_dims(args[i].N, ret.slice_flags, ret.slice_pos, ret.slice_pos, args[i].mpos);
	}

	for (int i = 0; i < MAX_DIMS; i++)
		for (int j = 0; j < N; j++)
			if ((1 != args[j].mdims[i]) && ((ret.dims[i] != args[j].mdims[i]) || (ret.slice_pos[i] != args[j].mpos[i])))
				ret.slice_flags &= ~MD_BIT(i);

	for (int i = 0; i < N; i++)
		args[i].lflags |= ret.slice_flags;


	// step 2: compute loop flags
	ret.loop_flags = lflags | ~(MD_BIT(D) - 1);

	for (int i = 0; i < N; i++) {

		ret.loop_flags &= args[i].lflags;
		md_max_dims(ret.D, ret.loop_flags, ret.dims, ret.dims, args[i].mdims);
	}

	for (int i = 0; i < N; i++)
		for (int j = 0; j < ret.D; j++)
			if ((1 != args[i].mdims[j]) && (ret.dims[j] != args[i].mdims[j]))
				ret.loop_flags &= ~MD_BIT(j);

	md_select_dims(MAX_DIMS, ret.loop_flags, ret.dims, ret.dims);

	for (int i = 0; i < N; i++)
		args[i].non_first_pos_flags &= md_nontriv_dims(ret.D, ret.dims);

	return ret;
}


static void delayed_op_init(delayed_op_t* op, int D, unsigned long lflags,
			  int N, struct delayed_op_arg_s args[N],
			  long mchange, long mpeak,
			  delayed_op_fun_t fun, delayed_op_del_t del, delayed_op_debug_t debug)
{
	op->loop = ldim_init(D, lflags, N, args);

	op->N = N;

	op->D = D;
	for (int i = 0; i < N; i++)
		op->D = MAX(op->D, args[i].N);

	op->args = ARR_CLONE(struct delayed_op_arg_s[N], args);
	for (int i = 0; i < N; i++)
		op->args[i].N = MAX(op->D, op->args[i].N);

	op->mchange = mchange;
	op->mpeak = mpeak;

	op->fun = fun;
	op->debug = debug;
	op->del = del;
}

static void delayed_op_free(const delayed_op_t* x)
{
	if (NULL != x->del)
		(x->del)(x);

	xfree(x->args);

	xfree(x);
}

static void ptr_append_print_args(const char** ret, delayed_op_t* op)
{
	assert(0 < op->N);

	ptr_append_printf(ret, "(");

	for (int i = 0; i < op->N; i++)
		ptr_append_printf(ret, "%s%s%s%s: %p%s",
			op->args[i].read ? "R": "",
			op->args[i].write ? "W" : "",
			op->args[i].fitting ? "F" : "",
			op->args[i].full_access ? "A" : "",
			op->args[i].ptr_base ,(i < op->N - 1) ? ", " : ")");
}

static void ptr_append_print_loopable_accessdims(const char** ret, delayed_op_t* op, int i)
{
	unsigned long loop_flags = op->loop.loop_flags & md_nontriv_dims(op->args[i].N, op->args[i].adims);

	ptr_append_printf(ret, "[");

	int N = MAX(op->loop.D, op->args[i].N);

	while ((1 < N) && (1 == op->args[i].adims[N - 1]) && (1 == op->loop.dims[N - 1]))
		N--;

	for (int j = 0; j < N; j++) {

		if (1 < op->loop.dims[j] && MD_IS_SET(op->loop.slice_flags, j))
			ptr_append_printf(ret, " %ldS%ld", op->loop.dims[j], op->loop.slice_pos[j]);
		else
			ptr_append_printf(ret, " %ld%s", op->args[i].adims[j], MD_IS_SET(loop_flags, j) ? "L" : "");
	}

	ptr_append_printf(ret, " ]");
}

static const char* print_delayed_fun_f(delayed_op_t* op, bool nested)
{
	const char* ret;

	if (NULL != op->debug) {

		ret = (op->debug)(op, nested);
	} else {

		ret = ptr_printf("%s ", op->TYPEID->name);
		ptr_append_print_args(&ret, op);
		ptr_append_printf(&ret, " ");
		ptr_append_print_loopable_accessdims(&ret, op, 0);
	}

	const char* ret2 = ptr_printf("%*s%s", delayed_nested_level * 4, "", ret);
	xfree(ret);
	return ret2;
}


static void delayed_op_exec_resolve(delayed_op_t* op, int D, int N, long dims[N][D], void* ptr[N], unsigned long slice_flags, long pos[MAX_DIMS])
{
	assert(D == op->D);
	assert(N == op->N);

	for (int i = 0; i < N; i++) {

		if (NULL != dims)
			md_select_dims(D, ~slice_flags, dims[i], op->args[i].adims);

		ptr[i] = op->args[i].ptr + md_calc_offset(D, op->args[i].astrs, pos);
	}
}

static void delayed_op_exec(delayed_op_t* op, unsigned long slice_flags, long pos[MAX_DIMS])
{
	delayed_nested_level++;
	bool exec = true;

	for (int i = 0; (i < MAX_DIMS) && exec; i++) {

		if (!MD_IS_SET(slice_flags, i))
			continue;

		if ((1 == op->loop.dims[i]) && (0 != pos[i]))
			exec = false;

		if (MD_IS_SET(op->loop.loop_flags & op->loop.slice_flags, i) && (i < op->loop.D) && (op->loop.slice_pos[i] != pos[i]))
			exec = false;
	}

	slice_flags &= md_nontriv_dims(op->loop.D, op->loop.dims);

#if 0
	if (exec) {
		const char* prefix = ptr_printf("Exec delayed op ");
		const char* op_str = print_delayed_fun_f(op, false);
		debug_printf(delayed_dl, "%s%s %lu %lu ", prefix, op_str, slice_flags, op->loop.loop_flags & op->loop.slice_flags);
		xfree(op_str);
		xfree(prefix);
		debug_print_dims(DP_INFO, 32, pos);
	}

	if (!delayed_op_is_chain(op) && exec) {

		long dims[op->N][op->D];
		void* ptr[op->N];

		delayed_op_exec_resolve(op, op->D, op->N, dims, ptr, slice_flags, pos);

		for (int i = 0; i < op->N; i++)
			assert(!op->args[i].read || vptr_check_init(op->D, dims[i], op->args[i].astrs, ptr[i]));
	}
#endif

	if (exec)
		(op->fun)(op, slice_flags, pos);

	delayed_nested_level--;
}



/******************************************************************************
 * Impplementation of specific delayed operations
 ******************************************************************************/



struct delayed_op_alloc_s {

	delayed_op_t super;
	bool tmp_buffer;
};

static DEF_TYPEID(delayed_op_alloc_s);

static void delayed_op_alloc_fun(delayed_op_t* op, unsigned long flags, long /*pos*/[MAX_DIMS])
{
	if (!vptr_is_mem_allocated(op->args[0].ptr_base))
		vptr_set_loop_flags(op->args[0].ptr_base, flags & op->loop.loop_flags);
}

static const char* delayed_op_alloc_debug(delayed_op_t* op, bool /*nested*/)
{
	bool clear = vptr_is_set_clear(op->args[0].ptr_base);
	const char* ret = ptr_printf("%s (+%s)%s ", op->TYPEID->name, STRING_MEM_SIZE(op->mchange), clear ? " (cleared)" : "");

	ptr_append_print_args(&ret, op);
	ptr_append_printf(&ret, " ");
	ptr_append_print_loopable_accessdims(&ret, op, 0);

	return ret;
}

static struct delayed_op_s* delayed_op_alloc_create(const void* ptr, int N, const long dims[N], size_t size)
{
	PTR_ALLOC(struct delayed_op_alloc_s, op);
	SET_TYPEID(delayed_op_alloc_s, op);

	long strs[N];
	md_calc_strides(N, strs, dims, size);

	struct delayed_op_arg_s arg[1] = { arg_create(N, dims, strs, ptr, size, false, true) };

	delayed_op_init(CAST_UP(op), N, ~0UL, 1, arg, md_calc_size(N, dims) * (long)size, 0, delayed_op_alloc_fun, NULL, delayed_op_alloc_debug);

	op->tmp_buffer = false;

	return CAST_UP(PTR_PASS(op));
}

bool delayed_op_is_alloc(const delayed_op_t* op)
{
	return NULL != CAST_MAYBE(delayed_op_alloc_s, op);
}



struct delayed_op_free_s {

	delayed_op_t super;
	bool tmp_buffer;
};

static DEF_TYPEID(delayed_op_free_s);

static void delayed_op_free_fun(delayed_op_t* op, unsigned long flags, long pos[MAX_DIMS])
{
	long dims[1][op->D];
	void* ptr[1];

	delayed_op_exec_resolve(op, op->D, 1, dims, ptr, flags, pos);
	vptr_free_mem(op->D, dims[0], op->args[0].astrs, ptr[0], op->args[0].asize);
}

static void delayed_op_free_del(const delayed_op_t* op)
{
	vptr_free(CAST_DOWN(delayed_op_free_s, op)->super.args[0].ptr);
}

static const char* delayed_op_free_debug(delayed_op_t* op, bool /*nested*/)
{
	const char* ret = ptr_printf("%s (%s) ", op->TYPEID->name, STRING_MEM_SIZE(op->mchange));

	ptr_append_print_args(&ret, op);
	ptr_append_printf(&ret, " ");
	ptr_append_print_loopable_accessdims(&ret, op, 0);

	return ret;
}

static struct delayed_op_s* delayed_op_free_create(const void* ptr, int N, const long dims[N], size_t size)
{
	PTR_ALLOC(struct delayed_op_free_s, op);
	SET_TYPEID(delayed_op_free_s, op);

	long strs[N];
	md_calc_strides(N, strs, dims, size);

	struct delayed_op_arg_s arg[1] = { arg_create(N, dims, strs, ptr, size, false, true) };
	delayed_op_init(CAST_UP(op), N, ~0UL, 1, arg, -md_calc_size(N, dims) * (long)size, 0, delayed_op_free_fun, delayed_op_free_del, delayed_op_free_debug);

	op->tmp_buffer = false;

	return CAST_UP(PTR_PASS(op));
}

bool delayed_op_is_free(const delayed_op_t* op)
{
	return NULL != CAST_MAYBE(delayed_op_free_s, op);
}



struct delayed_op_copy_s {

	delayed_op_t super;
};

static DEF_TYPEID(delayed_op_copy_s);

static void delayed_op_copy_fun(delayed_op_t* op, unsigned long flags, long pos[MAX_DIMS])
{
	long dims[2][op->D];
	void* ptr[2];

	delayed_op_exec_resolve(op, op->D, 2, dims, ptr, flags, pos);
	md_copy2(op->D, dims[0], op->args[0].astrs, ptr[0], op->args[1].astrs, ptr[1], op->args[0].asize);
}

static struct delayed_op_s* delayed_op_copy_create(int D, const long dim[D], const long ostr[D], void* optr, const long istr[D], const void* iptr, size_t size)
{
	struct delayed_op_arg_s arg[2];

	arg[0] = arg_create(D, dim, ostr, optr, size, false, true);
	arg[1] = arg_create(D, dim, istr, iptr, size, true, false);

	PTR_ALLOC(struct delayed_op_copy_s, op);
	SET_TYPEID(delayed_op_copy_s, op);

	delayed_op_init(CAST_UP(op), D, ~0UL, 2, arg, 0, 0, delayed_op_copy_fun, NULL, NULL);

	return CAST_UP(PTR_PASS(op));
}

bool delayed_op_is_copy(const delayed_op_t* op)
{
	return NULL != CAST_MAYBE(delayed_op_copy_s, op);
}



struct delayed_op_circ_shift_s {

	delayed_op_t super;

	const long* center;
};

static DEF_TYPEID(delayed_op_circ_shift_s);

static void delayed_op_circ_shift_free(const delayed_op_t* op)
{
	xfree(CAST_DOWN(delayed_op_circ_shift_s, op)->center);
}

static void delayed_op_circ_shift_fun(delayed_op_t* op, unsigned long flags, long pos[MAX_DIMS])
{
	long dims[2][op->D];
	void* ptr[2];

	delayed_op_exec_resolve(op, op->D, 2, dims, ptr, flags, pos);

	auto shift_op = CAST_DOWN(delayed_op_circ_shift_s, op);

	md_circ_shift2(op->D, dims[0], shift_op->center, op->args[0].astrs, ptr[0], op->args[1].astrs, ptr[1], op->args[0].asize);
}

static struct delayed_op_s* delayed_op_circ_shift_create(int D, const long dimensions[D], const long center[D], const long str1[D], void* dst, const long str2[D], const void* src, size_t size)
{
	long dims[D];
	unsigned long lflags = ~md_nontriv_strides(D, center);
	md_select_dims(D, lflags, dims, dimensions);

	struct delayed_op_arg_s arg[2] = {
		arg_create(D, dimensions, str1, dst, size, false, true),
		arg_create(D, dimensions, str2, src, size, true, false),
	};

	PTR_ALLOC(struct delayed_op_circ_shift_s, op);
	SET_TYPEID(delayed_op_circ_shift_s, op);

	op->center = ARR_CLONE(long[D], center);

	delayed_op_init(CAST_UP(op), D, lflags, 2, arg, 0, 0, delayed_op_circ_shift_fun, delayed_op_circ_shift_free, NULL);
	return CAST_UP(PTR_PASS(op));
}



struct delayed_op_clear_s {

	delayed_op_t super;
};

static DEF_TYPEID(delayed_op_clear_s);

static void delayed_op_clear_fun(delayed_op_t* op, unsigned long flags, long pos[MAX_DIMS])
{
	long dims[1][op->D];
	void* ptr[1];

	delayed_op_exec_resolve(op, op->D, 1, dims, ptr, flags, pos);
	md_clear2(op->D, dims[0], op->args[0].astrs, ptr[0], op->args[0].asize);
}

static struct delayed_op_s* delayed_op_clear_create(int D, const long dim[D], const long str[D], void* ptr, size_t size)
{

	struct delayed_op_arg_s arg[1] = { arg_create(D, dim, str, ptr, size, false, true) };

	PTR_ALLOC(struct delayed_op_clear_s, op);
	SET_TYPEID(delayed_op_clear_s, op);

	delayed_op_init(CAST_UP(op), D, ~0UL, 1, arg, 0, 0, delayed_op_clear_fun, NULL, NULL);
	return CAST_UP(PTR_PASS(op));
}

bool delayed_op_is_clear(const delayed_op_t* op)
{
	return NULL != CAST_MAYBE(delayed_op_clear_s, op);
}



struct delayed_op_md_fun_s {

	delayed_op_t super;

	bool hide_real_dim;
	long rstrs[3];

	enum delayed_md_fun_type type;
	size_t offset;
	unsigned long mpi_r_flags;
};

static DEF_TYPEID(delayed_op_md_fun_s);

static void delayed_op_md_fun(delayed_op_t* op, unsigned long flags, long pos[MAX_DIMS])
{
	void* ptr[op->N];

	delayed_op_exec_resolve(op, op->D, op->N, NULL, ptr, flags, pos);

	auto md_op = CAST_DOWN(delayed_op_md_fun_s, op);
	size_t offset = md_op->offset;

	int shift = md_op->hide_real_dim ? 1 : 0;
	int D = op->D + shift;

	long dims[D?:1];
	long strs[op->N][D?:1];

	dims[0] = 2;
	for (int i = 0; i < op->N; i++)
		strs[i][0] = md_op->rstrs[i];

	md_select_dims(op->D, ~flags, dims + shift, op->args[0].adims);

	for (int i = 0; i < op->N; i++)
		md_select_strides(op->D, ~flags, strs[i] + shift, op->args[i].astrs);

	for (int i = 0; i < op->N; i++)
		if (MD_IS_SET(md_op->mpi_r_flags, i))
			mpi_set_reduction_buffer(ptr[i]);

	switch (md_op->type) {

	case delayed_op_type_z3op:
		make_z3op(offset, D, dims, strs[0], ptr[0], strs[1], ptr[1], strs[2], ptr[2]);
		break;

	case delayed_op_type_3op:
		make_3op(offset, D, dims, strs[0], ptr[0], strs[1], ptr[1], strs[2], ptr[2]);
		break;

	case delayed_op_type_z3opd:
		make_z3opd(offset, D, dims, strs[0], ptr[0], strs[1], ptr[1], strs[2], ptr[2]);
		break;

	case delayed_op_type_3opd:
		make_3opd(offset, D, dims, strs[0], ptr[0], strs[1], ptr[1], strs[2], ptr[2]);
		break;

	case delayed_op_type_z2op:
		make_z2op(offset, D, dims, strs[0], ptr[0], strs[1], ptr[1]);
		break;

	case delayed_op_type_2op:
		make_2op(offset, D, dims, strs[0], ptr[0], strs[1], ptr[1]);
		break;

	case delayed_op_type_z2opd:
		make_z2opd(offset, D, dims, strs[0], ptr[0], strs[1], ptr[1]);
		break;

	case delayed_op_type_2opd:
		make_2opd(offset, D, dims, strs[0], ptr[0], strs[1], ptr[1]);
		break;

	case delayed_op_type_z2opf:
		make_z2opf(offset, D, dims, strs[0], ptr[0], strs[1], ptr[1]);
		break;

	case delayed_op_type_2opf:
		make_2opf(offset, D, dims, strs[0], ptr[0], strs[1], ptr[1]);
		break;

	default:
		error("Delayed md_fun type not implemented!\n");
	};

	for (int i = 0; i < op->N; i++)
		if (MD_IS_SET(md_op->mpi_r_flags, i))
			mpi_unset_reduction_buffer(ptr[i]);
}

static const char* delayed_op_md_fun_debug(delayed_op_t* op, bool /*nested*/)
{
	const char* name = NULL;
	switch(CAST_DOWN(delayed_op_md_fun_s, op)->offset) {
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

	const char* ret = ptr_printf("md_function %s%s ", CAST_DOWN(delayed_op_md_fun_s, op)->hide_real_dim ? "(zr)" : "", name);
	ptr_append_print_args(&ret, op);
	ptr_append_printf(&ret, " ");
	ptr_append_print_loopable_accessdims(&ret, op, 0);

	return ret;
}

static bool check_real_dim_hide(int D, int i, int N, const long dim[D], long rstrs[3], const long* strs[N], const void* ptr[N], size_t nsizes[N], const size_t sizes[N])
{
	if (2 != dim[i])
		return false;

	for (int k = 0; k < N; k++) {

		nsizes[k] = vptr_get_shape(ptr[k])->size;

		if (0 != strs[k][i] && (long)sizes[k] != strs[k][i])
			return false;

		if (0 == strs[k][i] && nsizes[k] != sizes[k])
			return false;

		if ((long)sizes[k] == strs[k][i] && (nsizes[k] != sizes[k] * 2))
			return false;
	}

	for (int k = 0; k < N; k++)
		rstrs[k] = strs[k][i];

	return true;
}

static struct delayed_op_s* delayed_op_md_fun_create(enum delayed_md_fun_type type, size_t offset, int D, const long dim[D], int N, const long* strs[N], const void* ptr[N], const size_t sizes[N])
{
	PTR_ALLOC(struct delayed_op_md_fun_s, op);
	SET_TYPEID(delayed_op_md_fun_s, op);

	struct delayed_op_arg_s arg[N];

	size_t nsizes[N];
	op->hide_real_dim = check_real_dim_hide(D, 0, N, dim, op->rstrs, strs, ptr, nsizes, sizes);
	int shift = op->hide_real_dim ? 1 : 0;

	op->hide_real_dim = op->hide_real_dim || check_real_dim_hide(D, D - 1, N, dim, op->rstrs, strs, ptr, nsizes, sizes);

	if (op->hide_real_dim)
		D--;

	arg[0] = arg_create(D, dim + shift, strs[0] + shift, ptr[0], op->hide_real_dim ? nsizes[0] : sizes[0], false, true);

	for (int i = 1; i < N; i++)
		arg[i] = arg_create(D, dim + shift, strs[i] + shift, ptr[i], op->hide_real_dim ? nsizes[i] : sizes[i], true, false);

	delayed_op_init(CAST_UP(op), D, ~0UL, N, arg, 0, 0, delayed_op_md_fun, NULL, delayed_op_md_fun_debug);

	op->offset = offset;
	op->type = type;
	op->mpi_r_flags = 0;

	for(int i = 0; i < N; i++)
		if (mpi_is_set_reduction_buffer(ptr[i]))
			op->mpi_r_flags |= MD_BIT(i);

	if (   (offset == offsetof(struct vec_ops, fmac))
	    || (offset == offsetof(struct vec_ops, fmacD))
	    || (offset == offsetof(struct vec_ops, zfmac))
	    || (offset == offsetof(struct vec_ops, zfmacD))
	    || (offset == offsetof(struct vec_ops, zfmacc))
	    || (offset == offsetof(struct vec_ops, zfmaccD)))
		op->super.args[0].read = true;

	return CAST_UP(PTR_PASS(op));
}



struct delayed_op_vptr_fun_s {

	delayed_op_t super;

	vptr_fun_t fun;
	struct vptr_fun_data_s* data;
	bool resolve;
};

static DEF_TYPEID(delayed_op_vptr_fun_s);

static void delayed_op_vptr_fun_free(const delayed_op_t* op)
{
	if (NULL != CAST_DOWN(delayed_op_vptr_fun_s, op)->data->del)
		CAST_DOWN(delayed_op_vptr_fun_s, op)->data->del(CAST_DOWN(delayed_op_vptr_fun_s, op)->data);

	xfree(CAST_DOWN(delayed_op_vptr_fun_s, op)->data);
}

static void delayed_op_vptr_fun_fun(delayed_op_t* op, unsigned long flags, long pos[MAX_DIMS])
{
	void* ptr[op->N];

	struct delayed_op_vptr_fun_s* fun = CAST_DOWN(delayed_op_vptr_fun_s, op);

	long tdims[op->N][op->D];
	long tstrs[op->N][op->D];
	size_t sizes[op->N];

	delayed_op_exec_resolve(op, op->D, op->N, tdims, ptr, flags, pos);

	const long* dims[op->N];
	const long* strs[op->N];

	for (int i = 0; i < op->N; i++) {

		md_select_strides(op->D, md_nontriv_dims(op->D, tdims[i]), tstrs[i], op->args[i].astrs);
		sizes[i] = op->args[i].asize;

		dims[i] = tdims[i];
		strs[i] = tstrs[i];
	}

	exec_vptr_fun_internal(fun->fun, fun->data, op->N, op->D, op->loop.loop_flags & ~op->loop.slice_flags, dims, strs, ptr, sizes, fun->resolve);
}

static const char* delayed_op_vptr_fun_debug(delayed_op_t* op, bool /*nested*/)
{
	const char* ret = ptr_printf("fun %s ", CAST_DOWN(delayed_op_vptr_fun_s, op)->data->TYPEID->name);
	ptr_append_print_args(&ret, op);

	for (int i = 0; i < op->N; i++) {

		ptr_append_printf(&ret, " %d: ", i);
		ptr_append_print_loopable_accessdims(&ret, op, i);
	}

	return ret;
}

static struct delayed_op_s* delayed_op_vptr_fun_create(vptr_fun_t fun, vptr_fun_data_t* data, int N, int D, unsigned long lflags, unsigned long wflags, unsigned long rflags, const long* dims[N], const long* strs[N], void* ptr[N], size_t sizes[N], bool resolve)
{
	PTR_ALLOC(struct delayed_op_vptr_fun_s, op);
	SET_TYPEID(delayed_op_vptr_fun_s, op);

	struct delayed_op_arg_s arg[N];
	for (int i = 0; i < N; i++)
		arg[i] = arg_create(D, dims[i], strs[i], ptr[i], sizes[i], MD_IS_SET(rflags, i), MD_IS_SET(wflags, i));

	delayed_op_init(CAST_UP(op), D, lflags, N, arg, 0, 0, delayed_op_vptr_fun_fun, delayed_op_vptr_fun_free, delayed_op_vptr_fun_debug);

	op->fun = fun;
	op->data = data;
	op->resolve = resolve;

	return CAST_UP(PTR_PASS(op));
}



struct delayed_op_chain_s {

	delayed_op_t super;

	unsigned long seq_flags;
	long ldims[MAX_DIMS];

	int M;
	delayed_op_t** ops;
};

static DEF_TYPEID(delayed_op_chain_s);

static void delayed_chain_fun(delayed_op_t* _op, unsigned long flags, long pos[MAX_DIMS])
{
	struct delayed_op_chain_s* op = CAST_DOWN(delayed_op_chain_s, _op);

	do {

		for (int i = 0; i < op->M; i++)
			delayed_op_exec(op->ops[i], flags | op->seq_flags, pos);

	} while (md_next(MAX_DIMS, op->ldims, op->seq_flags & ~flags, pos));
}

static void delayed_chain_free(const delayed_op_t* _op)
{
	struct delayed_op_chain_s* op = CAST_DOWN(delayed_op_chain_s, _op);

	for (int i = 0; i < op->M; i++)
		delayed_op_free(op->ops[i]);

	xfree(op->ops);
}

static const char* delayed_chain_debug(delayed_op_t* _op, bool nested)
{
	struct delayed_op_chain_s* op = CAST_DOWN(delayed_op_chain_s, _op);

	const char* ret = ptr_printf("chain (%d ops, %lu)", op->M, op->seq_flags);
	ptr_append_print_args(&ret, _op);

	int N = MAX_DIMS;
	while ((0 < N) && (1 == op->ldims[N - 1]))
		N--;

	for (int i = 0; i < N; i++)
		ptr_append_printf(&ret, " %ld%s", op->ldims[i], (1 == op->ldims[i]) || !MD_IS_SET(_op->loop.loop_flags, i) ? "" : MD_IS_SET(op->seq_flags, i) ? "S" : "L");

	ptr_append_printf(&ret, " ]");

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


static delayed_op_t* delayed_op_chain_create(list_t ops_list, unsigned long seq_flags)
{
	int D = ((delayed_op_t*)list_get_item(ops_list, 0))->D;
	for (int i = 0; i < list_count(ops_list); i++)
		D = MAX(D, ((delayed_op_t*)list_get_item(ops_list, i))->D);

	delayed_optimize_alloc(ops_list);
	delayed_optimize_free(ops_list);

	int Nmax = 0;
	int N = 0;

	for (int i = 0; i < list_count(ops_list); i++) {

		delayed_op_t* op = list_get_item(ops_list, i);
		Nmax += op->N;
	}

	struct delayed_op_arg_s args1[Nmax];
	struct delayed_op_arg_s* args = args1;

	for (int i = 0; i < list_count(ops_list); i++) {

		delayed_op_t* op = list_get_item(ops_list, i);

		for (int j = 0; j < op->N; j++) {

			args[N++] = op->args[j];

			for (int k = 0; k < N - 2; k++) {

				if (delayed_arg_same_access(args[k], args[N - 1])) {

					args[k].read = args[k].read || args[N - 1].read;
					args[k].write = args[k].write || args[N - 1].write;

					N--;
					break;
				}
			}
		}
	}

	struct delayed_op_arg_s args2[N];
	struct delayed_op_arg_s* targs = args2;

	for (int i = 0; i < list_count(ops_list); i++) {

		delayed_op_t* op = list_get_item(ops_list, i);

		if (!delayed_op_is_alloc(op))
			continue;

		for (int j = i; j < list_count(ops_list); j++) {

			delayed_op_t* fop = list_get_item(ops_list, j);
			if (!delayed_op_is_free(fop) || fop->args[0].ptr_base != op->args[0].ptr_base)
				continue;

			int nN = 0;
			for (int k = 0; k < N; k++)
				if (args[k].ptr_base != op->args[0].ptr_base)
					targs[nN++] = args[k];

			N=nN;
			SWAP(args, targs);
		}
	}

	for(int i = 0; i < N; i++)
		md_select_dims(args[i].N, md_nontriv_strides(args[i].N, args[i].astrs), args[i].adims, args[i].adims);

	PTR_ALLOC(struct delayed_op_chain_s, x);
	SET_TYPEID(delayed_op_chain_s, x);

	unsigned long loop_flags = queue_compute_loop_flags(x->ldims, ops_list);
	x->seq_flags = loop_flags & seq_flags;

	long mchange = 0;
	long mpeak = 0;

	for (int i = 0; i < list_count(ops_list); i++) {

		delayed_op_t* op = list_get_item(ops_list, i);

		if (delayed_op_is_alloc(op) || delayed_op_is_free(op)) {

			long dims[op->D];
			md_select_dims(op->D, x->seq_flags, dims, op->args[0].adims);
			op->mchange /= md_calc_size(op->D, dims);
		}

		mpeak = MAX(mpeak, op->mpeak + mchange);
		mchange += op->mchange;
	}

	delayed_op_init(CAST_UP(x), D, loop_flags, N, args, mchange, mpeak, delayed_chain_fun, delayed_chain_free, delayed_chain_debug);

	x->M = list_count(ops_list);
	x->ops = *TYPE_ALLOC(delayed_op_t*[x->M]);
	list_to_array(x->M, (void**)x->ops, ops_list);

	return CAST_UP(PTR_PASS(x));
}

bool delayed_op_is_chain(const delayed_op_t* op)
{
	return NULL != CAST_MAYBE(delayed_op_chain_s, op);
}



/******************************************************************************
 * Argument analysis
 ******************************************************************************/

static inline bool delayed_arg_same_access(struct delayed_op_arg_s arg1, struct delayed_op_arg_s arg2)
{
	return (   (arg1.ptr == arg2.ptr) && (arg1.asize == arg2.asize)
		&& md_check_equal_dims(MAX(arg1.N, arg2.N), arg1.adims, arg2.adims, ~0UL)
		&& md_check_equal_dims(MAX(arg1.N, arg2.N), arg1.astrs, arg2.astrs, ~0UL));
}

static inline bool delayed_arg_depends_on(struct delayed_op_arg_s arg1, struct delayed_op_arg_s arg2)
{
	if (arg1.ptr_base != arg2.ptr_base)
		return false;

	if (!arg1.write && !arg2.write)
		return false;

#if 0
	// More precise dependency analysis (slower)
	int N = MIN(arg1.N, arg2.N);
	unsigned long sflags = arg1.sflags & arg2.sflags;

	if (0 != sflags)
		for (int k = 0; k < N; k++)
			if (MD_IS_SET(sflags, k) && (arg1.mpos[k] != arg2.mpos[k]))
				return false;

	if (!md_overlap(arg1.N, arg1.adims, arg1.astrs, arg1.ptr, 1, arg2.N, arg2.adims, arg2.astrs, arg2.ptr, 1))
		return false;
#endif

	return true;
}

static inline bool delayed_arg_subset(struct delayed_op_arg_s a, struct delayed_op_arg_s b)
{
	if (a.ptr_base != b.ptr_base)
		return false;

	if (b.full_access)
		return true;

	return false;
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
		if (ptr == op->args[i].ptr_base && op->args[i].write)
			return true;

	return false;
}

static egraph_t delayed_op_queue_to_graph(list_t ops_queue)
{
	egraph_t graph = egraph_create();

	for (int i = 0; i < list_count(ops_queue); i++) {

		delayed_op_t* op = list_get_item(ops_queue, i);

		const char* name = (NULL != debug_graph_path  && (0 == mpi_get_rank())) ? print_delayed_fun_f(op, true) : NULL;
		egraph_add_node(graph, enode_create(name, op));

		egraph_reset_between(graph);

		for (int k = 0; k < op->N; k++) {

			for (int j = i - 1; j >= 0; j--) {

				enode_t node2 = egraph_get_node(graph, j);
				if (enode_is_ancestors(node2))
					continue;

				delayed_op_t* op2 = list_get_item(ops_queue, j);

				for (int l = 0; l < op2->N; l++) {

					if (delayed_arg_depends_on(op->args[k], op2->args[l])) {

						enode_add_dependency(egraph_get_node(graph, j), egraph_get_node(graph, i));
						egraph_set_ancestors(node2);
						break;
					}
				}
			}
		}
	}

	egraph_reset_between(graph);
	return graph;
}


/******************************************************************************
 * Looping analysis / optimization
 ******************************************************************************/

struct access_s {

	const void* ptr;
	unsigned long read;
	unsigned long written;
	long rpos[MAX_DIMS];
	long wpos[MAX_DIMS];
};

static bool cmp_write(const void* arg, const void* ref)
{
	return ((struct access_s*)arg)->ptr == ref;
}

static unsigned long queue_compute_loop_flags(long loop_dims[MAX_DIMS], list_t ops_queue)
{
	md_singleton_dims(MAX_DIMS, loop_dims);
	unsigned long loop_flags = ~0UL;

	list_t written = list_create();

	for (int i = 0; i < list_count(ops_queue); i++) {

		delayed_op_t* op = list_get_item(ops_queue, i);

		loop_flags &= op->loop.loop_flags;

		for (int j = 0; j < MAX_DIMS; j++) {

			if (!MD_IS_SET(loop_flags, j) || !MD_IS_SET(op->loop.loop_flags, j))
				continue;

			if (1 == loop_dims[j])
				loop_dims[j] = op->loop.dims[j];
			else if (loop_dims[j] != op->loop.dims[j])
				loop_flags = MD_CLEAR(loop_flags, j);
		}

		for (int j = 0; j < op->N; j++) {

			struct delayed_op_arg_s* arg = &op->args[j];
			struct access_s* w = list_get_first_item(written, arg->ptr_base, cmp_write, false);

			if (NULL == w)
				continue;

			if (arg->write) {

				unsigned long rflags = loop_flags & w->read;
				for (int i = 0; i < op->loop.D; i++)
					if ((MD_IS_SET(rflags & op->loop.slice_flags, i)) && (w->rpos <= op->loop.slice_pos))
						rflags = MD_CLEAR(rflags, i);

				loop_flags &= ~rflags;
			}

			if (arg->read)  {

				unsigned long wflags = loop_flags & w->written;
				for (int i = 0; i < op->loop.D; i++)
					if ((MD_IS_SET(wflags & op->loop.slice_flags, i)) && (w->wpos <= op->loop.slice_pos))
						wflags = MD_CLEAR(wflags, i);

				loop_flags &= ~wflags;
			}
		}

		for (int j = 0; j < op->N; j++) {

			struct delayed_op_arg_s* arg = &op->args[j];
			struct access_s* w = list_get_first_item(written, arg->ptr_base, cmp_write, false);

			if (NULL == w) {

				PTR_ALLOC(struct access_s, x);
				x->ptr = arg->ptr_base;
				x->written = 0UL;
				x->read = 0UL;
				memset(x->rpos, 0, sizeof(x->rpos));
				memset(x->wpos, 0, sizeof(x->wpos));
				w = PTR_PASS(x);
				list_push(written, w);
			}

			unsigned long flags = op->loop.loop_flags & arg->non_first_pos_flags;

			if (arg->write) {

				w->written |= flags;
				for (int i = 0; i < MAX_DIMS; i++) {

					if (!MD_IS_SET(flags, i))
						continue;

					long apos = MD_IS_SET(op->loop.slice_flags, i) ? op->loop.slice_pos[i] : (op->loop.dims[i] - 1);
					w->wpos[i] = MAX(w->wpos[i], apos);
				}

			}

			if (arg->read) {

				w->read |= flags;
				for (int i = 0; i < MAX_DIMS; i++) {

					if (!MD_IS_SET(flags, i))
						continue;

					long apos = MD_IS_SET(op->loop.slice_flags, i) ? op->loop.slice_pos[i] : (op->loop.dims[i] - 1);
					w->rpos[i] = MAX(w->rpos[i], apos);
				}
			}
		}
	}

	while (0 < list_count(written))
		xfree(list_pop(written));

	list_free(written);

	md_select_dims(MAX_DIMS, loop_flags, loop_dims, loop_dims);
	return loop_flags;
}

static unsigned long delayed_tmp_buffer_compute_loop_flags(egraph_t graph)
{
	unsigned long ret = 0;

	for (int i = 0; i < list_count(graph); i++) {

		delayed_op_t* op = enode_get_data(egraph_get_node(graph, i));

		if (!delayed_op_is_alloc(op) || !CAST_DOWN(delayed_op_alloc_s, op)->tmp_buffer)
			continue;

		int j = i;
		delayed_op_t* op2;

		while (true) {

			op2 = enode_get_data(egraph_get_node(graph, ++j));

			if (delayed_op_is_free(op2) && (op->args[0].ptr_base == op2->args[0].ptr_base))
				break;
		}

		egraph_reset_between(graph);
		egraph_set_descendants(egraph_get_node(graph, i));
		egraph_set_ancestors(egraph_get_node(graph, j));

		list_t between = list_create();
		for (int i = 0; i < list_count(graph); i++) {

			enode_t node = egraph_get_node(graph, i);

			if (enode_is_between(node))
				list_append(between, enode_get_data(node));
		}

		long loop_dims[MAX_DIMS];
		unsigned long lflags = queue_compute_loop_flags(loop_dims, between);
		op->loop.loop_flags &= lflags;
		op2->loop.loop_flags &= lflags;
		list_free(between);

		ret |= op->loop.loop_flags;
	}

	egraph_reset_between(graph);

	return ret;
}

static void delayed_optimize_queue_looping_flags(egraph_t graph, bool multiple, unsigned long lflags)
{

	for (int i = 0; i < list_count(graph); i++) {

		enode_t fnode = egraph_get_node(graph, i);
		delayed_op_t* fop = enode_get_data(fnode);

		if (!(delayed_op_is_free(fop) && (CAST_DOWN(delayed_op_free_s, fop)->tmp_buffer) && (0 == (lflags & ~fop->loop.loop_flags))))
			continue;

		egraph_reset_between(graph);

		list_t open_free = list_create();
		list_append(open_free, fop->args[0].ptr_base);

		list_t alloc_ops = list_create();
		list_append(alloc_ops, fop);

		egraph_set_ancestors(fnode);

		for (int j = i - 1; j >= 0 && (0 < list_count(open_free)); j--) {

			enode_t node = egraph_get_node(graph, j);
			delayed_op_t* op = enode_get_data(node);

			if (delayed_op_is_alloc(op) && (-1 != list_get_first_index(open_free, op->args[0].ptr_base, NULL))) {

				list_append(alloc_ops, op);
				egraph_set_descendants(node);
				list_remove_item(open_free, list_get_first_index(open_free, op->args[0].ptr_base, NULL));
			}

			if (   multiple && delayed_op_is_free(op) && (CAST_DOWN(delayed_op_free_s, op)->tmp_buffer)
			    && (0 == (lflags & ~op->loop.loop_flags))
			    && md_check_equal_dims(MAX(op->D, fop->D), op->loop.dims, fop->loop.dims, lflags)) {

				list_append(alloc_ops, op);
				list_append(open_free, op->args[0].ptr_base);
				egraph_set_ancestors(node);
			}
		}

		assert(0 == list_count(open_free));
		list_free(open_free);

		list_t tmp_graph = list_create();
		for (int i = 0; i < list_count(graph); i++)
			list_append(tmp_graph, list_get_item(graph, i));

		list_t chain = egraph_sort_between(tmp_graph);
		for (int i = 0; i < list_count(chain); i++)
			list_insert(chain, enode_get_data(list_remove_item(chain, i)), i);

		long loop_dims[MAX_DIMS];
		unsigned long loop_flags = queue_compute_loop_flags(loop_dims, chain);

		if (0 != (lflags & ~loop_flags)) {

			list_free(chain);
			list_free(tmp_graph);
			list_free(alloc_ops);
			continue;
		}

		long old_mpeak = compute_mpeak(graph, true);

		long lsize = md_calc_size(MAX_DIMS, loop_dims);

		for (int j = 0; j < list_count(alloc_ops); j++)
			((delayed_op_t*)list_get_item(alloc_ops, j))->mchange /= lsize;

		long new_mpeak = compute_mpeak(tmp_graph, true);

		for (int j = 0; j < list_count(alloc_ops); j++)
			((delayed_op_t*)list_get_item(alloc_ops, j))->mchange *= lsize;

		if (new_mpeak > old_mpeak) {

#if 0
			debug_printf(DP_INFO, "Cannot loop as new memory is not smaller than old\n");
			debug_printf(DP_INFO, " old chain:\n");
			debug_mpeak_queue(DP_INFO, graph, true);

			for (int j = 0; j < list_count(alloc_ops); j++)
				((delayed_op_t*)list_get_item(alloc_ops, j))->mchange /= lsize;

			debug_printf(DP_INFO, " new chain:\n");
			debug_mpeak_queue(DP_INFO, tmp_graph, true);

			for (int j = 0; j < list_count(alloc_ops); j++)
				((delayed_op_t*)list_get_item(alloc_ops, j))->mchange *= lsize;
#endif

			list_free(alloc_ops);
			list_free(chain);
			list_free(tmp_graph);

			continue;
		}

		int k = 0;
		while (!enode_is_between(egraph_get_node(tmp_graph, k)))
			k++;

		enode_t chain_node = enode_create(NULL, delayed_op_chain_create(chain, bart_delayed_loop_flags));
		list_insert(tmp_graph, chain_node, k++);

		while ((k < list_count(tmp_graph)) && enode_is_between(egraph_get_node(tmp_graph, k))) {

			enode_t node = egraph_remove_node(tmp_graph, k);

			list_t iedges = enode_get_iedges(node);
			for (int l = 0; l < list_count(iedges); l++) {

				enode_t inode = list_get_item(iedges, l);
				if (!enode_is_between(inode))
					enode_add_dependency(inode, chain_node);
			}

			list_t oedges = enode_get_oedges(node);
			for (int l = 0; l < list_count(oedges); l++) {

				enode_t onode = list_get_item(oedges, l);
				if (!enode_is_between(onode))
					enode_add_dependency(chain_node, onode);
			}

			enode_free(node);
		}

		list_free(chain);
		list_free(alloc_ops);

		while (0 < list_count(graph))
			list_pop(graph);

		while (0 < list_count(tmp_graph))
			egraph_add_node(graph, list_pop(tmp_graph));

		list_free(tmp_graph);
		i = -1; // restart
	}
}

void delayed_optimize_queue_looping(list_t ops_queue)
{
	if (0 == bart_delayed_loop_flags)
		return;


	egraph_t graph = delayed_op_queue_to_graph(ops_queue);

	unsigned long loop_flags = delayed_tmp_buffer_compute_loop_flags(graph);

	int N = 0;
	long loop_dims[ARRAY_SIZE(bart_delayed_loop_dims)];

	for (int i = ARRAY_SIZE(bart_delayed_loop_dims) - 1; i >= 0; i--) {

		if ((-1 == bart_delayed_loop_dims[i]) || !MD_IS_SET(loop_flags, bart_delayed_loop_dims[i]))
			continue;

		loop_dims[N++] = bart_delayed_loop_dims[i];
	}

	int order[MD_BIT(N)];
	for (int i = 0; i < (int)MD_BIT(N); i++)
		order[i] = i;

	NESTED(int, cmp_ord, (int a, int b))
	{
		int da = bitcount((unsigned long)a);
		int db = bitcount((unsigned long)a);

		if (da != db)
			return (db > da) - (db < da);
		else
			return (b > a) - (b < a);
	};

	quicksort(MD_BIT(N), order, cmp_ord);


	for (int i = 0; i < (int)MD_BIT(N) - 1; i++) {

		unsigned long loc_loop_flags = 0;

		for (int j = 0; j < N; j++)
			if (MD_IS_SET((unsigned long)order[i], j))
				loc_loop_flags |= MD_BIT(loop_dims[j]);

		delayed_optimize_queue_looping_flags(graph, true, loc_loop_flags);
		delayed_optimize_queue_looping_flags(graph, false, loc_loop_flags);
	}

	while (0 < list_count(ops_queue))
		list_pop(ops_queue);

	for (int i = 0; i < list_count(graph); i++)
		list_append(ops_queue, enode_get_data(egraph_get_node(graph, i)));

	egraph_free(graph);
}



/******************************************************************************
 * Queue optimization
 ******************************************************************************/



void delayed_optimize_queue(list_t ops_queue)
{
	int N = list_count(ops_queue);
	if (0 == N)
		return;

	double start_time = timestamp();

	float mpeak1 = compute_mpeak(ops_queue, false);

	debug_printf(delayed_dl, "Optimize queue with %d operations\n", list_count(ops_queue));
	debug_delayed_queue(delayed_dl, ops_queue, true);

	delayed_optimize_alloc(ops_queue);
	delayed_optimize_free(ops_queue);

	delayed_optimize_clear(ops_queue);
	delayed_optimize_copy(ops_queue);

	delayed_optimize_alloc(ops_queue);
	delayed_optimize_free(ops_queue);
	delayed_optimize_new_buffer_on_overwrite(ops_queue);

	delayed_optimize_alloc(ops_queue);
	delayed_optimize_free(ops_queue);
	delayed_optimize_set_tmp_buffer(ops_queue);

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

	debug_printf(delayed_dl, "First step: Optimized queue with %d operations\n", list_count(ops_queue));
	debug_delayed_queue(delayed_dl, ops_queue, true);

	double first_step = timestamp() - start_time;

	float mpeak2 = compute_mpeak(ops_queue, false);

	delayed_optimize_queue_looping(ops_queue);

	debug_printf(delayed_dl, "Second step: Optimized queue with %d operations\n", list_count(ops_queue));
	debug_delayed_queue(delayed_dl, ops_queue, true);

	double tot_time = timestamp() - start_time;

	float mpeak3 = compute_mpeak(ops_queue, false);
	debug_printf(MIN(DP_DEBUG3, delayed_dl), "Optimized queue with %d operations %s -> %s -> %s took %es (%es + %es)\n",
		N, STRING_MEM_SIZE(mpeak1), STRING_MEM_SIZE(mpeak2), STRING_MEM_SIZE(mpeak3), tot_time, first_step, tot_time - first_step);
}

static void delayed_optimize_set_tmp_buffer(list_t ops_queue)
{
	for (int i = 0; i < list_count(ops_queue); i++) {

		delayed_op_t* op_allo = list_get_item(ops_queue, i);

		if (!delayed_op_is_alloc(op_allo) || vptr_is_mem_allocated(op_allo->args[0].ptr_base))
			continue;

		for (int j = i + 1; j < list_count(ops_queue); j++) {

			delayed_op_t* op_free = list_get_item(ops_queue, j);

			if (!delayed_op_is_free(op_free))
				continue;

			if (op_allo->args[0].ptr == op_free->args[0].ptr && !vptr_is_mem_allocated(op_allo->args[0].ptr)) {

				CAST_DOWN(delayed_op_alloc_s, op_allo)->tmp_buffer = true;
				CAST_DOWN(delayed_op_free_s, op_free)->tmp_buffer = true;

				break;
			}
		}
	}
}

static void delayed_optimize_new_buffer_on_overwrite(list_t ops_queue)
{
	for (int i = 0; i < list_count(ops_queue) - 1; i++) {

		delayed_op_t* op_overwrite = list_get_item(ops_queue, i);

		if (delayed_op_is_alloc(op_overwrite) || delayed_op_is_free(op_overwrite))
			continue;

		for (int j = 0; j < op_overwrite->N; j++) {

			struct delayed_op_arg_s arg = op_overwrite->args[j];

			if (arg.read || !arg.write || !arg.full_access)
				continue;

			if (vptr_is_writeback(arg.ptr_base) || vptr_is_mem_allocated(arg.ptr_base))
				continue;

			bool overwrite = true;

			for (int k = 0; k < op_overwrite->N; k++)
				if (op_overwrite->args[k].read && op_overwrite->args[j].ptr_base == op_overwrite->args[k].ptr_base)
					overwrite = false;

			if (!overwrite)
				continue;

			delayed_op_t* op_prior = NULL;

			for (int k = i - 1; (k >= 0) && (NULL == op_prior); k--) {

				delayed_op_t* op2 = list_get_item(ops_queue, k);

				for (int k = 0; k < op2->N; k++)
					if (op2->args[k].ptr_base == op_overwrite->args[j].ptr_base)
						op_prior = op2;
			}

			if (NULL != op_prior && delayed_op_is_alloc(op_prior))
				continue;

			int k = i + 1;

			for (; k < list_count(ops_queue); k++) {

				delayed_op_t* op_free = list_get_item(ops_queue, k);
				if (delayed_op_is_free(op_free) && (arg.ptr_base == op_free->args[0].ptr_base))
					break;
			}

			if (list_count(ops_queue) == k)
				continue;

			void* optr = arg.ptr_base;
			void* nptr = vptr_alloc_sameplace(arg.N, arg.mdims, arg.msize, arg.ptr_base);

			if (delayed_dl >= debug_level) {

				const char* str_op = print_delayed_fun_f(op_overwrite, false);
				debug_printf(delayed_dl, "New alloc for overwrite: %p -> %p %s\n", optr, nptr, str_op);
				xfree(str_op);
			}


			list_insert(ops_queue, list_remove_item(ops_queue, k), i++);
			list_insert(ops_queue, delayed_op_alloc_create(nptr, arg.N, arg.mdims, arg.msize), i++);
			list_insert(ops_queue, delayed_op_free_create(nptr, arg.N, arg.mdims, arg.msize), k + 2);

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

static void delayed_optimize_alloc(list_t ops_queue)
{
	for (int i = list_count(ops_queue) - 1; i >= 0; i--) {

		delayed_op_t* op = list_get_item(ops_queue, i);
		if (delayed_op_is_alloc(op) && vptr_is_mem_allocated(op->args[0].ptr_base))
			op->mchange = 0;
	}

	for (int i = list_count(ops_queue) - 1; i >= 0; i--) {

		delayed_op_t* op = list_get_item(ops_queue, i);

		if (delayed_op_is_alloc(op))
			list_remove_item(ops_queue, i);
		else
			continue;

		bool reinsert = true;

		for (int j = i; j < list_count(ops_queue); j++) {

			if (!delayed_ptr_required(list_get_item(ops_queue, j), op->args[0].ptr_base))
				continue;

			if (delayed_op_is_clear(list_get_item(ops_queue, j)) && !vptr_is_mem_allocated(op->args[0].ptr_base)) {

				delayed_op_free(list_remove_item(ops_queue, j--));
				vptr_clear(op->args[0].ptr_base);
				continue;
			}

			if (delayed_op_is_free(list_get_item(ops_queue, j))) {

				delayed_op_free(list_remove_item(ops_queue, j));
				delayed_op_free(op);
			} else {

				list_insert(ops_queue, op, j);
			}

			reinsert = false;
			break;
		}

		if (reinsert)
			list_append(ops_queue, op);
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
				if (op2->args[k].write && op2->args[k].ptr_base != op->args[0].ptr_base)
					keep = true;

			if (!keep) {

				delayed_op_free(list_remove_item(ops_queue, j - 1));
				i = -1;
			}
		}
	}
}

static void replace_inplace(list_t ops_queue, delayed_op_t* op_allo, delayed_op_t* op_free)
{
	void* nptr = op_allo->args[0].ptr;
	void* optr = op_free->args[0].ptr;

	debug_printf(delayed_dl, "Inplace substitution: %p -> %p\n", optr, nptr);

	for (int i = 0; i < list_count(ops_queue); i++) {

		delayed_op_t* op = list_get_item(ops_queue, i);

		if (delayed_op_is_alloc(op) && optr == op->args[0].ptr_base) {

			op->args[0].ptr = nptr;
			op->args[0].ptr_base = nptr;

			if (vptr_is_set_clear(optr))
				vptr_clear(nptr);
			else
				vptr_unset_clear(nptr);

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

				if (   !(CAST_DOWN(delayed_op_alloc_s, op_allo)->tmp_buffer)
				    || !(CAST_DOWN(delayed_op_free_s, op_free)->tmp_buffer)
				    || (op_allo->args[0].ptr_base == op_free->args[0].ptr_base)
				    || (!vptr_is_same_type(op_allo->args[0].ptr_base, op_free->args[0].ptr_base))
				    || vptr_is_mem_allocated(op_allo->args[0].ptr_base) || vptr_is_mem_allocated(op_free->args[0].ptr_base)
				    || !((NULL != CAST_MAYBE(delayed_op_md_fun_s, op)) || (NULL != CAST_MAYBE(delayed_op_copy_s, op)) || (NULL != CAST_MAYBE(delayed_op_vptr_fun_s, op))))
					continue;

				if (vptr_is_set_clear(op_allo->args[0].ptr_base)) {

					bool read = false;

					for (int i = 0; i < op->N; i++)
						if (op->args[i].read && op->args[i].ptr_base == op_allo->args[0].ptr_base)
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
			op = delayed_op_clear_create(op->args[0].N, op->args[0].adims, op->args[0].astrs, op->args[0].ptr_base, op->args[0].asize);
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

				if (op2->args[k].read)
					read = true;

				if (op2->args[k].write && delayed_arg_subset(op->args[0], op2->args[k]))
					write = true;
			}
		}

		if (!read && write) {

			changed = true;
			delayed_op_free(op);
			continue;
		}

		if (   (delayed_op_is_copy(op2) || (NULL != CAST_MAYBE(delayed_op_circ_shift_s, op2)))
		    && op2->args[0].ptr_base != op->args[0].ptr_base
		    && delayed_arg_subset(op2->args[1], op->args[0])) {

			changed = true;
			list_insert(ops_queue, delayed_op_clear_create(op2->args[0].N, op2->args[0].adims, op2->args[0].astrs, op2->args[0].ptr, op2->args[0].asize), j + 1);
			delayed_op_free(list_remove_item(ops_queue, j));
			list_insert(ops_queue, op, j);
			i = j + 1;
			continue;
		}

		list_insert(ops_queue, op, j);
		auto md_op = CAST_MAYBE(delayed_op_md_fun_s, op2);

		if (NULL != md_op)  {

			bool stop = false;

			for (int k = 0; k < op2->N; k++) {

				if (op2->args[k].write && (op->args[0].ptr_base == op2->args[k].ptr_base)) {

					stop = true;
					break;
				}
			}

			if (stop)
				continue;

			delayed_op_t* nop = NULL;

			switch(md_op->offset) {

			case offsetof(struct vec_ops, add):

				if (delayed_arg_subset(op2->args[1], op->args[0])) {

					nop = delayed_op_copy_create(op2->args[1].N, op2->args[0].adims, op2->args[0].astrs, op2->args[0].ptr, op2->args[2].astrs, op2->args[2].ptr, op2->args[0].asize);
					break;
				}

				if (delayed_arg_subset(op2->args[2], op->args[0]))
					nop = delayed_op_copy_create(op2->args[1].N, op2->args[0].adims, op2->args[0].astrs, op2->args[0].ptr, op2->args[1].astrs, op2->args[1].ptr, op2->args[0].asize);

				break;

			case offsetof(struct vec_ops, sub):

				if (delayed_arg_subset(op2->args[2], op->args[0]))
					nop = delayed_op_copy_create(op2->args[1].N, op2->args[0].adims, op2->args[0].astrs, op2->args[0].ptr, op2->args[1].astrs, op2->args[1].ptr, op2->args[0].asize);
				break;

			case offsetof(struct vec_ops, mul):
			case offsetof(struct vec_ops, div):
			case offsetof(struct vec_ops, zmul):
			case offsetof(struct vec_ops, zdiv):
			case offsetof(struct vec_ops, zmulc):

				if ((delayed_arg_subset(op2->args[1], op->args[0])) || (delayed_arg_subset(op2->args[2], op->args[0])))
					nop = delayed_op_clear_create(op2->args[0].N, op2->args[0].adims, op2->args[0].astrs, op2->args[0].ptr, op2->args[0].asize);
				break;

			default:
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

				if (op2->args[k].read)
					read = true;

				if (op2->args[k].write && delayed_arg_subset(op->args[0], op2->args[k]))
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


				if (op2->args[k].ptr_base == op->args[0].ptr_base && !op2->args[k].write && op2->args[k].read) {

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
			op = enode_get_data(egraph_get_node(ops_queue, i));
		else
			op = list_get_item(ops_queue, i);

		mpeak = MAX(mpeak, mchange + op->mpeak);

		const char* ops = print_delayed_fun_f(op, false);
		mchange += op->mchange;
		debug_printf(dl, "%d: %ld %ld - %s\n", i, mchange, mpeak, ops);
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
			op = enode_get_data(egraph_get_node(ops_queue, i));
		else
			op = list_get_item(ops_queue, i);

		mpeak = MAX(mpeak, mchange + op->mpeak);
		mchange += op->mchange;
	}

	return mpeak;
}

long compute_mchange(list_t ops_queue, bool node)
{
	long mchange = 0;

	for (int i = 0; i < list_count(ops_queue); i++) {

		delayed_op_t* op = NULL;

		if (node)
			op = enode_get_data(egraph_get_node(ops_queue, i));
		else
			op = list_get_item(ops_queue, i);

		mchange += op->mchange;
	}

	return mchange;
}




static void delayed_op_append_node(egraph_t graph, delayed_op_t* op)
{
	if (CAST_MAYBE(delayed_op_chain_s, op)) {

		for (int i = 0; i < CAST_MAYBE(delayed_op_chain_s, op)->M; i++)
			delayed_op_append_node(graph, CAST_MAYBE(delayed_op_chain_s, op)->ops[i]);

		return;
	}

	const char* name = print_delayed_fun_f(op, true);
	egraph_add_node(graph, enode_create(name, op));

	for (int k = 0; k < op->N; k++) {

		bool done = false;

		for (int l = 0; l < k; l++)
			if (op->args[k].ptr_base == op->args[l].ptr_base)
				done = true;

		if (done)
			continue;

		for (int j = list_count(graph) - 2; j >= 0; j--) {

			enode_t node = egraph_get_node(graph, j);
			delayed_op_t* op2 = enode_get_data(node);

			bool found = false;
			for (int l = 0; l < op2->N; l++)
				if (op2->args[k].ptr_base == op->args[l].ptr_base)
					found = true;

			if (found) {

				enode_add_dependency(node, egraph_get_node(graph, list_count(graph) - 1));
				break;
			}
		}
	}
}


static void delayed_export_queue(list_t ops_queue, const char* name)
{
	list_t graph = egraph_create();

	for (int i = 0; i < list_count(ops_queue); i++)
		delayed_op_append_node(graph, list_get_item(ops_queue, i));

	export_egraph_dot(name, graph);

	egraph_free(graph);
}


void delayed_compute_debug(const char* name)
{
	struct queue_s* queue = get_global_queue();

	list_t ops = queue->ops;

	if (0 == list_count(ops))
		return;

	queue->compute = true;

	delayed_optimize_queue(ops);
	delayed_optimize_queue_looping(ops);

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

