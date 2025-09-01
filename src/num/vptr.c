/* Copyright 2023-2024. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal, Bernhard Rapp
 */

#include <assert.h>
#include <stdbool.h>
#include <signal.h>
#include <unistd.h>

#ifdef _WIN32
#include "win/mman.h"
#include "win/open_patch.h"
#else
#include <sys/mman.h>
#endif

#include "misc/types.h"
#include "misc/tree.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/shrdptr.h"
#include "misc/debug.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/mpi_ops.h"

#include "vptr.h"


#ifdef USE_DWARF
#undef assert
#define assert(expr) do { if (!(expr)) error("Assertion '" #expr "' failed in %s:%d\n",  __FILE__, __LINE__); } while (0)
#endif

struct vptr_hint_s {

	int N;
	long* dims;

	long* rank;
	unsigned long mpi_flags;

	struct shared_obj_s sptr;
};

static int hint_get_rank(int N, const long pos[N], struct vptr_hint_s* hint)
{
	long offset = 0;
	long stride = 1;

	for (int i = 0; i < MIN(N, hint->N); i++) {

		offset += stride * ((1 == hint->dims[i]) ?  0 : pos[i]);
		stride *= hint->dims[i];
	}

	return (int)hint->rank[offset];
}

struct vptr_hint_s* vptr_hint_ref(struct vptr_hint_s* hint)
{
	if (NULL != hint)
		shared_obj_ref(&hint->sptr);

	return hint;
}

static void vptr_hint_del(const struct shared_obj_s* sptr)
{
	auto hint = CONTAINER_OF(sptr, const struct vptr_hint_s, sptr);

	xfree(hint->dims);
	xfree(hint->rank);
	xfree(hint);
}

struct vptr_hint_s* hint_mpi_create(unsigned long mpi_flags, int N, const long dims[N])
{
	PTR_ALLOC(struct vptr_hint_s, x);

	long mdims[N];
	md_select_dims(N, mpi_flags, mdims, dims);

	int procs = mpi_get_num_procs();
	int max_proc = 1;

	for (int i = N - 1; i >= 0; i--) {

		procs = MAX(1, procs);
		max_proc *= MIN(procs, (int)mdims[i]);

		if (procs <= 1)
			mdims[i] = 1;

		procs /= (int)mdims[i];
	}

	mpi_flags &= md_nontriv_dims(N, mdims);

	x->N = N;
	x->dims = ARR_CLONE(long[N], mdims);

	long tdims[N];
	for (int i = 0; i < N; i++)
		tdims[N - 1 - i] = mdims[i];


	long tot = md_calc_size(N, mdims);

	long rank1[tot];
	long rank2[tot];

	for (long i = 0; i < tot; i++)
		rank1[i] = i % max_proc;

	int order[N];
	for (int i = 0; i < N; i++)
		order[i] = N - 1 - i;

	md_permute(N, order, mdims, rank2, tdims, rank1, sizeof(long));

	x->rank = ARR_CLONE(long[md_calc_size(N, mdims)], rank2);

	x->mpi_flags = mpi_flags;

	shared_obj_init(&x->sptr, vptr_hint_del);

	return PTR_PASS(x);
}

void vptr_hint_free(struct vptr_hint_s* hint)
{
	if (NULL == hint)
		return;

	shared_obj_destroy(&hint->sptr);
}



static tree_t vmap = NULL;



struct vptr_mem_s {

	const struct vptr_shape_s* shape;

	int num_blocks;
	void** mem;
	long block_size;
	unsigned long flags;
};

struct vptr_mem_s vptr_mem_default = { NULL, 1, NULL, 0, 0UL };

enum VPTR_LOC { VPTR_CPU, VPTR_GPU, VPTR_CFL, VPTR_ANY, VPTR_LOC_COUNT };
long vptr_size[VPTR_LOC_COUNT] = { 0 };
long vptr_peak[VPTR_LOC_COUNT] = { 0 };


static enum VPTR_LOC vptr_loc_sameplace(enum VPTR_LOC loc)
{
	switch (loc) {
		case VPTR_CPU: return VPTR_CPU;
		case VPTR_GPU: return VPTR_GPU;
		case VPTR_CFL: return VPTR_CPU;
		default: assert(0);
	}
}

static void vptr_update_size(enum VPTR_LOC loc, long size)
{
#pragma omp atomic
	vptr_size[loc] += size ;
	vptr_peak[loc] = MAX(vptr_peak[loc], vptr_size[loc]);

#pragma omp atomic
	vptr_size[VPTR_ANY] += size ;
	vptr_peak[VPTR_ANY] = MAX(vptr_peak[VPTR_ANY], vptr_size[VPTR_ANY]);
}

static void vptr_mem_block_init(struct vptr_mem_s* mem)
{
	if (NULL != mem->mem)
		return;

	assert(0 != mem->shape->N);

#pragma omp critical(vptr_mem_init)
	if (NULL == mem->mem) {

		long tdims[mem->shape->N];
		md_select_dims(mem->shape->N, mem->flags, tdims, mem->shape->dims);
		mem->num_blocks = md_calc_size(mem->shape->N, tdims);
		mem->mem = *TYPE_ALLOC(void*[mem->num_blocks]);

		for (int i = 0; i < mem->num_blocks; i++)
			mem->mem[i] = NULL;

		mem->block_size = (long)mem->shape->size * md_calc_size(mem->shape->N, mem->shape->dims) / mem->num_blocks;
	}
}

static void vptr_mem_block_alloc(struct vptr_mem_s* mem, int idx, enum VPTR_LOC loc, bool clear)
{
	vptr_mem_block_init(mem);

	if (NULL != mem->mem[idx])
		return;

	vptr_update_size(loc, mem->block_size);

	switch (loc) {

	case VPTR_CPU:
		mem->mem[idx] = xmalloc((size_t)mem->block_size);
		if (clear)
			memset(mem->mem[idx], 0, (size_t)mem->block_size);
		break;
#ifdef USE_CUDA
	case VPTR_GPU:
		mem->mem[idx] = cuda_malloc(mem->block_size);
		if (clear)
			cuda_clear(mem->block_size, mem->mem[idx]);
		break;
#endif
	default:
		assert(0);
	}
}

static void vptr_mem_block_free(struct vptr_mem_s* mem, int idx, enum VPTR_LOC loc, bool free)
{
	if ((NULL == mem->mem) || (NULL == mem->mem[idx]))
		return;

	vptr_update_size(loc, -mem->block_size);

	if (!free)
		return;

	switch (loc) {
	case VPTR_CFL:
		assert(0 == idx);
		unmap_cfl(mem->shape->N, mem->shape->dims, mem->mem[idx]);
		break;
	case VPTR_CPU:
		xfree(mem->mem[idx]);
		break;
#ifdef USE_CUDA
	case VPTR_GPU:
		cuda_free((void*)mem->mem[idx]);
		break;
#endif
	default:
		assert(0);
	}

	mem->mem[idx] = NULL;
}

static void vptr_mem_free(struct vptr_mem_s* mem, enum VPTR_LOC loc, bool free)
{
	if (NULL == mem->mem)
		return;

	for (int i = 0; i < mem->num_blocks; i++)
		vptr_mem_block_free(mem, i, loc, free);

	xfree(mem->mem);
	mem->mem = NULL;
}



static void* vptr_mem_block_resolve(struct vptr_mem_s* mem, enum VPTR_LOC loc, bool clear, long offset)
{
	unsigned long dflags = md_nontriv_dims(mem->shape->N, mem->shape->dims);

	long idx = md_reravel_index(mem->shape->N, mem->flags & dflags, dflags, mem->shape->dims, offset / (long)mem->shape->size);

	vptr_mem_block_init(mem);

#pragma omp critical(vptr_mem_resolve)
	vptr_mem_block_alloc(mem, idx, loc, clear);

	return mem->mem[idx] + md_reravel_index(mem->shape->N, ~mem->flags & dflags, dflags, mem->shape->dims, offset / (long)mem->shape->size) * (long)mem->shape->size + (offset % (long)mem->shape->size);
}


struct vptr_range_s {

	int D;
	struct mem_s** sub_ptr;
};



struct mem_s {

	void* ptr;
	size_t len;

	enum VPTR_LOC loc;
	bool clear;
	bool free;
	bool writeback;

	struct vptr_shape_s shape;

	struct vptr_mem_s blocks;
	struct vptr_range_s range;

	struct vptr_hint_s* hint;

	const char* backtrace;		// backtrace of allocation (for debugging)

	bool reduction_buffer;		// true if this pointer is used as reduction buffer
					// only one rank is allowed to write to it
					// => we can use a simple all_reduce to sum up the results
};

const char* str_byte[] = { "B", "KB", "MB", "GB", "TB" };

static const char* get_byte_unit(size_t size)
{
	double fsize = size;
	int idx = 0;

	while (idx + 1 < (int)ARRAY_SIZE(str_byte) && 1024 < fsize) {

		idx++;
		fsize /= 1024;
	}

	return str_byte[idx];
}

static double get_byte_size(size_t size)
{
	double fsize = size;
	int idx = 0;

	while (idx + 1 < (int)ARRAY_SIZE(str_byte) && 1024 < fsize) {

		idx++;
		fsize /= 1024;
	}

	return fsize;
}

static void vptr_debug_mem(int dl, const struct mem_s* mem)
{
	assert(0 != mem);
	debug_printf(dl, "Virtual pointer %p %s (%.1f%s)\n", mem->ptr,
		(VPTR_CFL == mem->loc) ? "wrapping CFL" : (VPTR_GPU == vptr_loc_sameplace(mem->loc) ? "on GPU" : "on CPU"),
		get_byte_size(mem->len), get_byte_unit(mem->len));

	if (0 < mem->shape.N) {

		debug_printf(dl, "size: %lu, dims: ", mem->shape.size);
		debug_print_dims(dl, mem->shape.N, mem->shape.dims);
	}

	if (0 < mem->range.D) {

		debug_printf(dl, "wraps %d virtual pointer:\n", mem->range.D);

		for (int i = 0; i < mem->range.D; i++) {

			debug_printf(dl, "%d: %p size: %lu, dims: ", i, mem->range.sub_ptr[i]->ptr, mem->range.sub_ptr[i]->shape.size);
			debug_print_dims(dl, mem->range.sub_ptr[i]->shape.N, mem->range.sub_ptr[i]->shape.dims);
		}
	}

	if ((0 == mem->shape.N) && (0 == mem->range.D))
		debug_printf(dl, "ptr not initialized\n");

	if (NULL != mem->backtrace)
		debug_printf(dl, "allocated at:\n%s", mem->backtrace);
}

static struct mem_s* search(const void* ptr, bool remove);

void vptr_debug(int dl, const void* ptr)
{
	struct mem_s* mem = search(ptr, false);

	if (NULL == mem) {

		debug_printf(dl, "Not a virtual pointer: %p\n", ptr);
		return;
	}

	vptr_debug_mem(dl, mem);
}

static int vptr_cmp(const void* _a, const void* _b)
{
	const struct mem_s* a = _a;
	const struct mem_s* b = _b;

	if (a->ptr == b->ptr)
		return 0;

	return (a->ptr > b->ptr) ? 1 : -1;
}


#ifndef BARTLIB_EXPORTS
static struct sigaction old_sa;

static void handler(int /*sig*/, siginfo_t *si, void*)
{
	struct mem_s* mem = search(si->si_addr, false);

	if (mem) {

		sleep((unsigned int)mpi_get_rank());
		vptr_debug_mem(DP_ERROR, mem);
		error("Virtual pointer at %p not resolved!\n", si->si_addr);
	}

#ifdef USE_CUDA
	if (cuda_ondevice(si->si_addr))
		error("Tried to access CUDA pointer at %x from CPU!\n", si->si_addr);
#endif
	error("Segfault!\n");
}
#endif

static void vptr_init(void)
{
	if (NULL != vmap)
		return;

#ifndef BARTLIB_EXPORTS
	struct sigaction sa;

	sa.sa_flags = SA_SIGINFO;
	sigemptyset(&sa.sa_mask);
	sa.sa_sigaction = handler;

	sigaction(SIGSEGV, &sa, &old_sa);

#pragma omp critical(bart_vmap)
	if (NULL == vmap)
		vmap = tree_create(vptr_cmp);
#endif
}


static int inside_p(const void* _rptr, const void* ptr)
{
	const struct mem_s* rptr = _rptr;

	if ((ptr >= rptr->ptr) && (ptr < rptr->ptr + rptr->len))
		return 0;

	return (rptr->ptr > ptr) ? 1 : -1;
}


static struct mem_s* search(const void* ptr, bool remove)
{
	if (NULL == vmap)
		return NULL;

	struct mem_s* mem;

#pragma omp critical(bart_vmap)
	mem = tree_find(vmap, ptr, inside_p, remove);

	return mem;
}

static struct mem_s* vptr_reserve_int(size_t len)
{
	assert(0 < len);

	void* ptr = mmap(NULL, len, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);

	PTR_ALLOC(struct mem_s, x);

	x->ptr = ptr;
	x->len = len;

	x->loc = VPTR_CPU;
	x->free = true;
	x->clear = false;

	x->shape.N = 0;
	x->shape.dims = NULL;
	x->shape.size = 0;

	x->blocks.shape = &x->shape;
	x->blocks.flags = 0UL;
	x->blocks.mem = NULL;

	x->range.D = 0;
	x->range.sub_ptr = NULL;

	x->writeback = false;
	x->hint = NULL;

	x->reduction_buffer = false;

	x->backtrace = NULL;

#ifdef VPTR_DEBUG
#ifdef USE_DWARF
	x->backtrace = debug_good_backtrace_string(4);
#endif // USE_DWARF
#endif


	vptr_init();
	tree_insert(vmap, x);

	return PTR_PASS(x);
}

void* vptr_alloc_size(size_t len)
{
	struct mem_s* mem = vptr_reserve_int(len);
	return mem->ptr;
}

bool vptr_is_init(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);

	if (NULL == mem)
		error("Cannot check initialization of non-virtual pointer!\n");

	return ((0 != mem->shape.N) || (0 != mem->range.D));
}

void vptr_clear(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);

	if (NULL == mem)
		error("Cannot clear non-virtual pointer!\n");

	for (int i = 0; (NULL != mem->blocks.mem) && (i < mem->blocks.num_blocks); i++)
		assert(NULL == mem->blocks.mem[i]);

	mem->clear = true;

	for (int i = 0; i < mem->range.D; i++) {

		struct mem_s* tmem = mem->range.sub_ptr[i];

		if (NULL == tmem)
			continue;

		for (int i = 0; (NULL != tmem->blocks.mem) && (i < tmem->blocks.num_blocks); i++)
			assert(NULL == tmem->blocks.mem[i]);

		tmem->clear = true;
	}
}

static void vptr_set_dims_int(struct mem_s* mem, int N, const long dims[N], size_t size, struct vptr_hint_s* hint)
{
	assert(0 == mem->range.D);

	if (0 == mem->blocks.shape->N) {

		assert(mem->len == (size_t)md_calc_size(N, dims) * size);

		mem->shape.N = N;
		mem->shape.dims = ARR_CLONE(long[N], dims);
		mem->shape.size = size;

		mem->hint = vptr_hint_ref(hint);

		if (NULL != hint) {

			assert(md_check_compat(MIN(N, hint->N), ~0UL, dims, hint->dims));
			mem->blocks.flags = hint->mpi_flags;
		}
	} else {

		assert(mem->shape.N == N);
		assert(mem->shape.size == size);
		assert(md_check_compat(N, ~0UL, mem->shape.dims, dims));
		assert(hint == mem->hint);
	}
}

void vptr_set_dims(const void* ptr, int N, const long dims[N], size_t size, struct vptr_hint_s* hint)
{
	struct mem_s* mem = search(ptr, false);

	if (NULL == mem)
		error("Cannot set dimensions of non-virtual pointer!\n");

	vptr_set_dims_int(mem, N, dims, size, hint);
}

void vptr_set_dims_sameplace(const void* x, const void* ref)
{
	struct mem_s* mem = search(x, false);
	struct mem_s* rmem = search(ref, false);
	assert(NULL != mem);
	assert(NULL != rmem);

	if (0 < mem->range.D || 0 < mem->shape.N)
		return;

	if (0 < rmem->range.D) {

		int D = rmem->range.D;
		struct mem_s* sub_mem[D];

		for (int i = 0; i < D; i++) {

			sub_mem[i] = vptr_reserve_int(rmem->range.sub_ptr[i]->len);
			sub_mem[i]->loc = vptr_loc_sameplace(rmem->loc);
			sub_mem[i]->clear = mem->clear;
			sub_mem[i]->reduction_buffer = mem->reduction_buffer;

			vptr_set_dims_int(sub_mem[i], rmem->range.sub_ptr[i]->shape.N, rmem->range.sub_ptr[i]->shape.dims, rmem->range.sub_ptr[i]->shape.size, rmem->range.sub_ptr[i]->hint);
		}

		mem->range.D = D;
		mem->range.sub_ptr = ARR_CLONE(struct mem_s*[D], sub_mem);
		mem->free = true;
		mem->hint = vptr_hint_ref(rmem->hint);
	} else {

		vptr_set_dims_int(mem, rmem->shape.N, rmem->shape.dims, rmem->shape.size, rmem->hint);
	}
}

static struct mem_s* vptr_create(int N, const long dims[N], size_t size, struct vptr_hint_s* hint)
{
	long len = md_calc_size(N, dims) * (long)size;

	struct mem_s* mem = vptr_reserve_int((size_t)len);
	vptr_set_dims_int(mem, N, dims, size, hint);

	return mem;
}

void* vptr_resolve_range(const void* ptr)
{
	if (NULL == ptr)
		return NULL;

	if (!is_vptr(ptr))
		return (void*)ptr;

	struct mem_s* mem = search(ptr, false);
	assert(NULL != mem);

	if((0 == mem->range.D) && (0 == mem->shape.N))
		error("Virtual pointer not initialized!");

	if (0 == mem->range.D)
		return (void*)ptr;

	long offset = ptr - mem->ptr;
	int i = 0;

	while (offset >= (long)mem->range.sub_ptr[i]->len)
		offset -= (long)mem->range.sub_ptr[i++]->len;

	return mem->range.sub_ptr[i]->ptr + offset;
}


static void* vptr_resolve_int(const void* ptr, bool assert_rank)
{
	ptr = vptr_resolve_range(ptr);

	struct mem_s* mem = search(ptr, false);

	if (NULL == mem)
		return (void*)ptr;

	if (!mpi_accessible(ptr)) {

		if (assert_rank)
			error("Trying to access %x from rank %d!\n", ptr, mpi_get_rank());

		return NULL;
	}

	return vptr_mem_block_resolve(&mem->blocks, mem->loc, mem->clear, ptr - mem->ptr);
}

void* vptr_resolve(const void* ptr)
{
	return vptr_resolve_int(ptr, true);
}

void* vptr_resolve_unchecked(const void* ptr)
{
	return vptr_resolve_int(ptr, false);
}


const struct vptr_shape_s* vptr_get_shape(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);
	assert(mem);
	assert(0 < mem->shape.N);
	return &mem->shape;
}


long vptr_get_offset(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);
	assert(mem);
	return ptr - mem->ptr;
}


size_t vptr_get_len(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);
	assert(mem);
	return mem->len;
}


bool is_vptr(const void* ptr)
{
	return NULL != search(ptr, false);
}

bool is_vptr_gpu(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);
	return mem && (VPTR_GPU == vptr_loc_sameplace(mem->loc));
}

bool is_vptr_cpu(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);
	return mem && (VPTR_CPU == vptr_loc_sameplace(mem->loc));
}

static void vptr_update_loc(struct mem_s* mem, enum VPTR_LOC loc)
{
	if (NULL == mem)
		error("Cannot change location of non-virtual pointer!\n");

	assert(loc == VPTR_CPU || loc == VPTR_GPU);

	if (vptr_loc_sameplace(mem->loc) == loc)
		return;

	mem->loc = loc;

	if (0 < mem->range.D) {

		for (int i = 0; i < mem->range.D; i++)
			vptr_update_loc(mem->range.sub_ptr[i], loc);

	} else {

		for (int i = 0; (NULL != mem->blocks.mem) && (i < mem->blocks.num_blocks); i++)
			if(NULL != mem->blocks.mem[i])
				error("Cannot change location of already allocated virtual pointer!\n");
	}
}

void vptr_set_gpu(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);
	vptr_update_loc(mem, VPTR_GPU);
}

void vptr_set_cpu(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);
	vptr_update_loc(mem, VPTR_CPU);
}




bool vptr_free(const void* ptr)
{
	//md_copy for writeback requires vptr to stay valid until after the copy, hence don't remove here
	struct mem_s* mem = search(ptr, false);

	if (NULL == mem)
		return false;

	if (mem->writeback)
		md_copy(mem->shape.N, mem->shape.dims, mem->blocks.mem[0], mem->ptr, mem->shape.size);

	mem = search(ptr, true);

	if (mem->free)
		for(int i = 0; i < mem->range.D; i++)
			vptr_free(mem->range.sub_ptr[i]->ptr);


	vptr_mem_free(&mem->blocks, mem->loc, mem->free);


	munmap((void*)ptr, mem->len);

	if (NULL != mem->shape.dims)
		xfree(mem->shape.dims);

	if (NULL != mem->range.sub_ptr)
		xfree(mem->range.sub_ptr);

	if (NULL != mem->backtrace)
		xfree(mem->backtrace);

	vptr_hint_free(mem->hint);

	xfree(mem);

	return true;
}

void* vptr_alloc(int N, const long dims[N], size_t size, struct vptr_hint_s* hint)
{
	struct mem_s* mem = vptr_create(N, dims, size, hint);
	return mem->ptr;
}

// returns NULL if ref is not a virtual pointer.
void* vptr_alloc_sameplace(int N, const long dims[N], size_t size, const void* ref)
{
	struct mem_s* mem = search(ref, false);

	if (NULL == mem)
		return NULL;

	auto ret = vptr_create(N, dims, size, mem->hint);
	ret->loc = vptr_loc_sameplace(mem->loc);

	return ret->ptr;
}

void* vptr_alloc_same(const void* ref)
{
	struct mem_s* mem = search(ref, false);

	if (NULL == mem)
		return NULL;

	void* ret = vptr_alloc_size(mem->len);
	vptr_set_dims_sameplace(ret, ref);

	if (is_vptr_gpu(ref))
		vptr_set_gpu(ret);

	return ret;
}

void* vptr_move_gpu(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);

	assert(mem && mem->ptr == ptr);

	auto ret = vptr_create(mem->shape.N, mem->shape.dims, mem->shape.size, mem->hint);
	ret->loc = VPTR_GPU;

	md_copy(mem->shape.N, mem->shape.dims, ret->ptr, mem->ptr, mem->shape.size);

	return ret->ptr;
}

void* vptr_move_cpu(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);

	assert(mem && mem->ptr == ptr);

	auto ret = vptr_create(mem->shape.N, mem->shape.dims, mem->shape.size, mem->hint);
	ret->loc = VPTR_CPU;

	md_copy(mem->shape.N, mem->shape.dims, ret->ptr, mem->ptr, mem->shape.size);

	return ret->ptr;
}

void* vptr_wrap(int N, const long dims[N], size_t size, const void* ptr, struct vptr_hint_s* hint, bool free, bool writeback)
{
	assert(!is_vptr(ptr));

	auto mem = vptr_create(N, dims, size, hint);

	mem->writeback = writeback;

	mem->blocks.flags = 0;
	vptr_mem_block_init(&mem->blocks);
	mem->blocks.mem[0] = (void*)ptr;

#ifdef USE_CUDA
	mem->loc = cuda_ondevice(ptr) ? VPTR_GPU : VPTR_CPU;
#endif
	mem->free = free;

	vptr_update_size(mem->loc, mem->blocks.block_size);

	return mem->ptr;
}

void* vptr_wrap_sameplace(int N, const long dims[N], size_t size, const void* ptr, const void* ref, bool free, bool writeback)
{
	assert(!is_vptr(ptr));

	struct mem_s* mem = search(ref, false);

	assert(NULL != mem);

	return vptr_wrap(N, dims, size, ptr, mem->hint, free, writeback);
}

void* vptr_wrap_cfl(int N, const long dims[N], size_t size, const void* ptr, struct vptr_hint_s* hint, bool free, bool writeback)
{
	assert(!is_vptr(ptr));

	auto mem = vptr_create(N, dims, size, hint);

	mem->writeback = writeback;

	mem->blocks.flags = 0;
	vptr_mem_block_init(&mem->blocks);
	mem->blocks.mem[0] = (void*)ptr;

	mem->loc = VPTR_CFL;
	mem->free = free;
	vptr_update_size(mem->loc, mem->blocks.block_size);

	return mem->ptr;
}


void* vptr_wrap_range(int D, void* ptr[D], bool free)
{
	assert(0 < D);
	size_t len = 0;

	struct mem_s* mem[D];

	for (int i = 0; i < D; i++) {

		mem[i] = search(ptr[i], false);

		assert(NULL != mem[i]);
		assert(is_vptr_gpu(ptr[i]) == is_vptr_gpu(ptr[0]));
		len += mem[i]->len;
	}

	struct mem_s* rmem = vptr_reserve_int(len);
	rmem->range.D = D;
	rmem->range.sub_ptr = ARR_CLONE(struct mem_s*[D], mem);
	rmem->free = free;
	rmem->hint = vptr_hint_ref(mem[0]->hint);
	rmem->loc = vptr_loc_sameplace(mem[0]->loc);

	return rmem->ptr;
}


/*
 * Assuming a loop over adims accesing a pointer allocated with mdims via astrs.
 * long apos[N];
 * do {
 * 	long mpos[D];
 * 	md_unravel_index(D, mpos, ~0UL, mdims, offset + md_calc_offset(N, astrs, apos));
 * 	...
 * } while (md_next(N, adims, ~0UL, apos));
 *
 * This function computes the positions in mpos which may change due to a change of apos[i].
 *
 * if a bit is not set in any return flag, we access only a slice of the memory
 * if a bit is only set in one return flag, we can access the memory in a single loop
*/
static void loop_access_dims(int N, unsigned long flags[N], const long adims[N], const long astrs[N], int D, const long mdims[D], long offset)
{
	long mstrs[D];
	md_calc_strides(D, mstrs, mdims, 1);

	long mpos[D];
	md_unravel_index(D, mpos, ~0UL, mdims, offset);

	long adims2[MIN(N, D)];
	for (int i = 0; i < MIN(N, D); i++)
		adims2[i] = adims[i] + mpos[i];

	if (   md_check_bounds(MIN(N, D), md_nontriv_dims(N, adims), adims2, mdims)
	    && md_check_equal_dims(MIN(N, D), astrs, mstrs, md_nontriv_dims(N, adims))
	    && (N < D || 1 == md_calc_size(N - D, adims + D))) {

		for (int i = 0; i < N; i++)
			flags[i] = 1 < adims[i] ? MD_BIT(i) : 0ul;

		return;
	}

	long dstrs[N][D];
	for (int i = 0; i < N; i++) {

		md_set_dims(D, dstrs[i], 0);

		if (0 != astrs[i])
			md_unravel_index(D, dstrs[i], ~0UL, mdims, labs(astrs[i]));

		if (0 > astrs[i])
			for (int j = 0; j < D; j++)
				dstrs[i][j] = -dstrs[i][j];
	}

	long mlpos[D];
	long mupos[D];

	memset(mlpos, 0, sizeof mlpos);	// GCC ANALYZER
	memset(mupos, 0, sizeof mupos);	// GCC ANALYZER

	for (int j = 0; j < D; j++) {

		mlpos[j] = mpos[j];
		mupos[j] = mpos[j];

		for (int i = 0; i < N; i++) {

			mlpos[j] = MIN(mlpos[j], dstrs[i][j] * (adims[i] - 1));
			mupos[j] = MAX(mupos[j], dstrs[i][j] * (adims[i] - 1));
		}
	}

	//which dimensions are affected by a move in this dimension
	unsigned long affect_flags[D];

	for (int i = 0; i < D; i++) {

		affect_flags[i] = MD_BIT(i);

		long mlposj = mlpos[i];
		long muposj = mupos[i];

		for (int j = i; j < D - 1; j++) {

			if ((0 > mlposj) || (muposj >= mdims[j])) {

				affect_flags[i] |= MD_BIT(j + 1);

				mlposj /= mdims[j];
				muposj /= mdims[j];

				mlposj += mlpos[j + 1];
				muposj += mupos[j + 1];
			} else {

				break;
			}
		}
	}

	for(int i = 0; i < N; i++) {

		flags[i] = 0ul;

		for(int j = 0; j < D; j++)
			if (0 != dstrs[i][j])
				flags[i] |= affect_flags[j];
	}
}

static bool check_valid_loop_access(const struct mem_s* mem , int N, const long dims[N], const long strs[N], size_t size, const void* ptr, bool throw_error)
{
	const void* minp = ptr;
	const void* maxp = ptr + size - 1;

	for (int i = 0; i < N; i++) {

		minp += MIN(0, (dims[i] - 1) * strs[i]);
		maxp += MAX(0, (dims[i] - 1) * strs[i]);
	}

	bool valid = !((maxp >= mem->ptr + mem->len) || minp < mem->ptr);

	if (throw_error && !valid) {

		debug_print_dims(DP_INFO, N, dims);
		debug_print_dims(DP_INFO, N, strs);
		vptr_debug_mem(DP_ERROR, mem);
		error("Invalid vptr access at %p!\n", ptr);
	}

	return valid;
}

static void size_to_dims(int N, long odims[N + 1], const long idims[N], size_t size)
{
	odims[0] = (long)size;
	md_copy_dims(N, odims + 1, idims);
}

static void size_to_strs(int N, long ostrs[N + 1], const long istrs[N], size_t /*size*/)
{
	ostrs[0] = 1;
	md_copy_dims(N, ostrs + 1, istrs);
}



/**
 * Returns which dimensions cannot be accessed using the same resolved pointer
 */
unsigned long vptr_block_loop_flags(int N, const long dims[N], const long strs[N], const void* ptr, size_t size, _Bool contiguous_strs)
{
	ptr = vptr_resolve_range(ptr);
	struct mem_s* mem = search(ptr, false);

	if (NULL == mem)
		return 0UL;

	vptr_mem_block_init(&mem->blocks);

	check_valid_loop_access(mem, N, dims, strs, size, ptr, true);

	unsigned long lflags = mem->blocks.flags;
	if (NULL != mem->hint)
		lflags |= mem->hint->mpi_flags;

	if (0 == lflags)
		return 0UL;

	long mdims[mem->shape.N + 1];
	long tdims[N + 1];
	long tstrs[N + 1];

	size_to_dims(mem->shape.N, mdims, mem->shape.dims, mem->shape.size);
	size_to_dims(N, tdims, dims, size);
	size_to_strs(N, tstrs, strs, size);

	md_select_dims(N + 1, md_nontriv_strides(N + 1, tstrs), tdims, tdims);
	md_select_strides(N + 1, md_nontriv_dims(N + 1, tdims), tstrs, tstrs);

	unsigned long flags[N + 1];
	memset(flags, 0, sizeof flags);	// GCC ANALYZER
	loop_access_dims(N + 1, flags, tdims, tstrs, mem->shape.N + 1, mdims, ptr - mem->ptr);

	unsigned long ret_flags = 0;

	for (int i = 1; i < N + 1; i++)
		if (0 != (lflags & (flags[i] / 2)))
			ret_flags |= MD_BIT(i);

	ret_flags /= 2;

	if (0 != ret_flags && !contiguous_strs) {

		for (int i = md_min_idx(ret_flags); i < N; i++)
			ret_flags |= MD_BIT(i);

		ret_flags &= md_nontriv_dims(N, dims);
	}

	return ret_flags;
}

void vptr_contiguous_strs(int N, const void* ptr, unsigned long lflags, long nstrs[N], const long ostrs[N])
{
	ptr = vptr_resolve_range(ptr);
	struct mem_s* mem = search(ptr, false);

	if (NULL == mem) {

		for (int i = 0; i < N; i++)
			nstrs[i] = ostrs[i];

		return;
	}

	int Nm = mem->shape.N + 1;
	long mdims[Nm];
	mdims[0] = (long)mem->shape.size;
	md_copy_dims(mem->shape.N, mdims + 1, mem->shape.dims);

	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(lflags, i) || (0 == ostrs[i])) {

			nstrs[i] = ostrs[i];
			continue;
		}

		nstrs[i] = md_reravel_index(Nm, ~(2 * mem->blocks.flags) & md_nontriv_dims(Nm, mdims), md_nontriv_dims(Nm, mdims), mdims, labs(ostrs[i])) * (ostrs[i] < 0 ? -1 : 1);
	}
}


/**
 * Returns true if ptr is distributed along multiple processes
 *
 * @param ptr pointer to check
 */
bool is_mpi(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);

	return mem && mem->hint && (mem->hint->mpi_flags != 0UL);
}


/**
 * Calculates rank of given pointer if it is distributed returns -1 else
 *
 * @param ptr pointer to check
 */
int mpi_ptr_get_rank(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);

	assert(mem && mem->hint && (0 != mem->hint->mpi_flags));

	auto h = mem->hint;

	int N = MAX(mem->shape.N, h->N);

	long pos[N];
	md_set_dims(N, pos, 0);

	// position in allocation
	md_unravel_index(mem->shape.N, pos, ~(0UL), mem->shape.dims, (ptr - mem->ptr) / (long)mem->shape.size);

	return hint_get_rank(h->N, pos, mem->hint);
}


static bool mpi_accessible_from_mem(const struct mem_s* mem, const void* ptr, int rank)
{
	if ((NULL == mem) || (NULL == mem->hint) || (0 == mem->hint->mpi_flags))
		return true;

	if (0 == mem->shape.N) {

		vptr_debug_mem(DP_ERROR, mem);
		error("Virtual pointer is range or not initialized!\n");
	}

	struct vptr_hint_s* h = mem->hint;
	int N = MAX(mem->shape.N, h->N);

	long pos[N];
	md_set_dims(N, pos, 0);

	md_unravel_index(mem->shape.N, pos, ~0UL, mem->shape.dims, (ptr - mem->ptr) / (long)mem->shape.size);


	unsigned long loop_flags = ~md_nontriv_dims(mem->shape.N, mem->shape.dims);

	loop_flags &= h->mpi_flags;

	do {
		if (hint_get_rank(h->N, pos, h) == rank)
			return true;

	} while (md_next(h->N, h->dims, loop_flags, pos));

	return false;
}

bool mpi_accessible_from(const void* ptr, int rank)
{
	ptr = vptr_resolve_range(ptr);
	auto mem = search(ptr, false);

	return mpi_accessible_from_mem(mem, ptr, rank);
}


bool mpi_accessible(const void* ptr)
{
	return mpi_accessible_from(ptr, mpi_get_rank());
}

static bool mpi_accessible_from_mult(int N, const struct mem_s* mem[N], const void* ptr[N], int rank)
{

	for (int i = 0; i < N; i++)
		if (!mpi_accessible_from_mem(mem[i], ptr[i], rank))
			return false;

	return true;
}


bool mpi_accessible_mult(int N, const void* tptr[N])
{
	const void* ptr[N];
	for (int i = 0; i < N; i++)
		ptr[i] = vptr_resolve_range(tptr[i]);

	const struct mem_s* mem[N];
	for (int i = 0; i < N; i++)
		mem[i] = search(ptr[i], false);

	if (!mpi_accessible_from_mult(N, mem, ptr, mpi_get_rank()))
		return false;

	bool reduce = false;

	for (int i = 0; i < N; i++)
		if (mem[i] && mem[i]->reduction_buffer)
			reduce = true;

	for (int i = 0; i < mpi_get_rank(); i++)
		if (reduce && mpi_accessible_from_mult(N, mem, ptr, i))
			return false;

	return true;
}


void vptr_assert_sameplace(int N, void* nptr[N])
{
	struct vptr_hint_s* hint_ref = NULL;

	struct mem_s* mem_ref = search(nptr[0], false);

	if (mem_ref)
		hint_ref = mem_ref->hint;

	for (int i = 1; i < N; i++) {

		struct mem_s* mem = search(nptr[i], false);

		if ((NULL == mem_ref) && (NULL != mem))
			error("Incompatible pointer: vptr(%x) at %d and normal pointer(%x) at 0!\n", nptr[i], i, nptr[0]);

		if ((NULL != mem_ref) && (NULL == mem))
			error("Incompatible pointer: vptr(%x) at 0 and normal pointer(%x) at %d!\n", nptr[0], nptr[i], i);

		if (NULL == mem_ref)
			continue;

		if ((hint_ref && !mem->hint) || (!hint_ref && mem->hint))
			error("Incompatible hints!\n");

		if (NULL == hint_ref)
			continue;

		if (   (hint_ref->N != mem->hint->N)
		    || !md_check_equal_dims(hint_ref->N, hint_ref->dims, mem->hint->dims, ~0UL)) {

			debug_print_dims(DP_INFO, hint_ref->N, hint_ref->dims);
			debug_print_dims(DP_INFO, mem->hint->N, mem->hint->dims);

			error("Incompatible MPI dist rule!\n");
		}
	}
}

void mpi_set_reduction_buffer(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);
	assert(mem);
	mem->reduction_buffer = true;

	for (int i = 0; i < mem->range.D; i++)
		if (NULL != mem->range.sub_ptr[i])
			mem->range.sub_ptr[i]->reduction_buffer = true;

}

void mpi_unset_reduction_buffer(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);
	assert(mem);
	mem->reduction_buffer = false;

	for (int i = 0; i < mem->range.D; i++)
		if (NULL != mem->range.sub_ptr[i])
			mem->range.sub_ptr[i]->reduction_buffer = false;
}

bool mpi_is_reduction(int N, const long dims[N], const long ostrs[N], const void* optr, size_t osize, const long istrs[N], const void* iptr, size_t isize)
{
	struct vptr_mapped_dims_s* mdims = vptr_map_dims(N, dims, 2, (const long*[2]) { ostrs, istrs }, (const size_t[2]){ osize, isize }, (void*[2]){ (void*)optr, (void*)iptr });

	if (NULL != mdims) {

		bool ret = false;

		while (NULL != mdims) {

			const long (*mstrs)[mdims->D][mdims->N] = (void*) mdims->strs;
			ret = ret || mpi_is_reduction(mdims->N, mdims->dims, (*mstrs)[0], mdims->ptr[0], osize, (*mstrs)[1], mdims->ptr[1], isize);
			mdims = vptr_mapped_dims_free_and_next(mdims);
		}

		return ret;
	}

	if (!is_mpi(iptr))
		return false;

	if (!is_mpi(optr))
		return true;

	struct mem_s* imem = search(iptr, false);
	struct mem_s* omem = search(optr, false);

	if (imem->hint != omem->hint)
		return true;

	if (0 == (imem->hint->mpi_flags & md_nontriv_dims(imem->shape.N, imem->shape.dims)))
		return false;

	unsigned long flags = 0;
	flags |= vptr_block_loop_flags(N, dims, istrs, iptr, isize, true);
	flags |= vptr_block_loop_flags(N, dims, ostrs, optr, osize, true);

	if (0 == flags)
		return false;

	long imem_strs[imem->shape.N];
	long omem_strs[omem->shape.N];

	md_calc_strides(imem->shape.N, imem_strs, imem->shape.dims, imem->shape.size);
	md_calc_strides(omem->shape.N, omem_strs, omem->shape.dims, omem->shape.size);

	unsigned long imem_flags = imem->hint->mpi_flags & md_nontriv_dims(imem->shape.N, imem->shape.dims);
	unsigned long omem_flags = omem->hint->mpi_flags & md_nontriv_dims(omem->shape.N, omem->shape.dims);

	for (int i = 0; i < N; i++) {

		if (!MD_IS_SET(flags, i))
			continue;

		if (i >= imem->shape.N || i >= omem->shape.N)
			return true;

		if (istrs[i] != imem_strs[i] || ostrs[i] != omem_strs[i])
			return true;

		if (MD_IS_SET(imem_flags, i) != MD_IS_SET(omem_flags, i))
			return true;
	}

	return false;
}


struct vptr_hint_s* vptr_get_hint(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);

	return (NULL != mem) ? mem->hint : NULL;
}




static struct vptr_mapped_dims_s* vptr_mem_map_dims(int N, const long odims[N], int D, const long* ostrs[D], const size_t size[D], void* ptr[D], const struct mem_s* mem[D], bool check_changed)
{
	int Nmem = 0;
	for (int i = 0; i < D; i++)
		if (NULL != mem[i])
			Nmem = MAX(Nmem, mem[i]->shape.N);

	int Nsize = MD_BIT(D);

	int Nred = bitcount(md_nontriv_dims(N, odims));

	long maxdims[Nmem + Nsize];
	md_singleton_dims(Nmem + Nsize, maxdims);

	long strs[D][Nred + Nmem + Nsize];

	for (int i = 0; i < D; i++) {

		md_singleton_strides(Nred + Nmem + Nsize, strs[i]);

		if (NULL == mem[i])
			continue;

		long tdims[Nmem + Nsize];
		md_singleton_dims(Nmem + Nsize, tdims);

		md_copy_dims(mem[i]->shape.N, tdims, mem[i]->shape.dims);
		md_calc_strides(Nmem, strs[i], tdims, mem[i]->shape.size);

		for (int j = 0; j < Nsize; j++) {

			if (MD_IS_SET((unsigned long)j, i))
				continue;

			strs[i][j + Nmem] = (long)size[i];
			tdims[j + Nmem] = (long)mem[i]->shape.size / strs[i][j + Nmem];
		}

		md_max_dims(Nmem + Nsize, ~0UL, maxdims, maxdims, tdims);
	}

	unsigned long set_flag = 0UL;
	bool split = false;

	long dims[Nred + Nmem + Nsize];
	md_singleton_dims(Nmem + Nsize, dims);

	for (int i = 0, ip = Nmem + Nsize; i < N; i++) {

		if (1 == odims[i])
			continue;

		for (int j = 0; j < D; j++)
			strs[j][ip] = ostrs[j][i];

		dims[ip] = odims[i];
		ip++;
	}

	do {
		split = false;

		for (int j = Nmem + Nsize; j < Nred + Nmem + Nsize; j++) {

			if (1 == dims[j])
				continue;

			for (int k = 0; k < Nmem + Nsize; k++) {

				bool match = true;
				for (int l = 0; l < D; l++) {

					if (NULL == mem[l] && !MD_IS_SET(set_flag, l))
						continue;

					match = match && (strs[l][k] * dims[k] == strs[l][j]);
				}

				if (!match)
					continue;

				long dim = MIN(maxdims[k], dims[j]);

				while ((0 != (dims[j] % dim)) || (0 != (maxdims[k] % dim)))
					dim--;

				if (1 < dim) {

					maxdims[k] /= dim;
					dims[k] *= dim;
					dims[j] /= dim;

					if (!MD_IS_SET(set_flag, k)) {

						for (int l = 0; l < D; l++)
							strs[l][k] = strs[l][j];

						set_flag = MD_SET(set_flag, k);
					}

					for (int l = 0; l < D; l++)
						strs[l][j] *= dim;

					split = true;
				}
			}
		}
	} while (split);

	if (1 == md_calc_size(Nmem, dims) && check_changed)
		return NULL;

	unsigned long flag = md_nontriv_dims(Nred + Nmem + Nsize, dims);

	int Nnew = Nred + Nmem + Nsize;

	long nstrs[D][Nnew];
	for (int i = 0; i < D; i++)
		md_select_strides(Nnew, flag, nstrs[i], strs[i]);

	if (check_changed) {

		bool same = (N >= Nmem);
		same = same && md_check_equal_dims(Nmem, dims, odims, ~0UL);

		for (int i = 0; i < D; i++)
			same = same && md_check_equal_dims(Nmem, nstrs[i], ostrs[i], flag);

		if (same)
			return NULL;
	}


	PTR_ALLOC(struct vptr_mapped_dims_s, x);

	x->N = Nnew;
	x->dims = ARR_CLONE(long[x->N], dims);
	x->D = D;
	x->strs = ARR_CLONE(long[x->D * x->N], nstrs);
	x->ptr = ARR_CLONE(void*[x->D], ptr);
	x->next = NULL;

	return PTR_PASS(x);
}


struct vptr_mapped_dims_s* vptr_map_dims(int N, const long dims[N], int D, const long* strs[D], const size_t size[D], void* ptr[D])
{
	const struct mem_s* mem[D];

	void* tptr[D];
	bool valid = true;
	bool vptr = false;
	bool vptr_range = false;


	for (int i = 0; i < D; i++) {

		tptr[i] = vptr_resolve_range(ptr[i]);
		mem[i] = search(tptr[i], false);

		valid = valid && ((NULL == mem[i]) || check_valid_loop_access(mem[i], N, dims, strs[i], size[i], tptr[i], false));
		vptr = vptr || (NULL != mem[i]);
		vptr_range = vptr_range || (tptr[i] != ptr[i]);
	}

	if (!vptr)
		return NULL;

	if (valid)
		return vptr_mem_map_dims(N, dims, D, strs, size, tptr, mem, !vptr_range);

	if (1 != N)
		error("Cannot split vptr range with non flat dimensions!\n");

	long dim = dims[0];

	for (int i = 0; i < D; i++) {

		if ((0 == strs[i][0]) || NULL == mem[i])
			continue;

		if (0 > strs[i][0])
			error("Cannot split vptr range with negative strides!\n"); //FIXME: should be possible

		dim = MIN(dim, ((long)mem[i]->len - (tptr[i] - mem[i]->ptr)) / strs[i][0]);
	}

	assert(0 < dim);

	struct vptr_mapped_dims_s* ret = vptr_mem_map_dims(1, &dim, D, strs, size, tptr, mem, false);

	long rdim = dims[0] - dim;
	for (int i = 0; i < D; i++)
		tptr[i] = ptr[i] + dim * strs[i][0];

	ret->next = 0 < rdim ? vptr_map_dims(1, &rdim, D, strs, size, tptr) : NULL;

	return ret;
}


struct vptr_mapped_dims_s* vptr_mapped_dims_free_and_next(struct vptr_mapped_dims_s* x)
{
	struct vptr_mapped_dims_s* ret = x->next;

	xfree(x->dims);
	xfree(x->strs);
	xfree(x->ptr);
	xfree(x);

	return ret;
}


