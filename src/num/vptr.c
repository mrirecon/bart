/* Copyright 2023-2024. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal, Bernhard Rapp
 */

#include <assert.h>
#include <stdbool.h>
#include <signal.h>

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

struct mem_s {

	void* ptr;
	void** mem;
	size_t len;
	long num_blocks;
	long block_size;

	bool gpu;

	bool free;		// mem should be free'd
	bool free_first_only;	// only first block needs to be free'd
	bool writeback;

	int N;
	long* dims;
	size_t size;

	struct vptr_hint_s* hint;
};

static int vptr_cmp(const void* _a, const void* _b)
{
	const struct mem_s* a = _a;
	const struct mem_s* b = _b;

	if (a->ptr == b->ptr)
		return 0;

	return (a->ptr > b->ptr) ? 1 : -1;
}

static struct mem_s* search(const void* ptr, bool remove);

static struct sigaction old_sa;

static void handler(int /*sig*/, siginfo_t *si, void*)
{
	struct mem_s* mem = search(si->si_addr, false);

	if (mem)
		error("Virtual pointer at %x not resolved!\n", si->si_addr);

#ifdef USE_CUDA
	if (cuda_ondevice(si->si_addr))
		error("Tried to access CUDA pointer at %x from CPU!\n", si->si_addr);
#endif
	error("Segfault!\n");
}

static void vptr_init(void)
{
	if (NULL != vmap)
		return;

	struct sigaction sa;

	sa.sa_flags = SA_SIGINFO;
	sigemptyset(&sa.sa_mask);
	sa.sa_sigaction = handler;

	sigaction(SIGSEGV, &sa, &old_sa);

#pragma omp critical(bart_vmap)
	if (NULL == vmap)
		vmap = tree_create(vptr_cmp);
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


static struct mem_s* vptr_create(int N, const long dims[N], size_t size, struct vptr_hint_s* hint)
{
	long len = md_calc_size(N, dims) * (long)size;

	void* ptr = mmap(NULL, (size_t)len, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);

	PTR_ALLOC(struct mem_s, x);

	x->ptr = ptr;
	x->mem = NULL;
	x->len = (size_t)len;

	x->num_blocks = 1;
	x->block_size = len;

	x->free = true;
	x->free_first_only = false;
	x->gpu = false;

	x->N = N;
	x->dims = ARR_CLONE(long[N], dims);
	x->size = size;
	x->writeback = false;
	x->hint = vptr_hint_ref(hint);

	if (NULL != hint) {

		assert(md_check_compat(MIN(N, hint->N), ~0UL, dims, hint->dims));

		x->block_size = (long)size;	// size of continuous blocks located on one rank

		for (int i = 0; (i < N) && !MD_IS_SET(md_nontriv_dims(N, dims) & hint->mpi_flags, i); i++)
			x->block_size *= x->dims[i];

		x->num_blocks = (long)x->len / x->block_size;
	}

	vptr_init();
	tree_insert(vmap, x);

	return PTR_PASS(x);
}


static void* vptr_resolve_int(const void* ptr, bool assert_rank)
{
	struct mem_s* mem = search(ptr, false);

	if (NULL == mem)
		return (void*)ptr;

	if (!mpi_accessible(ptr)) {

		if (assert_rank)
			error("Trying to access %x from rank %d!\n", ptr, mpi_get_rank());

		return NULL;
	}

#pragma omp critical(bart_vmap)
	if (NULL == mem->mem) {

		mem->mem = *TYPE_ALLOC(void*[mem->num_blocks]);

		for (long i = 0; i < mem->num_blocks; i++)
			mem->mem[i] = NULL;
	}

	long idx = (ptr - mem->ptr) / (mem->block_size);

#pragma omp critical(bart_vmap)
	if (NULL == (mem->mem[idx])) {

#ifdef USE_CUDA
		if (mem->gpu)
			mem->mem[idx] = cuda_malloc(mem->block_size);
		else
#endif
		mem->mem[idx] = xmalloc((size_t)mem->block_size);
	}

	return mem->mem[idx] + ((ptr - mem->ptr) % mem->block_size);
}

void* vptr_resolve(const void* ptr)
{
	return vptr_resolve_int(ptr, true);
}

void* vptr_resolve_unchecked(const void* ptr)
{
	return vptr_resolve_int(ptr, false);
}


bool is_vptr(const void* ptr)
{
	return NULL != search(ptr, false);
}

bool is_vptr_gpu(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);
	return mem && mem->gpu;
}

bool is_vptr_cpu(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);
	return mem && !mem->gpu;
}



bool vptr_free(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);

	if (NULL == mem)
		return false;

	if (mem->free && (NULL != mem->mem)) {

		md_free(mem->mem[0]);

		for (int i = 1; (i < mem->num_blocks) && !mem->free_first_only; i++)
			md_free(mem->mem[i]);

	} else {

		// only for continuous allocations
		if (mem->writeback)
			md_copy(mem->N, mem->dims, mem->mem[0], mem->ptr, mem->size);
	}

	mem = search(ptr, true);

	munmap((void*)ptr, mem->len);

	if (NULL != mem->dims)
		xfree(mem->dims);

	if (NULL != mem->mem)
		xfree(mem->mem);

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
	ret->gpu = mem->gpu;

	return ret->ptr;
}

void* vptr_move_gpu(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);

	assert(mem && mem->ptr == ptr);

	auto ret = vptr_create(mem->N, mem->dims, mem->size, mem->hint);
	ret->gpu = true;

	md_copy(mem->N, mem->dims, ret->ptr, mem->ptr, mem->size);

	return ret->ptr;
}

void* vptr_move_cpu(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);

	assert(mem && mem->ptr == ptr);

	auto ret = vptr_create(mem->N, mem->dims, mem->size, mem->hint);
	ret->gpu = false;

	md_copy(mem->N, mem->dims, ret->ptr, mem->ptr, mem->size);

	return ret->ptr;
}

void* vptr_wrap(int N, const long dims[N], size_t size, const void* ptr, struct vptr_hint_s* hint, bool free, bool writeback)
{
	assert(!is_vptr(ptr));

	auto mem = vptr_create(N, dims, size, hint);

	mem->free_first_only = true;
	mem->free = free;
	mem->writeback = writeback;

	mem->mem = *TYPE_ALLOC(void*[mem->num_blocks]);

	for (int i = 0; i < mem->num_blocks; i++)
		mem->mem[i] = (void*)ptr + i * mem->block_size;

#ifdef USE_CUDA
	mem->gpu = cuda_ondevice(ptr);
#endif
	return mem->ptr;
}

void* vptr_wrap_sameplace(int N, const long dims[N], size_t size, const void* ptr, const void* ref, bool free, bool writeback)
{
	assert(!is_vptr(ptr));

	struct mem_s* mem = search(ref, false);

	assert(NULL != mem);

	return vptr_wrap(N, dims, size, ptr, mem->hint, free, writeback);
}


/**
 * Returns which dimensions cannot be accessed using the same resolved pointer
 */
unsigned long vptr_block_loop_flags(int N, const long dims[N], const long strs[N], const void* ptr, size_t size)
{
	struct mem_s* mem = search(ptr, false);

	if ((NULL == mem) || ((mem->block_size == (long)(mem->len))))
		return 0UL;

	long tdims[N + 1];
	long tstrs[N + 1];

	tdims[0] = (long)size;
	tstrs[0] = 1;

	md_select_dims(N, md_nontriv_strides(N, strs), tdims + 1, dims);
	md_select_strides(N, md_nontriv_dims(N, tdims + 1), tstrs + 1, strs);


	//general case: To check if dimension i can be safely accessed from one rank:
	//		1.) Search for all positions with pos[i] = 0 for max(offset % block_size) and min(offset % block_size)
	//		2.) Check if by changing pos[i] the range [0, mpi_size - 1] is left

	long rstrs[N + 1];
	for (int i = 0; i < N + 1; i++)
		rstrs[i] = tstrs[i] % mem->block_size;

	unsigned long flags = 0;
	long offset = (ptr - mem->ptr) % mem->block_size;

	for (int i = 0; i < N + 1; i++) {

		if (mem->block_size <= tstrs[i]) {

			flags |= MD_BIT(i);
			continue;
		}

		if (0 == rstrs[i])
			continue;

		long pos[N + 1];
		md_set_dims(N + 1, pos, 0);

		long max = offset; // max of (offset % block_size) for all possible pos with pos[i] = 0
		long min = offset; // min of (offset % block_size) for all possible pos with pos[i] = 0

		for (int j = 0; j < N + 1; j++) {

			if (i == j)
				continue;

			// cheap search by just considering max position
			if (0 < rstrs[i])
				max += rstrs[j] * (tdims[j] - 1);
			else
				min += rstrs[j] * (tdims[j] - 1);
		}

		if (   ((0 < tstrs[i]) && (max >= mem->block_size))
		    || ((0 > tstrs[i]) && (0 < min))) {

		    	// Cheap search is not valid!
			// Expensive brute force search!
			max = offset;
			min = offset;

			do {
				long o = (md_calc_offset(N + 1, rstrs, pos) + offset) % mem->block_size;

				if (0 > o)
					o += mem->block_size;

				max = MAX(max, o);
				min = MIN(min, o);

			} while (md_next(N + 1, tdims, ~MD_BIT(i) & md_nontriv_strides(N + 1, rstrs), pos));
		}

		if ((tstrs[i] > 0) && (max + (tdims[i] - 1) * tstrs[i] >= mem->block_size))
			flags |= MD_BIT(i);

		if ((tstrs[i] < 0) && (min + (tdims[i] - 1) * tstrs[i] < 0))
			flags |= MD_BIT(i);
	}

	if (MD_IS_SET(flags, 0))
		error("Memory block overlaps MPI boundaries!\n");

	return flags / 2;
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

	int N = MAX(mem->N, h->N);

	long pos[N];
	md_set_dims(N, pos, 0);

	// position in allocation
	md_unravel_index(mem->N, pos, ~(0UL), mem->dims, (ptr - mem->ptr) / (long)mem->size);

	return hint_get_rank(h->N, pos, mem->hint);
}


bool mpi_accessible_from(const void* ptr, int rank)
{
	auto mem = search(ptr, false);

	if ((NULL == mem) || (NULL == mem->hint) || (0 == mem->hint->mpi_flags))
		return true;

	struct vptr_hint_s* h = mem->hint;
	int N = MAX(mem->N, h->N);

	long pos[N];
	md_set_dims(N, pos, 0);

	md_unravel_index(mem->N, pos, ~0UL, mem->dims, (ptr - mem->ptr) / (long)mem->size);


	unsigned long loop_flags = ~md_nontriv_dims(mem->N, mem->dims);

	loop_flags &= MD_BIT(mem->N) - 1;
	loop_flags &= h->mpi_flags;

	do {
		if (hint_get_rank(h->N, pos, h) == rank)
			return true;

	} while (md_next(h->N, h->dims, loop_flags, pos));

	return false;
}


bool mpi_accessible(const void* ptr)
{
	return mpi_accessible_from(ptr, mpi_get_rank());
}


int mpi_reduce_color(unsigned long reduce_flags, const void* ptr)
{
	// FIXME: duplicates a lot of code of mpi_accessible_from
	//
	struct mem_s* mem = search(ptr, false);

	assert(NULL != mem);

	struct vptr_hint_s* h = mem->hint;
	int N = MAX(mem->N, h->N);

	long pos[N];

	md_set_dims(N, pos, 0);

	//position in allocation
	md_unravel_index(mem->N, pos, ~0UL, mem->dims, (ptr - mem->ptr) / (long)mem->size);


	unsigned long loop_flags = ~md_nontriv_dims(mem->N, mem->dims);

	loop_flags &= MD_BIT(mem->N) - 1;
	loop_flags &= h->mpi_flags;

	do {
		if (hint_get_rank(h->N, pos, h) == mpi_get_rank())
			return 1 + (int)md_ravel_index(h->N, pos, ~reduce_flags, h->dims);

	} while (md_next(h->N, h->dims, loop_flags, pos));

	return 0;
}


unsigned long mpi_parallel_flags(int N, const long dims[N], const long strs[N], size_t size, const void* ptr)
{
	struct mem_s* mem = search(ptr, false);

	assert((NULL != mem) && (NULL != mem->hint));
	assert((size == mem->size / 2) || (size == mem->size));

	long tdims[N];

	if (size == mem->size) {

		md_select_dims(N, md_nontriv_strides(N, strs), tdims, dims);

	} else {

		N--;
		md_select_dims(N, md_nontriv_strides(N, strs + 1), tdims, dims + 1);
	}

	assert(md_check_equal_dims(MIN(N, mem->N), tdims, mem->dims, ~0UL));
	assert(0 == md_nontriv_dims(N - mem->N, tdims + mem->N));
	assert(0 == md_nontriv_dims(mem->N - N, mem->dims + N));

	return mem->hint->mpi_flags & ~md_nontriv_dims(mem->N, mem->dims);
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

