/* Copyright 2023-2024. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal, Bernhard Rapp
 */

#include <assert.h>
#include <stdbool.h>
#include <signal.h>
#include <errno.h>

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
#include "num/vptr_fun.h"
#include "num/mpi_ops.h"

#include "vptr.h"


#ifdef USE_DWARF
#undef assert
#define assert(expr) do { if (!(expr)) error("Assertion '" #expr "' failed in %s:%d\n",  __FILE__, __LINE__); } while (0)
#endif

//#define VPTR_DEBUG
#ifndef __EMSCRIPTEN__
#define VPTR_TAGGING
#endif

static tree_t vmap = NULL;

static long vptr_size = 0;
static long vptr_cpu_size = 0;
static long vptr_gpu_size = 0;
static long vptr_cfl_size = 0;

static long vptr_peak = 0;
static long vptr_cpu_peak = 0;
static long vptr_gpu_peak = 0;
static long vptr_cfl_peak = 0;

static void vptr_update_cpu(long size)
{
	#pragma omp atomic
	vptr_cpu_size += size;
	vptr_cpu_peak = MAX(vptr_cpu_peak, vptr_cpu_size);

	#pragma omp atomic
	vptr_size += size;
	vptr_peak = MAX(vptr_peak, vptr_size);
}

static void vptr_update_gpu(long size)
{
	#pragma omp atomic
	vptr_gpu_size += size;
	vptr_gpu_peak = MAX(vptr_gpu_peak, vptr_gpu_size);

	#pragma omp atomic
	vptr_size += size;
	vptr_peak = MAX(vptr_peak, vptr_size);
}

static void vptr_update_cfl(long size)
{
	#pragma omp atomic
	vptr_cfl_size += size;
	vptr_cfl_peak = MAX(vptr_cfl_peak, vptr_cfl_size);

	#pragma omp atomic
	vptr_size += size;
	vptr_peak = MAX(vptr_peak, vptr_size);
}


struct vptr_hint_s {

	//delayed
	unsigned long loop_flags;	// flags for delayed looping

	//mpi
	int N;
	long* dims;			// dstributed mpi dimensions, any allocation must be compatible with this

	long* rank;			// rank of distribute array, if memory has a singleton dimension, it is accessedfrom multiple ranks
	unsigned long mpi_flags;

	long* rdims;			// reduced dimensions, i.e. dimensions without singleton dimensions

	struct shared_obj_s sptr;
};

struct mem_s {

	/*General Properties****************************************************/

	bool init;		// mem is initialized, i.e. dimensions are set

	void* ptr;		// virtual pointer
	size_t len;		// length of memory range

	bool clear;		// memory should be cleared on allocation
	bool gpu;		// memory is on GPU
	bool host;		// pointer should be allocated with cudaMallocHost to allow fast transfers to and from GPU

	bool free;		// memory should be free'd on free of virtual pointer
	bool writeback;		// virtual pointer wraps memory that should contain the result on free of virtual pointer
	bool cfl;		// memory needs to be free'd with unmap_cfl instead of md_free

	struct vptr_hint_s* hint;	// hint for MPI distribution

	const char* backtrace;	// backtrace of allocation (for debugging)

	/*Virtual Range******************************************************/

	// A virtual pointer can map multiple virtual pointer in a continous
	// virtual address range. This is used for iterative algorithms only.

	int D;			// number of sub pointers
	void** sub_ptr;		// sub pointers

	/*Virtual md Array****************************************************/

	int N;			// rank of virtual md_array
	long* dims;		// dimensions of virtual md_array
	size_t size;		// size of elements
	unsigned long dflags;	// non-trivial dimensions (md_nontriv_dims(N, dims)) for fast access

	void** mem;		// memory blocks
	unsigned long mflags;	// flags indicating that memory is split along selected dimensions
	unsigned long lflags;	// flags indicating that memory should not be accessed with a single vptr_resolve
				// either because of blocking or because of MPI distribution which does not necessaryily correspoonds to blocks

	bool reduce;		// memory is used for MPI reduction: if in any md_function all involved pointers can be accessed
				// from multiple ranks, the operation is only performed on the lowest rank. By a subsequent reduction,
				// the result is then distributed to all ranks.
				// This property can be set and reset for any operation.

	long* rmpi_dims;	// reduced MPI dimensions, i.e. mpi dimesnions only
	long* rmpi_strs;	// strides corresponding to reduced MPI dimensions
				// both are used for fast detection if ptr can be accessed from a specific MPI rank
};


static struct mem_s* search(const void* ptr, bool remove);

static struct sigaction old_sa;

static void handler(int /*sig*/, siginfo_t *si, void*)
{
	struct mem_s* mem = search(si->si_addr, false);

	if (mem) {

		debug_vptr(DP_INFO, si->si_addr);
		error("Virtual pointer at %x not resolved!\n", si->si_addr);
	}

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
		vmap = tree_create(NULL);
}



static struct mem_s* search(const void* ptr, bool remove)
{
	if (NULL == vmap)
		return NULL;

	return tree_find(vmap, ptr, NULL, remove);
}

static void vptr_mem_set_dims(struct mem_s* x, int N, const long dims[N], size_t size, struct vptr_hint_s* hint)
{
	if (x->init) {

		if ((x->N != N) || !md_check_equal_dims(N, x->dims, dims, ~0UL) || (x->size != size)) {

			debug_print_dims(DP_INFO, N, dims);
			debug_vptr(DP_INFO, x->ptr);
		}

		assert(x->hint == hint);
		assert(x->N == N);
		assert(md_check_equal_dims(N, x->dims, dims, ~0UL));
		assert(x->size == size);

		return;
	}

	assert(x->len == (size_t)md_calc_size(N, dims) * size);

	x->init = true;

	x->mflags = 0;
	x->lflags = md_nontriv_dims(N, dims) & (hint ? hint->mpi_flags : 0UL);
	x->mem = NULL;

	x->free = true;

	x->N = N;
	x->dims = ARR_CLONE(long[N], dims);
	x->size = size;
	x->dflags = md_nontriv_dims(N, dims);

	x->writeback = false;

	x->hint = vptr_hint_ref(hint);

	if (NULL != hint && (0 != (hint->mpi_flags & x->dflags))) {

		long rmpi_dims[bitcount(hint->mpi_flags)];
		long rmpi_strs[bitcount(hint->mpi_flags)];

		long str = (long)size;
		for (int i = 0, j = 0; i < N; i++) {

			if (MD_IS_SET(hint->mpi_flags, i)) {

				rmpi_dims[j] = dims[i];
				rmpi_strs[j] = str;
				j++;
			}

			str *= dims[i];
		}

		x->rmpi_dims = ARR_CLONE(long[bitcount(hint->mpi_flags)], rmpi_dims);
		x->rmpi_strs = ARR_CLONE(long[bitcount(hint->mpi_flags)], rmpi_strs);
	}
}

static void vptr_mem_set_blocking(struct mem_s* x, unsigned long flags)
{
	assert(x->init);
	assert(NULL == x->mem);
	flags &= x->dflags;
	x->lflags |= flags;
	x->mflags |= flags;
}


static struct mem_s* vptr_create_mem(size_t len)
{
	void* ptr = mmap(NULL, len, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);

	if ((void*)-1 == ptr)
		error("mmap %s (len %lu)\n", strerror(errno), len);

	vptr_init();

#ifdef VPTR_TAGGING
	ptr = tree_tag_ptr(vmap, ptr);
#endif

	PTR_ALLOC(struct mem_s, x);

	x->ptr = ptr;
	x->len = len;
	x->init = false;
	x->D = 0;
	x->sub_ptr = NULL;
	x->reduce = false;
	x->clear = false;
	x->gpu = false;
	x->host = false;
	x->mem = NULL;
	x->backtrace = NULL;
	x->cfl = false;
	x->rmpi_dims = NULL;
	x->rmpi_strs = NULL;

#ifdef VPTR_DEBUG
#ifdef USE_DWARF
	x->backtrace = debug_good_backtrace_string(4);
#endif // USE_DWARF
#endif

	ptr_tree_insert(vmap, x, ptr, len);

	return PTR_PASS(x);
}

static struct mem_s* vptr_create(int N, const long dims[N], size_t size, struct vptr_hint_s* hint)
{
	size_t len = (size_t)md_calc_size(N, dims) * size;

	struct mem_s* x = vptr_create_mem(len);

	vptr_mem_set_dims(x, N, dims, size, hint);
	if (NULL != hint)
		vptr_mem_set_blocking(x, hint->mpi_flags);

	return x;
}

static const void* last_resolved = NULL;

static bool mpi_accessible_from_mem(const struct mem_s* mem, const void* ptr, int rank);

static void* vptr_resolve_int(const void* ptr, bool assert_rank, bool read)
{
	struct mem_s* mem = search(ptr, false);

	if (NULL == mem)
		return (void*)ptr;

	assert(mem->init);
	assert(NULL == mem->sub_ptr);

	if (!mpi_accessible_from_mem(mem, ptr, mpi_get_rank())) {

		if (assert_rank) {

			debug_vptr(DP_WARN, ptr);
			error("Trying to access %x from rank %d!\n", ptr, mpi_get_rank());
		}

		return NULL;
	}

	if (NULL == mem->mem) {

		assert(mem->clear || !read);

#pragma omp critical(bart_vmap)
		if (NULL == mem->mem) {

			long bdims[mem->N];
			md_select_dims(mem->N, mem->mflags, bdims, mem->dims);

			long num_blocks = md_calc_size(mem->N, bdims);

			mem->mem = *TYPE_ALLOC(void*[num_blocks]);

			for (long i = 0; i < num_blocks; i++)
				mem->mem[i] = NULL;
		}
	}

	long idx = md_reravel_index(mem->N, mem->mflags & mem->dflags, mem->dflags, mem->dims, (ptr - mem->ptr) / (long)mem->size);
	long offset = md_reravel_index(mem->N, ~mem->mflags & mem->dflags, mem->dflags, mem->dims, (ptr - mem->ptr) / (long)mem->size) * (long)mem->size;

	if (NULL == (mem->mem[idx])) {

		assert(mem->clear || !read);

#pragma omp critical(bart_vmap)
		if (NULL == (mem->mem[idx])) {

			long bdims[mem->N];
			md_select_dims(mem->N, ~mem->mflags, bdims, mem->dims);
			long block_size = md_calc_size(mem->N, bdims) * (long)mem->size;

			last_resolved = mem->ptr;

#ifdef USE_CUDA
			if (mem->gpu) {

				mem->mem[idx] = (mem->host ? cuda_malloc_host : cuda_malloc)(block_size);
				if (mem->clear)
					cuda_clear(block_size, mem->mem[idx]);

				vptr_update_gpu(block_size);
			} else
#endif
			{
				mem->mem[idx] = xmalloc((size_t)block_size);
				if (mem->clear)
					memset(mem->mem[idx], 0, (size_t)block_size);

				vptr_update_cpu(block_size);
			}
		}
	}

	return mem->mem[idx] + offset;
}


void* vptr_resolve(const void* ptr)
{
	return vptr_resolve_int(ptr, true, false);
}

void* vptr_resolve_read(const void* ptr)
{
	return vptr_resolve_int(ptr, true, false);
}

void* vptr_resolve_unchecked(const void* ptr)
{
	return vptr_resolve_int(ptr, false, false);
}

void* vptr_resolve_range(const void* ptr)
{
	if (NULL == ptr)
		return NULL;

	if (!is_vptr(ptr))
		return (void*)ptr;

	struct mem_s* mem = search(ptr, false);
	assert(NULL != mem);

	if (NULL == mem->sub_ptr)
		return (void*)ptr;

	long offset = ptr - mem->ptr;
	int i = 0;

	while (offset >= (long)vptr_get_len(mem->sub_ptr[i]))
		offset -= (long)vptr_get_len(mem->sub_ptr[i++]);

	return mem->sub_ptr[i] + offset;
}


bool vptr_overlap(const void *ptr1, const void *ptr2)
{
	auto vptr2 = search(ptr2, false);
	assert(NULL != vptr2);

	return ptr1 >= vptr2->ptr && (ptr1 < vptr2->ptr + vptr2->len);
}

bool vptr_is_same_type(const void *ptr1, const void *ptr2)
{
	auto vptr1 = search(ptr1, false);
	auto vptr2 = search(ptr2, false);

	assert(NULL != vptr2);
	assert(NULL != vptr2);

	return vptr_hint_same(vptr1->hint, vptr2->hint)
	    && md_check_equal_dims(MIN(vptr1->N, vptr2->N), vptr1->dims, vptr2->dims, ~0ul)
	    && (md_calc_size(vptr1->N, vptr1->dims) == md_calc_size(vptr2->N, vptr2->dims))
	    && (vptr1->size == vptr2->size)
	    && vptr1->gpu == vptr2->gpu
	    && vptr1->host == vptr2->host;
}




bool is_vptr(const void* ptr)
{
#ifdef VPTR_TAGGING
	return 0 != tree_get_tag(NULL, ptr);
#else
	return NULL != search(ptr, false);
#endif
}

bool is_vptr_gpu(const void* ptr)
{
	if (!is_vptr(ptr))
		return false;

	ptr = vptr_resolve_range(ptr);

	struct mem_s* mem = search(ptr, false);
	return mem && mem->gpu;
}

bool is_vptr_cpu(const void* ptr)
{
	if (!is_vptr(ptr))
		return false;

	ptr = vptr_resolve_range(ptr);

	struct mem_s* mem = search(ptr, false);
	return mem && !mem->gpu;
}

bool is_vptr_host(const void* ptr)
{
	if (!is_vptr(ptr))
		return false;

	ptr = vptr_resolve_range(ptr);

	struct mem_s* mem = search(ptr, false);
	return mem && mem->host;
}

bool vptr_is_init(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);
	assert(mem);
	return mem->init;
}

void vptr_free_mem(int N, const long dims[N], const long strs[N], const void *ptr, size_t size)
{
	struct mem_s* mem = search(ptr, false);

	if (mem->sub_ptr || mem->cfl || mem->writeback || !mem->free )
		return;

	assert(mem);
	assert(N == mem->N);
	assert(size == mem->size);

	if (!md_check_equal_dims(N, dims, mem->dims, md_nontriv_dims(N, dims))) {

		debug_print_dims(DP_INFO, N, dims);
		debug_vptr(DP_INFO, ptr);
	}

	assert(md_check_equal_dims(N, dims, mem->dims, md_nontriv_dims(N, dims)));
	assert(md_check_equal_dims(N, strs, MD_STRIDES(N, mem->dims, mem->size), md_nontriv_dims(N, dims)));

	if ((~mem->mflags & md_nontriv_dims(mem->N, mem->dims)) & ~md_nontriv_dims(N, dims))
		return;

	long pos[N];
	md_set_dims(N, pos, 0);
	md_unravel_index(N, pos, ~0UL, mem->dims, (ptr - mem->ptr) / (long)mem->size);

	assert(0 == md_ravel_index(N, pos, ~mem->mflags, mem->dims));

	unsigned long lflags = md_nontriv_dims(N, dims);
	for (int i = 0; i < N; i++)
		assert(!MD_IS_SET(lflags, i)  || 0 == pos[i]);

	if (NULL == mem->mem)
		return;

	long bdims[mem->N];
	md_select_dims(mem->N, ~mem->mflags, bdims, mem->dims);
	long block_size = md_calc_size(mem->N, bdims) * (long)mem->size;

	do {
		long idx = md_ravel_index(N, pos, mem->mflags, mem->dims);

		if (NULL != mem->mem[idx]) {

			if (mem->gpu)
				vptr_update_gpu(-block_size);
			else
				vptr_update_cpu(-block_size);
		}

		md_free(mem->mem[idx]);
		mem->mem[idx] = NULL;
	} while (md_next(N, pos, lflags, mem->dims));
}

bool vptr_is_writeback(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);
	assert(mem);
	return mem->writeback;
}


bool vptr_free(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);

	if (NULL == mem)
		return false;

	if (NULL != mem->sub_ptr) {

		if (mem->free)
			for (int i = 0; i < mem->D; i++)
				vptr_free(mem->sub_ptr[i]);

		vptr_hint_free(mem->hint);
		xfree(mem->sub_ptr);

		mem = search(ptr, true);

		munmap(tree_untag_ptr(vmap, mem->ptr), mem->len);

		if (NULL != mem->backtrace)
			xfree(mem->backtrace);

		xfree(mem);

		return true;
	}

	if (!mem->init) {

		mem = search(ptr, true);

		xfree(mem);
		return true;
	}

	if (mem->free && (NULL != mem->mem)) {

		if (mem->cfl) {

			// only for continuous allocations
			if (mem->writeback)
				md_copy(mem->N, mem->dims, mem->mem[0], mem->ptr, mem->size);

			unmap_cfl(mem->N, mem->dims, mem->mem[0]);

			vptr_update_cfl(-(long)mem->len);

		} else {

			long bdims[mem->N];
			md_select_dims(mem->N, mem->mflags, bdims, mem->dims);
			long num_blocks = md_calc_size(mem->N, bdims);

			long block_size = (long)mem->len / num_blocks;

			for (int i = 0; (i < num_blocks); i++) {

				if (NULL != mem->mem[i]) {

					if (mem->gpu)
						vptr_update_gpu(-block_size);
					else
						vptr_update_cpu(-block_size);
				}

				md_free(mem->mem[i]);
			}
		}

	} else {

		if (!mem->free) {

			if (mem->cfl) {

				vptr_update_cfl(-(long)mem->len);
			} else {

				if (mem->gpu)
					vptr_update_gpu(-(long)mem->len);
				else
					vptr_update_cpu(-(long)mem->len);
			}
		}

		// only for continuous allocations
		if (mem->writeback)
			md_copy(mem->N, mem->dims, mem->mem[0], mem->ptr, mem->size);
	}

	mem = search(ptr, true);

	munmap(tree_untag_ptr(vmap, mem->ptr), mem->len);

	if (NULL != mem->dims)
		xfree(mem->dims);

	if (NULL != mem->mem)
		xfree(mem->mem);

	vptr_hint_free(mem->hint);

	if (NULL != mem->backtrace)
		xfree(mem->backtrace);

	if (NULL != mem->rmpi_dims)
		xfree(mem->rmpi_dims);

	if (NULL != mem->rmpi_strs)
		xfree(mem->rmpi_strs);

	xfree(mem);

	return true;
}

void* vptr_alloc(int N, const long dims[N], size_t size, struct vptr_hint_s* hint)
{
	struct mem_s* mem = vptr_create(N, dims, size, hint);
	return mem->ptr;
}

void* vptr_alloc_size(size_t size)
{
	struct mem_s* mem = vptr_create_mem(size);
	return mem->ptr;
}

void vptr_set_dims(const void* x, int N, const long dims[N], size_t size, struct vptr_hint_s* hint)
{
	if (! is_vptr(x))
		return;

	x = vptr_resolve_range(x);

	struct mem_s* mem = search(x, false);
	assert(NULL != mem);

	bool init = mem->init;

	vptr_mem_set_dims(mem, N, dims, size, hint);

	if (!init)
		vptr_mem_set_blocking(mem, mem->lflags);
}

void vptr_set_clear(const void* x)
{
	struct mem_s* mem = search(x, false);
	assert(NULL != mem);
	mem->clear = true;
}

void vptr_unset_clear(const void* x)
{
	struct mem_s* mem = search(x, false);
	assert(NULL != mem);
	mem->clear = false;
}

bool vptr_is_set_clear(const void* x)
{
	struct mem_s* mem = search(x, false);
	assert(NULL != mem);
	return mem->clear;
}


bool vptr_is_mem_allocated(const void* x)
{
	struct mem_s* mem = search(x, false);
	assert(NULL != mem);
	return (NULL != mem->mem);
}

void vptr_set_loop_flags(const void* x, unsigned long flags)
{
	if (0 == flags)
		return;

	struct mem_s* mem = search(x, false);
	assert(NULL != mem);
	assert(NULL == mem->mem);

	vptr_mem_set_blocking(mem, flags);
}

void mpi_set_reduce(const void* ptr)
{
	if (!is_vptr(ptr))
		return;

	ptr = vptr_resolve_range(ptr);

	struct mem_s* mem = search(ptr, false);
	assert(NULL != mem);

	mem->reduce = true;
}

void mpi_unset_reduce(const void* ptr)
{
	if (!is_vptr(ptr))
		return;

	ptr = vptr_resolve_range(ptr);

	struct mem_s* mem = search(ptr, false);
	assert(NULL != mem);

	mem->reduce = false;
}

bool mpi_is_set_reduce(const void* ptr)
{
	if (!is_vptr(ptr))
		return false;

	struct mem_s* mem = search(ptr, false);
	assert(NULL != mem);

	return mem->reduce;
}

bool mpi_is_reduction(int N, const long dims[N], const long ostrs[N], const void* optr, size_t osize, const long istrs[N], const void* iptr, size_t isize)
{
	iptr = vptr_resolve_range(iptr);
	optr = vptr_resolve_range(optr);

	if (!is_mpi(iptr))
		return false;

	if (!is_mpi(optr))
		return true;

	struct mem_s* imem = search(iptr, false);
	struct mem_s* omem = search(optr, false);

	if ((NULL == imem->rmpi_dims) != (NULL == omem->rmpi_dims))
		return true;

	if (bitcount(imem->hint->mpi_flags) != bitcount(omem->hint->mpi_flags))
		return true;

	if (!md_check_equal_dims(bitcount(imem->hint->mpi_flags), imem->rmpi_dims, omem->rmpi_dims, md_nontriv_dims(bitcount(imem->hint->mpi_flags), imem->rmpi_dims)))
		return true;

	unsigned long flags = 0;
	flags |= vptr_block_loop_flags(N, dims, istrs, iptr, isize);
	flags |= vptr_block_loop_flags(N, dims, ostrs, optr, osize);

	if (0 == flags)
		return false;

	long imem_strs[imem->N];
	long omem_strs[omem->N];

	md_calc_strides(imem->N, imem_strs, imem->dims, imem->size);
	md_calc_strides(omem->N, omem_strs, omem->dims, omem->size);

	unsigned long imem_flags = imem->hint->mpi_flags & imem->dflags;
	unsigned long omem_flags = omem->hint->mpi_flags & omem->dflags;

	for (int i = 0; i < N; i++) {

		if (!MD_IS_SET(flags, i))
			continue;

		if (i >= imem->N || i >= omem->N)
			return true;

		if (istrs[i] != imem_strs[i] || ostrs[i] != omem_strs[i])
			return true;

		if (MD_IS_SET(imem_flags, i) != MD_IS_SET(omem_flags, i))
			return true;
	}

	return false;
}

void vptr_set_dims_sameplace(const void* x, const void* ref)
{
	struct mem_s* mem = search(x, false);
	struct mem_s* mem_ref = search(ref, false);
	assert(NULL != mem);
	assert(NULL != mem_ref);

	if (mem_ref->sub_ptr) {

		if (mem->init) {

			assert(mem->sub_ptr || 1 == mem->N);

			if (mem->sub_ptr) {

				return;
			} else {

				assert(NULL == mem->mem);

				xfree(mem->dims);
				mem->dims = NULL;

				vptr_hint_free(mem->hint);
			}
		}

		void* sub_ptr[mem_ref->D];
		for (int i = 0; i < mem_ref->D; i++) {

			struct mem_s* sub_mem = search(mem_ref->sub_ptr[i], false);
			sub_ptr[i] = vptr_alloc_sameplace(sub_mem->N, sub_mem->dims, sub_mem->size, mem_ref->sub_ptr[i]);

			if (mem->clear)
				vptr_set_clear(sub_ptr[i]);
		}

		if (!mem->init)
			mem->gpu = mem_ref->gpu;

		mem->init = true;
		mem->D = mem_ref->D;
		mem->sub_ptr = ARR_CLONE(void*[mem->D], sub_ptr);
		mem->free = true;
		mem->hint = vptr_hint_ref(mem_ref->hint);
	} else {

		bool init = mem->init;
		vptr_mem_set_dims(mem, mem_ref->N, mem_ref->dims, mem_ref->size, mem_ref->hint);
		if (!init)
			vptr_mem_set_blocking(mem, mem_ref->lflags);
	}
}


int vptr_get_N(const void* x)
{
	struct mem_s* mem = search(x, false);
	assert(NULL != mem);
	assert(mem->init);
	assert(!mem->sub_ptr);

	return mem->N;
}

void vptr_get_dims(const void* x, int N, long dims[N])
{
	struct mem_s* mem = search(x, false);
	assert(NULL != mem);
	assert(mem->init);
	assert(N == mem->N);
	assert(!mem->sub_ptr);

	md_copy_dims(N, dims, mem->dims);
}

size_t vptr_get_size(const void* x)
{
	struct mem_s* mem = search(x, false);
	assert(NULL != mem);
	assert(mem->init);
	assert(!mem->sub_ptr);

	return mem->size;
}

size_t vptr_get_len(const void* x)
{
	struct mem_s* mem = search(x, false);
	assert(NULL != mem);
	assert(mem->init);

	return mem->len;
}

long vptr_get_offset(const void* x)
{
	struct mem_s* mem = search(x, false);
	assert(NULL != mem);
	assert(mem->init);

	return x - mem->ptr;
}

struct vptr_hint_s* vptr_get_hint(const void* x)
{
	struct mem_s* mem = search(x, false);

	if (NULL == mem)
		return NULL;

	return mem->hint;
}

bool vptr_compat(const void* x, int N, const long dims[N], size_t size)
{
	if (!is_vptr(x))
		return true;

	struct mem_s* mem = search(x, false);

	if (mem->sub_ptr)
		return vptr_compat(vptr_resolve_range(x), N, dims, size);

	assert(mem->init);
	assert(!mem->sub_ptr);

	if (!md_check_equal_dims(MIN(N, mem->N), dims, mem->dims, ~0UL))
		return false;

	if (1 != md_calc_size(N - MIN(N, mem->N), dims + MIN(N, mem->N)))
		return false;

	if (1 != md_calc_size(mem->N - MIN(N, mem->N), mem->dims + MIN(N, mem->N)))
		return false;

	if (size != mem->size)
		return false;

	return true;
}

static long vptr_memcount(struct mem_s* mem)
{
	if (!mem->init)
		return 0;

	if (NULL == mem->mem)
		return 0;

	if (mem->sub_ptr)
		return 0;

	long bdims[mem->N];
	md_select_dims(mem->N, ~mem->mflags, bdims, mem->dims);
	long block_size = md_calc_size(mem->N, bdims) * (long)mem->size;
	long num_blocks = (long)mem->len / block_size;

	long ret = 0;

	for (long i = 0; i < num_blocks; i++)
		if (NULL != mem->mem[i])
			ret += block_size;

	return ret;
}


void debug_vptr(int dl, const void* x)
{
	if (-1 != debug_level && dl > debug_level)
		return;

	struct mem_s* mem = search(x, false);

	if (NULL == mem) {

		debug_printf(dl, "%p is not a virtual pointer.\n", x);
		return;
	}

	if (mem->init) {

		const char* p;

		if (mem->sub_ptr) {

			p = ptr_printf("Virtual pointer %p (on %s) wrapping other vptr", x, mem->gpu ? "GPU" : "CPU");
		} else {
			p = ptr_printf("Virtual pointer %p (on %s), backed by %.1fGB: N=%d, size=%zu, len=%zu block=%lu mpi=%lu dims=", x, mem->host ? "GPU+CPU" : mem->gpu ? "GPU" : mem->cfl ? "CFL" : "CPU", (float)vptr_memcount(mem) / (1024 * 1024 * 1024), mem->N, mem->size, mem->len, mem->mflags, mem->hint ? mem->hint->mpi_flags : 0);
			ptr_append_print_dims(&p, mem->N, mem->dims);
		}

		debug_printf(dl, "%s\n", p);
		xfree(p);
	} else {

		assert(NULL == mem->mem);
		debug_printf(dl, "Virtual pointer %p: len=%zu not initialized\n", x, mem->len);
	}

	if (NULL != mem->backtrace)
		debug_printf(DP_INFO, "Virtual pointer %p allocated at\n%s\n", x, mem->backtrace);
}

void print_vptr_cache(int dl)
{
	if (-1 != debug_level && dl > debug_level)
		return;

	if (NULL == vmap) {

		debug_printf(DP_INFO, "vptr cache empty!\n");
		return;
	}

	int N = tree_count(vmap);
	struct mem_s* mems[N];
	tree_to_array(vmap, N, (void*)mems);

	struct mem_s* mems_cpu[N];
	struct mem_s* mems_gpu[N];
	struct mem_s* mems_cfl[N];

	int num_cpu = 0;
	int num_gpu = 0;
	int num_cfl = 0;

	long memsize_cpu = 0;
	long memsize_gpu = 0;
	long memsize_cfl = 0;

	for (int i = 0; i < N; i++) {

		if (mems[i]->gpu && !mems[i]->host)
			mems_gpu[num_gpu++] = mems[i];
		else if (mems[i]->cfl)
			mems_cfl[num_cfl++] = mems[i];
		else
			mems_cpu[num_cpu++] = mems[i];

		if (mems[i]->gpu && !mems[i]->host)
			memsize_gpu += vptr_memcount(mems[i]);
		else if (mems[i]->cfl)
			memsize_cfl += vptr_memcount(mems[i]);
		else
			memsize_cpu += vptr_memcount(mems[i]);
	}

	debug_printf(dl, "%d virtual pointers on CPU using %.1fGB of memory:\n", num_cpu, (float)memsize_cpu / (1024 * 1024 * 1024));
	for (int i = 0; i < num_cpu; i++)
		debug_vptr(dl, mems_cpu[i]->ptr);

	debug_printf(dl, "%d virtual pointers on GPU using %.1fGB of memory:\n", num_gpu, (float)memsize_gpu / (1024 * 1024 * 1024));
	for (int i = 0; i < num_gpu; i++)
		debug_vptr(dl, mems_gpu[i]->ptr);

	debug_printf(dl, "%d virtual pointers on CFL using %.1fGB of memory:\n", num_cfl, (float)memsize_cfl / (1024 * 1024 * 1024));
	for (int i = 0; i < num_cfl; i++)
		debug_vptr(dl, mems_cfl[i]->ptr);

}

void print_vptr_stats(int dl)
{
	if (-1 != debug_level && dl > debug_level)
		return;

	if (0 == vptr_peak)
		return;

	debug_printf(dl, "Vptr use %.1f/ %.1f/ %.1f/ %.1f GB CPU/GPU/CFL/Total, peak was at %.1f/ %.1f/ %.1f/ %.1f GB CPU/GPU/CFL/Total\n",
			(float)vptr_cpu_size / (1024 * 1024 * 1024),
			(float)vptr_gpu_size / (1024 * 1024 * 1024),
			(float)vptr_cfl_size / (1024 * 1024 * 1024),
			(float)vptr_size / (1024 * 1024 * 1024),
			(float)vptr_cpu_peak / (1024 * 1024 * 1024),
			(float)vptr_gpu_peak / (1024 * 1024 * 1024),
			(float)vptr_cfl_peak / (1024 * 1024 * 1024),
			(float)vptr_peak / (1024 * 1024 * 1024));

	print_vptr_cache(dl);
}






// returns NULL if ref is not a virtual pointer.
void* vptr_alloc_sameplace(int N, const long dims[N], size_t size, const void* ref)
{
	struct mem_s* mem = search(ref, false);

	if (NULL == mem)
		return NULL;

	if (mem->sub_ptr) {

		size_t len = (size_t)md_calc_size(N, dims) * size;
		if (1 == N && len == mem->len)
			return vptr_alloc_same(ref);

		ref = vptr_resolve_range(ref);
		mem = search(ref, false);
	}

	auto ret = vptr_create(N, dims, size, mem->hint);
	ret->gpu = mem->gpu;
	ret->clear = mem->clear;

	return ret->ptr;
}

void* vptr_alloc_same(const void* ref)
{
	struct mem_s* mem = search(ref, false);

	if (NULL == mem)
		return NULL;

	void* ret = vptr_alloc_size(mem->len);
	vptr_set_dims_sameplace(ret, ref);
	vptr_set_gpu(ret, is_vptr_gpu(ref));

	return ret;
}


void vptr_set_gpu(const void* ptr, bool gpu)
{
	struct mem_s* mem = search(ptr, false);
	assert(NULL != mem);
	assert(NULL == mem->mem);

	mem->gpu = gpu;

	if (NULL != mem->sub_ptr)
		for (int i = 0; i < mem->D; i++)
			vptr_set_gpu(mem->sub_ptr[i], gpu);
}

void vptr_set_host(const void* ptr, bool host)
{
	struct mem_s* mem = search(ptr, false);
	assert(NULL != mem);
	assert(NULL == mem->mem);

	mem->host = host;

	if (NULL != mem->sub_ptr)
		for (int i = 0; i < mem->D; i++)
			vptr_set_host(mem->sub_ptr[i], host);
}

void* vptr_move_gpu(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);

	assert(mem && mem->ptr == ptr);
	assert(!mem->sub_ptr);

	auto ret = vptr_create(mem->N, mem->dims, mem->size, mem->hint);
	ret->gpu = true;

	md_copy(mem->N, mem->dims, ret->ptr, mem->ptr, mem->size);

	return ret->ptr;
}

void* vptr_move_cpu(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);

	assert(mem && mem->ptr == ptr);
	assert(!mem->sub_ptr);

	auto ret = vptr_create(mem->N, mem->dims, mem->size, mem->hint);
	ret->gpu = false;

	md_copy(mem->N, mem->dims, ret->ptr, mem->ptr, mem->size);

	return ret->ptr;
}

void* vptr_wrap(int N, const long dims[N], size_t size, const void* ptr, struct vptr_hint_s* hint, bool free, bool writeback)
{
	assert(!is_vptr(ptr));

	auto mem = vptr_create_mem((size_t)md_calc_size(N, dims) * size);
	vptr_mem_set_dims(mem, N, dims, size, hint);

	mem->free = free;
	mem->writeback = writeback;

	mem->mflags = 0;
	mem->mem = *TYPE_ALLOC(void*[1]);
	mem->mem[0] = (void*)ptr;

#ifdef USE_CUDA
	mem->gpu = cuda_ondevice(ptr);
#endif

	if (mem->gpu)
		vptr_update_gpu((long)mem->len);
	else
		vptr_update_cpu((long)mem->len);

	return mem->ptr;
}

void* vptr_wrap_sameplace(int N, const long dims[N], size_t size, const void* ptr, const void* ref, bool free, bool writeback)
{
	assert(!is_vptr(ptr));
	if (!is_vptr(ref))
		return (void*)ptr;

	struct mem_s* mem = search(ref, false);

	assert(NULL != mem);

	return vptr_wrap(N, dims, size, ptr, mem->hint, free, writeback);
}


void* vptr_wrap_cfl(int N, const long dims[N], size_t size, const void* ptr, struct vptr_hint_s* hint, bool free, bool writeback)
{
	if (NULL == hint)
		return (void*)ptr;

	assert(!is_vptr(ptr));

	auto mem = vptr_create_mem((size_t)md_calc_size(N, dims) * size);
	vptr_mem_set_dims(mem, N, dims, size, hint);

	mem->free = free;
	mem->writeback = writeback;
	mem->cfl = true;

	mem->mflags = 0;
	mem->mem = *TYPE_ALLOC(void*[1]);
	mem->mem[0] = (void*)ptr;

#ifdef USE_CUDA
	mem->gpu = cuda_ondevice(ptr);
#endif

	vptr_update_cfl((long)mem->len);

	return mem->ptr;
}

void* vptr_wrap_range(int D, void* ptr[D], bool free)
{
	assert(0 < D);
	size_t len = 0;

	for (int i = 0; i < D; i++) {

		struct mem_s* mem = search(ptr[i], false);
		assert(NULL != mem);
		assert(is_vptr_gpu(ptr[i]) == is_vptr_gpu(ptr[0]));
		len += mem->len;
	}

	auto mem = vptr_create_mem(len);
	mem->init = true;
	mem->D = D;
	mem->sub_ptr = ARR_CLONE(void*[D], ptr);
	mem->free = free;
	mem->hint = vptr_hint_ref(vptr_get_hint(ptr[0]));
	mem->gpu = is_vptr_gpu(ptr[0]);

	return mem->ptr;
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
void loop_access_dims(int N, unsigned long flags[N], const long adims[N], const long astrs[N], int D, const long mdims[D], long offset)
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
	    && 1 == md_calc_size(N - D, adims + D)) {

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

void check_vptr_valid_access(int N, const long dims[N], const long strs[N], const void* ptr, size_t size)
{
	if (!is_vptr(ptr))
		return;

	struct mem_s* mem = search(ptr, false);
	assert(NULL != mem);

	const void* minp = ptr;
	const void* maxp = ptr + size - 1;

	for (int i = 0; i < N; i++) {

		if (0 == strs[i])
			continue;

		if (0 > strs[i])
			minp += (dims[i] - 1) * strs[i];
		else
			maxp += (dims[i] - 1) * strs[i];
	}

	if ((maxp >= mem->ptr + mem->len) || minp < mem->ptr) {

		debug_vptr(DP_INFO, mem->ptr);
		debug_print_dims(DP_INFO, N, dims);
		debug_print_dims(DP_INFO, N, strs);
		error("Invalid vptr access at %p!\n", ptr);
	}

	assert(maxp < mem->ptr + mem->len);
	assert(minp >= mem->ptr);
}



/**
 * Returns which dimensions cannot be accessed using the same resolved pointer
 */
unsigned long vptr_block_loop_flags(int N, const long dims[N], const long strs[N], const void* ptr, size_t size)
{
	if (!is_vptr(ptr))
		return 0UL;

	struct mem_s* mem = search(ptr, false);

	if (NULL == mem)
		return 0UL;

	const void* minp = ptr;
	const void* maxp = ptr + size - 1;

	for (int i = 0; i < N; i++) {

		if (0 == strs[i])
			continue;

		if (0 > strs[i])
			minp += (dims[i] - 1) * strs[i];
		else
			maxp += (dims[i] - 1) * strs[i];
	}

	assert(maxp < mem->ptr + mem->len);
	assert(minp >= mem->ptr);


	assert(!mem->sub_ptr);

	unsigned long lflags = mem->lflags;

	if (0 == lflags)
		return 0UL;

	long mdims[mem->N + 1];
	long tdims[N + 1];
	long tstrs[N + 1];

	tdims[0] = (long)size;
	tstrs[0] = 1;
	mdims[0] = (long)mem->size;

	md_copy_dims(mem->N, mdims + 1, mem->dims);
	md_select_dims(N, md_nontriv_strides(N, strs), tdims + 1, dims);
	md_select_strides(N, md_nontriv_dims(N, tdims + 1), tstrs + 1, strs);

	unsigned long flags[N + 1];
	memset(flags, 0, sizeof flags);	// GCC ANALYZER
	loop_access_dims(N + 1, flags, tdims, tstrs, mem->N + 1, mdims, ptr - mem->ptr);

	unsigned long ret_flags = 0;

	for (int i = 1; i < N + 1; i++)
		if (0 != (lflags & (flags[i] / 2)))
			ret_flags |= MD_BIT(i);

	return ret_flags /= 2;
}

void vptr_continous_strs(int N, const void* ptr, unsigned long lflags, long nstrs[N], const long ostrs[N])
{
	struct mem_s* mem = search(ptr, false);

	if (NULL == mem) {

		for (int i = 0; i < N; i++)
			nstrs[i] = ostrs[i];

		return;
	}

	assert(!mem->sub_ptr);

	int Nm = mem->N + 1;
	long mdims[Nm];
	mdims[0] = (long)mem->size;
	md_copy_dims(mem->N, mdims + 1, mem->dims);

	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(lflags, i) || (0 == ostrs[i])) {

			nstrs[i] = ostrs[i];
			continue;
		}

		nstrs[i] = md_reravel_index(Nm, ~(2 * mem->mflags) & (2 * mem->dflags + 1), (2 * mem->dflags + 1), mdims, labs(ostrs[i])) * (ostrs[i] < 0 ? -1 : 1);
	}
}


static void vptr_assert_sameplace_mem(int N, const void* ptr[N], const struct mem_s* mem_arr[N])
{
	struct vptr_hint_s* hint_ref = NULL;

	const struct mem_s* mem_ref = mem_arr[0];

	assert(!mem_ref || !mem_ref->sub_ptr);

	if (mem_ref)
		hint_ref = mem_ref->hint;

	for (int i = 1; i < N; i++) {

		const struct mem_s* mem = mem_arr[i];

		if ((NULL == mem_ref) && (NULL != mem))
			error("Incopatible pointer: vptr(%x) at %d and normal pointer(%x) at 0!\n", mem->ptr, i, ptr[0]);

		if ((NULL != mem_ref) && (NULL == mem))
			error("Incopatible pointer: vptr(%x) at 0 and normal pointer(%x) at %d!\n", mem_ref->ptr, ptr[i], i);

		if (NULL == mem_ref)
			continue;

		if (!vptr_hint_compat(hint_ref, mem->hint)) {

			debug_print_dims(DP_INFO, hint_ref->N, hint_ref->dims);
			debug_print_dims(DP_INFO, mem->hint->N, mem->hint->dims);

			error("Incopatible MPI dist rule!\n");
		}
	}
}

void vptr_assert_sameplace(int N, void* nptr[N])
{
	const struct mem_s* mem[N];

	for (int i = 0; i < N; i++)
		mem[i] = search(nptr[i], false);

	vptr_assert_sameplace_mem(N, (const void**)nptr, mem);
}

/**
 * Returns true if ptr is distributed along multiple processes
 *
 * @param ptr pointer to check
 */
bool is_mpi(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);

	assert(!mem || !mem->sub_ptr);

	return mem && mem->hint && ((mem->hint->mpi_flags & mem->dflags) != 0UL);
}

static int hint_get_rank(int N, const long pos[N], struct vptr_hint_s* hint);


/**
 * Calculates rank of given pointer if it is distributed returns -1 else
 *
 * @param ptr pointer to check
 */
int mpi_ptr_get_rank(const void* ptr)
{
	struct mem_s* mem = search(ptr, false);

	assert(NULL != mem);
	assert(!mem->sub_ptr);

	auto h = mem->hint;
	int N = bitcount(h->mpi_flags);
	long pos[N];
	md_unravel_index2(N, pos, ~(0UL), mem->rmpi_dims, mem->rmpi_strs, ptr - mem->ptr);

	return hint_get_rank(N, pos, mem->hint);
}

static bool mpi_accessible_from_mem(const struct mem_s* mem, const void* ptr, int rank)
{
	if ((NULL == mem) || (NULL == mem->hint) || (0 == (mem->hint->mpi_flags & mem->dflags)))
		return true;

	assert(!mem->sub_ptr);

	struct vptr_hint_s* h = mem->hint;
	int N = bitcount(h->mpi_flags);
	long pos[N];
	md_unravel_index2(N, pos, ~(0UL), mem->rmpi_dims, mem->rmpi_strs, ptr - mem->ptr);
	unsigned long loop_flags = ~md_nontriv_dims(N, mem->rmpi_dims);

	do {
		if (hint_get_rank(N, pos, h) == rank)
			return true;

	} while (md_next(N, h->rdims, loop_flags, pos));

	return false;
}


bool mpi_accessible_from(const void* ptr, int rank)
{
	auto mem = search(ptr, false);

	return mpi_accessible_from_mem(mem, ptr, rank);
}


bool mpi_accessible(const void* ptr)
{
	if (!is_vptr(ptr))
		return true;

	return mpi_accessible_from(ptr, mpi_get_rank());
}


static bool mpi_accessible_from_mult(int N, const struct mem_s* mem[N], const void* ptr[N], int rank)
{

	for (int i = 0; i < N; i++)
		if (!mpi_accessible_from_mem(mem[i], ptr[i], rank))
			return false;

	return true;
}

bool mpi_accessible_mult(int N, const void* ptr[N])
{
	bool check = false;

	for (int i = 0; i < N; i++)
		if (is_vptr(ptr[i]))
			check = true;

	if (!check)
		return true;

	const struct mem_s* mem[N];

	for (int i = 0; i < N; i++) {

		mem[i] = search(ptr[i], false);
		assert(NULL != mem[i]);
	}

	vptr_assert_sameplace_mem(N, ptr, mem);

	if (!mpi_accessible_from_mult(N, mem, ptr, mpi_get_rank()))
		return false;

	bool reduce = false;

	for (int i = 0; i < N; i++)
		if (mem[i]->reduce)
			reduce = true;

	for (int i = 0; i < mpi_get_rank(); i++)
		if (reduce && mpi_accessible_from_mult(N, mem, ptr, i))
			return false;

	return true;
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


static int hint_get_rank(int N, const long pos[N], struct vptr_hint_s* hint)
{
	long offset = 0;
	long stride = 1;

	assert(N == bitcount(hint->mpi_flags));

	for (int i = 0; i < N; i++) {

		offset += stride * pos[i];
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

	if (NULL != hint->dims)
		xfree(hint->dims);

	if (NULL != hint->rank)
		xfree(hint->rank);

	if (NULL != hint->rdims)
		xfree(hint->rdims);

	xfree(hint);
}

struct vptr_hint_s* vptr_hint_create(unsigned long mpi_flags, int N, const long dims[N], unsigned long delayed_flags)
{
	PTR_ALLOC(struct vptr_hint_s, x);

	x->loop_flags = delayed_flags;

	long mdims[N];
	md_select_dims(N, mpi_flags, mdims, dims);

	if (1 < md_calc_size(N, mdims) && mpi_get_num_procs() > md_calc_size(N, mdims))
		error("BART started with more MPI processes than MPI dimensions. Reduce number of MPI processes to %d\n", md_calc_size(N, mdims));

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

	if (0 == mpi_flags) {

		xfree(PTR_PASS(x));
		return hint_delayed_create(delayed_flags);
	}

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

	long rdims[bitcount(mpi_flags)];
	for (int i = 0, j = 0; i < N; i++)
		if (MD_IS_SET(mpi_flags, i))
			rdims[j++] = mdims[i];

	x->rdims = ARR_CLONE(long[bitcount(mpi_flags)], rdims);

	shared_obj_init(&x->sptr, vptr_hint_del);

	return PTR_PASS(x);
}


struct vptr_hint_s* hint_mpi_create(unsigned long mpi_flags, int N, const long dims[N])
{
	return vptr_hint_create(mpi_flags, N, dims, 0);
}

struct vptr_hint_s* hint_delayed_create(unsigned long delayed_flags)
{
	PTR_ALLOC(struct vptr_hint_s, x);

	x->loop_flags = delayed_flags;
	x->N = 0;
	x->dims = NULL;
	x->rank = NULL;
	x->mpi_flags = 0;
	x->rdims = NULL;

	shared_obj_init(&x->sptr, vptr_hint_del);

	return PTR_PASS(x);
}

bool vptr_hint_compat(const struct vptr_hint_s* hint1, const struct vptr_hint_s* hint2)
{
	if (hint1 == hint2)
		return true;

	if (NULL == hint1)
		return 0 == hint2->N;

	if (NULL == hint2)
		return 0 == hint1->N;

	if (hint1->N != hint2->N)
		return false;

	if (!md_check_equal_dims(hint1->N, hint1->dims, hint2->dims, ~0UL))
		return false;

	if (hint1->mpi_flags != hint2->mpi_flags)
		return false;

	return true;
}

bool vptr_hint_same(const struct vptr_hint_s* hint1, const struct vptr_hint_s* hint2)
{
	unsigned long lflags1 = (NULL == hint1) ? 0 : hint1->loop_flags;
	unsigned long lflags2 = (NULL == hint2) ? 0 : hint2->loop_flags;

	return (lflags1 == lflags2) && vptr_hint_compat(hint1, hint2);
}

void vptr_hint_free(struct vptr_hint_s* hint)
{
	if (NULL == hint)
		return;

	shared_obj_destroy(&hint->sptr);
}

unsigned long vptr_delayed_loop_flags(const void* ptr)
{
	struct vptr_hint_s* hint = vptr_get_hint(ptr);
	return hint ? hint->loop_flags : 0;
}


