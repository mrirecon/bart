
#include <stddef.h>

#include "misc/cppwrap.h"

struct vptr_hint_s;
extern struct vptr_hint_s* vptr_hint_ref(struct vptr_hint_s* hint);
extern void vptr_hint_free(struct vptr_hint_s* hint);

extern void* vptr_alloc(int N, const long dims[__VLA(N)], size_t size, struct vptr_hint_s* hint);
extern void* vptr_alloc_sameplace(int N, const long dims[__VLA(N)], size_t size, const void* ref);
extern void* vptr_alloc_same(const void* ref);
extern void* vptr_alloc_size(size_t size);
extern void vptr_set_clear(const void* x);
extern void vptr_unset_clear(const void* x);
extern _Bool vptr_is_set_clear(const void* x);
extern void vptr_set_dims(const void* x, int N, const long dims[__VLA(N)], size_t size, struct vptr_hint_s* hint);
extern void vptr_set_dims_sameplace(const void* x, const void* ref);
extern void vptr_set_loop_flags(const void* x, unsigned long flags);
extern _Bool vptr_is_mem_allocated(const void* x);
extern void* vptr_wrap(int N, const long dims[__VLA(N)], size_t size, const void* ptr, struct vptr_hint_s* hint, _Bool free, _Bool writeback);
extern void* vptr_wrap_sameplace(int N, const long dims[__VLA(N)], size_t size, const void* ptr, const void* ref, _Bool free, _Bool writeback);
extern void* vptr_wrap_cfl(int N, const long dims[__VLA(N)], size_t size, const void* ptr, struct vptr_hint_s* hint, _Bool free, _Bool writeback);
extern void* vptr_wrap_range(int D, void* ptr[__VLA(D)], _Bool free);
extern _Bool vptr_is_writeback(const void* ptr);

extern void check_vptr_valid_access(int N, const long dims[N], const long strs[N], const void* ptr, size_t size);
extern void debug_vptr(int dl, const void* x);
extern void print_vptr_cache(int dl);
extern void print_vptr_stats(int dl);

extern _Bool vptr_is_init(const void* ptr);
extern int vptr_get_N(const void* x);
extern void vptr_get_dims(const void* x, int N, long dims[__VLA(N)]);
extern size_t vptr_get_size(const void* x);
extern size_t vptr_get_len(const void* x);
extern long vptr_get_offset(const void* x);
extern struct vptr_hint_s* vptr_get_hint(const void* x);
extern _Bool vptr_compat(const void* x, int N, const long dims[__VLA(N)], size_t size);

extern _Bool vptr_free(const void* ptr);
extern void vptr_free_mem(int N, const long dims[__VLA(N)], const long strs[__VLA(N)], const void *ptr, size_t size);
extern void* vptr_resolve(const void* ptr);
extern void* vptr_resolve_read(const void* ptr);
extern void* vptr_resolve_unchecked(const void* ptr);
extern void* vptr_resolve_range(const void* ptr);

extern _Bool vptr_overlap(const void* ptr1, const void* ptr2);
extern _Bool vptr_is_same_type(const void* ptr1, const void* ptr2);

extern _Bool is_vptr(const void* ptr);
extern _Bool is_vptr_cpu(const void* ptr);
extern _Bool is_vptr_gpu(const void* ptr);
extern _Bool is_vptr_host(const void* ptr);

extern void* vptr_move_cpu(const void* ptr);
extern void* vptr_move_gpu(const void* ptr);
extern void vptr_set_gpu(const void* ptr, _Bool gpu);
extern void vptr_set_host(const void* ptr, _Bool host);

extern void loop_access_dims(int N, unsigned long flags[__VLA(N)], const long adims[__VLA(N)], const long astrs[__VLA(N)], int D, const long mdims[__VLA(D)], long offset);
extern unsigned long vptr_block_loop_flags(int N, const long dims[__VLA(N)], const long strs[__VLA(N)], const void* ptr, size_t size);
extern void vptr_continous_strs(int N, const void* ptr, unsigned long lflags, long nstrs[__VLA(N)], const long ostrs[__VLA(N)]);
extern void vptr_assert_sameplace(int N, void* nptr[__VLA(N)]);

extern void mpi_set_reduce(const void* ptr);
extern void mpi_unset_reduce(const void* ptr);
extern _Bool mpi_is_set_reduce(const void* ptr);
extern _Bool mpi_is_reduction(int N, const long dims[__VLA(N)], const long ostrs[__VLA(N)], const void* optr, size_t osize, const long istrs[__VLA(N)], const void* iptr, size_t isize);

extern _Bool is_mpi(const void* ptr);
extern struct vptr_hint_s* hint_mpi_create(unsigned long mpi_flags, int N, const long dims[__VLA(N)]);
extern _Bool mpi_accessible_from(const void* ptr, int rank);
extern _Bool mpi_accessible(const void* ptr);
extern _Bool mpi_accessible_mult(int N, const void* ptr[__VLA(N)]);
extern int mpi_ptr_get_rank(const void* ptr);

extern struct vptr_hint_s* hint_delayed_create(unsigned long delayed_flags);
extern unsigned long vptr_delayed_loop_flags(const void* ptr);

extern unsigned long mpi_parallel_flags(int N, const long dims[__VLA(N)], const long strs[__VLA(N)], size_t size, const void* ptr);

extern struct vptr_hint_s* vptr_hint_create(unsigned long mpi_flags, int N, const long dims[__VLA(N)], unsigned long delayed_flags);
extern _Bool vptr_hint_compat(const struct vptr_hint_s* hint1, const struct vptr_hint_s* hint2);
extern _Bool vptr_hint_same(const struct vptr_hint_s* hint1, const struct vptr_hint_s* hint2);

#include "misc/cppwrap.h"

