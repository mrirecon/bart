
#include <stddef.h>

#include "misc/cppwrap.h"

struct vptr_shape_s {

	int N;
	const long* dims;
	size_t size;
};

struct vptr_hint_s;
extern struct vptr_hint_s* vptr_hint_ref(struct vptr_hint_s* hint);
extern void vptr_hint_free(struct vptr_hint_s* hint);
extern struct vptr_hint_s* vptr_get_hint(const void* ptr);

extern void* vptr_alloc(int N, const long dims[N], size_t size, struct vptr_hint_s* hint);
extern void* vptr_alloc_sameplace(int N, const long dims[__VLA(N)], size_t size, const void* ref);
extern void* vptr_wrap(int N, const long dims[__VLA(N)], size_t size, const void* ptr, struct vptr_hint_s* hint, _Bool free, _Bool writeback);
extern void* vptr_wrap_sameplace(int N, const long dims[__VLA(N)], size_t size, const void* ptr, const void* ref, _Bool free, _Bool writeback);
extern void* vptr_wrap_cfl(int N, const long dims[N], size_t size, const void* ptr, struct vptr_hint_s* hint, _Bool free, _Bool writeback);

extern _Bool vptr_free(const void* ptr);
extern void* vptr_resolve(const void* ptr);
extern void* vptr_resolve_unchecked(const void* ptr);

extern const struct vptr_shape_s* vptr_get_shape(const void* ptr);
extern long vptr_get_offset(const void* ptr);

extern _Bool is_vptr(const void* ptr);
extern _Bool is_vptr_cpu(const void* ptr);
extern _Bool is_vptr_gpu(const void* ptr);

extern _Bool is_mpi(const void* ptr);

extern void* vptr_move_cpu(const void* ptr);
extern void* vptr_move_gpu(const void* ptr);

extern unsigned long vptr_block_loop_flags(int N, const long dims[__VLA(N)], const long strs[__VLA(N)], const void* ptr, size_t size, _Bool contiguous_strs);
extern void vptr_contiguous_strs(int N, const void* ptr, unsigned long lflags, long nstrs[__VLA(N)], const long ostrs[__VLA(N)]);

extern void vptr_assert_sameplace(int N, void* nptr[__VLA(N)]);


extern struct vptr_hint_s* hint_mpi_create(unsigned long mpi_flags, int N, const long dims[__VLA(N)]);
extern _Bool mpi_accessible_from(const void* ptr, int rank);
extern _Bool mpi_accessible(const void* ptr);
extern _Bool mpi_accessible_mult(int N, const void* ptr[__VLA(N)]);
extern int mpi_ptr_get_rank(const void* ptr);

extern void mpi_set_reduction_buffer(const void* ptr);
extern void mpi_unset_reduction_buffer(const void* ptr);
extern _Bool mpi_is_reduction(int N, const long dims[__VLA(N)], const long ostrs[__VLA(N)], const void* optr, size_t osize, const long istrs[__VLA(N)], const void* iptr, size_t isize);




#include "misc/cppwrap.h"

