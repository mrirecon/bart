#include "misc/cppwrap.h"


#include <stdint.h>
#include <stddef.h>



extern void init_mpi(int* argc, char*** argv);
extern void deinit_mpi(void);
extern void mpi_signoff_proc(_Bool signof);

extern int mpi_get_rank(void);
extern int mpi_get_num_procs(void);
extern _Bool mpi_is_main_proc(void);

extern void mpi_sync(void);

extern void mpi_sync_val(void* pval, long size);

extern void mpi_scatter_batch(void* dst, long count, const void* src, size_t type_size);
extern void mpi_gather_batch(void* dst, long count, const void* src, size_t type_size);

#include "misc/cppwrap.h"
