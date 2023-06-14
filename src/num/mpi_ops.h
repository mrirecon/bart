#ifdef USE_MPI
#include <stdint.h>
#include <limits.h>
#include <mpi.h>

#if SIZE_MAX == UCHAR_MAX
   #define BART_MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
   #define BART_MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
   #define BART_MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
   #define BART_MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
   #define BART_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
   #error "Could not define BART_MPI_SIZE_T"
#endif

#define COMM_CFL_BCAST 1


extern MPI_Comm mpi_get_comm(void);
extern void mpi_split_comm(MPI_Comm base_comm, int tag);
extern void mpi_comm_subset_activate(void);

#endif

extern void init_mpi(int* argc, char*** argv);
extern void deinit_mpi(void);

extern int mpi_get_rank(void);
extern int mpi_get_num_procs(void);
extern _Bool mpi_is_main_proc(void);
