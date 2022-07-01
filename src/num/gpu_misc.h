

#include <cuda_runtime_api.h>

extern dim3 getBlockSize3(const long dims[3], const void* func);
extern dim3 getGridSize3(const long dims[3], const void* func);

extern dim3 getBlockSize(long dims, const void* func);
extern dim3 getGridSize(long dims, const void* func);

extern void print_dim3(dim3 dims);

