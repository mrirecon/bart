
#ifdef __cplusplus
#error This file does not support C++
#endif

#include <stdlib.h>

extern int simplify_dims(int D, int N, long dims[N], long (*strs[D])[N]);
extern int optimize_dims(int D, int N, long dims[N], long (*strs[D])[N]);
extern int optimize_dims_gpu(int D, int N, long dims[N], long (*strs[D])[N]);
extern int min_blockdim(int D, int N, const long dims[N], long (*strs[D])[N], size_t size[D]);
extern void compute_permutation(int N, int ord[N], const long strs[N]);
extern unsigned long dims_parallel(int D, unsigned long io, int N, const long dims[N], long (*strs[D])[N], size_t size[D]);
extern unsigned long parallelizable(int D, unsigned int io, int N, const long dims[N], const long (*strs[D])[N], size_t size[D]);

struct vec_ops;

struct nary_opt_data_s {

	long size;
	const struct vec_ops* ops;
};



typedef void CLOSURE_TYPE(md_nary_opt_fun_t)(struct nary_opt_data_s* data, void* ptr[]);

extern void optimized_nop(int N, unsigned long io, int D, const long dim[D], const long (*nstr[N])[D?:1], void* const nptr[N], size_t sizes[N], md_nary_opt_fun_t too);

