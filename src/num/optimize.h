
#ifdef __cplusplus
#error This file does not support C++
#endif

#include <stdlib.h>

extern void merge_dims(int D, int N, long dims[N], long (*ostrs[D])[N]);
extern int remove_empty_dims(int D, int N, long dims[N], long (*ostrs[D])[N]);

extern int simplify_dims(int D, int N, long dims[N], long (*strs[D])[N]);
extern int optimize_dims(int D, int N, long dims[N], long (*strs[D])[N]);
extern int optimize_dims_gpu(int D, int N, long dims[N], long (*strs[D])[N]);
extern int min_blockdim(int D, int N, const long dims[N], long (*strs[D])[N], size_t size[D]);
extern unsigned long dims_parallel(int D, unsigned int io, int N, const long dims[N], long (*strs[D])[N], size_t size[D]);


struct vec_ops;

struct nary_opt_data_s {

	long size;
	const struct vec_ops* ops;
};



typedef void CLOSURE_TYPE(md_nary_opt_fun_t)(struct nary_opt_data_s* data, void* ptr[]);

extern void optimized_nop(int N, unsigned int io, int D, const long dim[D], const long (*nstr[N])[D?:1], void* const nptr[N], size_t sizes[N], md_nary_opt_fun_t too);

