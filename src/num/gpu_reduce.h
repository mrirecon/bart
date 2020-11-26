#ifdef __cplusplus
extern "C" {
#endif

extern void cuda_reduce_zadd_inner(long dim_reduce, long dim_batch, _Complex float* dst, const _Complex float* src);
extern void cuda_reduce_zadd_outer(long dim_reduce, long dim_batch, _Complex float* dst, const _Complex float* src);

extern void cuda_reduce_zmax_inner(long dim_reduce, long dim_batch, _Complex float* dst, const _Complex float* src);
extern void cuda_reduce_zmax_outer(long dim_reduce, long dim_batch, _Complex float* dst, const _Complex float* src);

extern void cuda_reduce_add_inner(long dim_reduce, long dim_batch, float* dst, const float* src);
extern void cuda_reduce_add_outer(long dim_reduce, long dim_batch, float* dst, const float* src);

#ifdef __cplusplus
}
#endif
