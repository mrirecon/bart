
#ifdef __cplusplus
extern "C" {
#endif

extern void cuda_xpay_bat(long Bi, long N, long Bo, const float* beta, float* a, const float* x);
extern void cuda_axpy_bat(long Bi, long N, long Bo, float* a, const float* alpha, const float* x);
extern void cuda_dot_bat(long Bi, long N, long Bo, float* dst, const float* x, const float* y);

#ifdef __cplusplus
}
#endif
