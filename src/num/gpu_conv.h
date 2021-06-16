
#ifdef __cplusplus
extern "C" {
#endif

extern void cuda_im2col(_Complex float* dst, const _Complex float* src, const long odims[5], const long idims[5], const long kdims[5], const long dilation[5], const long strides[5]);
extern void cuda_im2col_transp(_Complex float* dst, const _Complex float* src, const long odims[5], const long idims[5], const long kdims[5], const long dilation[5], const long strides[5]);

#ifdef __cplusplus
}
#endif
