#ifdef __cplusplus
extern "C" {
#endif

extern void cuda_copy_ND(int D, const long dims[], const long ostrs[], void* dst, const long istrs[], const void* src, unsigned long size);

#ifdef __cplusplus
}
#endif
