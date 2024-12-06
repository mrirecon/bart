#ifdef __cplusplus
extern "C" {
#endif

extern void cuda_copy_ND(int D, const long dims[], const long ostrs[], void* dst, const long istrs[], const void* src, unsigned long size);

extern void cuda_decompress(long stride, long N, long dcstrs, void* dst, long istrs, const long* index, const void* src, unsigned long size);
extern void cuda_compress(long stride, long N, void* dst, long istrs, const long* index, long dcstrs, const void* src, unsigned long size);


#ifdef __cplusplus
}
#endif
