

#ifdef __cplusplus
extern "C" {
#ifndef __VLA
#define __VLA(x) 
#endif
#else
#ifndef __VLA
#define __VLA(x) static x
#endif
#endif

extern void normalizel1(int N, unsigned int flags, const long dims[N], complex float* maps);
extern void normalize(int N, unsigned int flags, const long dims[__VLA(N)], _Complex float* maps);
extern _Complex float* compute_mask(unsigned int N, const long msk_dims[__VLA(N)], const float restrict_fov[__VLA(N)]);
extern void apply_mask(unsigned int N, const long dims[__VLA(N)], _Complex float* x, const float restrict_fov[__VLA(N)]);
extern void fixphase(unsigned int D, const long dims[__VLA(D)], unsigned int dim, _Complex float* out, const _Complex float* in);
extern void fixphase2(unsigned int D, const long dims[__VLA(D)], unsigned int dim, const _Complex float rot[dims[dim]], _Complex float* out, const _Complex float* in);

#ifdef __cplusplus
}
#endif


