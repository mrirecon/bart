

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


extern void normalize(int N, unsigned int flags, const long dims[__VLA(N)], _Complex float* maps);
extern _Complex float* compute_mask(unsigned int N, const long msk_dims[__VLA(N)], const float restrict_fov[__VLA(N)]);
extern void apply_mask(unsigned int N, const long dims[__VLA(N)], _Complex float* x, const float restrict_fov[__VLA(N)]);
extern void linear_phase(unsigned int N, const long dims[__VLA(N)], const float pos[__VLA(N)], _Complex float* out);

#ifdef __cplusplus
}
#endif


