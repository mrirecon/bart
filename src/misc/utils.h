

#include "misc/cppwrap.h"

extern void normalizel1(int N, unsigned long flags, const long dims[__VLA(N)], _Complex float* maps);
extern void normalize(int N, unsigned long flags, const long dims[__VLA(N)], _Complex float* maps);
extern _Complex float* compute_mask(int N, const long msk_dims[__VLA(N)], const float restrict_fov[__VLA(N)]);
extern void apply_mask(int N, const long dims[__VLA(N)], _Complex float* x, const float restrict_fov[__VLA(N)]);
extern void fixphase(int D, const long dims[__VLA(D)], int dim, _Complex float* out, const _Complex float* in);
extern void fixphase2(int D, const long dims[__VLA(D)], int dim, const _Complex float rot[__VLA(dims[dim])], _Complex float* out, const _Complex float* in);

#include "misc/cppwrap.h"


