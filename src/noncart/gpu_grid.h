#include "misc/cppwrap.h"

extern void cuda_apply_linphases_3D(int N, const long img_dims[__VLA(N)], const float shifts[3], _Complex float* dst, const _Complex float* src, _Bool conj, _Bool fmac, float scale);
extern void cuda_apply_rolloff_correction(float os, float width, float beta, int N, const long dims[__VLA(N)], _Complex float* dst, const _Complex float* src);

#include "misc/cppwrap.h"


