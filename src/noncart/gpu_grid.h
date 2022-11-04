#include "misc/cppwrap.h"

extern void cuda_apply_linphases_3D(int N, const long img_dims[__VLA(N)], const float shifts[3], _Complex float* dst, const _Complex float* src, _Bool conj, _Bool fmac, float scale);


#include "misc/cppwrap.h"


