
#include "misc/cppwrap.h"

extern long upper_triag_idx(long i, long j);

extern _Complex float* hermite_to_uppertriag(int dim1, int dim2, int dimt, int N, long out_dims[__VLA(N)], const long* dims, const _Complex float* src);
extern _Complex float* uppertriag_to_hermite(int dim1, int dim2, int dimt, int N, long out_dims[__VLA(N)], const long* dims, const _Complex float* src);

extern float* symmetric_to_uppertriag(int dim1, int dim2, int dimt, int N, long out_dims[__VLA(N)], const long* dims, const float* src);
extern float* uppertriag_to_symmetric(int dim1, int dim2, int dimt, int N, long out_dims[__VLA(N)], const long* dims, const float* src);

extern void md_tenmul_upper_triag2(int dim1, int dim2, int N, const long dims[__VLA(N)], const long ostrs[__VLA(N)], float* dst, const long istrs[__VLA(N)], const float* src, const long mdims[__VLA(N)], const long mstrs[__VLA(N)], const float* mat);
extern void md_tenmul_upper_triag(int dim1, int dim2, int N, const long odims[__VLA(N)], float* dst, const long idims[__VLA(N)], const float* src, const long mdims[__VLA(N)], const float* mat);

extern void md_ztenmul_upper_triag2(int dim1, int dim2, int N, const long dims[__VLA(N)], const long ostrs[__VLA(N)], _Complex float* dst, const long istrs[__VLA(N)], const _Complex float* src, const long mdims[__VLA(N)], const long mstrs[__VLA(N)], const _Complex float* mat);
extern void md_ztenmul_upper_triag(int dim1, int dim2, int N, const long odims[__VLA(N)], _Complex float* dst, const long idims[__VLA(N)], const _Complex float* src, const long mdims[__VLA(N)], const _Complex float* mat);

#include "misc/cppwrap.h"
