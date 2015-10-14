


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

#include <complex.h>

extern float median_float(int N, float ar[N]);
extern complex float median_complex_float(int N, complex float ar[N]);


extern void md_medianz2(int D, int M, long dim[D], long ostr[D], complex float* optr, long istr[D], complex float* iptr);
extern void md_medianz(int D, int M, long dim[D], complex float* optr, complex float* iptr);

extern void linear_phase(unsigned int N, const long dims[__VLA(N)], const float pos[__VLA(N)], _Complex float* out);
extern void centered_gradient(unsigned int N, const long dims[__VLA(N)], const _Complex float grad[__VLA(N)], _Complex float* out);
extern void klaplace(unsigned int N, const long dims[__VLA(N)], unsigned int flags, _Complex float* out);

#ifdef __cplusplus
}
#endif

