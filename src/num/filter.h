
#include <complex.h>

#include "misc/cppwrap.h"

extern float median_float(int N, const float ar[N]);
extern complex float median_complex_float(int N, const complex float ar[N]);
extern void weiszfeld(int iter, int N, int D, float out[D], const float in[N][D]);


extern void md_medianz2(int D, int M, const long dim[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr);
extern void md_medianz(int D, int M, const long dim[D], complex float* optr, const complex float* iptr);

extern void md_geometric_medianz2(int D, int M, const long dim[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr);
extern void md_geometric_medianz(int D, int M, const long dim[D], complex float* optr, const complex float* iptr);

extern void md_moving_avgz2(int D, int M, const long dim[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr);
extern void md_moving_avgz(int D, int M, const long dim[D], complex float* optr, const complex float* iptr);

extern void linear_phase(unsigned int N, const long dims[__VLA(N)], const float pos[__VLA(N)], _Complex float* out);
extern void centered_gradient(unsigned int N, const long dims[__VLA(N)], const _Complex float grad[__VLA(N)], _Complex float* out);
extern void klaplace(int N, const long dims[__VLA(N)], long flags, _Complex float* out);
void klaplace_scaled(int N, const long dims[N], long flags, const float sc[N], complex float* out);

extern void md_zhamming(const unsigned int D, const long dims[__VLA(D)], const long flags, complex float* optr, const complex float* iptr);
extern void md_zhamming2(const unsigned int D, const long dims[__VLA(D)], const long flags, const long ostr[__VLA(D)], complex float* optr, const long istr[__VLA(D)], const complex float* iptr);

extern void md_zhann(const unsigned int D, const long dims[__VLA(D)], const long flags, complex float* optr, const complex float* iptr);
extern void md_zhann2(const unsigned int D, const long dims[__VLA(D)], const long flags, const long ostr[__VLA(D)], complex float* optr, const long istr[__VLA(D)], const complex float* iptr);

#include "misc/cppwrap.h"

