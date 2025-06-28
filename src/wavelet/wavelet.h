
#include <complex.h>
#include <stdbool.h>

extern const float wavelet_haar[2][2][2];
extern const float wavelet_dau2[2][2][4];
extern const float wavelet_cdf44[2][2][10];

// layer 1

extern void fwt1(int N, int d, const long dims[N], const long ostr[N], complex float* low, complex float* hgh, const long istr[N], const complex float* in, const long flen, const float filter[2][2][flen]);
extern void iwt1(int N, int d, const long dims[N], const long ostr[N], complex float* out, const long istr[N], const complex float* low, const complex float* hgh, const long flen, const float filter[2][2][flen]);

// layer 2

extern void fwtN(int N, unsigned long flags, const long shifts[N], const long dims[N], const long ostr[2 * N], complex float* out, const long istr[N], const complex float* in, const long flen, const float filter[2][2][flen]);
extern void iwtN(int N, unsigned long flags, const long shifts[N], const long dims[N], const long ostr[N], complex float* out, const long istr[2 * N], const complex float* in, const long flen, const float filter[2][2][flen]);

extern void wavelet_dims(int N, unsigned long flags, long odims[2 * N], const long dims[N], const long flen);

// layer 3

extern void fwt(int N, unsigned long flags, const long shifts[N], const long odims[N], complex float* out, const long idims[N], const complex float* in, const long minsize[N], const long flen, const float filter[2][2][flen]);
extern void iwt(int N, unsigned long flags, const long shifts[N], const long odims[N], complex float* out, const long idims[N], const complex float* in, const long minsize[N], const long flen, const float filter[2][2][flen]);

extern int wavelet_num_levels(int N, unsigned long flags, const long dims[N], const long min[N], const long flen);
extern long wavelet_coeffs(int N, unsigned long flags, const long dims[N], const long min[N], const long flen);

extern void wavelet_coeffs2(int N, unsigned long flags, long odims[N], const long dims[N], const long min[N], const long flen);

extern void fwt2(int N, unsigned long flags, const long shifts[N], const long odims[N], const long ostr[N], complex float* out, const long idims[N], const long istr[N], const complex float* in, const long minsize[N], long flen, const float filter[2][2][flen]);
extern void iwt2(int N, unsigned long flags, const long shifts[N], const long odims[N], const long ostr[N], complex float* out, const long idims[N], const long istr[N], const complex float* in, const long minsize[N], const long flen, const float filter[2][2][flen]);

extern void wavelet_thresh(int N, float lambda, unsigned long flags, unsigned long jflags, const long shifts[N], const long dims[N], complex float* out, const complex float* in, const long minsize[N], long flen, const float filter[2][2][flen]);


