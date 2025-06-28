
#include "misc/cppwrap.h"

extern void sinc_resize(int D, const long out_dims[__VLA(D)], _Complex float* out, const long in_dims[__VLA(D)], const _Complex float* in);
extern void sinc_zeropad(int D, const long out_dims[__VLA(D)], _Complex float* out, const long in_dims[__VLA(D)], const _Complex float* in);
extern void fft_zeropad(int D, unsigned long flags, const long out_dims[__VLA(D)], _Complex float* out, const long in_dims[__VLA(D)], const _Complex float* in);
extern void fft_zeropadH(int D, unsigned long flags, const long out_dims[__VLA(D)], _Complex float* out, const long in_dims[__VLA(D)], const _Complex float* in);

#include "misc/cppwrap.h"



