
#include "misc/cppwrap.h"

#ifndef WTYPE
#define WTYPE
enum wtype { WAVELET_HAAR, WAVELET_DAU2, WAVELET_CDF44 };
#endif

extern struct linop_s* linop_wavelet_create(int N, unsigned long flags, const long dims[__VLA(N)], const long istr[__VLA(N)],
						enum wtype wtype, const long minsize[__VLA(N)], _Bool randshift);

#include "misc/cppwrap.h"

