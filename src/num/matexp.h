
#include <complex.h>

#include "misc/nested.h"

extern void mat_exp(int N, float t, float out[N][N], const float in[N][N]);
extern void zmat_exp(int N, float t, complex float out[N][N], const complex float in[N][N]);
extern void mat_to_exp(int N, float st, float en, float out[N][N], float tol,
		void CLOSURE_TYPE(f)(float* out, float t, const float* yn));

