
extern void simplex(unsigned int D, unsigned int N, float x[N], const float c[N], const float b[D], const float A[D][N]);

#if __GNUC__ < 5
#include "misc/pcaa.h"

#define simplex(D, N, x, c, b, A) \
	simplex(D, N, x, c, b, AR2D_CAST(float, D, N, A))
#endif

