
#ifndef __QUADRATURE_H
#define __QUADRATURE_H 1

#include "misc/nested.h"

extern void quadrature_trapezoidal(int N, const float t[static N + 1], int P, float out[P],
		void CLOSURE_TYPE(sample)(float out[P], int i));

extern void quadrature_simpson_ext(int N, float T, int P, float out[P],
		void CLOSURE_TYPE(sample)(float out[P], int i));

#endif // __QUADRATURE_H

