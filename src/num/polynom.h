
#include <complex.h>

extern complex double polynom_eval(complex double x, int N, const complex double coeff[N + 1]);
extern void polynom_derivative(int N, complex double out[N], const complex double in[N + 1]);
extern void polynom_integral(int N, complex double out[N + 2], const complex double in[N + 1]);
extern complex double polynom_integrate(complex double st, complex double end, int N, const complex double coeff[N + 1]);
extern void polynom_monomial(int N, complex double coeff[N + 1], int O);
extern void polynom_from_roots(int N, complex double coeff[N + 1], const complex double root[N]);
extern void polynom_scale(int N, complex double out[N + 1], complex double scale, const complex double in[N + 1]);
extern void polynom_shift(int N, complex double out[N + 1], complex double shift, const complex double in[N + 1]);

extern void quadratic_formula(complex double x[2], complex double coeff[3]);
extern void cubic_formula(complex double x[3], complex double coeff[4]);


