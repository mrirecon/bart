
#include "misc/misc.h"

#include "num/polynom.h"

#include "utest.h"


static bool test_polynom_eval(void)
{
	const complex double coeff[3] = { 1., 0., 1. };

	bool ok = true;

	ok &= (1. == polynom_eval(0., 2, coeff));
	ok &= (2. == polynom_eval(1., 2, coeff));
	ok &= (2. == polynom_eval(-1., 2, coeff));

	return ok;
}

UT_REGISTER_TEST(test_polynom_eval);


static bool array_eq(int N, const complex double c1[N], const complex double c2[N], double eps)
{
	return (0 == N) ? true : ((cabs(c1[0] - c2[0]) <= eps) && array_eq(N - 1, c1 + 1, c2 + 1, eps));
}


static bool test_polynom_derivative(void)
{
	const complex double coeff[3] = { 1., 0., 1. };
	complex double coeff2[2];
	
	polynom_derivative(2, coeff2, coeff);

	return array_eq(2, coeff2, (const complex double[]){ 0., 2. }, 0.);
}


UT_REGISTER_TEST(test_polynom_derivative);


static bool test_polynom_integral(void)
{
	const complex double coeff[3] = { 1., 0., 1. };
	complex double coeff2[2];
	complex double coeff3[3];
	
	polynom_derivative(2, coeff2, coeff);
	polynom_integral(1, coeff3, coeff2);

	return ((0. == coeff3[1]) && (1. == coeff3[2]));
}


UT_REGISTER_TEST(test_polynom_integral);


static bool test_polynom_integrate(void)
{
	const complex double coeff2[2] = { 0., 1. };

	return (0.5 == polynom_integrate(0., 1., 1, coeff2));
}


UT_REGISTER_TEST(test_polynom_integrate);


static bool test_polynom_from_roots1(void)
{
	const complex double roots[2] = { 1., -2.i };

	// (x - 1.) * (x + 2.i) == x^2 + (-1 + 2.i) x -2i.
	const complex double coeff0[3] = { -2.i, -1. + 2.i, 1. };
	complex double coeff[3];

	polynom_from_roots(2, coeff, roots);

	return array_eq(3, coeff0, coeff, 0.);
}

UT_REGISTER_TEST(test_polynom_from_roots1);


static bool test_polynom_from_roots(void)
{
	const complex double roots[3] = { 1., 2., 3. };

	complex double coeff[4];
	polynom_from_roots(3, coeff, roots);

	bool ok = true;

	for (unsigned int i = 0; i < ARRAY_SIZE(roots); i++)
		ok &= (0. == polynom_eval(roots[i], 3, coeff));

	complex double prod = 1.;

	for (unsigned int i = 0; i < ARRAY_SIZE(roots); i++)
		prod *= -roots[i];

	ok &= (prod == polynom_eval(0., 3, coeff));

	return ok;
}


UT_REGISTER_TEST(test_polynom_from_roots);


static bool test_polynom_scale(void)
{
	const complex double coeff[3] = { 1., 0., 1. };
	complex double coeff2[3];

	polynom_scale(2, coeff2, 2., coeff);

	return (5. == polynom_eval(1., 2, coeff2));	
}


UT_REGISTER_TEST(test_polynom_scale);


static bool test_polynom_shift(void)
{
	const complex double coeff[3] = { 1., 0., 1. };

	// 1. + (x + 1)^2 = 2 + 2 * x + x^2
	complex double coeff2[3];

	polynom_shift(2, coeff2, 1., coeff);

	return array_eq(3, coeff2, (const complex double[]){ 2., 2., 1. }, 0.);
}


UT_REGISTER_TEST(test_polynom_shift);



static bool test_quadratic_formula(void)
{
	const complex double roots[2] = { 1., -2.i };
	// (x - 1.) * (x + 2.i) == x^2 + (-1 + 2.i) x -2i.
	complex double coeff[3] = { -2.i, -1. + 2.i, 1. };

	complex double r2[2];
	quadratic_formula(r2, coeff);

	return array_eq(2, r2, roots, 0.);
}

UT_REGISTER_TEST(test_quadratic_formula);



static bool test_cubic_formula(void)
{
	const complex double roots[3] = { 1., 0.5, -2.i };

	complex double coeff[4];
	polynom_from_roots(3, coeff, roots);

	complex double r2[3];
	cubic_formula(r2, coeff);

	return array_eq(3, r2, roots, 1.E-15);
}

UT_REGISTER_TEST(test_cubic_formula);








