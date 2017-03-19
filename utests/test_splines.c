/* Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */


#include <math.h>
#include <stdio.h>

#include "num/splines.h"

#include "utest.h"

static const double coords[5] = { 0., 1., 0.5, 1., 0.5 };

static bool test_bezier_curve(void)
{
	bool ret = true;

	ret = ret && (coords[0] == bezier_curve(0., 4, coords));
	ret = ret && (coords[4] == bezier_curve(1., 4, coords));

	return ret;
}

UT_REGISTER_TEST(test_bezier_curve);


static bool test_bezier_increase_degree(void)
{
	double coords2[6];
	bezier_increase_degree(4, coords2, coords);

	double err = 0.;

	for (double x = 0.; x < 1.; x += 0.01) {

		double a = bezier_curve(x, 4, coords);
		double b = bezier_curve(x, 5, coords2);

		err += pow(a - b, 2);
	}

	return (err < 1.E-28);
}

UT_REGISTER_TEST(test_bezier_increase_degree);


static bool test_bezier_split(void)
{
	double coords[5] = { 0., 1., 0.5, 1., 0.5 };

	double coordsA[5];
	double coordsB[5];
	bezier_split(0.5, 4, coordsA, coordsB, coords);

	double err = 0.;

	for (double x = 0.; x < 1.; x += 0.01) {

		double a = bezier_curve(x, 4, coords);
		double b = (x <= 0.5) 	? bezier_curve(2. * x, 4, coordsA)
					: bezier_curve(2. * (x - 0.5), 4, coordsB);

		err += pow(a - b, 2);
	}

	return (err < 1.E-28);
}

UT_REGISTER_TEST(test_bezier_split);



static bool test_cspline(void)
{
	const double coeff[4] = { 0., 1., 1., 1. };

	bool ok = true;

	for (double x = 0.; x < 1.; x += 0.1)
		ok &= (fabs(x - cspline(x, coeff)) < 1.E-15);

	return ok;
}


UT_REGISTER_TEST(test_cspline);


static bool test_bspline(void)
{
	const double knots[11] = { 0., 0.0, 0.0, 0., 0.25, 0.5, 0.75, 1., 1., 1., 1. };

	bool ok = true;

	for (int i = 0; i < 7; i++) {

		double coord[7] = { 0., 0., 0., 0., 0., 0., 0. };
		coord[i] = 1.;

		double err = 0.;

		for (double x = 0.; x <= 1.; x += 0.01) {

			double a = bspline(10, i, 3, knots, x);
			double b = bspline_curve(10, 3, knots, coord, x);

			err += pow(a - b, 2);
		}

		ok &= (err < 1.E-28);
	}

	return ok;
}


UT_REGISTER_TEST(test_bspline);


static bool test_bspline_knot_insert(void)
{
	const double knots[11] = { 0., 0.0, 0.0, 0., 0.25, 0.5, 0.75, 1., 1., 1., 1. };
	double coord[7] = { 0., 0., 0.75, 0.5, 0.25, 0., 0 };

	double knots2[12];
	double coord2[8];
	bspline_knot_insert(0.6, 10, 3, knots2, coord2, knots, coord);

	double err = 0.;

	for (double x = 0.; x < 1.; x += 0.01) {

		double a = bspline_curve(10, 3, knots, coord, x);
		double b = bspline_curve(11, 3, knots2, coord2, x);

		err += pow(a - b, 2);
	}

	return (err < 1.E-28);
}

UT_REGISTER_TEST(test_bspline_knot_insert);



static bool test_bspline_derivative(void)
{
	const double knots[11] = { 0., 0.0, 0.0, 0., 0.25, 0.5, 0.75, 1., 1., 1., 1. };

	bool ok = true;

	for (int i = 0; i < 7; i++) {

		double coord[7] = { 0., 0., 0., 0., 0., 0., 0. };
		coord[i] = 1.;

		double err = 0.;

		for (double x = 0.; x <= 1.; x += 0.01) {

			double a = bspline_derivative(10, i, 3, knots, x);
			double b = bspline_curve_derivative(1, 10, 3, knots, coord, x);

			err += pow(a - b, 2);
		}

		ok &= (err < 1.E-1);
	}

	return ok;
}

UT_REGISTER_TEST(test_bspline_derivative);


static bool test_bspline_zero(void)
{
	const double knots[11] = { 0., 0.0, 0.0, 0., 0.25, 0.5, 0.75, 1., 1., 1., 1. };
	const double z0[7] = { 0., 0., 0.75, 0.5, 0.25, 0., 0 };

	bool ok = true;

	for (int i = 2; i < 5; i++) { // FIXME

		double coord[7] = { 0., 0., 0., 0., 0., 0., 0. };
		coord[i] = 1.;

		double k2[9];
		double c2[6];

		bspline_coeff_derivative_n(1, 10, 3, k2, c2, knots, coord);
		double z = bspline_curve_zero(8, 2, k2, c2);

		ok &= (fabs(z - z0[i]) < 1.E-5);

	}

	return ok;
}

UT_REGISTER_TEST(test_bspline_zero);




static bool test_nurbs(void)
{
	const double knots[11] = { 0., 0.0, 0.0, 0., 0.25, 0.5, 0.75, 1., 1., 1., 1. };

	bool ok = true;

	for (int i = 0; i < 7; i++) {

		double coord[7] = { 0., 0., 0., 0., 0., 0., 0. };
		double weights[7] = { 1., 1., 1., 1., 1., 1., 1. };
		coord[i] = 1.;

		double err = 0.;

		for (double x = 0.; x <= 1.; x += 0.01) {

			double a = nurbs(10, 3, knots, coord, weights, x);
			double b = bspline_curve(10, 3, knots, coord, x);
			double c = bspline_curve(10, 3, knots, weights, x);

			err += pow(a - b / c, 2);
		}

		ok &= (err < 1.E-28);
	}

	return ok;
}


UT_REGISTER_TEST(test_nurbs);


static bool test_nurbs_arc(void)
{
	const double knots[6] = { 0., 0., 0., 1., 1., 1. };

	double coordx[3] = { 0., 1., 1. };
	double coordy[3] = { 1., 1., 0. };
	double weights[3] = { sqrt(2.), 1., sqrt(2.) };

	bool ok = true;

	for (double t = 0.; t <= 1.; t += 0.01) {

		double x = nurbs(5, 2, knots, coordx, weights, t);
		double y = nurbs(5, 2, knots, coordy, weights, t);

		ok &= fabs(pow(x, 2.) + pow(y, 2.) - 1.) < 1.E-15;
	}

	return ok;
}


UT_REGISTER_TEST(test_nurbs_arc);



static bool test_nurbs_circle(void)
{
	const double knots[10] = { 0., 0., 0., 1., 1., 2., 2., 3., 3., 3. };

	double coordx[7] = {
		cos(0. * 2. * M_PI / 3.) + cos(1. * 2. * M_PI / 3.),
		cos(1. * 2. * M_PI / 3.) * 2.,
		cos(1. * 2. * M_PI / 3.) + cos(2. * 2. * M_PI / 3.),
		cos(2. * 2. * M_PI / 3.) * 2.,
		cos(2. * 2. * M_PI / 3.) + cos(3. * 2. * M_PI / 3.),
		cos(3. * 2. * M_PI / 3.) * 2.,
		cos(3. * 2. * M_PI / 3.) + cos(1. * 2. * M_PI / 3.),
	};
	double coordy[7] = {
		sin(0. * 2. * M_PI / 3.) + sin(1. * 2. * M_PI / 3.),
		sin(1. * 2. * M_PI / 3.) * 2.,
		sin(1. * 2. * M_PI / 3.) + sin(2. * 2. * M_PI / 3.),
		sin(2. * 2. * M_PI / 3.) * 2.,
		sin(2. * 2. * M_PI / 3.) + sin(3. * 2. * M_PI / 3.),
		sin(3. * 2. * M_PI / 3.) * 2.,
		sin(3. * 2. * M_PI / 3.) + sin(1. * 2. * M_PI / 3.),
	};
	double weights[7] = { 1., 0.5, 1., 0.5, 1., 0.5, 1. };

	bool ok = true;

	for (double t = 0.; t <= 3.; t += 0.01) {

		double x = nurbs(9, 2, knots, coordx, weights, t);
		double y = nurbs(9, 2, knots, coordy, weights, t);

		ok &= fabs(pow(x, 2.) + pow(y, 2.) - 1.) < 1.E-15;
	}

	return ok;
}


UT_REGISTER_TEST(test_nurbs_circle);

