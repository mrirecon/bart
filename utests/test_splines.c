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



static bool test_bspline(void)
{
	const double knots[11] = { 0., 0.0, 0.0, 0., 0.25, 0.5, 0.75, 1., 1., 1., 1. };
	const double coord[7] = { 0., 0., 0., 1., 0., 0., 0. };

	double err = 0.;

	for (double x = 0.; x < 1.; x += 0.01) {

		double a = bspline(10, 3, 2, knots, x);
		double b = bspline_curve(10, 2, knots, coord, x);

		err += pow(a - b, 2);
	}

	return (err < 1.E-28);
}


UT_REGISTER_TEST(test_bspline);

