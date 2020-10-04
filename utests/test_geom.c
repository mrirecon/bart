/* Copyright 2020. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2019-2020 Martin Uecker <muecker@gwdg.de>
 */

#include <math.h>

#include "geom/polygon.h"
#include "geom/polyhedron.h"
#include "geom/triangle.h"

#include "utest.h"


static bool test_winding_number(void)
{
	double p[2] = { 0., 0. };

	double pg[4][2] = {
		{  1., -1. },
		{  1.,  1. },
		{ -1.,  1. },
		{ -1., -1. },
	};

	return (1 == polygon_winding_number(4, pg, p));
}

UT_REGISTER_TEST(test_winding_number);


static bool test_triangle_intersect(void)
{
	float o[3] = { 0., 0., 0. };
	float d[3] = { 1., 0., 0. };
	float t[3][3] = {

		{ 1., -1., -1. },
		{ 1., -1.,  1. },
		{ 1.,  1.,  0. },
	};

	float uv[2];

	return (1. == triangle_intersect(uv, o, d, t));
}


UT_REGISTER_TEST(test_triangle_intersect);



static bool test_polygon_area(void)
{
	double pg[4][2] = { { 0., 0. }, { 1., 0. }, { 1., 1. }, { 0., 1. } };

	return (1. == polygon_area(4, pg));
}

UT_REGISTER_TEST(test_polygon_area);




static bool test_polyhedron_volume(void)
{
	double ph[4][3][3] = {
		{ { +0.5, +0.5, +0.5 }, { +0.5, -0.5, +0.5 }, { +0.5, +0.5, -0.5 } },
		{ { +0.5, +0.5, +0.5 }, { +0.5, +0.5, -0.5 }, { -0.5, +0.5, +0.5 } },
		{ { +0.5, -0.5, +0.5 }, { +0.5, +0.5, +0.5 }, { -0.5, +0.5, +0.5 } },
		{ { +0.5, -0.5, +0.5 }, { -0.5, +0.5, +0.5 }, { +0.5, +0.5, -0.5 } },
	};

	return fabs(1. / 6. - polyhedron_vol(4, ph)) < 1.E-16;
}

UT_REGISTER_TEST(test_polyhedron_volume);



