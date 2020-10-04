/* Copyright 2020. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2019-2020 Martin Uecker <muecker@gwdg.de>
 */

#include "polyhedron.h"

static void vec3d_sub(double n[3], const double a[3], const double b[3])
{
	for (int j = 0; j < 3; j++)
		n[j] = a[j] - b[j];
}

static void vec3d_rot(double n[3], const double a[3], const double b[3])
{
	n[0] = a[1] * b[2] - a[2] * b[1];
	n[1] = a[2] * b[0] - a[0] * b[2];
	n[2] = a[0] * b[1] - a[1] * b[0];
}

static double vec3d_dot(const double a[3], const double b[3])
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}


double polyhedron_vol(int N, const double tri[N][3][3])
{
	double sum = 0.;

	for (int i = 0; i < N; i++) {

		double a[3];
		double b[3];

		vec3d_sub(a, tri[i][0], tri[i][1]);
		vec3d_sub(b, tri[i][0], tri[i][2]);

		double n[3];
		vec3d_rot(n, a, b);
		double r = vec3d_dot(n, tri[i][0]);

		sum += r / 6.;
	}

	return sum;
}



