/* Copyright 2019-2020. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2019-2020 Martin Uecker <muecker@gwdg.de>
 */

#include <math.h>
#include <stdbool.h>

#include "num/vec3.h"

#include "triangle.h"


// Moeller-Trumbore

float (triangle_intersect)(float uv[2], const vec3_t o, const vec3_t d, const vec3_t tri[3])
{
	vec3_t e1, e2;
	vec3_sub(e1, tri[1], tri[0]);	
	vec3_sub(e2, tri[2], tri[0]);

	vec3_t pvec;
	vec3_rot(pvec, d, e2);
	float det = vec3_sdot(e1, pvec);

	if (fabsf(det) < 1.E-8)	// parallel
		return 0.;

	float idet = 1. / det;

	vec3_t tvec;
	vec3_sub(tvec, o, tri[0]);
	float u = vec3_sdot(tvec, pvec) * idet;

	if ((u < 0.) || (u > 1.))
		return 0.;

	vec3_t qvec;
	vec3_rot(qvec, tvec, e1);
	float v = vec3_sdot(d, qvec) * idet;
	
	if ((v < 0.) || (u + v > 1.))
		return 0.;

	uv[0] = u;
	uv[1] = v;

	return vec3_sdot(e2, qvec) * idet;
}


static float det2d(const float a[2], const float b[2])
{
	return a[0] * b[1] - a[1] * b[0];
}

bool (triangle2d_inside)(const float tri[3][2], const float p[2])
{
	float ab[2] = { tri[1][0] - tri[0][0], tri[1][1] - tri[0][1] };
	float ac[2] = { tri[2][0] - tri[0][0], tri[2][1] - tri[0][1] };
	float d = det2d(ab, ac);
	float u = (det2d(p, ac) - det2d(tri[0], ac)) / d;
	float v = (det2d(p, ab) - det2d(tri[0], ab)) / d;
	return (0. < u) && (0. < v) && (u + v < 1.);
}


