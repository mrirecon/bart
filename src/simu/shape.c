/* Copyright 2020-2025. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2023-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <assert.h>
#include <math.h>

#include "num/flpmath.h"
#include "num/specfun.h"
#include "num/multind.h"

#include "geom/polygon.h"

#include "shape.h"


complex double xpolygon(int N, const double pg[N][2], const double p[3])
{
	assert(0. == p[2]);

	int w = polygon_winding_number(N, pg, p);

	return w;
}

static double sdot(const double a[2], const double b[2])
{
	return a[0] * b[0] + a[1] * b[1];
}


// https://arxiv.org/pdf/1703.00255.pdf
// FIXME: series expansion cases are missing

static complex double kpolygon1(int N, const double pg[N][2], const double c[2], const double q0[2])
{
#if 0
	double pwn = polygon_winding_number(N, pg, c);

	assert(    (1. == pwn)
		|| (-1. == pwn));
#endif
	double q[2];

	q[0] = -q0[0] * M_PI;
	q[1] = -q0[1] * M_PI;

	double q2 = q[0] * q[0] + q[1] * q[1];

	if (0. == q2)
//	if (1.E-6 > q2)
		return polygon_area(N, pg) / 4.;

	double qx[2] = { -q[1], q[0] };

	complex double sum[2] = { 0., 0. };

	for (int i = 0; i < N; i++) {

		double E[2] = { pg[i][0] - pg[(i - 1 + N) % N][0],  pg[i][1] - pg[(i - 1 + N) % N][1] };
		double R[2] = { pg[i][0] + pg[(i - 1 + N) % N][0],  pg[i][1] + pg[(i - 1 + N) % N][1] };

		R[0] -= 2. * c[0];
		R[1] -= 2. * c[1];

		complex double x = sinc(sdot(q, E)) * cexp(1.i * sdot(q, R));

		sum[0] += E[0] * x;
		sum[1] += E[1] * x;
	}

	return (qx[0] * sum[0] + qx[1] * sum[1]) / (8.i * q2);
}

complex double kpolygon(int N, const double pg0[N][2], const double q0[3])
{
	double pg[N][2];

	for (int i = 0; i < N; i++) {

		pg[i][0] = pg0[i][0];
		pg[i][1] = pg0[i][1];
	}

	double cg[2] = { 0., 0. };

	for (int i = 0; i < N; i++) {

		cg[0] += pg[i][0];
		cg[1] += pg[i][1];
	}

	cg[0] /= N;
	cg[1] /= N;

	assert(0. == q0[2]);

	return cexp(-M_PI * 2.i * sdot(cg, q0)) * kpolygon1(N, pg, cg, q0);
//	return kpolygon1(N, pg, (double[]){ 0., 0. }, q0);
}

// compute trigonometric polynomial defined in tri_poly and evaluate at p in x-space
complex double xtripoly(const struct tri_poly* t, const long C, const double p[4])
{
	long D = t->D;

	for (int i = 0; i < 3; i++)
		assert(t->cdims[i] == t->cpdims[i + 1]);

	long cstrs[D], cpstrs[D], cpos[D], pos[D], ccpos[D];

	md_calc_strides(D, cstrs, t->cdims, CFL_SIZE);
	md_calc_strides(D, cpstrs, t->cpdims, FL_SIZE);

	md_set_dims(D, pos, 0);
	md_set_dims(D, cpos, 0);

	complex float z = 0;
	double ip = 0;

	do {
		for (int i = 0; i < 3; i++)
			pos[i + 1] = cpos[i];

		ip = 0;
		for (pos[0] = 0; pos[0] < 3; pos[0]++)
			ip += MD_ACCESS(D, cpstrs, pos, t->cpos) * p[pos[0]];

		md_copy_dims(D, ccpos, cpos);
		ccpos[COIL_DIM] = C;

		z += MD_ACCESS(D, cstrs, ccpos, t->coeff) * cexpf(2.i * M_PI * ip);

	} while(md_next(D, t->cdims, ~COIL_FLAG, cpos));

	return z;
}

// evaluate representation of trigonometric polynomial in k-space at position p
complex double ktripoly(const struct tri_poly* t, const long C, const double p[4])
{
	long D = t->D;
	float* ccpos = t->cpos;
	complex float* ccoeff = t->coeff;

	long pos[D], cpos[D], pstrs[D], cstrs[D];

	md_calc_strides(D, pstrs, t->cpdims, FL_SIZE);
	md_calc_strides(D, cstrs, t->cdims, CFL_SIZE);

	md_set_dims(D, pos, 0);

	do {
		bool t = true;

		for (pos[0] = 2; pos[0] >= 0; pos[0]--)
			if (p[pos[0]] != MD_ACCESS(D, pstrs, pos, ccpos))
				t = false;

		if (t) {

			md_set_dims(D, cpos, 0);

			for (int i = 0; i < 3; i++)
				cpos[i] = pos[i+1];

			cpos[COIL_DIM] = C;

			return MD_ACCESS(D, cstrs, cpos, ccoeff);
		}

	} while(md_next(D, t->cpdims, ~MD_BIT(0), pos));

	return 0;
}

