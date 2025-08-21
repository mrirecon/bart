/* Copyright 2020-2023. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef _SHAPE
#define _SHAPE

#include <complex.h>
#include "misc/mri.h"

struct tri_poly {

	long D;
	long cdims[DIMS];
	complex float* coeff; // k-space coeff
	long cpdims[DIMS];
	float* cpos;	// coordinates of k-space coeff
};

extern complex double xpolygon(int N, const double pg[N][2], const double p[3]);
extern complex double kpolygon(int N, const double pg[N][2], const double q[3]);

extern complex double xtripoly(const struct tri_poly* t, const long C, const double p[4]);
extern complex double ktripoly(const struct tri_poly* t, const long C, const double p[4]);

#endif
