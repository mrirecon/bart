/* Copyright 2014-2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#include "misc/cppwrap.h"

struct ellipsis_s {

	complex double intensity;
	double axis[2];
	double center[2];
	double angle;
};

extern const struct ellipsis_s shepplogan[10];
extern const struct ellipsis_s shepplogan_mod[10];
extern const struct ellipsis_s phantom_disc[1];
extern const struct ellipsis_s phantom_ring[4];


extern complex double xellipsis(const double center[2], const double axis[2], double angle, const double p[2]);
extern complex double kellipsis(const double center[2], const double axis[2], double angle, const double p[2]);
extern complex double xrectangle(const double center[2], const double axis[2], double angle, const double p[2]);
extern complex double krectangle(const double center[2], const double axis[2], double angle, const double p[2]);
    


extern complex double phantom(unsigned int N, const struct ellipsis_s arr[__VLA(N)], const double pos[2], _Bool ksp);
extern complex double phantomX(unsigned int N, const struct ellipsis_s arr[__VLA(N)], const double pos[2], _Bool ksp);

#include "misc/cppwrap.h"

