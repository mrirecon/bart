/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef __SHEPPLOGAN_H
#define __SHEPPLOGAN_H

#include <complex.h>

#include "misc/cppwrap.h"

struct ellipsis_s {

	complex double intensity;
	double axis[2];
	double center[2];
	double angle;
};

struct ellipsis3d_s {

	complex double intensity;
	double axis[3];
	double center[3];
	double angle;
};

struct ellipsis_bs {

	struct ellipsis_s geo;
	bool background;
};


extern const struct ellipsis_s shepplogan[10];
extern const struct ellipsis_s shepplogan_mod[10];
extern const struct ellipsis_s phantom_disc[1];
extern const struct ellipsis_s phantom_ring[4];


extern const struct ellipsis3d_s phantom_disc3d[1];
extern const struct ellipsis3d_s shepplogan3d[10];

extern const struct ellipsis_s phantom_geo1[3];
extern const struct ellipsis_s phantom_geo2[2];
extern const struct ellipsis_s phantom_geo3[7];
extern const struct ellipsis_s phantom_geo4[1];
extern const struct ellipsis_s phantom_geo5[1];

extern const struct ellipsis_bs phantom_tubes[21];

extern const struct ellipsis_bs nist_phantom_t2[29];

extern const struct ellipsis_bs phantom_sonar[15];


extern complex double xellipsis(const double center[2], const double axis[2], double angle, const double p[2]);
extern complex double kellipsis(const double center[2], const double axis[2], double angle, const double p[2]);

extern complex double xellipsis3d(const double center[3], const double axis[3], double angle, const double p[3]);
extern complex double kellipsis3d(const double center[3], const double axis[3], double angle, const double p[3]);

extern complex double xrectangle(const double center[2], const double axis[2], double angle, const double p[2]);
extern complex double krectangle(const double center[2], const double axis[2], double angle, const double p[2]);
    


extern complex double phantom(unsigned int N, const struct ellipsis_s arr[__VLA(N)], const double pos[2], _Bool ksp);
extern complex double phantomX(unsigned int N, const struct ellipsis_s arr[__VLA(N)], const double pos[2], _Bool ksp);

extern complex double phantom3d(unsigned int N, const struct ellipsis3d_s arr[__VLA(N)], const double pos[3], _Bool ksp);

#include "misc/cppwrap.h"

#endif	// __SHEPPLOGAN_H
