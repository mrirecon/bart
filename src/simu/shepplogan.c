/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 */
 
#include <complex.h>
#include <math.h>
#include <assert.h>

#ifdef USE_GSL
#include <gsl/gsl_specfunc.h>
#endif

#include "simu/shepplogan.h"

#include "misc/misc.h"

// AK Jain, 439.
// modified Toft 199-200 multiply -0.98 -> 0.8
// multiply the rest by ten
const struct ellipsis_s shepplogan[10] = {
	{	1.,   { .69,   .92   }, { 0.,     0. },	0. },
	{	-.98, { .6624, .8740 }, { 0., -.0184 }, 0. },
	{       -.02, { .1100, .3100 }, {  .22,    0. },    -18, },
	{       -.02, { .1600, .4100 }, { -.22,    0 },     18  },
	{        .01, { .2100, .2500 }, {  0,    .35 },    0   },
	{        .01, { .0460, .0460 }, {  0,    .1 },    0   },
	{        .01, { .0460, .0460 }, {  0,   -.1 },     0   },
	{        .01, { .0460, .0230 }, { -.08,  -.605 },   0   },
	{        .01, { .0230, .0230 }, {  0,   -.606 },   0   },
	{        .01, { .0230, .0460 }, { .06,  -.605 },  0   }
};

const struct ellipsis_s shepplogan_mod[10] = {
	{	1.,  { .69,   .92   }, { 0.,     0. },	0. },
	{	-.8, { .6624, .8740 }, { 0., -.0184 }, 0. },
	{       -.2, { .1100, .3100 }, {  .22,    0. },    -18, },
	{       -.2, { .1600, .4100 }, { -.22,    0 },     18  },
	{        .1, { .2100, .2500 }, {  0,    .35 },    0   },
	{        .1, { .0460, .0460 }, {  0,    .1 },    0   },
	{        .1, { .0460, .0460 }, {  0,   -.1 },     0   },
	{        .1, { .0460, .0230 }, { -.08,  -.605 },   0   },
	{        .1, { .0230, .0230 }, {  0,   -.606 },   0   },
	{        .1, { .0230, .0460 }, { .06,  -.605 },  0   }
};

const struct ellipsis_s phantom_disc[1] = {
	{	1., { 1., 1. }, { 0., 0. }, 0. }
};


// old: imaginary ring outside from 0.5 to 0.49
const struct ellipsis_s phantom_ring[4] = {
	{	1., { 0.75, 0.75 }, { 0., 0. }, 0. },
	{	-1. + 1.i, { 0.5, 0.5 }, { 0., 0. }, 0. },
	{	-1.i, { 0.48, 0.48 }, { 0., 0. }, 0. },
	{	1., { 0.48, 0.48 }, { 0., 0. }, 0. },
//	{	1., { 0.48, 0.48 }, { 0., 0. }, 0. },
};


static double sinc(double x)
{
	return (0. == x) ? 1. : (sin(x) / x);
}

static double jinc(double x)
{
#ifdef USE_GSL
	return (0. == x) ? 1. : (2. * gsl_sf_bessel_J1(x) / x);
#else
	UNUSED(x);
	assert(0);
#endif
}



static void rot2d(double x[2], const double in[2], double angle)
{
	x[0] = cos(angle) * in[0] + sin(angle) * in[1];
	x[1] = sin(angle) * in[0] - cos(angle) * in[1];
}

complex double xellipsis(const double center[2], const double axis[2], double angle, const double p[2])
{
	double p90[2];
	p90[0] = -p[1];
	p90[1] = p[0];

	double pshift[2];
	pshift[0] = p90[0] + center[0];
	pshift[1] = p90[1] + center[1];
	double prot[2];
	rot2d(prot, pshift, 2. * M_PI * angle / 360.);

	double radius = pow(prot[0] / axis[0], 2.) + pow(prot[1] / axis[1], 2.);

	return (radius <= 1.) ? 1. : 0.;
}

complex double kellipsis(const double center[2], const double axis[2], double angle, const double p[2])
{
	double p90[2];
	p90[0] = -p[1];
	p90[1] = p[0];

	double prot[2];
	rot2d(prot, p90, 2. * M_PI * angle / 360.);

	double radius = sqrt(pow(prot[0] * axis[0], 2.) + pow(prot[1] * axis[1], 2.));

	complex double res = jinc(2. * M_PI * radius) * (axis[0] * axis[1]);

	return res * cexp(2.i * M_PI * (p90[0] * center[0] + p90[1] * center[1]));
}

complex double xrectangle(const double center[2], const double axis[2], double angle, const double p[2])
{
	double p90[2];
	p90[0] = -p[1];
	p90[1] = p[0];

	double pshift[2];
	pshift[0] = p90[0] + center[0];
	pshift[1] = p90[1] + center[1];
	double prot[2];
	rot2d(prot, pshift, 2. * M_PI * angle / 360.);

	double radius = fabs(prot[0] / axis[0]) + fabs(prot[1] / axis[1]);

	return (radius <= 1.) ? 1. : 0.;
}

complex double krectangle(const double center[2], const double axis[2], double angle, const double p[2])
{
	double p90[2];
	p90[0] = -p[1];
	p90[1] = p[0];

	double prot[2];
	rot2d(prot, p90, 2. * M_PI * angle / 360.);

	complex double res = sinc(2. * M_PI * prot[0] * axis[0]) * sinc(2. * M_PI * prot[1] * axis[1]) * (axis[0] * axis[1]);

	return res * cexp(2.i * M_PI * (p90[0] * center[0] + p90[1] * center[1]));
}


complex double phantom(unsigned int N, const struct ellipsis_s arr[N], const double pos[2], _Bool ksp)
{
	complex double res = 0.;

	for (unsigned int i = 0; i < N; i++)
		res += arr[i].intensity * (ksp ? kellipsis : xellipsis)(arr[i].center, arr[i].axis, arr[i].angle, pos);

	return res;
}

complex double phantomX(unsigned int N, const struct ellipsis_s arr[N], const double pos[2], _Bool ksp)
{
	complex double res = 0.;

	for (unsigned int i = 0; i < N; i++)
		res += arr[i].intensity * (ksp ? krectangle : xrectangle)(arr[i].center, arr[i].axis, arr[i].angle, pos);

	return res;
}

