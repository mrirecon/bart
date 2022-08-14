/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2013-2016 Martin Uecker <martin.uecker@med.uni-goettinge.de>
 */

#define _GNU_SOURCE
#include <complex.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>

#include "simu/shepplogan.h"

#include "misc/misc.h"

// AK Jain, 439.
// modified Toft 199-200 multiply -0.98 -> 0.8
// multiply the rest by ten
const struct ellipsis_s shepplogan[10] = {
	{	1.,   { .69,   .92   }, { 0.,     0. },	   0.  },
	{	-.98, { .6624, .8740 }, { 0., -.0184 },    0.  },
	{       -.02, { .1100, .3100 }, {  .22,    0. }, -18. / 360. * 2. * M_PI },
	{       -.02, { .1600, .4100 }, { -.22,    0 },   18. / 360. * 2. * M_PI },
	{        .01, { .2100, .2500 }, {  0,    .35 },    0.   },
	{        .01, { .0460, .0460 }, {  0,    .1 },     0.   },
	{        .01, { .0460, .0460 }, {  0,   -.1 },     0.   },
	{        .01, { .0460, .0230 }, { -.08,  -.605 },  0.   },
	{        .01, { .0230, .0230 }, {  0,   -.606 },   0.   },
	{        .01, { .0230, .0460 }, { .06,  -.605 },   0.   }
};

const struct ellipsis_s shepplogan_mod[10] = {
	{	1.,  { .69,   .92   }, { 0.,     0. },	  0.  },
	{	-.8, { .6624, .8740 }, { 0., -.0184 },    0.  },
	{       -.2, { .1100, .3100 }, {  .22,    0. }, -18. / 360. * 2. * M_PI },
	{       -.2, { .1600, .4100 }, { -.22,    0 },   18. / 360. * 2. * M_PI },
	{        .1, { .2100, .2500 }, {  0,    .35 },    0   },
	{        .1, { .0460, .0460 }, {  0,    .1 },     0   },
	{        .1, { .0460, .0460 }, {  0,   -.1 },     0   },
	{        .1, { .0460, .0230 }, { -.08,  -.605 },  0   },
	{        .1, { .0230, .0230 }, {  0,   -.606 },   0   },
	{        .1, { .0230, .0460 }, { .06,  -.605 },   0   }
};

const struct ellipsis_s phantom_disc[1] = {
	{	1., { 1., 1. }, { 0., 0. }, 0. }
};

const struct ellipsis3d_s phantom_disc3d[1] = {
	{	1., { 1., 1., 1. }, { 0., 0., 0. }, 0. }
};


// old: imaginary ring outside from 0.5 to 0.49
const struct ellipsis_s phantom_ring[4] = {
	{	1., { 0.75, 0.75 }, { 0., 0. }, 0. },
	{	-1. + 1.i, { 0.5, 0.5 }, { 0., 0. }, 0. },
	{	-1.i, { 0.48, 0.48 }, { 0., 0. }, 0. },
	{	1., { 0.48, 0.48 }, { 0., 0. }, 0. },
//	{	1., { 0.48, 0.48 }, { 0., 0. }, 0. },
};

// Some geometric objects
const struct ellipsis_s phantom_geo1[3] = {
	{	.5,    { .2,   .2   }, { .6,      .5 },	   0        },
	{	.2,    { .2,   .4   }, { -.6,    -.6 },	   M_PI/7.  },
	{	.6,    { .3,   .1   }, { .6,     -.5 },	   M_PI/5.  },
};

const struct ellipsis_s phantom_geo2[2] = {
	{      0.5,   { .5,   -.5   }, { 0.,     0.  },	   M_PI/5  },
	{	.7,   { .2,   .2    }, { -.5,     .7 },	  -M_PI/4  },
};

const struct ellipsis_s phantom_geo3[7] = {
	{	1.,    { .2,   .2   }, { -.6,   .6   },	   M_PI/4 },
	{	1.,    { .2,   .2   }, { -.6,  -.6   },	   M_PI/4  },
	{	0.6,   { .07,   .07   }, { 0.15,   0   },	   0.  },
	{	0.7,   { .07,   .07   }, { 0,      0   },	   0.  },
	{	0.8,   { .07,   .07   }, { -0.15,  0   },	   0.  },
	{	0.9,   { .07,   .07   }, { 0.15,   0.15},  0.  },
	{	0.9,   { .07,   .07   }, { 0.15,  -0.15},  0.  },
};

const struct ellipsis_s phantom_geo4[1] = {
	{	0.8,   { .1,   .7   }, { .6,     0  },	   0},
};

const struct ellipsis_s phantom_geo5[1] = {
	{      1.,   { .5,   -.5   }, { 0.,     0.  },	   0  },
};

const struct ellipsis_bs phantom_tubes[21] = {
	{{	1.,	{ .125,	.125	},	{ -0.13, -0.19 },	0.,	}, false },
	{{	1.,	{ .125,	.125	},	{ -0.45, -0.32 },	0.,	}, false },
	{{	1.,	{ .125,	.125	},	{ -0.55,  0.05 },	0.,	}, false },
	{{	1.,	{ .125,	.125	},	{ -0.37,  0.37 },	0.,	}, false },
	{{	1.,	{ .125,	.125	},	{ -0.05,  0.55 },	0.,	}, false },
	{{	1.,	{ .125,	.125	},	{  0.33,  0.40 },	0.,	}, false },
	{{	1.,	{ .125,	.125	},	{  0.53,  0.12 },	0.,	}, false },
	{{	1.,	{ .125,	.125	},	{  0.50, -0.24 },	0.,	}, false },
	{{	1.,	{ .125,	.125	},	{  0.20, -0.05 },	0.,	}, false },
	{{	1.,	{ .125,	.125	},	{  -0.11, 0.16 },	0.,	}, false },
	{{	1.,	{ .75,	.75	},	{  0.00,  0.00 },	0.,	}, true },
	{{	-1.,	{ .16,	.16	},	{ -0.13, -0.19 },	0.,	}, true },
	{{	-1.,	{ .16,	.16	},	{ -0.45, -0.32 },	0.,	}, true },
	{{	-1.,	{ .16,	.16	},	{ -0.55,  0.05 },	0.,	}, true },
	{{	-1.,	{ .16,	.16	},	{ -0.37,  0.37 },	0.,	}, true },
	{{	-1.,	{ .16,	.16	},	{ -0.05,  0.55 },	0.,	}, true },
	{{	-1.,	{ .16,	.16	},	{  0.33,  0.40 },	0.,	}, true },
	{{	-1.,	{ .16,	.16	},	{  0.53,  0.12 },	0.,	}, true },
	{{	-1.,	{ .16,	.16	},	{  0.50, -0.24 },	0.,	}, true },
	{{	-1.,	{ .16,	.16	},	{  0.20, -0.05 },	0.,	}, true },
	{{	-1.,	{ .16,	.16	},	{  -0.11, 0.16 },	0.,	}, true },
};


// NIST Phantom Geometry of T2 Sphere
//	Stupic, KF, Ainslie, M, Boss, MA, et al.
//	A standard system phantom for magnetic resonance imaging.
//	Magn Reson Med. 2021; 86: 1194– 1211. https://doi.org/10.1002/mrm.28779
const struct ellipsis_bs nist_phantom_t2[29] = {
	/*01*/	{{	1.,	{ .074,	.074		},	{ 0., 0.254*2-0.074 },		0.,	}, false },
	/*02*/	{{	1.,	{ .074,	.074		},	{ 0.096*2+0.074, 0.212*2-0.074 },	0.,	}, false },
	/*03*/	{{	1.,	{ .074,	.074		},	{ 0.176*2+0.074, 0.104*2-0.074 },	0.,	}, false },
	/*04*/	{{	1.,	{ .074,	.074		},	{ 0.176*2+0.074, -0.104*2+0.074 },	0.,	}, false },
	/*05*/	{{	1.,	{ .074,	.074		},	{ 0.096*2+0.074, -0.212*2+0.074 },	0.,	}, false },
	/*06*/	{{	1.,	{ .074,	.074		},	{ 0., -0.254*2+0.074 },	0.,	}, false },
	/*07*/	{{	1.,	{ .074,	.074		},	{ -(0.096*2+0.074), -(0.212*2-0.074) },	0.,	}, false },
	/*08*/	{{	1.,	{ .074,	.074		},	{ -(0.176*2+0.074), -(0.104*2-0.074) },	0.,	}, false },
	/*09*/	{{	1.,	{ .074,	.074		},	{ -(0.176*2+0.074), 0.104*2-0.074 },	0.,	}, false },
	/*10*/	{{	1.,	{ .074,	.074		},	{ -(0.096*2+0.074), 0.212*2-0.074 },	0.,	}, false },
	/*11*/	{{	1.,	{ .074,	.074		},	{ -(0.053*2+0.074), 0.123*2-0.074 },	0.,	}, false },
	/*12*/	{{	1.,	{ .074,	.074		},	{ 0.053*2+0.074, 0.123*2-0.074 },	0.,	}, false },
	/*13*/	{{	1.,	{ .074,	.074		},	{ 0.053*2+0.074, -0.123*2+0.074 },	0.,	}, false },
	/*14*/	{{	1.,	{ .074,	.074		},	{ -(0.053*2+0.074), -(0.123*2-0.074) },	0.,	}, false },
	/*Back*/	{{	1.,	{ .81,	.81		},	{ 0.00,  0.00 },	0.,	}, true },
	/*01*/	{{	-1.,	{ .088,	.088		},	{ 0., 0.254*2-0.074 },		0.,	}, true },
	/*02*/	{{	-1.,	{ .088,	.088		},	{ 0.096*2+0.074, 0.212*2-0.074 },	0.,	}, true },
	/*03*/	{{	-1.,	{ .088,	.088		},	{ 0.176*2+0.074, 0.104*2-0.074 },	0.,	}, true },
	/*04*/	{{	-1.,	{ .088,	.088		},	{ 0.176*2+0.074, -0.104*2+0.074 },	0.,	}, true },
	/*05*/	{{	-1.,	{ .088,	.088		},	{ 0.096*2+0.074, -0.212*2+0.074 },	0.,	}, true },
	/*06*/	{{	-1.,	{ .088,	.088		},	{ 0., -0.254*2+0.074 },	0.,	}, true },
	/*07*/	{{	-1.,	{ .088,	.088		},	{ -(0.096*2+0.074), -(0.212*2-0.074) },	0.,	}, true },
	/*08*/	{{	-1.,	{ .088,	.088		},	{ -(0.176*2+0.074), -(0.104*2-0.074) },	0.,	}, true },
	/*09*/	{{	-1.,	{ .088,	.088		},	{ -(0.176*2+0.074), 0.104*2-0.074 },	0.,	}, true },
	/*10*/	{{	-1.,	{ .088,	.088		},	{ -(0.096*2+0.074), 0.212*2-0.074 },	0.,	}, true },
	/*11*/	{{	-1.,	{ .088,	.088		},	{ -(0.053*2+0.074), 0.123*2-0.074 },	0.,	}, true },
	/*12*/	{{	-1.,	{ .088,	.088		},	{ 0.053*2+0.074, 0.123*2-0.074 },	0.,	}, true },
	/*13*/	{{	-1.,	{ .088,	.088		},	{ 0.053*2+0.074, -0.123*2+0.074 },	0.,	}, true },
	/*14*/	{{	-1.,	{ .088,	.088		},	{ -(0.053*2+0.074), -(0.123*2-0.074) },	0.,	}, true },
};


// T1 T2 Phantom based on reference phantom by Diagnostic Sonar LTD (Scotland, UK)
const struct ellipsis_bs phantom_sonar[15] = {
	/*01*/	{{	1.,	{ .135,	.135		},	        { 0., 0.45 },		                        0.,	}, false },
	/*02*/	{{	1.,	{ .135,	.135		},	{ 0.866*0.45, 0.5*0.45 },	        0.,	}, false },
	/*03*/	{{	1.,	{ .135,	.135		},	{ 0.866*0.45, -0.5*0.45 },	0.,	}, false },
	/*04*/	{{	1.,	{ .135,	.135		},	        { 0., -0.45 },	                                0.,	}, false },
	/*05*/	{{	1.,	{ .135,	.135		},	{ -0.866*0.45, -0.5*0.45 },	0.,	}, false },
	/*06*/	{{	1.,	{ .135,	.135		},	{ -0.866*0.45, 0.5*0.45 },	0.,	}, false },
	/*07 (center)*/	{{	1.,	{ .16,	.16		},	{ 0.00,  0.00 },	                        0.,	}, false },
	/*Back*/	{{	1.,	{ .75,	.75		},	{ 0.00,  0.00 },	                        0.,	}, true },
	/*01*/	{{	-1.,	{ .15,	.15		},	        { 0., 0.45 },		                        0.,	}, true },
	/*02*/	{{	-1.,	{ .15,	.15		},	{ 0.866*0.45, 0.5*0.45 },	        0.,	}, true },
	/*03*/	{{	-1.,	{ .15,	.15		},	{ 0.866*0.45, -0.5*0.45 },	0.,	}, true },
	/*04*/	{{	-1.,	{ .15,	.15		},	        { 0., -0.45 },	                                0.,	}, true },
	/*05*/	{{	-1.,	{ .15,	.15		},	{ -0.866*0.45, -0.5*0.45 },	0.,	}, true },
	/*06*/	{{	-1.,	{ .15,	.15		},	{ -0.866*0.45, 0.5*0.45 },	0.,	}, true },
	/*07 (center)*/	{{	-1.,	{ .175,	.175		},	{ 0.00,  0.00 },	                        0.,	}, true },
};


/* Magnetic Resonance in Medicine 58:430--436 (2007)
 * Three-Dimensional Analytical Magnetic Resonance
 * Imaging Phantom in the Fourier Domain
 * Cheng Guan Koay, Joelle E. Sarlls, and Evren Özarslan
 */
const struct ellipsis3d_s shepplogan3d[10] = {

	{	2.,  { .6900, .9200, .9000 }, {  .000,  .000,  .000 }, 0. },
	{	-.8, { .6624, .8740, .8800 }, {  .000,  .000,  .000 }, 0. },
	{       -.2, { .4100, .1600, .2100 }, { -.220,  .000, -.250 }, 3. * M_PI / 5. },
	{       -.2, { .3100, .1100, .2200 }, {  .220,  .000, -.250 }, 2. * M_PI / 5. },
	{        .2, { .2100, .2500, .5000 }, {  .000,  .350, -.250 }, 0.  },
	{        .2, { .0460, .0460, .0460 }, {  .000,  .100, -.250 }, 0.  },
	{        .1, { .0460, .0230, .0200 }, { -.080, -.650, -.250 }, 0.  },
	{        .1, { .0460, .0230, .0200 }, {  .060, -.650, -.250 }, M_PI / 2. },
	{        .2, { .0560, .0400, .1000 }, {  .060, -.105,  .625 }, M_PI / 2.  },
	{       -.2, { .0560, .0560, .1000 }, {  .000,  .100,  .625 }, 0.  }
};


static double sinc(double x)
{
	return (0. == x) ? 1. : (sin(x) / x);
}

static double jinc(double x)
{
	return (0. == x) ? 1. : (2. * j1(x) / x);
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
	rot2d(prot, pshift, angle);

	double radius = pow(prot[0] / axis[0], 2.) + pow(prot[1] / axis[1], 2.);

	return (radius <= 1.) ? 1. : 0.;
}

complex double kellipsis(const double center[2], const double axis[2], double angle, const double p[2])
{
	double p90[2];
	p90[0] = -p[1];
	p90[1] = p[0];

	double prot[2];
	rot2d(prot, p90, angle);

	double radius = sqrt(pow(prot[0] * axis[0], 2.) + pow(prot[1] * axis[1], 2.));

	complex double res = jinc(2. * M_PI * radius) * (axis[0] * axis[1]);

	return res * cexp(2.i * M_PI * (p90[0] * center[0] + p90[1] * center[1])) / sqrtf(2. * M_PI) * 2.;
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
	rot2d(prot, pshift, M_PI/4 + angle);

	double radius = fabs(prot[0] / axis[0] / sqrt(2)) + fabs(prot[1] / axis[1] / sqrt(2));

	return (radius <= 1.) ? 1. : 0.;
}

complex double krectangle(const double center[2], const double axis[2], double angle, const double p[2])
{
	double p90[2];
	p90[0] = -p[1];
	p90[1] = p[0];

	double prot[2];
	rot2d(prot, p90, angle);

	complex double res = sinc(2. * M_PI * prot[0] * axis[0]) * sinc(2. * M_PI * prot[1] * axis[1]) * (axis[0] * axis[1]);

	return res * cexp(2.i * M_PI * (p90[0] * center[0] + p90[1] * center[1])) / sqrtf(2. * M_PI) * 2.;
}


complex double phantom(unsigned int N, const struct ellipsis_s arr[N], const double pos[2], bool ksp)
{
	complex double res = 0.;

	for (unsigned int i = 0; i < N; i++)
		res += arr[i].intensity * (ksp ? kellipsis : xellipsis)(arr[i].center, arr[i].axis, arr[i].angle, pos);

	return res;
}

complex double phantomX(unsigned int N, const struct ellipsis_s arr[N], const double pos[2], bool ksp)
{
	complex double res = 0.;

	for (unsigned int i = 0; i < N; i++)
		res += arr[i].intensity * (ksp ? krectangle : xrectangle)(arr[i].center, arr[i].axis, arr[i].angle, pos);

	return res;
}



static double ksphere3(double x)
{
	return (0. == x) ? (1. / 3.) : ((sin(x) - x * cos(x)) / pow(x, 3.));
}

complex double xellipsis3d(const double center[3], const double axis[3], double angle, const double p[3])
{
	double p90[3];
	p90[0] = -p[1];
	p90[1] = p[0];
	p90[2] = p[2];

	double pshift[3];
	pshift[0] = p90[0] + center[0];
	pshift[1] = p90[1] + center[1];
	pshift[2] = p90[2] + center[2];
	double prot[3];
	rot2d(prot, pshift, angle);
	prot[2] = pshift[2];

	double radius = pow(prot[0] / axis[0], 2.) + pow(prot[1] / axis[1], 2.) + pow(prot[2] / axis[2], 2.);

	return (radius <= 1.) ? 1. : 0.;
}

complex double kellipsis3d(const double center[3], const double axis[3], double angle, const double p[3])
{
	double p90[3];
	p90[0] = -p[1];
	p90[1] = p[0];
	p90[2] = p[2];

	double pshift[3];
	pshift[0] = p90[0] + center[0];
	pshift[1] = p90[1] + center[1];
	pshift[2] = p90[2] + center[2];
	double prot[3];
	rot2d(prot, pshift, angle);
	prot[2] = pshift[2];

	double radius = sqrt(pow(prot[0] * axis[0], 2.) + pow(prot[1] * axis[1], 2.) + pow(prot[2] * axis[2], 2.));

	complex double res = ksphere3(2. * M_PI * radius) * (axis[0] * axis[1] * axis[2]);

	return res * cexp(2.i * M_PI * (p90[0] * center[0] + p90[1] * center[1] + p90[2] * center[2])) / sqrtf(M_PI) * sqrtf(8.);
}


complex double phantom3d(unsigned int N, const struct ellipsis3d_s arr[N], const double pos[3], bool ksp)
{
	complex double res = 0.;

	for (unsigned int i = 0; i < N; i++)
		res += arr[i].intensity * (ksp ? kellipsis3d : xellipsis3d)(arr[i].center, arr[i].axis, arr[i].angle, pos);

	return res;
}

