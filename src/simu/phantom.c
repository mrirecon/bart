/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2012-2013 Martin Uecker <uecker@eecs.berkeley.edu>
 *
 * Simple numerical phantom which simulates image-domain or
 * k-space data with multiple channels.
 *
 */

#include <math.h>
#include <complex.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>

#include "num/multind.h"
#include "num/loop.h"

#include "misc/misc.h"
#include "misc/mri.h"

#include "simu/shepplogan.h"
#include "simu/sens.h"

#include "phantom.h"




#define MAX_COILS 8
#define COIL_COEFF 5


static complex float xsens(void* _data, const long pos[])
{
	const long* dims = _data;

	long c = pos[COIL_DIM];
	assert(c < MAX_COILS);

	complex float val = 0.;

	double mpos[2] = { (double)(2 * pos[1] - dims[1]) / (1. * (double)dims[1]), 
                           (double)(2 * pos[2] - dims[2]) / (1. * (double)dims[2]) };

	long sh = (COIL_COEFF - 1) / 2;

	for (int i = 0; i < COIL_COEFF; i++)
		for (int j = 0; j < COIL_COEFF; j++)
			val += sens_coeff[c][i][j] * cexpf(2.i * M_PI * ((i - sh) * mpos[0] + (j - sh) * mpos[1]) / 4.);

	return val;
}


static complex float xkernel(void* _data, const long pos[])
{
	const long* dims = _data;

	double mpos[2] = { (double)(2 * pos[1] - dims[1]) / (1. * (double)dims[1]),
                           (double)(2 * pos[2] - dims[2]) / (1. * (double)dims[2]) };

	return phantom(10, shepplogan_mod, mpos, false);
}

static complex float xkernel2(void* _data, const long pos[])
{
	return xkernel(_data, pos) * xsens(_data, pos);
}


static complex float kkernel(void* _data, const long pos[])
{
	const long* dims = _data;

	double mpos[2] = { (double)(2 * pos[1] - dims[1]) / 4., 
			   (double)(2 * pos[2] - dims[2]) / 4. };

	return phantom(10, shepplogan_mod, mpos, true);
}

static complex float kkernel2(void* _data, const long pos[])
{
	const long* dims = _data;
	complex float val = 0.;

	long c = pos[COIL_DIM];

	for (int i = 0; i < COIL_COEFF; i++) {
		for (int j = 0; j < COIL_COEFF; j++) {

			long sh = (COIL_COEFF - 1) / 2;

			double mpos[2] = { (double)(2 * pos[1] + (i - sh) - dims[1]) / 4.,
					   (double)(2 * pos[2] + (j - sh) - dims[2]) / 4. };

			val += sens_coeff[c][i][j] * phantom(10, shepplogan_mod, mpos, true);
		}
	}

	return val;
}


void calc_phantom(const long dims[DIMS], complex float* out, _Bool kspace)
{
	md_zsample(DIMS, dims, out, (void*)dims, (1 == dims[COIL_DIM]) ? (kspace ? kkernel : xkernel)
								       : (kspace ? kkernel2 : xkernel2));
}



struct data2 {

	const complex float* traj;
	long istrs[DIMS];
};

static complex float kkernel_nc(void* _data, const long pos[])
{
	struct data2* data = _data;
	double mpos[3];
	mpos[0] = data->traj[md_calc_offset(3, data->istrs, pos) + 0] / 2.;
	mpos[1] = data->traj[md_calc_offset(3, data->istrs, pos) + 1] / 2.;
//	mpos[2] = data->traj[md_calc_offset(3, data->istrs, pos) + 2];

	return phantom(10, shepplogan_mod, mpos, true);
}


/*
 * To simulate channels, we simply convovle with a few Fourier coefficients
 * for sensitivities. See:
 *
 * M Guerquin-Kern, L Lejeune, KP Pruessmann, and M Unser, 
 * Realistic Analytical Phantoms for Parallel Magnetic Resonance Imaging
 * IEEE TMI 31:626-636 (2012)
 */
static complex float kkernel_nc2(void* _data, const long pos[])
{
	struct data2* data = _data;

	complex float val = 0.;

	long c = pos[COIL_DIM];

	for (int i = 0; i < COIL_COEFF; i++) {
		for (int j = 0; j < COIL_COEFF; j++) {

			long sh = (COIL_COEFF - 1) / 2;

			double mpos[3];
			mpos[0] = ((i - sh) + data->traj[md_calc_offset(DIMS, data->istrs, pos) + 0]) / 2.;
			mpos[1] = ((j - sh) + data->traj[md_calc_offset(DIMS, data->istrs, pos) + 1]) / 2.;

			val += sens_coeff[c][i][j] * phantom(10, shepplogan_mod, mpos, true);
		}
	}

	return val;
}


void calc_phantom_noncart(const long dims[DIMS], complex float* out, const complex float* traj)
{
	struct data2 data;
	data.traj = traj;

	assert(3 == dims[0]);

	long odims[DIMS];
	md_select_dims(DIMS, 2 + 4 + 8, odims, dims);

	long sdims[DIMS];
	md_select_dims(DIMS, 1 + 2, sdims, dims);
	md_calc_strides(DIMS, data.istrs, sdims, 1);

	md_zsample(DIMS, odims, out, &data, (dims[COIL_DIM] == 1) ? kkernel_nc : kkernel_nc2);
}



void calc_sens(const long dims[DIMS], complex float* sens)
{
	md_zsample(DIMS, dims, sens, (void*)dims, xsens);
}



static complex float xdisc(void* _data, const long pos[])
{
	const long* dims = _data;

	double mpos[2] = { (double)(2 * pos[1] - dims[1]) / (1. * (double)dims[1]),
                           (double)(2 * pos[2] - dims[2]) / (1. * (double)dims[2]) };

	return phantom(1, phantom_disc, mpos, false);
}


static complex float kdisc(void* _data, const long pos[])
{
	const long* dims = _data;

	double mpos[2] = { (double)(2 * pos[1] - dims[1]) / 4.,
			   (double)(2 * pos[2] - dims[2]) / 4. };

	return phantom(1, phantom_disc, mpos, true);
}



void calc_circ(const long dims[DIMS], complex float* out, _Bool kspace)
{
	md_zsample(DIMS, dims, out, (void*)dims, (kspace ? kdisc : xdisc));
}


