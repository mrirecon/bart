/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2012-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
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
#include "simu/coil.h"

#include "phantom.h"




#define MAX_COILS 8
#define COIL_COEFF 5

typedef complex float (*krn_t)(void* _data, const double mpos[3]);

static complex float xsens(unsigned int c, double mpos[3], void* data, krn_t fun)
{
	assert(c < MAX_COILS);
#if 1
	complex float val = 0.;

	long sh = (COIL_COEFF - 1) / 2;

	for (int i = 0; i < COIL_COEFF; i++)
		for (int j = 0; j < COIL_COEFF; j++)
			val += sens_coeff[c][i][j] * cexpf(-2.i * M_PI * ((i - sh) * mpos[0] + (j - sh) * mpos[1]) / 4.);
#else
	float p[3] = { mpos[0], mpos[1], mpos[2] };
	complex float val = coil(p, MAX_COILS, c);
#endif
	return val * fun(data, mpos);
}

/*
 * To simulate channels, we simply convolve with a few Fourier coefficients
 * of the sensitivities. See:
 *
 * M Guerquin-Kern, L Lejeune, KP Pruessmann, and M Unser, 
 * Realistic Analytical Phantoms for Parallel Magnetic Resonance Imaging
 * IEEE TMI 31:626-636 (2012)
 */
static complex float ksens(unsigned int c, double mpos[3], void* data, krn_t fun)
{
	assert(c < MAX_COILS);

	complex float val = 0.;

	for (int i = 0; i < COIL_COEFF; i++) {
		for (int j = 0; j < COIL_COEFF; j++) {

			long sh = (COIL_COEFF - 1) / 2;

			double mpos2[3] = { mpos[0] + (double)(i - sh) / 4.,
					    mpos[1] + (double)(j - sh) / 4.,
					    mpos[2] };

			val += sens_coeff[c][i][j] * fun(data, mpos2);
		}
	}

	return val;
}

static complex float nosens(unsigned int c, double mpos[3], void* data, krn_t fun)
{
	UNUSED(c);
	return fun(data, mpos);
}

struct data1 {

	bool sens;
	const long dims[3];
	void* data;
	krn_t fun;
};

static complex float xkernel(void* _data, const long pos[])
{
	struct data1* data = _data;

	double mpos[3] = { (double)(pos[0] - data->dims[0] / 2) / (0.5 * (double)data->dims[0]),
                           (double)(pos[1] - data->dims[1] / 2) / (0.5 * (double)data->dims[1]),
                           (double)(pos[2] - data->dims[2] / 2) / (0.5 * (double)data->dims[2]) };

	return (data->sens ? xsens : nosens)(pos[COIL_DIM], mpos, data->data, data->fun);
}

static complex float kkernel(void* _data, const long pos[])
{
	struct data1* data = _data;

	double mpos[3] = { (double)(pos[0] - data->dims[0] / 2) / 2.,
			   (double)(pos[1] - data->dims[1] / 2) / 2.,
			   (double)(pos[2] - data->dims[2] / 2) / 2. };

	return (data->sens ? ksens : nosens)(pos[COIL_DIM], mpos, data->data, data->fun);
}

struct data2 {

	const complex float* traj;
	long istrs[DIMS];
	bool sens;
	void* data;
	krn_t fun;
};

static complex float nkernel(void* _data, const long pos[])
{
	struct data2* data = _data;
	double mpos[3];
	mpos[0] = data->traj[md_calc_offset(3, data->istrs, pos) + 0] / 2.;
	mpos[1] = data->traj[md_calc_offset(3, data->istrs, pos) + 1] / 2.;
	mpos[2] = data->traj[md_calc_offset(3, data->istrs, pos) + 2] / 2.;

	return (data->sens ? ksens : nosens)(pos[COIL_DIM], mpos, data->data, data->fun);
}

struct krn_data {

	bool kspace;
	unsigned int N;
	const struct ellipsis_s* el;
};

static complex float krn(void* _data, const double mpos[3])
{
	struct krn_data* data = _data;
	return phantom(data->N, data->el, mpos, data->kspace);
}

struct krn3d_data {

	bool kspace;
	unsigned int N;
	const struct ellipsis3d_s* el;
};

static complex float krn3d(void* _data, const double mpos[3])
{
	struct krn3d_data* data = _data;
	return phantom3d(data->N, data->el, mpos, data->kspace);
}

static void sample(unsigned int N, const long dims[N], complex float* out, unsigned int D, const struct ellipsis_s* el, bool kspace)
{
	struct data1 data = {

		.sens = (dims[COIL_DIM] > 1),
		.dims = { dims[0], dims[1], dims[2] },
		.data = &(struct krn_data){ kspace, D, el },
		.fun = krn,
	};

	md_parallel_zsample(N, dims, out, &data, kspace ? kkernel : xkernel);
}


void calc_phantom(const long dims[DIMS], complex float* out, bool kspace)
{
	sample(DIMS, dims, out, 10, shepplogan_mod, kspace);
}


static void sample3d(unsigned int N, const long dims[N], complex float* out, unsigned int D, const struct ellipsis3d_s* el, bool kspace)
{
	struct data1 data = {

		.sens = (dims[COIL_DIM] > 1),
		.dims = { dims[0], dims[1], dims[2] },
		.data = &(struct krn3d_data){ kspace, D, el },
		.fun = krn3d,
	};

	md_parallel_zsample(N, dims, out, &data, kspace ? kkernel : xkernel);
}


void calc_phantom3d(const long dims[DIMS], complex float* out, bool kspace)
{
	sample3d(DIMS, dims, out, 10, shepplogan3d, kspace);
}


static void sample_noncart(const long dims[DIMS], complex float* out, const complex float* traj, unsigned int D, const struct ellipsis_s* el)
{
	struct data2 data = {

		.traj = traj,
		.sens = (dims[COIL_DIM] > 1),
		.data = &(struct krn_data){ true, D, el },
		.fun = krn,
	};

	assert(3 == dims[0]);

	long odims[DIMS];
	md_select_dims(DIMS, 2 + 4 + 8, odims, dims);

	long sdims[DIMS];
	md_select_dims(DIMS, 1 + 2 + 4, sdims, dims);
	md_calc_strides(DIMS, data.istrs, sdims, 1);

	md_parallel_zsample(DIMS, odims, out, &data, nkernel);
}


static void sample3d_noncart(const long dims[DIMS], complex float* out, const complex float* traj, unsigned int D, const struct ellipsis3d_s* el)
{
	struct data2 data = {

		.traj = traj,
		.sens = (dims[COIL_DIM] > 1),
		.data = &(struct krn3d_data){ true, D, el },
		.fun = krn3d,
	};

	assert(3 == dims[0]);

	long odims[DIMS];
	md_select_dims(DIMS, 2 + 4 + 8, odims, dims);

	long sdims[DIMS];
	md_select_dims(DIMS, 1 + 2 + 4, sdims, dims);
	md_calc_strides(DIMS, data.istrs, sdims, 1);

	md_parallel_zsample(DIMS, odims, out, &data, nkernel);
}


void calc_phantom_noncart(const long dims[DIMS], complex float* out, const complex float* traj)
{
	sample_noncart(dims, out, traj, 10, shepplogan_mod);
}

void calc_phantom3d_noncart(const long dims[DIMS], complex float* out, const complex float* traj)
{
	sample3d_noncart(dims, out, traj, 10, shepplogan3d);
}


static complex float cnst_one(void* _data, const double mpos[2])
{
	UNUSED(_data);
	UNUSED(mpos);
	return 1.;
}

void calc_sens(const long dims[DIMS], complex float* sens)
{
	struct data1 data = {

		.sens = true,
		.dims = { dims[0], dims[1], dims[2] },
		.data = NULL,
		.fun = cnst_one,
	};

	md_parallel_zsample(DIMS, dims, sens, &data, xkernel);
}




void calc_circ(const long dims[DIMS], complex float* out, bool kspace)
{
	sample(DIMS, dims, out, 1, phantom_disc, kspace);
}

void calc_circ3d(const long dims[DIMS], complex float* out, bool kspace)
{
	sample3d(DIMS, dims, out, 1, phantom_disc3d, kspace);
}

void calc_ring(const long dims[DIMS], complex float* out, bool kspace)
{
	sample(DIMS, dims, out, 4, phantom_ring, kspace);
}

void calc_moving_circ(const long dims[DIMS], complex float* out, bool kspace)
{
	struct ellipsis_s disc[1] = { phantom_disc[0] };
	disc[0].axis[0] /= 3;
	disc[0].axis[1] /= 3;

	long strs[DIMS];
	md_calc_strides(DIMS, strs, dims, sizeof(complex float));

	long dims1[DIMS];
	md_select_dims(DIMS, ~MD_BIT(TE_DIM), dims1, dims);

	for (int i = 0; i < dims[TE_DIM]; i++) {

		disc[0].center[0] = 0.5 * sin(2. * M_PI * (float)i / (float)dims[TE_DIM]);
		disc[0].center[1] = 0.5 * cos(2. * M_PI * (float)i / (float)dims[TE_DIM]);
		sample(DIMS, dims1, (void*)out + strs[TE_DIM] * i, 1, disc, kspace);
	}
}




