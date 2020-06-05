/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015-2020. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2012-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 *
 * Simple numerical phantoms that simulate image-domain or
 * k-space data with multiple channels.
 */

#include <math.h>
#include <complex.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>

#include "num/multind.h"
#include "num/loop.h"
#include "num/flpmath.h"
#include "num/splines.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "geom/logo.h"

#include "simu/sens.h"
#include "simu/coil.h"
#include "simu/shape.h"
#include "simu/shepplogan.h"

#include "phantom.h"



#define MAX_COILS 8
#define COIL_COEFF 5

typedef complex float (*krn_t)(void* _data, int s, const double mpos[3]);

static complex float xsens(int c, int s, double mpos[3], void* data, krn_t fun)
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
	complex float val = coil(&coil_defaults, p, MAX_COILS, c);
#endif
	return val * fun(data, s, mpos);
}

/*
 * To simulate channels, we simply convolve with a few Fourier coefficients
 * of the sensitivities. See:
 *
 * M Guerquin-Kern, L Lejeune, KP Pruessmann, and M Unser, 
 * Realistic Analytical Phantoms for Parallel Magnetic Resonance Imaging
 * IEEE TMI 31:626-636 (2012)
 */
static complex float ksens(int c, int s, double mpos[3], void* data, krn_t fun)
{
	assert(c < MAX_COILS);

	complex float val = 0.;

	for (int i = 0; i < COIL_COEFF; i++) {
		for (int j = 0; j < COIL_COEFF; j++) {

			long sh = (COIL_COEFF - 1) / 2;

			double mpos2[3] = { mpos[0] + (double)(i - sh) / 4.,
					    mpos[1] + (double)(j - sh) / 4.,
					    mpos[2] };

			val += sens_coeff[c][i][j] * fun(data, s, mpos2);
		}
	}

	return val;
}

static complex float nosens(int c, int s, double mpos[3], void* data, krn_t fun)
{
	UNUSED(c);
	return fun(data, s, mpos);
}

struct data {

	const complex float* traj;
	const long* tstrs;

	bool sens;
	const long dims[3];
	void* data;
	krn_t fun;
};

static complex float xkernel(void* _data, const long pos[])
{
	struct data* data = _data;

	double mpos[3] = { (double)(pos[0] - data->dims[0] / 2) / (0.5 * (double)data->dims[0]),
                           (double)(pos[1] - data->dims[1] / 2) / (0.5 * (double)data->dims[1]),
                           (double)(pos[2] - data->dims[2] / 2) / (0.5 * (double)data->dims[2]) };

	return (data->sens ? xsens : nosens)(pos[COIL_DIM], pos[COEFF_DIM], mpos, data->data, data->fun);
}

static complex float kkernel(void* _data, const long pos[])
{
	struct data* data = _data;

	double mpos[3];

	if (NULL == data->traj) {

		mpos[0] = (double)(pos[0] - data->dims[0] / 2) / 2.;
		mpos[1] = (double)(pos[1] - data->dims[1] / 2) / 2.;
		mpos[2] = (double)(pos[2] - data->dims[2] / 2) / 2.;

	} else {

		assert(0 == pos[0]);
		mpos[0] = (&MD_ACCESS(DIMS, data->tstrs, pos, data->traj))[0] / 2.;
		mpos[1] = (&MD_ACCESS(DIMS, data->tstrs, pos, data->traj))[1] / 2.;
		mpos[2] = (&MD_ACCESS(DIMS, data->tstrs, pos, data->traj))[2] / 2.;
	}

	return (data->sens ? ksens : nosens)(pos[COIL_DIM], pos[COEFF_DIM], mpos, data->data, data->fun);
}




static void sample(const long dims[DIMS], complex float* out, const long tstrs[DIMS], const complex float* traj, void* krn_data, krn_t krn, bool kspace)
{
	struct data data = {

		.traj = traj,
		.sens = (dims[COIL_DIM] > 1),
		.dims = { dims[0], dims[1], dims[2] },
		.data = krn_data,
		.tstrs = tstrs,
		.fun = krn,
	};

	md_parallel_zsample(DIMS, dims, out, &data, kspace ? kkernel : xkernel);
}


struct krn2d_data {

	bool kspace;
	bool coeff;
	int N;
	const struct ellipsis_s* el;
};

static complex float krn2d(void* _data, int s, const double mpos[3])
{
	struct krn2d_data* data = _data;

	if (data->coeff) {

		assert(s < data->N);
		return phantom(1, &data->el[s], mpos, data->kspace);

	} else {

		return phantom(data->N, data->el, mpos, data->kspace);
	}
}

static complex float krnX(void* _data, int s, const double mpos[3])
{
	struct krn2d_data* data = _data;

	if (data->coeff) {

		assert(s < data->N);
		return phantomX(1, &data->el[s], mpos, data->kspace);

	} else {

		return phantomX(data->N, data->el, mpos, data->kspace);
	}
}

struct krn3d_data {

	bool kspace;
	bool coeff;
	int N;
	const struct ellipsis3d_s* el;
};

static complex float krn3d(void* _data, int s, const double mpos[3])
{
	struct krn3d_data* data = _data;

	if (data->coeff) {

		assert(s < data->N);
		return phantom3d(1, &data->el[s], mpos, data->kspace);

	} else {

		return phantom3d(data->N, data->el, mpos, data->kspace);
	}
}


void calc_phantom(const long dims[DIMS], complex float* out, bool d3, bool kspace, const long tstrs[DIMS], const _Complex float* traj)
{
	bool coeff = (dims[COEFF_DIM] > 1);

	if (!d3)
		sample(dims, out, tstrs, traj, &(struct krn2d_data){ kspace, coeff, ARRAY_SIZE(shepplogan_mod), shepplogan_mod }, krn2d, kspace);
	else
		sample(dims, out, tstrs, traj, &(struct krn3d_data){ kspace, coeff, ARRAY_SIZE(shepplogan3d), shepplogan3d }, krn3d, kspace);
}


void calc_geo_phantom(const long dims[DIMS], complex float* out, bool kspace, int phtype, const long tstrs[DIMS], const _Complex float* traj)
{
	bool coeff = (dims[COEFF_DIM] > 1);
	complex float* round = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* angular = md_alloc(DIMS, dims, CFL_SIZE);

	switch (phtype) {

	case 1:
		sample(dims, round, tstrs, traj, &(struct krn2d_data){ kspace, coeff, ARRAY_SIZE(phantom_geo1), phantom_geo1 }, krn2d, kspace);
		sample(dims, angular, tstrs, traj, &(struct krn2d_data){ kspace, coeff, ARRAY_SIZE(phantom_geo2), phantom_geo2 }, krnX, kspace);
		md_zadd(DIMS, dims, out, round, angular);
		break;

	case 2:
		sample(dims, round, tstrs, traj, &(struct krn2d_data){ kspace, coeff, ARRAY_SIZE(phantom_geo4), phantom_geo4 }, krn2d, kspace);
		sample(dims, angular, tstrs, traj, &(struct krn2d_data){ kspace, coeff, ARRAY_SIZE(phantom_geo3), phantom_geo3 }, krnX, kspace);
		md_zadd(DIMS, dims, out, round, angular);
		break;

	default:
		assert(0);
	}

	md_free(round);
	md_free(angular);
}

static complex float cnst_one(void* _data, int s, const double mpos[2])
{
	UNUSED(_data);
	UNUSED(mpos);
	UNUSED(s);
	return 1.;
}

void calc_sens(const long dims[DIMS], complex float* sens)
{
	struct data data = {

		.traj = NULL,
		.sens = true,
		.dims = { dims[0], dims[1], dims[2] },
		.data = NULL,
		.fun = cnst_one,
	};

	md_parallel_zsample(DIMS, dims, sens, &data, xkernel);
}




void calc_circ(const long dims[DIMS], complex float* out, bool d3, bool kspace, const long tstrs[DIMS], const complex float* traj)
{
	bool coeff = (dims[COEFF_DIM] > 1);

	if (!d3)
		sample(dims, out, tstrs, traj, &(struct krn2d_data){ kspace, coeff, ARRAY_SIZE(phantom_disc), phantom_disc }, krn2d, kspace);
	else
		sample(dims, out, tstrs, traj, &(struct krn3d_data){ kspace, coeff, ARRAY_SIZE(phantom_disc3d), phantom_disc3d }, krn3d, kspace);
}

void calc_ring(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj)
{
	bool coeff = (dims[COEFF_DIM] > 1);

	sample(dims, out, tstrs, traj, &(struct krn2d_data){ kspace, coeff, ARRAY_SIZE(phantom_ring), phantom_ring }, krn2d, kspace);
}


struct moving_ellipsis_s {

	struct ellipsis_s geom;
	complex float fourier_coeff_size[2][3];
	complex float fourier_coeff_pos[2][3];
};

static complex float fourier_series(float t, unsigned int N, const complex float coeff[static N])
{
	complex float val = 0.;

	for (unsigned int i = 0; i < N; i++)
		val += coeff[i] * cexpf(2.i * M_PI * t * (float)i);

	return val;
}

static void calc_moving_discs(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj,
				int N, const struct moving_ellipsis_s disc[N])
{
	bool coeff = (dims[COEFF_DIM] > 1);

	long strs[DIMS];
	md_calc_strides(DIMS, strs, dims, sizeof(complex float));

	long dims1[DIMS];
	md_select_dims(DIMS, ~MD_BIT(TE_DIM), dims1, dims);

	for (int i = 0; i < dims[TE_DIM]; i++) {
#if 1
		struct ellipsis_s disc2[N];

		for (int j = 0; j < N; j++) {

			disc2[j] = disc[j].geom;
			disc2[j].center[0] = crealf(fourier_series(i / (float)dims[TE_DIM], 3, disc[j].fourier_coeff_pos[0]));
			disc2[j].center[1] = crealf(fourier_series(i / (float)dims[TE_DIM], 3, disc[j].fourier_coeff_pos[1]));
			disc2[j].axis[0] = crealf(fourier_series(i / (float)dims[TE_DIM], 3, disc[j].fourier_coeff_size[0]));
			disc2[j].axis[1] = crealf(fourier_series(i / (float)dims[TE_DIM], 3, disc[j].fourier_coeff_size[1]));
		}
#endif
		void* traj2 = (NULL == traj) ? NULL : ((void*)traj + i * tstrs[TE_DIM]);

		sample(dims1, (void*)out + i * strs[TE_DIM], tstrs, traj2, &(struct krn2d_data){ kspace, coeff, N, disc2 }, krn2d, kspace);
	}
}


void calc_moving_circ(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj)
{
	struct moving_ellipsis_s disc[1] = { {
			.geom = phantom_disc[0],
			.fourier_coeff_size = { { 0.3, 0., 0, }, { 0.3, 0., 0. }, },
			.fourier_coeff_pos = { { 0, 0.5, 0., }, { 0., 0.5i, 0. } },
	} };

	calc_moving_discs(dims, out, kspace, tstrs, traj, ARRAY_SIZE(disc), disc);
}


struct poly1 {

	int N;
	complex float coeff;
	double (*pg)[][2];
};

struct poly {

	bool kspace;
	bool coeff;
	int P;
	struct poly1 (*p)[];
};

static complex float krn_poly(void* _data, int s, const double mpos[3])
{
	struct poly* data = _data;

	complex float val = 0.;

	for (int p = 0; p < data->P; p++) {

		if (data->coeff && (s == p))
			continue;

		val += (*data->p)[p].coeff * (data->kspace ? kpolygon : xpolygon)((*data->p)[p].N, *(*data->p)[p].pg, mpos);
	}

	return val;
}

void calc_star(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj)
{
	bool coeff = (dims[COEFF_DIM] > 1);

	struct poly poly = {
		kspace,
		coeff,
		1,
		&(struct poly1[]){
			{
			8,
			1.,
			&(double[][2]){
				{ -0.5, -0.5 },
				{  0.0, -0.3 },
				{ +0.5, -0.5 },
				{  0.3,  0.0 },
				{ +0.5, +0.5 },
				{  0.0, +0.3 },
				{ -0.5, +0.5 },
				{ -0.3,  0.0 },
			}
			}
		}
	};

	struct data data = {

		.traj = traj,
		.tstrs = tstrs,
		.sens = (dims[COIL_DIM] > 1),
		.dims = { dims[0], dims[1], dims[2] },
		.data = &poly,
		.fun = krn_poly,
	};

	md_parallel_zsample(DIMS, dims, out, &data, kspace ? kkernel : xkernel);
}

#define ARRAY_SLICE(x, a, b) ({ __auto_type __x = &(x); assert((0 <= a) && (a < b) && (b <= ARRAY_SIZE(*__x))); ((__typeof__((*__x)[0]) (*)[b - a])&((*__x)[a])); })

void calc_bart(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj)
{
	bool coeff = (dims[COEFF_DIM] > 1);

	int N = 11 + 6 + 6 + 8 + 4 + 16 + 6 + 8 + 6 + 6;
	double points[N * 11][2];

	struct poly poly = {
		kspace,
		coeff,
		10,
		&(struct poly1[]){
			{ 11 * 11, -1., ARRAY_SLICE(points,  0 * 11, 11 * 11) },
			{  6 * 11, -1., ARRAY_SLICE(points, 11 * 11, 17 * 11) },
			{  6 * 11, -1., ARRAY_SLICE(points, 17 * 11, 23 * 11) },
			{  8 * 11, -1., ARRAY_SLICE(points, 23 * 11, 31 * 11) },
			{  4 * 11, -1., ARRAY_SLICE(points, 31 * 11, 35 * 11) },
			{ 16 * 11, -1., ARRAY_SLICE(points, 35 * 11, 51 * 11) },
			{  6 * 11, -1., ARRAY_SLICE(points, 51 * 11, 57 * 11) },
			{  8 * 11, -1., ARRAY_SLICE(points, 57 * 11, 65 * 11) },
			{  6 * 11, -1., ARRAY_SLICE(points, 65 * 11, 71 * 11) },
			{  6 * 11, -1., ARRAY_SLICE(points, 71 * 11, 77 * 11) },
		}
	};

	for (int i = 0; i < N; i++) {

		for (int j = 0; j <= 10; j++) {

			double t = j * 0.1;
			int n = i * 11 + j;

			points[n][1] = cspline(t, bart_logo[i][0]) / 250. - 0.50;
			points[n][0] = cspline(t, bart_logo[i][1]) / 250. - 0.75;
		}
	}

	struct data data = {

		.traj = traj,
		.tstrs = tstrs,
		.sens = (dims[COIL_DIM] > 1),
		.dims = { dims[0], dims[1], dims[2] },
		.data = &poly,
		.fun = krn_poly,
	};

	md_parallel_zsample(DIMS, dims, out, &data, kspace ? kkernel : xkernel);
}



void calc_phantom_arb(int N, const struct ellipsis_s data[N], const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj)
{
	bool coeff = (dims[COEFF_DIM] > 1);

	assert((!coeff) || (0 == tstrs[COEFF_DIM]));
	assert((!coeff) || (N == dims[COEFF_DIM]));

	sample(dims, out, tstrs, traj, &(struct krn2d_data){ kspace, coeff, N, data }, krn2d, kspace);
}


static void separate_bckgrd(int Nb, struct ellipsis_s bckgrd[Nb], int Nf, struct ellipsis_s frgrd[Nf], int N, const struct ellipsis_bs geometry[N])
{
	// FIXME: Do not pass unused variables

	for(int j = 0, jb = 0, jf = 0 ; j < N; j++) {

		if (geometry[j].background) {

			bckgrd[jb] = geometry[j].geo;
			jb++;

		} else {

			frgrd[jf] = geometry[j].geo;
			jf++;
		}
	}
}

void calc_phantom_tubes(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj)
{
	if (1 < dims[COEFF_DIM]) {

		// Define geometry parameter -> see src/shepplogan.c

		struct ellipsis_s tubes_bkgrd[11];
		struct ellipsis_s tubes_frgrd[10];

		assert(dims[COEFF_DIM] == ARRAY_SIZE(tubes_frgrd) + 1); // foreground + 1 background image!

		separate_bckgrd(ARRAY_SIZE(tubes_bkgrd), tubes_bkgrd, ARRAY_SIZE(tubes_frgrd), tubes_frgrd, ARRAY_SIZE(phantom_tubes), phantom_tubes);

		// Determine basis functions of the background

		long dims2[DIMS];
		md_copy_dims(DIMS, dims2, dims);
		dims2[COEFF_DIM] = ARRAY_SIZE(tubes_bkgrd);

		complex float* bkgrd = md_alloc(DIMS, dims2, CFL_SIZE);

		calc_phantom_arb(dims2[COEFF_DIM], tubes_bkgrd, dims2, bkgrd, kspace, tstrs, traj);

		// Save background to output

		md_zsum(DIMS, dims2, COEFF_FLAG, out, bkgrd);

		md_free(bkgrd);

		// Determine basis functions of the foreground

		dims2[COEFF_DIM] = ARRAY_SIZE(tubes_frgrd); // remove background

		complex float* frgrd = md_alloc(DIMS, dims2, CFL_SIZE);

		calc_phantom_arb(dims2[COEFF_DIM], tubes_frgrd, dims2, frgrd, kspace, tstrs, traj);

		// Add foreground basis functions to out

		long pos[DIMS] = { [0 ... DIMS - 1] = 0 };
		pos[COEFF_DIM] = 1;

		md_copy_block(DIMS, pos, dims, out, dims2, frgrd, CFL_SIZE);

		md_free(frgrd);

	} else { // sum up all objects

		long tdims[DIMS];
		md_copy_dims(DIMS, tdims, dims);

		tdims[COEFF_DIM] = 11;	// Number of elements of tubes phantom with rings see src/shepplogan.c

		complex float* tmp = md_alloc(DIMS, tdims, CFL_SIZE);

		calc_phantom_tubes(tdims, tmp, kspace, tstrs, traj);

		md_zsum(DIMS, tdims, COEFF_FLAG, out, tmp);
		md_free(tmp);
	}
}

