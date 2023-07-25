/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015-2020. Martin Uecker.
 * Copyright 2021-2023. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
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
#include "num/rand.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "geom/logo.h"
#include "geom/brain.h"

#include "simu/sens.h"
#include "simu/coil.h"
#include "simu/shape.h"
#include "simu/shepplogan.h"

#include "phantom.h"



#define MAX_COILS 64
#define COIL_COEFF 5

struct pha_opts pha_opts_defaults = {

	.stype = DEFAULT,
};


struct krn2d_data {

	bool kspace;
	bool coeff;
	enum coil_type stype;
	int N;
	const struct ellipsis_s* el;
};

typedef complex float (*krn_t)(void* _data, int s, const double mpos[3]);

static complex float xsens(int c, int s, double mpos[3], void* data, krn_t fun)
{
	assert(c < MAX_COILS);
#if 1
	struct krn2d_data* krn_data = data;

	complex float val = 0.;

	long sh = (COIL_COEFF - 1) / 2;

	for (int i = 0; i < COIL_COEFF; i++)
		for (int j = 0; j < COIL_COEFF; j++) {

			switch (krn_data->stype) {

			case HEAD_3D_64CH:

				for (int m = 0; m < COIL_COEFF; m++)
					val += sens64_coeff[c][i][j][m] * cexpf(-2.i * M_PI * ((i - sh) * mpos[0] + (j - sh) * mpos[1] + (m - sh) * mpos[2]) / 4.);

				break;

			case HEAD_2D_8CH:

				val += sens_coeff[c][i][j] * cexpf(-2.i * M_PI * ((i - sh) * mpos[0] + (j - sh) * mpos[1]) / 4.);
				break;

			case DEFAULT:
			default:

				error("Please specify a sensitivity type.");
				break;
			}
		}

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

	struct krn2d_data* krn_data = data;

	complex float val = 0.;

	for (int i = 0; i < COIL_COEFF; i++) {
		for (int j = 0; j < COIL_COEFF; j++) {

			long sh = (COIL_COEFF - 1) / 2;

			switch (krn_data->stype) {

			case HEAD_3D_64CH:

				for (int m = 0; m < COIL_COEFF; m++) {

					double mpos2[3] = { mpos[0] + (double)(i - sh) / 4.,
						mpos[1] + (double)(j - sh) / 4.,
						mpos[2] + (double)(m - sh) / 4.};

					val += sens64_coeff[c][i][j][m] * fun(data, s, mpos2);
				}

				break;

			case HEAD_2D_8CH: ;

				double mpos2[3] = { mpos[0] + (double)(i - sh) / 4.,
					mpos[1] + (double)(j - sh) / 4.,
					mpos[2] };

				val += sens_coeff[c][i][j] * fun(data, s, mpos2);

				break;

			case DEFAULT:
			default:

				error("Please specify a sensitivity type.");
				break;
			}
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


static void my_sample(const long dims[DIMS], complex float* out, void* data, complex float (*krn)(void* data, const long pos[]))
{
	NESTED(complex float, kernel, (const long pos[]))
	{
		return krn(data, pos);
	};

	md_parallel_zsample(DIMS, dims, out, kernel);
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

	my_sample(dims, out, &data, kspace ? kkernel : xkernel);
}

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
	enum coil_type stype;
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


void calc_phantom(const long dims[DIMS], complex float* out, bool d3, bool kspace, const long tstrs[DIMS], const _Complex float* traj, struct pha_opts* popts)
{
	bool coeff = (dims[COEFF_DIM] > 1);

	if (!d3)
		sample(dims, out, tstrs, traj, &(struct krn2d_data){ kspace, coeff, popts->stype, ARRAY_SIZE(shepplogan_mod), shepplogan_mod }, krn2d, kspace);
	else
		sample(dims, out, tstrs, traj, &(struct krn3d_data){ kspace, coeff, popts->stype, ARRAY_SIZE(shepplogan3d), shepplogan3d }, krn3d, kspace);
}


void calc_geo_phantom(const long dims[DIMS], complex float* out, bool kspace, int phtype, const long tstrs[DIMS], const _Complex float* traj, struct pha_opts* popts)
{
	bool coeff = (dims[COEFF_DIM] > 1);

	complex float* round = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* angular = md_alloc(DIMS, dims, CFL_SIZE);

	switch (phtype) {

	case 1:
		sample(dims, round, tstrs, traj, &(struct krn2d_data){ kspace, coeff, popts->stype, ARRAY_SIZE(phantom_geo1), phantom_geo1 }, krn2d, kspace);
		sample(dims, angular, tstrs, traj, &(struct krn2d_data){ kspace, coeff, popts->stype, ARRAY_SIZE(phantom_geo2), phantom_geo2 }, krnX, kspace);
		md_zadd(DIMS, dims, out, round, angular);
		break;

	case 2:
		sample(dims, round, tstrs, traj, &(struct krn2d_data){ kspace, coeff, popts->stype, ARRAY_SIZE(phantom_geo4), phantom_geo4 }, krn2d, kspace);
		sample(dims, angular, tstrs, traj, &(struct krn2d_data){ kspace, coeff, popts->stype, ARRAY_SIZE(phantom_geo3), phantom_geo3 }, krnX, kspace);
		md_zadd(DIMS, dims, out, round, angular);
		break;

	case 3:
		sample(dims, out, tstrs, traj, &(struct krn2d_data){ kspace, coeff, popts->stype, ARRAY_SIZE(phantom_geo5), phantom_geo5 }, krnX, kspace);
		break;

	default:
		assert(0);
	}

	md_free(round);
	md_free(angular);
}

static complex float cnst_one(void* _data, int s, const double mpos[3])
{
	UNUSED(_data);
	UNUSED(mpos);
	UNUSED(s);
	return 1.;
}

void calc_sens(const long dims[DIMS], complex float* sens, struct pha_opts* popts)
{
	struct data data = {

		.traj = NULL,
		.sens = true,
		.dims = { dims[0], dims[1], dims[2] },
		.data = &(struct krn2d_data){ false, false, popts->stype, DIMS, NULL },
		.fun = cnst_one,
	};

	my_sample(dims, sens, &data, xkernel);
}




void calc_circ(const long dims[DIMS], complex float* out, bool d3, bool kspace, const long tstrs[DIMS], const complex float* traj, struct pha_opts* popts)
{
	bool coeff = (dims[COEFF_DIM] > 1);

	if (!d3)
		sample(dims, out, tstrs, traj, &(struct krn2d_data){ kspace, coeff, popts->stype, ARRAY_SIZE(phantom_disc), phantom_disc }, krn2d, kspace);
	else
		sample(dims, out, tstrs, traj, &(struct krn3d_data){ kspace, coeff, popts->stype, ARRAY_SIZE(phantom_disc3d), phantom_disc3d }, krn3d, kspace);
}

void calc_ring(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, struct pha_opts* popts)
{
	bool coeff = (dims[COEFF_DIM] > 1);

	sample(dims, out, tstrs, traj, &(struct krn2d_data){ kspace, coeff, popts->stype, ARRAY_SIZE(phantom_ring), phantom_ring }, krn2d, kspace);
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
				int N, const struct moving_ellipsis_s disc[N], struct pha_opts* popts)
{
	bool coeff = (dims[COEFF_DIM] > 1);

	long strs[DIMS];
	md_calc_strides(DIMS, strs, dims, sizeof(complex float));

	long dims1[DIMS];
	md_select_dims(DIMS, ~MD_BIT(TIME_DIM), dims1, dims);

	for (int i = 0; i < dims[TIME_DIM]; i++) {

		struct ellipsis_s disc2[N];

		for (int j = 0; j < N; j++) {

			disc2[j] = disc[j].geom;
			disc2[j].center[0] = crealf(fourier_series(i / (float)dims[TIME_DIM], 3, disc[j].fourier_coeff_pos[0]));
			disc2[j].center[1] = crealf(fourier_series(i / (float)dims[TIME_DIM], 3, disc[j].fourier_coeff_pos[1]));
			disc2[j].axis[0] = crealf(fourier_series(i / (float)dims[TIME_DIM], 3, disc[j].fourier_coeff_size[0]));
			disc2[j].axis[1] = crealf(fourier_series(i / (float)dims[TIME_DIM], 3, disc[j].fourier_coeff_size[1]));
		}

		void* traj2 = (NULL == traj) ? NULL : ((void*)traj + i * tstrs[TIME_DIM]);

		sample(dims1, (void*)out + i * strs[TIME_DIM], tstrs, traj2, &(struct krn2d_data){ kspace, coeff, popts->stype, N, disc2 }, krn2d, kspace);
	}
}


void calc_moving_circ(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, struct pha_opts* popts)
{
	struct moving_ellipsis_s disc[1] = { {
			.geom = phantom_disc[0],
			.fourier_coeff_size = { { 0.3, 0., 0, }, { 0.3, 0., 0. }, },
			.fourier_coeff_pos = { { 0, 0.5, 0., }, { 0., 0.5i, 0. } },
	} };

	calc_moving_discs(dims, out, kspace, tstrs, traj, ARRAY_SIZE(disc), disc, popts);
}


struct poly1 {

	int N;
	complex float coeff;
	double (*pg)[][2];
};

struct poly {

	bool kspace;
	bool coeff;
	enum coil_type stype;
	int P;
	struct poly1 (*p)[];
};

static complex float krn_poly(void* _data, int s, const double mpos[3])
{
	struct poly* data = _data;

	complex float val = 0.;

	if (data->coeff)
		// Constrain to 1 allows geometry separation by intensity
		val = (*data->p)[s].coeff / cabsf((*data->p)[s].coeff) * (data->kspace ? kpolygon : xpolygon)((*data->p)[s].N, *(*data->p)[s].pg, mpos);
	else
		for (int p = 0; p < data->P; p++)
			val += (*data->p)[p].coeff * (data->kspace ? kpolygon : xpolygon)((*data->p)[p].N, *(*data->p)[p].pg, mpos);

	return val;
}

void calc_star(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, struct pha_opts* popts)
{
	bool coeff = (dims[COEFF_DIM] > 1);

	struct poly poly = {
		kspace,
		coeff,
		popts->stype,
		1,
		&(struct poly1[]){ { 8, 1.,
				&(double[][2]){
					{ -0.5, -0.5 },
					{  0.0, -0.3 },
					{ +0.5, -0.5 },
					{  0.3,  0.0 },
					{ +0.5, +0.5 },
					{  0.0, +0.3 },
					{ -0.5, +0.5 },
					{ -0.3,  0.0 },
				} } }
	};

	struct data data = {

		.traj = traj,
		.tstrs = tstrs,
		.sens = (dims[COIL_DIM] > 1),
		.dims = { dims[0], dims[1], dims[2] },
		.data = &poly,
		.fun = krn_poly,
	};

	my_sample(dims, out, &data, kspace ? kkernel : xkernel);
}

#define ARRAY_SLICE(x, a, b) ({ __auto_type __x = &(x); assert((0 <= a) && (a < b) && (b <= ARRAY_SIZE(*__x))); ((__typeof__((*__x)[0]) (*)[b - a])&((*__x)[a])); })

static void calc_bart2(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, struct pha_opts* popts)
{
	bool coeff = (dims[COEFF_DIM] > 1);

	int N = 11 + 6 + 6 + 8 + 4 + 16 + 6 + 8 + 6 + 6;
	double points[N * 11][2];

	struct poly poly = {
		kspace,
		coeff,
		popts->stype,
		10,
		&(struct poly1[]){
			{ 11 * 11, -1., ARRAY_SLICE(points,  0 * 11, 11 * 11) }, // B background
			{  6 * 11, -1., ARRAY_SLICE(points, 11 * 11, 17 * 11) }, // B hole 1
			{  6 * 11, -1., ARRAY_SLICE(points, 17 * 11, 23 * 11) }, // B hole 1
			{  8 * 11, -1., ARRAY_SLICE(points, 23 * 11, 31 * 11) }, // A background
			{  4 * 11, -1., ARRAY_SLICE(points, 31 * 11, 35 * 11) }, // A hole
			{ 16 * 11, -1., ARRAY_SLICE(points, 35 * 11, 51 * 11) }, // R background
			{  6 * 11, -1., ARRAY_SLICE(points, 51 * 11, 57 * 11) }, // R hole
			{  8 * 11, -1., ARRAY_SLICE(points, 57 * 11, 65 * 11) }, // T
			{  6 * 11, -1., ARRAY_SLICE(points, 65 * 11, 71 * 11) }, // bracket left
			{  6 * 11, -1., ARRAY_SLICE(points, 71 * 11, 77 * 11) }, // bracket right
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

	sample(dims, out, tstrs, traj, &poly, krn_poly, kspace);
}


static void combine_geom_components(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj,
		void (*fun)(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, struct pha_opts* popts),
		const int N_all, const int N_reduc, const int components[N_reduc], struct pha_opts* popts)
{
	if (1 < dims[COEFF_DIM]) {

		assert(N_reduc == dims[COEFF_DIM]);

		// Create full length phantom with all components of geometry

		long full_dims[DIMS];
		md_copy_dims(DIMS, full_dims, dims);
		full_dims[COEFF_DIM] = N_all;

		complex float* full_pha = md_alloc(DIMS, full_dims, CFL_SIZE);

		fun(full_dims, full_pha, kspace, tstrs, traj, popts);

		// Sum up individual components of all objects in the phantom

		long tmp_pos[DIMS] = { [0 ... DIMS - 1] = 0 };
		long out_pos[DIMS] = { [0 ... DIMS - 1] = 0 };

		long map_dims[DIMS];
		md_select_dims(DIMS, ~COEFF_FLAG, map_dims, dims);

		complex float* tmp_map = md_alloc(DIMS, map_dims, CFL_SIZE);

		long tmp_dims[DIMS];
		md_select_dims(DIMS, ~COEFF_FLAG, tmp_dims, dims);

		for (int i = 0; i < dims[COEFF_DIM]; i++) {

			tmp_dims[COEFF_DIM] = components[i];

			complex float* tmp = md_alloc(DIMS, tmp_dims, CFL_SIZE);

			// Extract block of components belonging to individual objects

			md_copy_block(DIMS, tmp_pos, tmp_dims, tmp, full_dims, full_pha, CFL_SIZE);

			// Sum up components belonging to individual object

			md_zsum(DIMS, tmp_dims, COEFF_FLAG, tmp_map, tmp);

			md_free(tmp);

			tmp_pos[COEFF_DIM] += components[i];

			// Copy object to output

			md_copy_block(DIMS, out_pos, dims, out, map_dims, tmp_map, CFL_SIZE);

			out_pos[COEFF_DIM] += 1;
		}

		md_free(tmp_map);
		md_free(full_pha);

	} else {

		fun(dims, out, kspace, tstrs, traj, popts);
	}
}


void calc_bart(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, struct pha_opts* popts)
{
	const int N_all = 10;		// There are overall 10 geometric components in the BART logo
	const int N_reduc = 6;		// But the BART logo consists of only 6 characters: B, A, R, T, _, _

	const int components[] = { 3, 2, 2, 1, 1, 1 }; // Defines how many geometric components reduce to a single character

	combine_geom_components(dims, out, kspace, tstrs, traj, calc_bart2, N_all, N_reduc, components, popts);
}


static void calc_brain2(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, struct pha_opts* popts)
{
	bool coeff = (dims[COEFF_DIM] > 1);

	int N = 3897;
	double points[N * 11][2];

	struct poly poly = {
		kspace,
		coeff,
		popts->stype,
		64,
		&(struct poly1[]){
			{ 55 , 1, ARRAY_SLICE(points, 0, 55) },
			{ 66 , 1, ARRAY_SLICE(points, 55, 121) },
			{ 44 , 1, ARRAY_SLICE(points, 121, 165) },
			{ 2596 , -1, ARRAY_SLICE(points, 165, 2761) },
			{ 902 , 1, ARRAY_SLICE(points, 2761, 3663) },
			{ 187 , 1, ARRAY_SLICE(points, 3663, 3850) },
			{ 132 , 1, ARRAY_SLICE(points, 3850, 3982) },
			{ 286 , -1, ARRAY_SLICE(points, 3982, 4268) },
			{ 44 , 1, ARRAY_SLICE(points, 4268, 4312) },
			{ 495 , -1, ARRAY_SLICE(points, 4312, 4807) },
			{ 110 , 2, ARRAY_SLICE(points, 4807, 4917) },
			{ 154 , 2, ARRAY_SLICE(points, 4917, 5071) },
			{ 143 , 2, ARRAY_SLICE(points, 5071, 5214) },
			{ 165 , 2, ARRAY_SLICE(points, 5214, 5379) },
			{ 110 , 2, ARRAY_SLICE(points, 5379, 5489) },
			{ 110 , 2, ARRAY_SLICE(points, 5489, 5599) },
			{ 88 , 2, ARRAY_SLICE(points, 5599, 5687) },
			{ 154 , -2, ARRAY_SLICE(points, 5687, 5841) },
			{ 561 , -2, ARRAY_SLICE(points, 5841, 6402) },
			{ 7656 , 2, ARRAY_SLICE(points, 6402, 14058) },
			{ 286 , 2, ARRAY_SLICE(points, 14058, 14344) },
			{ 187 , -2, ARRAY_SLICE(points, 14344, 14531) },
			{ 462 , -2, ARRAY_SLICE(points, 14531, 14993) },
			{ 121 , 2, ARRAY_SLICE(points, 14993, 15114) },
			{ 209 , 2, ARRAY_SLICE(points, 15114, 15323) },
			{ 88 , 2, ARRAY_SLICE(points, 15323, 15411) },
			{ 88 , 2, ARRAY_SLICE(points, 15411, 15499) },
			{ 176 , 2, ARRAY_SLICE(points, 15499, 15675) },
			{ 88 , 2, ARRAY_SLICE(points, 15675, 15763) },
			{ 88 , 2, ARRAY_SLICE(points, 15763, 15851) },
			{ 132 , 2, ARRAY_SLICE(points, 15851, 15983) },
			{ 66 , 2, ARRAY_SLICE(points, 15983, 16049) },
			{ 88 , -2, ARRAY_SLICE(points, 16049, 16137) },
			{ 121 , 2, ARRAY_SLICE(points, 16137, 16258) },
			{ 286 , 2, ARRAY_SLICE(points, 16258, 16544) },
			{ 110 , 3, ARRAY_SLICE(points, 16544, 16654) },
			{ 154 , 3, ARRAY_SLICE(points, 16654, 16808) },
			{ 132 , 3, ARRAY_SLICE(points, 16808, 16940) },
			{ 154 , 3, ARRAY_SLICE(points, 16940, 17094) },
			{ 110 , 3, ARRAY_SLICE(points, 17094, 17204) },
			{ 484 , -3, ARRAY_SLICE(points, 17204, 17688) },
			{ 110 , 3, ARRAY_SLICE(points, 17688, 17798) },
			{ 88 , 3, ARRAY_SLICE(points, 17798, 17886) },
			{ 154 , 3, ARRAY_SLICE(points, 17886, 18040) },
			{ 7656 , -3, ARRAY_SLICE(points, 18040, 25696) },
			{ 2596 , 3, ARRAY_SLICE(points, 25696, 28292) },
			{ 187 , 3, ARRAY_SLICE(points, 28292, 28479) },
			{ 297 , 3, ARRAY_SLICE(points, 28479, 28776) },
			{ 110 , 3, ARRAY_SLICE(points, 28776, 28886) },
			{ 198 , 3, ARRAY_SLICE(points, 28886, 29084) },
			{ 88 , 3, ARRAY_SLICE(points, 29084, 29172) },
			{ 88 , 3, ARRAY_SLICE(points, 29172, 29260) },
			{ 5676 , -3, ARRAY_SLICE(points, 29260, 34936) },
			{ 176 , 3, ARRAY_SLICE(points, 34936, 35112) },
			{ 88 , 3, ARRAY_SLICE(points, 35112, 35200) },
			{ 88 , 3, ARRAY_SLICE(points, 35200, 35288) },
			{ 132 , 3, ARRAY_SLICE(points, 35288, 35420) },
			{ 66 , 3, ARRAY_SLICE(points, 35420, 35486) },
			{ 88 , 3, ARRAY_SLICE(points, 35486, 35574) },
			{ 110 , 3, ARRAY_SLICE(points, 35574, 35684) },
			{ 484 , 4, ARRAY_SLICE(points, 35684, 36168) },
			{ 561 , 4, ARRAY_SLICE(points, 36168, 36729) },
			{ 462 , 4, ARRAY_SLICE(points, 36729, 37191) },
			{ 5676 , 4, ARRAY_SLICE(points, 37191, 42867) },
		}
	};

	for (int i = 0; i < N; i++) {

		for (int j = 0; j <= 10; j++) {

			double t = j * 0.1;
			int n = i * 11 + j;

			// Scale geometry with 1.04 to match underlying raw data
			points[n][1] = +1.04 * cspline(t, brain_geom[i][0]);
			points[n][0] = -1.04 * cspline(t, brain_geom[i][1]); // -1 to reflect over y axis
		}
	}

	sample(dims, out, tstrs, traj, &poly, krn_poly, kspace);
}

void calc_brain(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, struct pha_opts* popts)
{
	enum { N_all = 64 };		// 64 overall geometric components in the brain geometry
	enum { N_reduc = 4 };		// Combine them to 4

	const int components[N_reduc] = { 10, 25, 25, 4 }; // Defines how many geometric components reduce to a single object

	combine_geom_components(dims, out, kspace, tstrs, traj, calc_brain2, N_all, N_reduc, components, popts);
}


#define ARRAY_SLICE2(x, a, b) ({ __auto_type __x = &(x); ((__typeof__((*__x)[0]) (*)[b - a])&((*__x)[a])); })

void calc_cfl_geom(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, int N_max, int D_max, long hdims[N_max][D_max], complex float* x[N_max], struct pha_opts* popts)
{
	bool coeff = (dims[COEFF_DIM] > 1);

	long cstrs[DIMS] = { 0 };
	md_calc_strides(DIMS, cstrs, hdims[0], sizeof(complex float));

	long mstrs[DIMS] = { 0 };
	md_calc_strides(DIMS, mstrs, hdims[1], sizeof(complex float));

	int N = hdims[0][0];

	double data[N][2][4];

	long pos[DIMS];
	md_copy_dims(DIMS, pos, hdims[0]);

	for (int s = 0; s < hdims[0][0]; s++) {
		for (int c = 0; c < hdims[0][1]; c++) {
			for (int p = 0; p < hdims[0][2]; p++) {

				pos[0] = s;
				pos[1] = c;
				pos[2] = p;

				long ind = md_calc_offset(DIMS, cstrs, pos) / CFL_SIZE;

				data[s][c][p] = x[0][ind];
			}
		}
	}

	double points[N * 11][2];

	long point_index = 0;

	struct poly1 paths[hdims[1][0]];

	md_copy_dims(DIMS, pos, hdims[1]);

	for (int i = 0; i < hdims[1][0]; i++) {

		pos[0] = i;
		pos[1] = 1;

		long ind_cp = md_calc_offset(DIMS, mstrs, pos) / CFL_SIZE;

		pos[1] = 2;
		long ind_color = md_calc_offset(DIMS, mstrs, pos) / CFL_SIZE;

		int cp = (int) cabsf(x[1][ind_cp]);
		int color = cabsf(x[1][ind_color]);

		struct poly1 tmp = { cp * 11, color, ARRAY_SLICE2(points, point_index * 11, (point_index + cp) * 11) };

		paths[i] = tmp;

		point_index += cp;
	}

	struct poly poly = {

		kspace,
		coeff,
		popts->stype,
		hdims[1][0],
		(struct poly1 (*)[])paths,
	};

	for (int i = 0; i < N; i++) {

		for (int j = 0; j <= 10; j++) {

			double t = j * 0.1;
			int n = i * 11 + j;

			points[n][1] = cspline(t, data[i][1]);
			points[n][0] = cspline(t, data[i][0]);
		}
	}

	sample(dims, out, tstrs, traj, &poly, krn_poly, kspace);
}


void calc_phantom_arb(int N, const struct ellipsis_s* data /*[N]*/, const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj, float rotation_angle, struct pha_opts* popts)
{
	bool coeff = (dims[COEFF_DIM] > 1);

	assert((!coeff) || (0 == tstrs[COEFF_DIM]));
	assert((!coeff) || (N == dims[COEFF_DIM]));

	long strs[DIMS];
	md_calc_strides(DIMS, strs, dims, sizeof(complex float));

	long dims1[DIMS];
	md_select_dims(DIMS, ~MD_BIT(TIME_DIM), dims1, dims);

	for (int i = 1; i < dims[TIME_DIM] + 1; i++) {

		struct ellipsis_s data2[N];

		complex float position = 0.;

		for (int j = 0; j < N; j++) {

			position = (data[j].center[0] + data[j].center[1]*I) * cexpf(-2.i * M_PI * rotation_angle / 360. * (float)i);

			data2[j] = data[j];
			data2[j].center[0] = crealf(position);
			data2[j].center[1] = cimagf(position);
			data2[j].axis[0] = data[j].axis[0];
			data2[j].axis[1] = data[j].axis[1];
		}

		void* traj2 = (NULL == traj) ? NULL : ((void*)traj + (i - 1) * tstrs[TIME_DIM]);

		sample(dims1, (void*)out + (i - 1) * strs[TIME_DIM], tstrs, traj2, &(struct krn2d_data){ kspace, coeff, popts->stype, N, data2 }, krn2d, kspace);
	}
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


static bool circ_intersection(float s1, float px1, float py1, float s2, float px2, float py2)
{
	float dist2 = (px1 - px2) * (px1 - px2) + (py1 - py2) * (py1 - py2);

	if (dist2 < (s1 + s2) * (s1 + s2))
		return true;

	return false;
}

static bool circ_in_background(float s1, float px1, float py1, float s2, float px2, float py2)
{
	float dist2 = (px1 - px2) * (px1 - px2) + (py1 - py2) * (py1 - py2);

	if (sqrtf(dist2) + s1 < s2)
		return true;

	return false;
}

void calc_phantom_tubes(const long dims[DIMS], complex float* out, bool kspace, bool random, float rotation_angle, int N, const long tstrs[DIMS], const complex float* traj, struct pha_opts* popts)
{
	if (1 < dims[COEFF_DIM]) {

		struct ellipsis_bs phantom_tubes_N[2 * N - 1];

		if (random) {

			unsigned int max_count = 10000;

			// background circle position and radius
			float sx_bg = .9;
			float px_bg = 0.;
			float py_bg = 0.;

			// min and max tube radius (0. to 1.)
			float smin = .025;
			float smax = .4;

			// min and max center position (-1. to 1.)
			float pmin = -1.;
			float pmax = 1.;

			// dead zone between tubes
			float edge_scale = 1.2;

			// generate random ellipse
			for (int i = 0; i < 2 * N - 2; i+=2) {

				double sx = 0;
				double px = 0;
				double py = 0;

				bool overlap = true;

				unsigned int total_count = 0;
				unsigned int count = 0;

				while (overlap) {

					if (count > max_count) {
					
						// shrink ellipse
						smax *= .95;
						smin *= .95;
						total_count += count;
						count = 0;
					}

					sx = smin +  (smax - smin) * uniform_rand();
					px = pmin + (pmax - pmin) * uniform_rand();
					py = pmin + (pmax - pmin) * uniform_rand();

					// check that ellipse fits within background circle
					overlap = !circ_in_background(edge_scale * sx, px, py, sx_bg, px_bg, py_bg);

					// check that new ellipse does not overlap with existing ellipses
					// FIXME: change from circle to ellipse intersection
					if ((i > 0) && !overlap) {

						for (int j = 1; j < i; j+=2) {

							float sx2 = phantom_tubes_N[j].geo.axis[0];
							float px2 = phantom_tubes_N[j].geo.center[0];
							float py2 = phantom_tubes_N[j].geo.center[1];

							overlap = circ_intersection(edge_scale * sx, px, py, sx2, px2, py2);

							if (overlap)
								break;
						}
					}

					count++;
					
					if (total_count > 100 * max_count)
						error("Could not fit tube in phantom (requested %d, stopped at %d after %d trials\n", N, i/2, total_count);
				}

				debug_printf(DP_DEBUG4, "i=%d, (%f, %f), (%f, %f)\n", i, sx, sx, px, py);

				struct ellipsis_bs ebs = { { 1., { sx, sx }, { px, py }, 0.}, false };
				phantom_tubes_N[i] = ebs;

				struct ellipsis_bs ebs2 = { { -1., { edge_scale * sx, edge_scale * sx }, { px, py }, 0.}, true };
				phantom_tubes_N[i + 1] = ebs2;
			}

			struct ellipsis_bs ebsb = { { 1., { sx_bg, sx_bg }, { px_bg, py_bg }, 0. }, true };
			phantom_tubes_N[2 * N - 2] = ebsb;

		} else {

			assert((8 == N) || (11 == N) || (15 == N));

			for (int i = 0; i < 2 * N - 1; i++)
				phantom_tubes_N[i] = (8 == N ? phantom_sonar : (15 == N ? nist_phantom_t2 : phantom_tubes))[i];
		}


		// Define geometry parameter -> see src/shepplogan.c

		struct ellipsis_s tubes_bkgrd[N];
		struct ellipsis_s tubes_frgrd[N-1];

		assert(dims[COEFF_DIM] == (unsigned int)ARRAY_SIZE(tubes_frgrd) + 1); // foreground + 1 background image!

		separate_bckgrd(ARRAY_SIZE(tubes_bkgrd), tubes_bkgrd, ARRAY_SIZE(tubes_frgrd), tubes_frgrd, ARRAY_SIZE(phantom_tubes_N), phantom_tubes_N);

		// Determine basis functions of the background

		long dims2[DIMS];
		md_copy_dims(DIMS, dims2, dims);
		dims2[COEFF_DIM] = ARRAY_SIZE(tubes_bkgrd);

		complex float* bkgrd = md_alloc(DIMS, dims2, CFL_SIZE);

		calc_phantom_arb(dims2[COEFF_DIM], tubes_bkgrd, dims2, bkgrd, kspace, tstrs, traj, rotation_angle, popts);

		// Sum up all spatial coefficients

		long dims3[DIMS];
		md_select_dims(DIMS, ~COEFF_FLAG, dims3, dims);

		complex float* tmp = md_alloc(DIMS, dims3, CFL_SIZE);

		md_zsum(DIMS, dims2, COEFF_FLAG, tmp, bkgrd);

		// Save summed up coefficients to out

		long pos[DIMS] = { [0 ... DIMS - 1] = 0 };

		md_copy_block(DIMS, pos, dims2, out, dims3, tmp, CFL_SIZE);

		md_free(bkgrd);
		md_free(tmp);

		// Determine basis functions of the foreground

		dims2[COEFF_DIM] = ARRAY_SIZE(tubes_frgrd); // remove background

		complex float* frgrd = md_alloc(DIMS, dims2, CFL_SIZE);

		calc_phantom_arb(dims2[COEFF_DIM], tubes_frgrd, dims2, frgrd, kspace, tstrs, traj, rotation_angle, popts);

		// Add foreground basis functions to out

		pos[COEFF_DIM] = 1;

		md_copy_block(DIMS, pos, dims, out, dims2, frgrd, CFL_SIZE);

		md_free(frgrd);

	} else { // sum up all objects

		long tdims[DIMS];
		md_copy_dims(DIMS, tdims, dims);

		tdims[COEFF_DIM] = N;	// Number of elements of tubes phantom with rings see src/shepplogan.c

		complex float* tmp = md_alloc(DIMS, tdims, CFL_SIZE);

		calc_phantom_tubes(tdims, tmp, kspace, random, rotation_angle, N, tstrs, traj, popts);

		md_zsum(DIMS, tdims, COEFF_FLAG, out, tmp);
		md_free(tmp);
	}
}

