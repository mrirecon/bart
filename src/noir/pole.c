/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2025 Moritz Blumenthal
 */

#include <complex.h>
#include <math.h>

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/morph.h"
#include "num/flpmath.h"
#include "num/loop.h"
#include "num/vec3.h"

#include "pole.h"

struct pole_config_s pole_config_default = {

	.diameter = 0.05,
	.closing = -1.,
	.thresh = 0.5,
	.segments = 10,
	.avg_flag = COIL_FLAG,
	.normal = -1,
};

static void get_circle_coords(struct pole_config_s* conf, long pos[3], int index, int normal, int diameter, bool twoD)
{
	float angle = 2. * M_PI * index / conf->segments;

	for (int i = 0; i < 3; i++)
		pos[i] = 0;

	float e1[3][3] = { { 0., 1., 0. }, { 0., 0., 1. }, { 1., 0., 0. } };
	float e2[3][3] = { { 0., 0., 1. }, { 1., 0., 0. }, { 0., 1., 0. } };

	float fpos[3] = { diameter / 2., diameter / 2., diameter / 2. };

	vec3_saxpy(fpos, fpos, 0.5 * diameter * cosf(angle), e1[normal]);
	vec3_saxpy(fpos, fpos, 0.5 * diameter * sinf(angle), e2[normal]);

	for (int i = 0; i < 3; i++)
		pos[i] = MAX(0, MIN(diameter, lroundf(fpos[i])));

	if (twoD)
		pos[normal] = 0;
}

static void compute_curl_map_normal(struct pole_config_s conf, int N, const long dims[N], complex float* curl_map, const complex float* sens, int normal)
{
	assert(2 <= bitcount(md_nontriv_dims(MIN(N, 3), dims)));
	assert(0 < conf.diameter);
	assert(0 <= normal && normal < 3);

	long pos1[N];
	long pos2[N];

	md_set_dims(N, pos1, 0);
	md_set_dims(N, pos2, 0);

	if (dims[(normal + 1) % 3] != dims[(normal + 2) % 3])
		debug_printf(DP_DEBUG1, "Non-square dimensions detected (%ld, %ld, %ld): ", dims[0], dims[1], dims[2]);

	int diameter = roundf(ceil(conf.diameter * MAX(dims[(normal + 1) % 3], dims[(normal + 2) % 3])));
	debug_printf(DP_DEBUG1, "Circle diameter set to %d (%.3f * %ld).\n", diameter, conf.diameter,  MAX(dims[(normal + 1) % 3], dims[(normal + 2) % 3]));

	bool twoD = false;

	long odims[N];
	md_copy_dims(N, odims, dims);
	for (int i = 0; i < 3; i++) {

		if (1 != dims[i]) {

			odims[i] -= diameter;
		} else {

			assert(i == normal);
			twoD = true;
		}

		assert(0 < odims[i]);
	}

	complex float* tmp_angle = md_alloc_sameplace(N, odims, CFL_SIZE, sens);
	complex float* angle = md_alloc_sameplace(N, odims, CFL_SIZE, sens);
	md_clear(N, odims, angle, CFL_SIZE);

	complex float* sens1 = md_alloc_sameplace(N, odims, CFL_SIZE, sens);
	complex float* sens2 = md_alloc_sameplace(N, odims, CFL_SIZE, sens);

	get_circle_coords(&conf, pos1, 0, normal, diameter, twoD);
	md_copy_block(N, pos1, odims, sens1, dims, sens, CFL_SIZE);

	for (int i = 0; i < conf.segments; i++) {

		get_circle_coords(&conf, pos1, i + 0, normal, diameter, twoD);
		get_circle_coords(&conf, pos2, i + 1, normal, diameter, twoD);

		if (md_check_equal_dims(N, pos1, pos2, MD_BIT(3) - 1))
			continue;

		md_copy_block(N, pos2, odims, sens2, dims, sens, CFL_SIZE);

		md_zmulc(N, odims, tmp_angle, sens1, sens2);
		md_zphsr(N, odims, tmp_angle, tmp_angle); // to handle 0.+0.i case
		md_zarg(N, odims, tmp_angle, tmp_angle);
		md_zadd(N, odims, angle, angle, tmp_angle);

		SWAP(sens1, sens2);
	}

	md_free(sens1);
	md_free(sens2);

	md_free(tmp_angle);
	md_zsmul(N, odims, angle, angle, -1. / (2. * M_PI));

	md_resize_center(N, dims, curl_map, odims, angle, CFL_SIZE);
	md_free(angle);
}

void compute_curl_map(struct pole_config_s conf, int N, const long curl_dims[N], int dim, complex float* curl_map, const long sens_dims[N], const complex float* sens)
{
	assert(dim < N);
	assert(md_check_compat(N, MD_BIT(dim), curl_dims, sens_dims));

	if (1 == curl_dims[dim]) {

		int normal = conf.normal;
		for (int i = 0; i < 3; i++)
			if (1 == sens_dims[i])
				normal = i;

		assert(-1 != normal && normal < 3);

		compute_curl_map_normal(conf, N, sens_dims, curl_map, sens, normal);
	} else {

		for (int i = 0; i < curl_dims[dim]; i++)
			compute_curl_map_normal(conf, N, sens_dims, curl_map + md_calc_size(N, sens_dims) * i, sens, i);
	}
}

void compute_curl_weighting(struct pole_config_s conf, int N, const long curl_dims[N], int dim, complex float* wgh_map, const long col_dims[N], const complex float* sens)
{
	assert(dim < N);
	assert(md_check_compat(N, MD_BIT(dim), curl_dims, col_dims));

	complex float* wgh = md_alloc_sameplace(N, col_dims, CFL_SIZE, wgh_map);

	md_clear(N, col_dims, wgh, CFL_SIZE);
	md_zss(N, col_dims, 0UL, wgh, sens);

	long rdims[N];
	md_select_dims(N, ~conf.avg_flag, rdims, col_dims);

	complex float* tmp = md_alloc_sameplace(N, rdims, CFL_SIZE, wgh_map);
	md_zss(N, col_dims, conf.avg_flag, tmp, sens);

	complex float* one = md_alloc_sameplace(N, rdims, CFL_SIZE, wgh_map);
	md_zfill(N, rdims, one, 1.);

	md_zdiv(N, rdims, tmp, one, tmp);
	md_free(one);

	md_zmul2(N, col_dims, MD_STRIDES(N, col_dims, CFL_SIZE), wgh, MD_STRIDES(N, col_dims, CFL_SIZE), wgh, MD_STRIDES(N, rdims, CFL_SIZE), tmp);
	md_free(tmp);

	md_copy2(N, curl_dims, MD_STRIDES(N, curl_dims, CFL_SIZE), wgh_map, MD_STRIDES(N, col_dims, CFL_SIZE), wgh, CFL_SIZE);
	md_free(wgh);
}

void average_curl_map(int N, const long pmap_dims[N], complex float* red_curl_map, const long curl_dims[N], int dim, complex float* curl_map, complex float* wgh_map)
{
	long tmp_dims[N];
	md_select_dims(N, MD_BIT(dim) | md_nontriv_dims(N, pmap_dims), tmp_dims, curl_dims);

	complex float* tmp = md_alloc_sameplace(N, tmp_dims, CFL_SIZE, curl_map);

	if (NULL != wgh_map)
		md_ztenmul(N, tmp_dims, tmp, curl_dims, curl_map, curl_dims, wgh_map);
	else
		md_zavg(N, curl_dims, ~md_nontriv_dims(N, tmp_dims), tmp, curl_map);

	if (1 < curl_dims[dim]) {

		md_zabs(N, tmp_dims, tmp, tmp);
		md_reduce_zmax(N, tmp_dims, MD_BIT(dim), red_curl_map, tmp);
	} else {

		md_copy(N, tmp_dims, red_curl_map, tmp, CFL_SIZE);
	}

	md_free(tmp);
}


static struct lseg_s extract_phase_poles_2d_sign(struct pole_config_s conf, int N, const long dims[N], const _Complex float* curl_map, bool pos)
{
	assert((3 == bitcount(md_nontriv_dims(3, dims))) || (2 == bitcount(md_nontriv_dims(3, dims))));
	assert(1 == md_calc_size(N - 3, dims + 3));

	int normal = conf.normal;

	for (int i = 0; i < 3; i++)
		if (1 == dims[i])
			normal = i;

	assert((0 <= normal) && (3 > normal));

	complex float* binary = md_alloc_sameplace(3, dims, CFL_SIZE, curl_map);

	if (pos)
		md_zsgreatequal(3, dims, binary, curl_map, conf.thresh);
	else
		md_zslessequal(3, dims, binary, curl_map, -conf.thresh);


	struct lseg_s ret = { .N = 0, .pos = NULL };

	if (0 == md_znorm(3, dims, binary)) {

		md_free(binary);
		return ret;
	}

	complex float* wgh = md_alloc_sameplace(3, dims, CFL_SIZE, curl_map);
	md_zmul(3, dims, wgh, binary, curl_map);

	int dmin = lroundf(ceilf(((-1 == conf.closing) ? conf.diameter / 2. : conf.closing) * MAX(dims[(normal + 1) % 3], dims[(normal + 2) % 3])));
	long mdims[3];
	complex float* mask = md_structuring_element_cube(3, mdims, dmin, md_nontriv_dims(3, dims), curl_map);

	md_closing(3, mdims, mask, dims, binary, binary, CONV_TRUNCATED);
	md_free(mask);

	long sdims[3];
	complex float* strc = md_structuring_element_cube(3, sdims, 1, md_nontriv_dims(3, dims), curl_map);

	complex float* labels = md_alloc_sameplace(3, dims, CFL_SIZE, curl_map);
	long nlabel = md_label(3, dims, labels, binary, sdims, strc);

	md_free(binary);
	md_free(strc);

	float com[nlabel][3];
	md_center_of_mass(nlabel, 3, com, dims, labels, wgh);
	md_free(labels);
	md_free(wgh);

	ret.N = 0;
	ret.pos = xmalloc(sizeof(vec3_t[nlabel][2]));

	int diameter = roundf(ceil(conf.diameter * MAX(dims[(normal + 1) % 3], dims[(normal + 2) % 3])));
	float offset = (diameter / 2. - diameter / 2.);

	for(int i = 0; i < nlabel; i++) {

		for (int j = 0; j < 3; j++) {

			com[i][j] = (com[i][j] + (1 != dims[j] ? offset : 0.) - dims[j] / 2) / (float)dims[j];
		}

		debug_printf(DP_DEBUG1, "Found%s pole at %f %f %f.\n", pos ? "" : " conjugate", com[i][0], com[i][1], com[i][2]);

		vec3_copy((ret.pos)[ret.N][0], com[i]);
		vec3_copy((ret.pos)[ret.N][1], com[i]);

		(ret.pos)[ret.N][0][normal] = (pos) ? -1. : 1.;
		(ret.pos)[ret.N][1][normal] = (pos) ? 1. : -1.;

		ret.N++;
	}

	return ret;
}


struct lseg_s extract_phase_poles_2D(struct pole_config_s conf, int N, const long dims[N], const _Complex float* curl_map)
{
	struct lseg_s pos = extract_phase_poles_2d_sign(conf, N, dims, curl_map, true);
	struct lseg_s neg = extract_phase_poles_2d_sign(conf, N, dims, curl_map, false);

	struct lseg_s ret = { .N = pos.N + neg.N, .pos = NULL };

	if (0 == ret.N)
		return ret;

	ret.pos = xmalloc(sizeof(vec3_t[ret.N][2]));

	for (int i = 0; i < pos.N; i++) {

		vec3_copy(ret.pos[i][0], pos.pos[i][0]);
		vec3_copy(ret.pos[i][1], pos.pos[i][1]);
	}

	for (int i = 0; i < neg.N; i++) {

		vec3_copy(ret.pos[i + pos.N][0], neg.pos[i][0]);
		vec3_copy(ret.pos[i + pos.N][1], neg.pos[i][1]);
	}

	if (NULL != pos.pos)
		xfree(pos.pos);

	if (NULL != neg.pos)
		xfree(neg.pos);

	return ret;
}


static void get_coord_transform(vec3_t evec[3], const vec3_t r1, const vec3_t r2)
{
	vec3_copy(evec[2], r2);
	vec3_sub(evec[2], evec[2], r1);
	vec3_smul(evec[2], evec[2],  1. / vec3_norm(evec[2]));

	evec[0][0] = 1.;
	evec[0][1] = 0;
	evec[0][2] = 0.;

	if (fabsf(vec3_sdot(evec[0], evec[2])) > 0.8) {

		evec[0][0] = 0.;
		evec[0][1] = 1.;
		evec[0][2] = 0.;
	}

	vec3_saxpy(evec[0], evec[0], -vec3_sdot(evec[0], evec[2]), evec[2]);
	vec3_smul(evec[0], evec[0], 1. / vec3_norm(evec[0]));

	vec3_rot(evec[1], evec[2], evec[0]);
}



void sample_phase_pole_2D(int N, const long dims[N], _Complex float* dst, int D, const float r[D][2][3])
{
	assert(2 == bitcount(md_nontriv_dims(3, dims)));

	vec3_t evec[D][3];
	vec3_t center[D];
	float L[D];

	for (int i = 0; i < D; i++) {

		vec3_t r1;
		vec3_t r2;

		vec3_copy(r1, r[i][0]);
		vec3_copy(r2, r[i][1]);

		vec3_t diff;
		vec3_sub(diff, r2, r1);

		if (0. == vec3_norm(diff)) {

			for (int j = 0; j < 3; j++) {

				if (1 == dims[j]) {

					r1[j] = 0.5;
					r2[j] = -0.5;
				}
			}

			vec3_sub(diff, r2, r1);
			assert(0. != vec3_norm(diff));
		}

		get_coord_transform(evec[i], r1, r2);

		L[i] = vec3_norm(diff) / 2.;

		vec3_add(center[i], r1, r2);
		vec3_smul(center[i], center[i], 0.5);

		for (int j = 0; j < 3; j++)
			center[i][j] += (dims[j] / 2) / (float)dims[j];
	}

	const long* dimsp = dims;
	float* centerp0 = &(center[0][0]);
	float* evecp0 = &(evec[0][0][0]);
	float* Lp = L;


	NESTED(complex float, pole_kernel, (const long pos[]))
	{
		complex float ret = 1.;

		float (*centerp)[D][3] = (float (*)[D][3])centerp0;
		float (*evecp)[D][3][3] = (float (*)[D][3][3])evecp0;

		for (int i = 0; i < D; i++) {

			vec3_t fpos;
			for (int j = 0; j < 3; j++)
				fpos[j] = pos[j] / (float)dimsp[j] - (*centerp)[i][j];

			if (Lp[i] < fabsf(vec3_sdot((*evecp)[i][2], fpos)))
				continue;

			complex float val = vec3_sdot(fpos, (*evecp)[i][0]) + vec3_sdot(fpos, (*evecp)[i][1]) * 1.i;
			complex float mag = cabsf(val);
			val = (0 == mag) ? 1. : val / mag;
			ret *= val;
		}

		return ret;
	};

	md_parallel_zsample(N, dims, dst, pole_kernel);
}


bool phase_pole_correction(struct pole_config_s conf, int N, const long pmap_dims[N], complex float* phase, const long sens_dims[N], const complex float* sens)
{
	long curl_dims[N];
	md_copy_dims(N, curl_dims, sens_dims);

	int normal = conf.normal;
	for (int i = 0; i < 3; i++)
		if (1 == curl_dims[i])
			normal = i;

	assert((0 <= normal) && (3 > normal));

	complex float* curl_map = md_alloc_sameplace(N, curl_dims, CFL_SIZE, sens);
	complex float* curl_wgh = md_alloc_sameplace(N, curl_dims, CFL_SIZE, sens);

	assert(ITER_DIM < N);

	compute_curl_map(conf, N, curl_dims, ITER_DIM, curl_map, sens_dims, sens);
	compute_curl_weighting(conf, N, curl_dims, ITER_DIM, curl_wgh, sens_dims, sens);

	long rcurl_map_dims[N];
	md_select_dims(N, ~(conf.avg_flag | MD_BIT(ITER_DIM)), rcurl_map_dims, curl_dims);

	complex float* red_curl_map = md_alloc_sameplace(N, rcurl_map_dims, CFL_SIZE, sens);
	average_curl_map(N, rcurl_map_dims, red_curl_map, curl_dims, ITER_DIM, curl_map, curl_wgh);

	struct lseg_s pos = extract_phase_poles_2D(conf, N, rcurl_map_dims, red_curl_map);

	md_zfill(N, pmap_dims, phase, 1.);

	if (0 < pos.N)
		sample_phase_pole_2D(N, pmap_dims, phase, pos.N, pos.pos);

	xfree(pos.pos);

	md_free(red_curl_map);
	md_free(curl_map);
	md_free(curl_wgh);

	return 0 < pos.N;
}

bool phase_pole_correction_loop(struct pole_config_s conf, int N, unsigned long lflags, const long pmap_dims[N], complex float* phase, const long sens_dims[N], const complex float* sens)
{
	bool ret = false;

	long npmap_dims[N];
	long nsens_dims[N];

	md_select_dims(N, ~lflags, npmap_dims, pmap_dims);
	md_select_dims(N, ~lflags, nsens_dims, sens_dims);

	long pmap_strs[N];
	long sens_strs[N];

	md_calc_strides(N, pmap_strs, pmap_dims, CFL_SIZE);
	md_calc_strides(N, sens_strs, sens_dims, CFL_SIZE);

	long pos[N];
	md_set_dims(N, pos, 0);

	do {
		ret = phase_pole_correction(conf, N, npmap_dims, &MD_ACCESS(N, pmap_strs, pos, phase), nsens_dims, &MD_ACCESS(N, sens_strs, pos, sens)) || ret;

	} while (md_next(N, pmap_dims, lflags, pos));

	return ret;
}


void phase_pole_normalize(int N, const long pdims[N], complex float* phase, const long idims[N], const complex float* image)
{
	complex float* timage = md_alloc_sameplace(N, idims, CFL_SIZE, image);

	md_ztenmul(N, idims, timage, pdims, phase, pdims, image);

	long tpdims[N];
	md_select_dims(N, ~7UL, tpdims, pdims);

	complex float* dot = md_alloc_sameplace(N, tpdims, CFL_SIZE, image);
	md_ztenmulc(N, tpdims, dot, idims, image, idims, timage);
	md_zphsr(N, tpdims, dot, dot);

	md_zmul2(N, pdims, MD_STRIDES(N, pdims, CFL_SIZE), phase, MD_STRIDES(N, pdims, CFL_SIZE), phase, MD_STRIDES(N, tpdims, CFL_SIZE), dot);

	md_free(timage);
	md_free(dot);
}
