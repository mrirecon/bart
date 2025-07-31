/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2025 Moritz Blumenthal
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/loop.h"
#include "num/conv.h"

#include "morph.h"


complex float* md_structuring_element_cube(int N, long dims[N], int radius, unsigned long flags, const void* ref)
{
	for (int i = 0; i < N; i++)
		dims[i] = (MD_IS_SET(flags, i)) ? 1 + 2 * radius : 1;

	complex float* structure = md_alloc_sameplace(N, dims, CFL_SIZE, ref);
	md_zfill(N, dims, structure, 1.);

	return structure;
}

complex float* md_structuring_element_ball(int N, long dims[N], int radius, unsigned long flags, const void* ref)
{
	for (int i = 0; i < N; i++)
		dims[i] = (MD_IS_SET(flags, i)) ? 1 + 2 * radius : 1;

	complex float* structure = md_alloc_sameplace(N, dims, CFL_SIZE, ref);
	const long* dimsp = dims;

	NESTED(complex float, ball_kernel, (const long pos[]))
	{
		complex float val = 0.;

		float rad = 0.;
		for (int i = 0; i < N; i++)
			rad += powf(labs(pos[i] - dimsp[i] / 2), 2);

		if (rad <= powf(radius, 2))
			val = 1.;


		return val;
	};

	md_zsample(N, dims, structure, ball_kernel);

	return structure;
}

complex float* md_structuring_element_cross(int N, long dims[N], int radius, unsigned long flags, const void* ref)
{
	for (int i = 0; i < N; i++)
		dims[i] = (MD_IS_SET(flags, i)) ? 1 + 2 * radius : 1;

	complex float* structure = md_alloc_sameplace(N, dims, CFL_SIZE, ref);
	const long* dimsp = dims;

	NESTED(complex float, ball_kernel, (const long pos[]))
	{
		complex float val = 0.;

		for (int i = 0; i < N; i++)
			if (pos[i] == dimsp[i] / 2)
				val = 1.;

		return val;
	};

	md_zsample(N, dims, structure, ball_kernel);

	return structure;
}


static void mask_conv(int D, const long mask_dims[D], complex float* mask, const long dims[D], complex float* out, const complex float* in, enum conv_type ctype)
{
	conv(D, md_nontriv_dims(D, mask_dims), ctype, CONV_SYMMETRIC, dims, out, dims, in, mask_dims, mask);
}

void md_erosion(int D, const long mask_dims[D], complex float* mask, const long dims[D], complex float* out, const complex float* in, enum conv_type ctype)
{
	complex float* tmp = md_alloc_sameplace(D, dims, CFL_SIZE, in);

	mask_conv(D, mask_dims, mask, dims, tmp, in, ctype);

	// take relative error into account due to floating points
	md_zsgreatequal(D, dims, out, tmp, (1 - 0.0001) * md_zasum(D, mask_dims, mask));

	md_free(tmp);
}


void md_dilation(int D, const long mask_dims[D], complex float* mask, const long dims[D], complex float* out, const complex float* in, enum conv_type ctype)
{
	complex float* tmp = md_alloc_sameplace(D, dims, CFL_SIZE, in);

	mask_conv(D, mask_dims, mask, dims, tmp, in, ctype);

	// take relative error into account due to floating points
	md_zsgreatequal(D, dims, out, tmp, (1 - md_zasum(D, mask_dims, mask) * 0.0001));

	md_free(tmp);
}

void md_opening(int D, const long mask_dims[D], complex float* mask, const long dims[D], complex float* out, const complex float* in, enum conv_type ctype)
{
	complex float* tmp = md_alloc_sameplace(D, dims, CFL_SIZE, in);

	md_erosion(D, mask_dims, mask, dims, tmp, in, ctype);

	md_dilation(D, mask_dims, mask, dims, out, tmp, ctype);

	md_free(tmp);
}

void md_closing(int D, const long mask_dims[D], complex float* mask, const long dims[D], complex float* out, const complex float* in, enum conv_type ctype)
{
	complex float* tmp = md_alloc_sameplace(D, dims, CFL_SIZE, in);

	md_dilation(D, mask_dims, mask, dims, tmp, in, ctype);

	md_erosion(D, mask_dims, mask, dims, out, tmp, ctype);

	md_free(tmp);
}



// this assumes a zero padded input such that if pos exceeds the input it evaluates to false
static bool extend_label(int N, const long lstrs[N], complex float* labels, const long istrs[N], const complex float* in, const long sdims[N], const long sstrs[N], const complex float* structure, long label)
{
	if (0. == *in)
		return false;

	if (0. != *labels)
		return false;

	*labels = label;

	long pos[N];
	md_set_dims(N, pos, 0);

	long offset[N];
	for (int i = 0; i < N; i++)
		offset[i] = -sdims[i] / 2;

	labels = &MD_ACCESS(N, lstrs, offset, labels);
	in = &MD_ACCESS(N, istrs, offset, in);

	while (md_next(N, sdims, ~0UL, pos)) {

		if (0. == MD_ACCESS(N, sstrs, pos, structure))
			continue;

		extend_label(N, lstrs, &MD_ACCESS(N, lstrs, pos, labels), istrs, &MD_ACCESS(N, istrs, pos, in), sdims, sstrs, structure, label);
	}

	return true;
}

static long md_label_int2(int N, const long dims[N], const long lstrs[N], complex float* labels, const long istrs[N], const _Complex float* in, const long sdims[N], const complex float* structure)
{
	md_clear2(N, dims, lstrs, labels, CFL_SIZE);

	long pos[N];
	md_set_dims(N, pos, 0);

	long sstrs[N];
	md_calc_strides(N, sstrs, sdims, CFL_SIZE);

	long label = 0;

	do {

		label++;
		if (!extend_label(N, lstrs, &MD_ACCESS(N, lstrs, pos, labels), istrs, &MD_ACCESS(N, istrs, pos, in), sdims, sstrs, structure, label))
			label--;

	} while (md_next(N, dims, ~0UL, pos));

	return label;
}


long md_label(int N, const long dims[N], _Complex float* labels, const _Complex float* src, const long sdims[N], const complex float* structure)
{
	long ndims[N];
	for (int i = 0; i < N; i++)
		ndims[i] = dims[i] + sdims[i] - 1;

	complex float* tin = md_alloc(N, ndims, CFL_SIZE);
	md_resize_center(N, ndims, tin, dims, src, CFL_SIZE);

	long tstrs[N];
	md_calc_strides(N, tstrs, ndims, CFL_SIZE);

	long offset[N];
	for (int i = 0; i < N; i++)
		offset[i] = (sdims[i] / 2);

	complex float* cpu_structure = md_alloc(N, sdims, CFL_SIZE);
	md_copy(N, sdims, cpu_structure, structure, CFL_SIZE);

	complex float* cpu_labels = md_alloc(N, dims, CFL_SIZE);

	long label = md_label_int2(N, dims, MD_STRIDES(N, dims, CFL_SIZE), cpu_labels, tstrs, &MD_ACCESS(N, tstrs, offset, tin), sdims, cpu_structure);
	md_free(tin);

	md_copy(N, dims, labels, cpu_labels, CFL_SIZE);
	md_free(cpu_labels);
	md_free(cpu_structure);

	return label;
}


complex float* md_label_simple_connection(int N, long dims[N], float radius, unsigned long flags)
{
	int r = (int)floor(radius);
	assert(0 < r);

	for (int i = 0; i < N; i++)
		dims[i] = (MD_IS_SET(flags, i)) ? 1 + 2 * r : 1;

	complex float* structure = md_alloc(N, dims, CFL_SIZE);

	long pos[N];
	md_set_dims(N, pos, 0);

	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	long center[N];
	for (int i = 0; i < N; i++)
		center[i] = dims[i] / 2;

	do {
		long sum = 0;
		for (int i = 0; i < N; i++)
			sum += pow(labs(pos[i] - center[i]), 2);

		MD_ACCESS(N, strs, pos, structure) = (pow(radius, 2) >= sum) ? 1. : 0.;
	} while (md_next(N, dims, ~0UL, pos));

	return structure;
}

void md_center_of_mass(int N_labels, int N, float com[N_labels][N], const long dims[N], const complex float* labels, const complex float* wgh)
{
	complex float* labels_cpu = md_alloc(N, dims, CFL_SIZE);
	md_copy(N, dims, labels_cpu, labels, CFL_SIZE);

	complex float* wgh_cpu = NULL;
	if (NULL != wgh) {

		wgh_cpu = md_alloc(N, dims, CFL_SIZE);
		md_copy(N, dims, wgh_cpu, wgh, CFL_SIZE);
	}

	float count[N_labels];

	long pos[N];
	md_set_dims(N, pos, 0);

	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	for (int i = 0; i < N_labels; i++) {

		count[i] = 0.;

		for (int j = 0; j < N; j++)
			com[i][j] = 0.;
	}

	do {
		long label = MD_ACCESS(N, strs, pos, labels_cpu) - 1;

		if (-1 == label)
			continue;

		assert(label < N_labels);

		float val = 1.;
		if (NULL != wgh_cpu)
			val = crealf(MD_ACCESS(N, strs, pos, wgh_cpu));

		count[label] += val;

		for (int j = 0; j < N; j++)
			com[label][j] += pos[j] * val;

	} while (md_next(N, dims, ~0UL, pos));

	for (int i = 0; i < N_labels; i++) {

		assert(0. != count[i]);

		for (int j = 0; j < N; j++)
			com[i][j] /= count[i];
	}

	md_free(labels_cpu);
	if (NULL != wgh_cpu)
		md_free(wgh_cpu);
}
