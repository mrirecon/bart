/* Copyright 2016-2017. Martin Uecker.
 * Copyright 2023-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/filter.h"
#include "math.h"

#include "linops/linop.h"

#include "nudft.h"



static void get_coord(int N, unsigned long flags, float coord[N], long pos[N], const long tdims[N], const long tstrs[N], const complex float* traj)
{
	assert(0 == pos[0]);
	int j = 0;

	for (int i = 0; i < N; i++) {

		coord[i] = 0.;

		if (MD_IS_SET(flags, i)) {

			pos[0] = j;
			coord[i] = -crealf(MD_ACCESS(N, tstrs, pos, traj));
			j++;
		}
	}

	assert(tdims[0] == j);
	pos[0] = 0;
}

/** \brief Correction term for B0 inhomogeneity
 *
 * out = exp(j * fieldmap(r) * sample_time)
 *
 *        j ⋅ ω(r) ⋅ t
 * out = e
 *
 *	where ω(r) is the field map at point r in kspace
 *	t is the time that this point in kspace was sampled at
 *
 *	this calculates the phase accrual for the whole grid at the given sample_time
 *	and returns the result in out.
 */
static void phase_correction_term(int N, const long out_dims[N], complex float* out, const float sample_time, const complex float* field_map)
{
	// FIXME:  integrate into linear phase and do it inplace to be faster
	md_zsmul(N, out_dims, out, field_map, sample_time);
	md_zexpj(N, out_dims, out, out);
}


/** \brief nuDFT Forward Operator
 *
 * perform DFT analysis to go from image to ksp
 *
 */
void nudft_forward2(int N, unsigned long flags,
			const long kdims[N], const long kstrs[N], complex float* ksp,
			const long idims[N], const long istrs[N], const complex float* img,
			const long tdims[N], const long tstrs[N], const complex float* traj,
			const complex float* fieldmap,
			const long tmstrs[N], const complex float* timemap)
{
	assert(1 == kdims[0]);
	assert(md_check_compat(N, ~0UL, kdims, tdims));

	long tmp_dims[N];
	long tmp_strs[N];

	md_select_dims(N, flags, tmp_dims, idims);
	md_calc_strides(N, tmp_strs, tmp_dims, CFL_SIZE);

	complex float* tmp = md_alloc_sameplace(N, tmp_dims, CFL_SIZE, ksp);

	// TODO: tmp_pc can be eliminated later, just for now to test
	long tmp_pc_dims[N];
	long tmp_pc_strs[N];
	md_select_dims(N, flags, tmp_pc_dims, idims);
	md_calc_strides(N, tmp_pc_strs, tmp_pc_dims, CFL_SIZE);
	complex float* tmp_pc = md_alloc(N, tmp_pc_dims, CFL_SIZE);

	long kstrs2[N];
	for (int i = 0; i < N; i++)
		kstrs2[i] = MD_IS_SET(flags, i) ? 0 : kstrs[i];

	md_clear2(N, kdims, kstrs, ksp, CFL_SIZE);

	long pos[N];
	for (int i = 0; i < N; i++)
		pos[i] = 0;

	do {
		float coord[N];
		get_coord(N, flags, coord, pos, tdims, tstrs, traj);

		linear_phase(N, tmp_dims, coord, tmp);

		if ((NULL != fieldmap) && (NULL != timemap)) {

			phase_correction_term(N, tmp_dims, tmp_pc, MD_ACCESS(N, tmstrs, pos, timemap), fieldmap);
			md_zmul(N, tmp_dims, tmp, tmp_pc, tmp);
		}

		md_zfmac2(N, idims, kstrs2, &MD_ACCESS(N, kstrs, pos, ksp), istrs, img, tmp_strs, tmp);

	} while (md_next(N, tdims, ~MD_BIT(0), pos));

	// scale to be unitary
	float scale = 1. / sqrtf((float)md_calc_size(N, idims));
	md_zsmul(N, kdims, ksp, ksp, scale);

	md_free(tmp);
	md_free(tmp_pc);
}


/**	\brief nuDFT adjoint operator
 *
 *	perform DFT synthesis to go from kspace to image
 */
void nudft_adjoint2(int N, unsigned long flags,
			const long idims[N], const long istrs[N], complex float* img,
			const long kdims[N], const long kstrs[N], const complex float* ksp,
			const long tdims[N], const long tstrs[N], const complex float* traj,
			const complex float* fieldmap,
			const long tmstrs[N], const complex float* timemap)
{
	assert(1 == kdims[0]);
	assert(md_check_compat(N, ~0UL, kdims, tdims));

	long tmp_dims[N];
	long tmp_strs[N];

	md_select_dims(N, flags, tmp_dims, idims);
	md_calc_strides(N, tmp_strs, tmp_dims, CFL_SIZE);

	complex float* tmp = md_alloc_sameplace(N, tmp_dims, CFL_SIZE, img);

	// TODO: tmp_pc can be eliminated later, just for now to test
	long tmp_pc_dims[N];
	long tmp_pc_strs[N];
	md_select_dims(N, flags, tmp_pc_dims, idims);
	md_calc_strides(N, tmp_pc_strs, tmp_pc_dims, CFL_SIZE);

	complex float* tmp_pc = md_alloc(N, tmp_pc_dims, CFL_SIZE);

	long kstrs2[N];
	for (int i = 0; i < N; i++)
		kstrs2[i] = MD_IS_SET(flags, i) ? 0 : kstrs[i];

	md_clear2(N, idims, istrs, img, CFL_SIZE);

	long pos[N];
	for (int i = 0; i < N; i++)
		pos[i] = 0;

	do { /* iterates over ksp in all selected dimensions */

		float coord[N];
		get_coord(N, flags, coord, pos,	tdims, tstrs, traj);

		linear_phase(N, tmp_dims, coord, tmp);

		if ((NULL != fieldmap) && (NULL != timemap)) {

			phase_correction_term(N, tmp_dims, tmp_pc, MD_ACCESS(N, tmstrs, pos, timemap), fieldmap);
			md_zmul(N, tmp_dims, tmp, tmp_pc, tmp);
		}

		md_zfmacc2(N, idims, istrs, img, kstrs2, &MD_ACCESS(N, kstrs, pos, ksp), tmp_strs, tmp);

	} while (md_next(N, tdims, ~MD_BIT(0), pos));

	// scale to be unitary
	float scale = 1. / sqrtf((float)md_calc_size(N, idims));
	md_zsmul(N, idims, img, img, scale);


	md_free(tmp);
	md_free(tmp_pc);
}


void nudft_forward(int N, unsigned long flags,
			const long odims[N], complex float* out,
			const long idims[N], const complex float* in,
			const long tdims[N], const complex float* traj,
			const complex float* fieldmap,
			const long tmdims[N], const complex float* timemap)
{
	long ostrs[N];
	long istrs[N];
	long tstrs[N];
	long tmstrs[N];

	md_calc_strides(N, ostrs, odims, CFL_SIZE);
	md_calc_strides(N, istrs, idims, CFL_SIZE);
	md_calc_strides(N, tstrs, tdims, CFL_SIZE);	// FL_SIZE
	md_calc_strides(N, tmstrs, tmdims, CFL_SIZE);

	nudft_forward2(N, flags, odims, ostrs, out, idims, istrs, in, tdims, tstrs, traj, fieldmap, tmstrs, timemap);
}


struct nudft_s {

	linop_data_t base;

	int N;
	unsigned long flags;

	long* kdims;
	long* idims;
	long* tdims;
	long* kstrs;
	long* istrs;
	long* tstrs;
	long* fmdims;
	long* fmstrs;
	long* tmdims;
	long* tmstrs;

	const complex float* traj;
	const complex float* fieldmap;
	const complex float* timemap;
};


static void nudft_apply(const linop_data_t* _data, complex float* out, const complex float* in)
{
	const struct nudft_s* data = CONTAINER_OF_CONST(_data, const struct nudft_s, base);
	int N = data->N;

	nudft_forward2(N, data->flags,
			data->kdims, data->kstrs, out,
			data->idims, data->istrs, in,
			data->tdims, data->tstrs, data->traj,
			data->fieldmap, data->tmstrs, data->timemap);
}

static void nudft_adj(const linop_data_t* _data, complex float* out, const complex float* in)
{
	const struct nudft_s* data = CONTAINER_OF_CONST(_data, const struct nudft_s, base);
	int N = data->N;

	nudft_adjoint2(N, data->flags,
			data->idims, data->istrs, out,
			data->kdims, data->kstrs, in,
			data->tdims, data->tstrs, data->traj,
			data->fieldmap, data->tmstrs, data->timemap);
}

static void nudft_delete(const linop_data_t* _data)
{
	const struct nudft_s* data = CONTAINER_OF_CONST(_data, const struct nudft_s, base);

	xfree(data->kdims);
	xfree(data->idims);
	xfree(data->tdims);
	xfree(data->kstrs);
	xfree(data->istrs);
	xfree(data->tstrs);
	xfree(data->fmdims);
	xfree(data->fmstrs);
	xfree(data->tmdims);
	xfree(data->tmstrs);

	xfree(data);
}

const struct linop_s* nudft_create2(int N, unsigned long flags,
					const long odims[N], const long ostrs[N],
					const long idims[N], const long istrs[N],
					const long tdims[N], const complex float* traj,
					const long fmdims[N], const complex float* fieldmap,
					const long tmdims[N], const complex float* timemap)
{
	PTR_ALLOC(struct nudft_s, data);

	data->N = N;
	data->flags = flags;
	data->traj = traj;
	data->fieldmap = fieldmap;
	data->timemap = timemap;

	data->kdims = *TYPE_ALLOC(long[N]);
	data->kstrs = *TYPE_ALLOC(long[N]);

	md_copy_dims(N, data->kdims, odims);
	md_copy_strides(N, data->kstrs, ostrs);

	data->idims = *TYPE_ALLOC(long[N]);
	data->istrs = *TYPE_ALLOC(long[N]);

	md_copy_dims(N, data->idims, idims);
	md_copy_strides(N, data->istrs, istrs);

	data->tdims = *TYPE_ALLOC(long[N]);
	data->tstrs = *TYPE_ALLOC(long[N]);

	md_copy_dims(N, data->tdims, tdims);
	md_calc_strides(N, data->tstrs, tdims, CFL_SIZE);

	if (NULL != fieldmap) {

		data->fmdims = *TYPE_ALLOC(long[N]);
		data->fmstrs = *TYPE_ALLOC(long[N]);

		md_copy_dims(N, data->fmdims, fmdims);
		md_calc_strides(N, data->fmstrs, fmdims, CFL_SIZE);

		data->tmdims = *TYPE_ALLOC(long[N]);
		data->tmstrs = *TYPE_ALLOC(long[N]);

		md_copy_dims(N, data->tmdims, tmdims);
		md_calc_strides(N, data->tmstrs, tmdims, CFL_SIZE);

	} else {

		data->fmdims = NULL;
		data->fmstrs = NULL;

		data->tmdims = NULL;
		data->tmstrs = NULL;
	}

	return linop_create2(N, odims, ostrs, N, idims, istrs, &PTR_PASS(data)->base,
			nudft_apply, nudft_adj, NULL, NULL, nudft_delete);
}

const struct linop_s* nudft_create(int N, unsigned long flags, const long odims[N], const long idims[N], const long tdims[N], const complex float* traj, const long fmdims[N], const complex float* fieldmap, const long tmdims[N], const complex float* timemap)
{
	return nudft_create2(N, flags, odims, MD_STRIDES(N, odims, CFL_SIZE), idims, MD_STRIDES(N, idims, CFL_SIZE), tdims, traj, fmdims, fieldmap, tmdims, timemap);
}


