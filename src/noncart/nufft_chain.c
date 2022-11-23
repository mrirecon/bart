/* Copyright 2022. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <math.h>

#include "linops/someops.h"
#include "misc/misc.h"
#include "misc/types.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/multiplace.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/fmac.h"

#include "noncart/grid.h"
#include "nufft_chain.h"


struct kb_rolloff_s {

	INTERFACE(linop_data_t);

	int N;
	const long* dims;
	const long* odims;

	struct grid_conf_s conf;
};

static DEF_TYPEID(kb_rolloff_s);


static void rolloff_apply(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	auto d = CAST_DOWN(kb_rolloff_s, _d);

	long ostrs[d->N];
	long istrs[d->N];

	md_calc_strides(d->N, ostrs, d->odims, CFL_SIZE);
	md_calc_strides(d->N, istrs, d->dims, CFL_SIZE);
 
	long pos[d->N];
	for (int i = 0; i < d->N; i++)
		pos[i] = (i < 3) ? labs((d->odims[i] / 2) - (d->dims[i] / 2)) : 0;

	md_clear(d->N, d->odims, dst, CFL_SIZE);
	apply_rolloff_correction2(d->conf.os, d->conf.width, d->conf.beta, d->N, d->dims,
				  ostrs, &(MD_ACCESS(d->N, ostrs, pos, dst)),
				  istrs, src);
}

static void rolloff_adjoint(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	auto d = CAST_DOWN(kb_rolloff_s, _d);

	long ostrs[d->N];
	long istrs[d->N];

	md_calc_strides(d->N, ostrs, d->odims, CFL_SIZE);
	md_calc_strides(d->N, istrs, d->dims, CFL_SIZE);
 
	long pos[d->N];
	for (int i = 0; i < d->N; i++)
		pos[i] = (i < 3) ? labs((d->odims[i] / 2) - (d->dims[i] / 2)) : 0;

	apply_rolloff_correction2(d->conf.os, d->conf.width, d->conf.beta, d->N, d->dims,
				  istrs, dst,
				  ostrs, &(MD_ACCESS(d->N, ostrs, pos, src)));
}

static void rolloff_normal(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	auto d = CAST_DOWN(kb_rolloff_s, _d);

	apply_rolloff_correction(1, d->conf.width, d->conf.beta, d->N, d->dims, dst, src);
	apply_rolloff_correction(1, d->conf.width, d->conf.beta, d->N, d->dims, dst, dst);
}

static void rolloff_free(const linop_data_t* _d)
{
	const auto d = CAST_DOWN(kb_rolloff_s, _d);

	xfree(d->dims);
	xfree(d->odims);

	xfree(d);
}

struct linop_s* linop_kb_rolloff_create(int N, const long dims[N], unsigned long flags, struct grid_conf_s* conf)
{
	PTR_ALLOC(struct kb_rolloff_s, d);
	SET_TYPEID(kb_rolloff_s, d);

	d->N = N;
	d->dims = ARR_CLONE(long[N], dims);

	flags &= md_nontriv_dims(N, dims);
	assert(0 == (flags & ~FFT_FLAGS));
	
	long odims[N];
	for (int i = 0; i < N; i++)
		odims[i] = (MD_IS_SET(flags, i)) ? lround(conf->os * dims[i]) : dims[i];

	d->odims = ARR_CLONE(long[N], odims);
	d->conf = *conf;

	return linop_create(N, odims, N, dims, CAST_UP(PTR_PASS(d)), rolloff_apply, rolloff_adjoint, rolloff_normal, NULL, rolloff_free);
}





struct kb_iterpolate_s {

	INTERFACE(linop_data_t);

	int N;
	const long* tdims;
	const long* kdims;
	const long* gdims;

	struct multiplace_array_s* traj;

	struct grid_conf_s conf;
};

static DEF_TYPEID(kb_iterpolate_s);


static void interpolate_apply(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	auto d = CAST_DOWN(kb_iterpolate_s, _d);

	md_clear(d->N, d->kdims, dst, CFL_SIZE);
	grid2H(&(d->conf), d->N, d->tdims, multiplace_read(d->traj, dst), d->kdims, dst, d->gdims, src);
}

static void interpolate_adjoint(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	auto d = CAST_DOWN(kb_iterpolate_s, _d);

	md_clear(d->N, d->gdims, dst, CFL_SIZE);
	grid2(&(d->conf), d->N, d->tdims, multiplace_read(d->traj, dst), d->gdims, dst, d->kdims, src);
}


static void interpolate_free(const linop_data_t* _d)
{
	auto d = CAST_DOWN(kb_iterpolate_s, _d);

	xfree(d->gdims);
	xfree(d->tdims);
	xfree(d->kdims);

	multiplace_free(d->traj);

	xfree(d);
}

struct linop_s* linop_interpolate_create(int N, unsigned long flags, const long ksp_dims[N], const long grd_dims[N], const long trj_dims[N], const complex float* traj, struct grid_conf_s* conf)
{
	PTR_ALLOC(struct kb_iterpolate_s, data);
	SET_TYPEID(kb_iterpolate_s, data);

	data->N = N;
	data->kdims = ARR_CLONE(long[N], ksp_dims);
	data->gdims = ARR_CLONE(long[N], grd_dims);
	data->tdims = ARR_CLONE(long[N], trj_dims);

	assert(0 == (flags & ~FFT_FLAGS));

	data->traj = multiplace_move(N, trj_dims, CFL_SIZE, traj);
	data->conf = *conf;

	return linop_create(N, ksp_dims, N, grd_dims, CAST_UP(PTR_PASS(data)), interpolate_apply, interpolate_adjoint, NULL, NULL, interpolate_free);
}




extern struct linop_s* nufft_create_chain(int N,
			     const long ksp_dims[N],
			     const long cim_dims[N],
			     const long traj_dims[N],
			     const complex float* traj,
			     const long wgh_dims[N],
			     const complex float* weights,
			     const long bas_dims[N],
			     const complex float* basis,
			     struct grid_conf_s* conf)
{
	unsigned long flags = FFT_FLAGS & md_nontriv_dims(N, cim_dims);

	auto ret = linop_kb_rolloff_create(N, cim_dims, flags, conf);

	long os_cim_dims[N];
	for (int i = 0; i < N; i++)
		os_cim_dims[i] = (MD_IS_SET(flags, i)) ? lround(conf->os * cim_dims[i]) : cim_dims[i];
	
	ret = linop_chain_FF(ret, linop_fftc_create(N, os_cim_dims, flags));

	if (0 != basis) {

		long ksp_max_dims[N];
		md_max_dims(N, ~0, ksp_max_dims, ksp_dims, bas_dims);
		assert(md_check_compat(N, ~0, ksp_dims, bas_dims));

		ret = linop_chain_FF(ret, linop_interpolate_create(N, flags, ksp_max_dims, os_cim_dims, traj_dims, traj, conf));
		ret = linop_chain_FF(ret, linop_fmac_create(N, ksp_max_dims, ~md_nontriv_dims(N, ksp_dims), 0, ~md_nontriv_dims(N, bas_dims), basis));
	} else {

		ret = linop_chain_FF(ret, linop_interpolate_create(N, flags, ksp_dims, os_cim_dims, traj_dims, traj, conf));
	}

	if (NULL != weights) {

		assert(md_check_compat(N, ~0, ksp_dims, wgh_dims));
		complex float* tmp = md_alloc_sameplace(N, wgh_dims, CFL_SIZE, weights);
		md_zsmul(N, ksp_dims, tmp, weights, powf(conf->os, bitcount(flags) / 2.));
		ret = linop_chain_FF(ret, linop_cdiag_create(N, ksp_dims, md_nontriv_dims(N, wgh_dims), tmp));
		md_free(tmp);
	} else {

		ret = linop_chain_FF(ret, linop_scale_create(N, ksp_dims, powf(conf->os, bitcount(flags) / 2.)));
	}

	return ret;
}
