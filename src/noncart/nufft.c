/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2016-2021. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2022. Institute of Biomedical Imaging. Graz University of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2017 Frank Ong
 * 2014-2022 Martin Uecker
 * 2018      Sebastian Rosenzweig
 *
 */

#include <math.h>
#include <complex.h>
#include <assert.h>
#include <stdbool.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/types.h"
#include "misc/version.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/filter.h"
#include "num/fft.h"
#include "num/shuffle.h"
#include "num/ops.h"
#include "num/multiplace.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/fmac.h"

#include "noncart/grid.h"

#include "nufft.h"

#define FFT_FLAGS (MD_BIT(0)|MD_BIT(1)|MD_BIT(2))

struct nufft_conf_s nufft_conf_defaults = {

	.toeplitz = true,
	.pcycle = false,
	.periodic = false,
	.lowmem = false,
	.loopdim = -1,
	.flags = FFT_FLAGS,
	.cfft = 0u,
	.decomp = true,
	.nopsf = false,
};

#include "nufft_priv.h"

DEF_TYPEID(nufft_data);



static complex float* compute_linphases(int N, long lph_dims[N + 1], unsigned long flags, const long img_dims[N + 1])
{
	int T = bitcount(flags);
	float shifts[1 << T][T];

	int s = 0;
	for(int i = 0; i < (1 << T); i++) {

		bool skip = false;

		for(int j = 0; j < T; j++) {

			shifts[s][j] = 0.;

			if (MD_IS_SET(i, j)) {

				skip = skip || (1 == img_dims[j]);
				shifts[s][j] = -0.5;
			}
		}

		if (!skip)
			s++;
	}

	int ND = N + 1;
	md_select_dims(ND, flags, lph_dims, img_dims);
	lph_dims[N] = s;

	complex float* linphase = md_alloc(ND, lph_dims, CFL_SIZE);

	#pragma omp parallel for shared(linphase)
	for(int i = 0; i < s; i++) {

		float shifts2[ND];
		for (int j = 0; j < ND; j++)
			shifts2[j] = 0.;

		for (int j = 0, t = 0; j < N; j++)
			if (MD_IS_SET(flags, j))
				shifts2[j] = shifts[i][t++];

		linear_phase(ND, img_dims, shifts2,
				linphase + i * md_calc_size(ND, img_dims));
	}

	return linphase;
}



static void compute_kern_basis(int N, unsigned long flags, const long pos[N],
				const long krn_dims[N], complex float* krn,
				const long bas_dims[N], const complex float* basis,
				const long wgh_dims[N], const complex float* weights)
{
// 	assert(1 == krn_dims[N - 1]);
	assert(1 == wgh_dims[N - 1]);
	assert(1 == bas_dims[N - 1]);

	long baT_dims[N];
	md_copy_dims(N, baT_dims, bas_dims);
	baT_dims[N - 1] = bas_dims[5];
	baT_dims[5] = 1;

	long wgT_dims[N];
	md_copy_dims(N, wgT_dims, wgh_dims);
	wgT_dims[N - 1] = wgh_dims[5];
	wgT_dims[5] = 1;

	long max_dims[N];
	md_max_dims(N, ~0u, max_dims, baT_dims, wgT_dims);

	long max_strs[N];
	md_calc_strides(N, max_strs, max_dims, CFL_SIZE);

	long bas_strs[N];
	md_calc_strides(N, bas_strs, bas_dims, CFL_SIZE);

	long baT_strs[N];
	md_copy_strides(N, baT_strs, bas_strs);
	baT_strs[N - 1] = bas_strs[5];
	baT_strs[5] = 0;

	long wgh_strs[N];
	md_calc_strides(N, wgh_strs, wgh_dims, CFL_SIZE);

	long wgT_strs[N];
	md_copy_strides(N, wgT_strs, wgh_strs);
	wgT_strs[N - 1] = wgh_strs[5];
	wgT_strs[5] = 0;

	debug_printf(DP_DEBUG1, "Allocating %ld\n", md_calc_size(N, max_dims));

	complex float* tmp = md_alloc_sameplace(N, max_dims, CFL_SIZE, krn);

	md_copy2(N, max_dims, max_strs, tmp, baT_strs, basis, CFL_SIZE);

	md_zmul2(N, max_dims, max_strs, tmp, max_strs, tmp, wgT_strs, weights);
	md_zmulc2(N, max_dims, max_strs, tmp, max_strs, tmp, wgT_strs, weights);

	baT_dims[5] = baT_dims[6];
	baT_dims[6] = 1;

	baT_strs[5] = baT_strs[6];
	baT_strs[6] = 0;

	long krn_strs[N];
	md_calc_strides(N, krn_strs, krn_dims, CFL_SIZE);

	long ma2_dims[N];
	md_tenmul_dims(N, ma2_dims, krn_dims, max_dims, baT_dims);

	long ma3_dims[N];
	md_select_dims(N, flags, ma3_dims, ma2_dims);

	long tmp_off = md_calc_offset(N, max_strs, pos);
	long bas_off = md_calc_offset(N, baT_strs, pos);

	if (use_compat_to_version("v0.7.00"))
		md_zsmul(N, max_dims, tmp, tmp, (double)bas_dims[6]);

	md_ztenmulc2(N, ma3_dims, krn_strs, krn,
			max_strs, (void*)tmp + tmp_off,
			baT_strs, (void*)basis + bas_off);

	md_free(tmp);
}



static void compute_kern(int N, unsigned long flags, const long pos[N],
				const long krn_dims[N], complex float* krn,
				const long bas_dims[N], const complex float* basis,
				const long wgh_dims[N], const complex float* weights)
{
	if (NULL != basis)
		return compute_kern_basis(N, flags, pos, krn_dims, krn, bas_dims, basis, wgh_dims, weights);

	assert(~0u == flags);

	md_zfill(N, krn_dims, krn, 1.);

	if (NULL != weights) {

		long krn_strs[N];
		md_calc_strides(N, krn_strs, krn_dims, CFL_SIZE);

		long wgh_strs[N];
		md_calc_strides(N, wgh_strs, wgh_dims, CFL_SIZE);

		md_zmul2(N, krn_dims, krn_strs, krn, krn_strs, krn, wgh_strs, weights);
		md_zmulc2(N, krn_dims, krn_strs, krn, krn_strs, krn, wgh_strs, weights);
	}

	return;
}



complex float* compute_psf(int N, const long img_dims[N], const long trj_dims[N], const complex float* traj,
				const long bas_dims[N], const complex float* basis,
				const long wgh_dims[N], const complex float* weights,
				bool periodic, bool lowmem)
{
	long img2_dims[N + 1];
	md_copy_dims(N, img2_dims, img_dims);
	img2_dims[N] = 1;

	long trj2_dims[N + 1];
	md_copy_dims(N, trj2_dims, trj_dims);
	trj2_dims[N] = 1;

	long bas2_dims[N + 1];
	md_copy_dims(N, bas2_dims, bas_dims);
	bas2_dims[N] = 1;

	long wgh2_dims[N + 1];
	md_copy_dims(N, wgh2_dims, wgh_dims);
	wgh2_dims[N] = 1;

	N++;

	long ksp2_dims[N];
	md_copy_dims(N, ksp2_dims, img2_dims);
	md_select_dims(3, ~MD_BIT(0), ksp2_dims, trj2_dims);

	if (NULL != basis) {

		assert(1 == trj2_dims[6]);
		assert(1 == trj2_dims[N - 1]);
		ksp2_dims[N - 1] = trj2_dims[5];
		trj2_dims[N - 1] = trj2_dims[5];
		trj2_dims[5] = 1;	// FIXME copy?
	}

	struct nufft_conf_s conf = nufft_conf_defaults;
	conf.periodic = periodic;
	conf.toeplitz = false;	// avoid infinite loop
	conf.lowmem = lowmem;


	debug_printf(DP_DEBUG2, "nufft kernel dims: ");
	debug_print_dims(DP_DEBUG2, N, ksp2_dims);

	debug_printf(DP_DEBUG2, "nufft psf dims:    ");
	debug_print_dims(DP_DEBUG2, N, img2_dims);

	debug_printf(DP_DEBUG2, "nufft traj dims:   ");
	debug_print_dims(DP_DEBUG2, N, trj2_dims);

	complex float* psft = NULL;

	long pos[N];
	for (int i = 0; i < N; i++)
		pos[i] = 0;

	long A = md_calc_size(N, ksp2_dims);
	long B = md_calc_size(N - 1, ksp2_dims) + md_calc_size(N - 1, img2_dims);
	long C = md_calc_size(N, img2_dims);

	if ((A <= B) || !lowmem) {

		debug_printf(DP_DEBUG1, "Allocating %ld (vs. %ld) + %ld\n", A, B, C);

		complex float* ones = md_alloc_sameplace(N, ksp2_dims, CFL_SIZE, traj);

		compute_kern(N, ~0u, pos, ksp2_dims, ones, bas2_dims, basis, wgh2_dims, weights);

		psft = md_alloc_sameplace(N, img2_dims, CFL_SIZE, traj);

		struct linop_s* op2 = nufft_create(N, ksp2_dims, img2_dims, trj2_dims, traj, NULL, conf);

		linop_adjoint_unchecked(op2, psft, ones);

		linop_free(op2);

		md_free(ones);

	} else {

		debug_printf(DP_DEBUG1, "Allocating %ld (vs. %ld) + %ld\n", B, A, C);

		psft = md_alloc_sameplace(N, img2_dims, CFL_SIZE, traj);
		md_clear(N, img2_dims, psft, CFL_SIZE);

		long trj2_strs[N];
		md_calc_strides(N, trj2_strs, trj2_dims, CFL_SIZE);

		complex float* ones = md_alloc_sameplace(N - 1, ksp2_dims, CFL_SIZE, traj);
		complex float* tmp = md_alloc_sameplace(N - 1, img2_dims, CFL_SIZE, traj);

		assert(!((1 != trj2_dims[N - 1]) && (NULL == basis)));

		for (long i = 0; i < trj2_dims[N - 1]; i++) {

			debug_printf(DP_DEBUG1, "KERN %03ld\n", i);

			long flags = ~0UL;

			if (1 != trj2_dims[N - 1])
				flags = ~(1u << (N - 1u));

			pos[N - 1] = i;
			compute_kern(N, flags, pos, ksp2_dims, ones, bas2_dims, basis, wgh2_dims, weights);

			struct linop_s* op2 = nufft_create(N - 1, ksp2_dims, img2_dims, trj2_dims, (void*)traj + i * trj2_strs[N - 1], NULL, conf);

			linop_adjoint_unchecked(op2, tmp, ones);
			md_zadd(N - 1, img2_dims, psft, psft, tmp);

			linop_free(op2);
		}

		md_free(ones);
		md_free(tmp);
	}

	return psft;
}


static complex float* compute_psf2(int N, const long psf_dims[N + 1], unsigned long flags, const long trj_dims[N + 1], const complex float* traj,
				const long bas_dims[N + 1], const complex float* basis, const long wgh_dims[N + 1], const complex float* weights,
				bool periodic, bool lowmem)
{
	int ND = N + 1;

	long img_dims[ND];
	long img_strs[ND];

	md_select_dims(ND, ~MD_BIT(N + 0), img_dims, psf_dims);
	md_calc_strides(ND, img_strs, img_dims, CFL_SIZE);

	// PSF 2x size

	long img2_dims[ND];
	long img2_strs[ND];

	md_copy_dims(ND, img2_dims, img_dims);

	for (int i = 0; i < N; i++)
		if (MD_IS_SET(flags, i))
			img2_dims[i] = (1 == img_dims[i]) ? 1 : (2 * img_dims[i]);

	md_calc_strides(ND, img2_strs, img2_dims, CFL_SIZE);

	complex float* traj2 = md_alloc_sameplace(ND, trj_dims, CFL_SIZE, traj);
	md_zsmul(ND, trj_dims, traj2, traj, 2.);

	complex float* psft = compute_psf(ND, img2_dims, trj_dims, traj2, bas_dims, basis, wgh_dims, weights, periodic, lowmem);
	md_free(traj2);

	fftuc(ND, img2_dims, flags, psft, psft);

	float scale = 1.;
	for (int i = 0; i < N; i++)
		scale *= ((img2_dims[i] > 1) && (MD_IS_SET(flags, i))) ? 4. : 1.;

	md_zsmul(ND, img2_dims, psft, psft, scale);

	// reformat

	complex float* psf = md_alloc_sameplace(ND, psf_dims, CFL_SIZE, traj);

	long factors[N];

	for (int i = 0; i < N; i++)
		factors[i] = ((img_dims[i] > 1) && (MD_IS_SET(flags, i))) ? 2 : 1;

	md_decompose(N + 0, factors, psf_dims, psf, img2_dims, psft, CFL_SIZE);

	md_free(psft);
	return psf;
}


static struct nufft_data* nufft_create_data(int N,
			const long cim_dims[N], bool basis,
			struct nufft_conf_s conf)
{
	PTR_ALLOC(struct nufft_data, data);
	SET_TYPEID(nufft_data, data);

	data->N = N;

	// extend internal dimensions by one for linear phases
	int ND = N + 1;

	data->ksp_dims = *TYPE_ALLOC(long[ND]);
	data->cim_dims = *TYPE_ALLOC(long[ND]);
	data->cml_dims = *TYPE_ALLOC(long[ND]);
	data->img_dims = *TYPE_ALLOC(long[ND]);
	data->trj_dims = *TYPE_ALLOC(long[ND]);
	data->lph_dims = *TYPE_ALLOC(long[ND]);
	data->psf_dims = *TYPE_ALLOC(long[ND]);
	data->wgh_dims = *TYPE_ALLOC(long[ND]);
	data->bas_dims = *TYPE_ALLOC(long[ND]);
	data->out_dims = *TYPE_ALLOC(long[ND]);
	data->ciT_dims = *TYPE_ALLOC(long[ND]);
	data->cmT_dims = *TYPE_ALLOC(long[ND]);
	data->cm2_dims = *TYPE_ALLOC(long[ND]);

	data->factors = *TYPE_ALLOC(long[ND]);

	md_singleton_dims(ND, data->factors);

	data->ksp_strs = *TYPE_ALLOC(long[ND]);
	data->cim_strs = *TYPE_ALLOC(long[ND]);
	data->cml_strs = *TYPE_ALLOC(long[ND]);
	data->img_strs = *TYPE_ALLOC(long[ND]);
	data->trj_strs = *TYPE_ALLOC(long[ND]);
	data->lph_strs = *TYPE_ALLOC(long[ND]);
	data->psf_strs = *TYPE_ALLOC(long[ND]);
	data->wgh_strs = *TYPE_ALLOC(long[ND]);
	data->bas_strs = *TYPE_ALLOC(long[ND]);
	data->out_strs = *TYPE_ALLOC(long[ND]);

	data->traj = NULL;
	data->psf = NULL;
	data->weights = NULL;
	data->basis = NULL;


	data->conf = conf;
	data->flags = conf.flags;

	data->width = 3.;
	data->beta = calc_beta(2., data->width);

	// dim 0 must be transformed (we treat this special in the trajectory)
	assert(MD_IS_SET(data->flags, 0));

	assert(0 == (data->flags & conf.cfft));
	assert(!((!conf.decomp) && conf.toeplitz));

	struct grid_conf_s grid_conf = {

		.width = data->width,
		.os = 2.,
		.periodic = data->conf.periodic,
		.beta = data->beta,
	};

	data->grid_conf = grid_conf;



	md_copy_dims(N, data->cim_dims, cim_dims);
	data->cim_dims[N] = 1;

	md_copy_dims(ND, data->ciT_dims, data->cim_dims);

	md_select_dims(ND, data->flags, data->img_dims, data->cim_dims);

	md_calc_strides(ND, data->cim_strs, data->cim_dims, CFL_SIZE);
	md_calc_strides(ND, data->img_strs, data->img_dims, CFL_SIZE);


	complex float* fftm = md_alloc(ND, data->img_dims, CFL_SIZE);

	md_zfill(ND, data->img_dims, fftm, 1.);
	fftmod(ND, data->img_dims, data->flags, fftm, fftm);

	data->fftmod = multiplace_move(ND, data->img_dims, CFL_SIZE, fftm);

	md_free(fftm);



	complex float* roll = md_alloc(ND, data->img_dims, CFL_SIZE);

	rolloff_correction(conf.decomp ? 1. : data->grid_conf.os, data->width, data->beta, data->img_dims, roll);

	data->roll = multiplace_move(ND, data->img_dims, CFL_SIZE, roll);

	md_free(roll);


	complex float* linphase;
	float scx;

	if (conf.decomp) {

		linphase = compute_linphases(N, data->lph_dims, data->flags, data->img_dims);

		md_calc_strides(ND, data->lph_strs, data->lph_dims, CFL_SIZE);

		if (!conf.toeplitz)
			md_zmul2(ND, data->lph_dims, data->lph_strs, linphase, data->lph_strs, linphase, data->img_strs, multiplace_read(data->roll, linphase));

		for (int i = 0; i < (int)data->N; i++)
			if ((data->img_dims[i] > 1) && MD_IS_SET(data->flags, i))
				data->factors[i] = 2;

		scx = 0.5;

	} else {

		linphase = md_alloc(ND, data->img_dims, CFL_SIZE);

		md_copy_dims(ND, data->lph_dims, data->img_dims);

		md_calc_strides(ND, data->lph_strs, data->lph_dims, CFL_SIZE);

		md_copy(ND, data->lph_dims, linphase, multiplace_read(data->roll, linphase), CFL_SIZE);

		multiplace_free(data->roll);
		data->roll = NULL;

		scx = sqrtf(0.5);
	}


	fftmod(ND, data->lph_dims, data->flags, linphase, linphase);
	fftscale(ND, data->lph_dims, data->flags, linphase, linphase);

	float scale = powf(scx, bitcount((data->flags) & md_nontriv_dims(N, data->lph_dims)));

	md_zsmul(ND, data->lph_dims, linphase, linphase, scale);

	data->linphase = multiplace_move(ND, data->lph_dims, CFL_SIZE, linphase);

	md_free(linphase);


	md_copy_dims(ND, data->cml_dims, data->cim_dims);
	data->cml_dims[N + 0] = data->lph_dims[N + 0];

	md_copy_dims(ND, data->cmT_dims, data->cml_dims);

	if (basis) {

		assert(1 == data->cml_dims[5]);
		data->cmT_dims[5] = data->cml_dims[6];
		data->cmT_dims[6] = 1;

		assert(1 == data->cim_dims[5]);
		data->ciT_dims[5] = data->cim_dims[6];
		data->ciT_dims[6] = 1;
	}

	md_calc_strides(ND, data->cml_strs, data->cml_dims, CFL_SIZE);


	md_copy_dims(ND, data->cm2_dims, data->cim_dims);

	for (int i = 0; i < (int)N; i++)
		if (conf.decomp && MD_IS_SET(data->flags, i))
			data->cm2_dims[i] = (1 == cim_dims[i]) ? 1 : (2 * cim_dims[i]);



	data->fft_op = linop_fft_create(ND, data->cml_dims, data->flags | data->conf.cfft);

	if (conf.pcycle || conf.lowmem) {

		debug_printf(DP_DEBUG1, "NUFFT: %s mode\n", conf.lowmem ? "low-mem" : "pcycle");
		data->cycle = 0;
		data->cfft_op = linop_fft_create(N, data->cim_dims, data->flags | data->conf.cfft);
	}

	return PTR_PASS(data);
}


static void nufft_set_traj(struct nufft_data* data, int N,
			   const long trj_dims[N], const complex float* traj,
			   const long wgh_dims[N], const complex float* weights,
			   const long bas_dims[N], const complex float* basis)
{
	int ND = N + 1;

	if (NULL != traj) {

		assert(md_check_equal_dims(N, trj_dims, data->trj_dims, ~0));

		multiplace_free(data->traj);

		data->traj = multiplace_move(N, trj_dims, CFL_SIZE, traj);
	}

	if (NULL != basis) {

		//	conf.toeplitz = false;
		debug_print_dims(DP_DEBUG1, N, bas_dims);

		assert(!md_check_dimensions(N, bas_dims, (1 << 5) | (1 << 6)));

		data->out_dims[5] = bas_dims[5];	// TE
		data->out_dims[6] = 1;			// COEFF

		if (1 == data->ksp_dims[6])
			data->ksp_dims[6] = bas_dims[6];

		assert(data->ksp_dims[6] == bas_dims[6]);

		// recompute
		md_calc_strides(ND, data->out_strs, data->out_dims, CFL_SIZE);
		md_calc_strides(ND, data->ksp_strs, data->ksp_dims, CFL_SIZE);

		md_copy_dims(N, data->bas_dims, bas_dims);
		data->bas_dims[N] = 1;

		md_calc_strides(ND, data->bas_strs, data->bas_dims, CFL_SIZE);

		multiplace_free(data->basis);

		data->basis = multiplace_move(ND, data->bas_dims, CFL_SIZE, basis);
	}

	if (NULL != weights) {

		md_copy_dims(N, data->wgh_dims, wgh_dims);
		data->wgh_dims[N] = 1;

		md_calc_strides(ND, data->wgh_strs, data->wgh_dims, CFL_SIZE);

		multiplace_free(data->weights);

		data->weights = multiplace_move(ND, data->wgh_dims, CFL_SIZE, weights);
	}

	if (data->conf.toeplitz) {

		debug_printf(DP_DEBUG1, "NUFFT: Toeplitz mode\n");

		md_copy_dims(ND, data->psf_dims, data->lph_dims);

		for (int i = 0; i < (int)N; i++)
			if (!MD_IS_SET(data->flags, i))
				data->psf_dims[i] = MAX(data->trj_dims[i], ((NULL != weights) ? data->wgh_dims[i] : 0));

		if (NULL != basis) {

			debug_printf(DP_DEBUG3, "psf_dims: ");
			debug_print_dims(DP_DEBUG3, N, data->psf_dims);

			data->psf_dims[6] = data->bas_dims[6];
			data->psf_dims[5] = data->bas_dims[6];
		}

		md_calc_strides(ND, data->psf_strs, data->psf_dims, CFL_SIZE);

		if (!data->conf.nopsf && (NULL != data->traj)) {

			const complex float* psf = compute_psf2(N, data->psf_dims, data->flags, data->trj_dims, traj,
						data->bas_dims, multiplace_read(data->basis, traj), data->wgh_dims, multiplace_read(data->weights, traj),
						true /*conf.periodic*/, data->conf.lowmem);

			multiplace_free(data->psf);

			data->psf = multiplace_move(ND, data->psf_dims, CFL_SIZE, psf);

			md_free(psf);
		}
	}
}


static void nufft_free_data(const linop_data_t* data);
static void nufft_apply(const linop_data_t* _data, complex float* dst, const complex float* src);
static void nufft_apply_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src);
static void nufft_apply_normal(const linop_data_t* _data, complex float* dst, const complex float* src);



static struct linop_s* nufft_create3(int N,
			     const long ksp_dims[N],
			     const long cim_dims[N],
			     const long traj_dims[N],
			     const complex float* traj,
			     const long wgh_dims[N],
			     const complex float* weights,
			     const long bas_dims[N],
			     const complex float* basis,
			     struct nufft_conf_s conf)
{
	debug_printf(DP_DEBUG1, "ksp : ");
	debug_print_dims(DP_DEBUG1, N, ksp_dims);
	debug_printf(DP_DEBUG1, "cim : ");
	debug_print_dims(DP_DEBUG1, N, cim_dims);
	debug_printf(DP_DEBUG1, "traj: ");
	debug_print_dims(DP_DEBUG1, N, traj_dims);

	if (NULL != weights) {

		debug_printf(DP_DEBUG1, "wgh : ");
		debug_print_dims(DP_DEBUG1, N, wgh_dims);
	}

	if (NULL != basis) {

		debug_printf(DP_DEBUG1, "bas : ");
		debug_print_dims(DP_DEBUG1, N, bas_dims);
	}


//	assert(md_check_compat(N, ~data->flags, ksp_dims, cim_dims));
//	assert(md_check_bounds(N, ~data->flags, cim_dims, ksp_dims));
	assert((1 == md_calc_size(N, ksp_dims)) || md_check_bounds(N, ~(conf.flags | (NULL == basis ? 0 : (1 << 6))), cim_dims, ksp_dims));

	// extend internal dimensions by one for linear phases
	int ND = N + 1;

	assert((1 == md_calc_size(N, traj_dims)) || (bitcount(conf.flags) == traj_dims[0]));

	long chk_dims[N];
	md_select_dims(N, ~conf.flags, chk_dims, traj_dims);
	assert((1 == md_calc_size(N, ksp_dims)) || md_check_compat(N, ~0ul, chk_dims, ksp_dims));
//	assert(md_check_bounds(N, ~0ul, chk_dims, ksp_dims));


	auto data = nufft_create_data(N, cim_dims, NULL != basis, conf);


	md_copy_dims(N, data->ksp_dims, ksp_dims);
	data->ksp_dims[N] = 1;

	md_copy_dims(N, data->out_dims, ksp_dims);
	data->out_dims[N] = 1;

	md_copy_dims(N, data->trj_dims, traj_dims);
	data->trj_dims[N] = 1;

	md_calc_strides(ND, data->trj_strs, data->trj_dims, CFL_SIZE);
	md_calc_strides(ND, data->ksp_strs, data->ksp_dims, CFL_SIZE);
	md_calc_strides(ND, data->out_strs, data->out_dims, CFL_SIZE);

	nufft_set_traj(data, N, traj_dims, traj, wgh_dims, weights, bas_dims, basis);

	long out_dims[N];
	md_copy_dims(N, out_dims, data->out_dims);

	return linop_create(N, out_dims, N, cim_dims,
			CAST_UP(data), nufft_apply, nufft_apply_adjoint, nufft_apply_normal, NULL, nufft_free_data);
}


struct linop_s* nufft_create2(int N,
			     const long ksp_dims[N],
			     const long cim_dims[N],
			     const long traj_dims[N],
			     const complex float* traj,
			     const long wgh_dims[N],
			     const complex float* weights,
			     const long bas_dims[N],
			     const complex float* basis,
			     struct nufft_conf_s conf)
{
	if (0 <= conf.loopdim) {

		int d = conf.loopdim;
		const long L = ksp_dims[d];

		assert(d < (int)N);
		assert((NULL == weights) || (1 == wgh_dims[d]));
		assert((NULL == basis) || (1 == bas_dims[d]));
		assert(1 == traj_dims[d]);
		assert(L == cim_dims[d]);

		if (1 < L) {

			debug_printf(DP_WARN, "NEW NUFFT LOOP CODE\n");

			long ksp1_dims[N];
			md_select_dims(N, ~MD_BIT(d), ksp1_dims, ksp_dims);

			long cim1_dims[N];
			md_select_dims(N, ~MD_BIT(d), cim1_dims, cim_dims);

			auto nu = nufft_create2(N, ksp1_dims, cim1_dims, traj_dims, traj, wgh_dims, weights, bas_dims, basis, conf);

			long out_dims[N];
			md_copy_dims(N, out_dims, ksp_dims);

			if (NULL != basis)
				out_dims[6] = 1;

			long istrs[N];
			long ostrs[N];

			md_calc_strides(N, istrs, cim_dims, CFL_SIZE);
			md_calc_strides(N, ostrs, out_dims, CFL_SIZE);

			istrs[d] = 0;
			ostrs[d] = 0;

			auto nu1 = linop_copy_wrapper(N, istrs, ostrs, nu);

			long loop_dims[N];
			md_select_dims(N, MD_BIT(d), loop_dims, out_dims);

			auto nu2 = linop_loop(N, loop_dims, nu1);

			linop_free(nu);
			linop_free(nu1);

			return nu2;
		}
	}

	return nufft_create3(N, ksp_dims, cim_dims,
			traj_dims, traj, wgh_dims, weights,
			bas_dims, basis, conf);
}

static void nufft_normal_only(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	UNUSED(_data);
	UNUSED(src);
	UNUSED(dst);

	error("NuFFT with normal operator only!");
}

struct linop_s* nufft_create_normal(int N, const long cim_dims[N],
				    int ND, const long psf_dims[ND], const complex float* psf,
				    bool basis, struct nufft_conf_s conf)
{
	debug_printf(DP_DEBUG1, "cim : ");
	debug_print_dims(DP_DEBUG1, N, cim_dims);

	debug_printf(DP_DEBUG1, "psf : ");
	debug_print_dims(DP_DEBUG1, ND, psf_dims);

	auto data = nufft_create_data(N, cim_dims, basis, conf);

	md_copy_dims(ND, data->psf_dims, psf_dims);

	assert(md_check_equal_dims(ND, data->psf_dims, data->lph_dims, data->flags));
	assert(conf.toeplitz);

	long out_dims[N];
	md_singleton_dims(N, out_dims);

	auto result = linop_create(N, out_dims, N, cim_dims,
			CAST_UP(data), nufft_normal_only, nufft_normal_only, nufft_apply_normal, NULL, nufft_free_data);

	if (NULL != psf)
		nufft_update_psf(result, ND, psf_dims, psf);

	return result;
}

struct linop_s* nufft_create(int N,				///< Number of dimension
			     const long ksp_dims[N],		///< kspace dimension
			     const long cim_dims[N],		///< Coil images dimension
			     const long traj_dims[N],		///< Trajectory dimension
			     const complex float* traj,		///< Trajectory
			     const complex float* weights,	///< Weights, ex, soft-gating or density compensation
			     struct nufft_conf_s conf)		///< NUFFT configuration options
{
	long wgh_dims[N];
	md_select_dims(N, ~MD_BIT(0), wgh_dims, traj_dims);

	return nufft_create2(N, ksp_dims, cim_dims, traj_dims, traj, wgh_dims, weights, NULL, NULL, conf);
}






static void nufft_free_data(const linop_data_t* _data)
{
	auto data = CAST_DOWN(nufft_data, _data);

	xfree(data->ksp_dims);
	xfree(data->cim_dims);
	xfree(data->cml_dims);
	xfree(data->img_dims);
	xfree(data->trj_dims);
	xfree(data->lph_dims);
	xfree(data->psf_dims);
	xfree(data->wgh_dims);
	xfree(data->bas_dims);
	xfree(data->out_dims);
	xfree(data->ciT_dims);
	xfree(data->cmT_dims);

	xfree(data->ksp_strs);
	xfree(data->cim_strs);
	xfree(data->cml_strs);
	xfree(data->img_strs);
	xfree(data->trj_strs);
	xfree(data->lph_strs);
	xfree(data->psf_strs);
	xfree(data->wgh_strs);
	xfree(data->bas_strs);
	xfree(data->out_strs);

	xfree(data->cm2_dims);
	xfree(data->factors);

	multiplace_free(data->linphase);
	multiplace_free(data->psf);
	multiplace_free(data->fftmod);
	multiplace_free(data->weights);
	multiplace_free(data->roll);
	multiplace_free(data->basis);
	multiplace_free(data->traj);

	linop_free(data->fft_op);

	if (data->conf.pcycle || data->conf.lowmem)
		linop_free(data->cfft_op);

	xfree(data);
}




// Forward: from image to kspace
static void nufft_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(nufft_data, _data);

	int ND = data->N + 1;

	if (data->conf.toeplitz) {

		complex float* tmp = md_alloc_sameplace(ND, data->cim_dims, CFL_SIZE, dst);
		md_zmul2(ND, data->cim_dims, data->cim_strs, tmp, data->cim_strs, src, data->img_strs, multiplace_read(data->roll, src));
		src = tmp;
	}


	complex float* grid = md_alloc_sameplace(ND, data->cml_dims, CFL_SIZE, dst);

	md_zmul2(ND, data->cml_dims, data->cml_strs, grid, data->cim_strs, src, data->lph_strs, multiplace_read(data->linphase, src));

	if (data->conf.toeplitz)
		md_free(src);

	linop_forward(data->fft_op, ND, data->cml_dims, grid, ND, data->cml_dims, grid);

	md_zmul2(ND, data->cml_dims, data->cml_strs, grid, data->cml_strs, grid, data->img_strs, multiplace_read(data->fftmod, src));


	complex float* gridX = md_alloc_sameplace(data->N, data->cm2_dims, CFL_SIZE, dst);

	md_recompose(data->N, data->factors, data->cm2_dims, gridX, data->cml_dims, grid, CFL_SIZE);
	md_free(grid);


	complex float* tmp = dst;

	if (NULL != data->basis)
		tmp = md_alloc_sameplace(ND, data->ksp_dims, CFL_SIZE, dst);

	md_clear(ND, data->ksp_dims, tmp, CFL_SIZE);

	grid2H(&data->grid_conf, ND, data->trj_dims, multiplace_read(data->traj, src), data->ksp_dims, tmp, data->cm2_dims, gridX);

	md_free(gridX);

	if (NULL != data->basis) {

		md_ztenmul(data->N, data->out_dims, dst, data->ksp_dims, tmp, data->bas_dims, multiplace_read(data->basis, src));
		md_free(tmp);
	}

	if (NULL != data->weights)
		md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->out_strs, dst, data->wgh_strs, multiplace_read(data->weights, src));
}


static void split_nufft_adjoint(const struct nufft_data* data, int ND, complex float* grid, const complex float* src)
{
	debug_printf(DP_DEBUG1, "nufft_adj split calculation for lowmem\n");

	// FFT_FLAGS, because the image dimensions can always occur in the trajectory

	long nontriv_traj_flags = FFT_FLAGS | md_nontriv_dims(data->N, data->trj_dims);

	long cm2_reduced_dims[ND];
	md_select_dims(ND, nontriv_traj_flags, cm2_reduced_dims, data->cm2_dims);


	// everything not in traj dims is done separately

	long max_dims[ND];
	md_set_dims(ND, max_dims, 1);
	md_max_dims(ND, ~nontriv_traj_flags, max_dims, data->cm2_dims, data->ksp_dims);

	long iter_dims[data->N];

	// All dimension not in the nontriv_traj_flags and all dimensions in ksp dims but not in cm2 dims
	// We need to exclude these last dimensions, because we have to sum over sum in the gridding procedure

	long iter_flags = ~(  nontriv_traj_flags
			    | (  md_nontriv_dims(ND, data->ksp_dims)
			       & ~md_nontriv_dims(ND, data->cm2_dims)));

	md_select_dims(data->N, iter_flags, iter_dims, max_dims);

	long ksp_reduced_dims[ND];
	md_select_dims(ND, nontriv_traj_flags, ksp_reduced_dims, data->ksp_dims);

	long ksp_reduced_strs[ND];
	md_calc_strides(ND, ksp_reduced_strs, ksp_reduced_dims, CFL_SIZE);

	long ksp_strs[ND];
	md_calc_strides(ND, ksp_strs, data->ksp_dims, CFL_SIZE);

	long cml_reduced_dims[ND];
	cml_reduced_dims[data->N] = data->cml_dims[data->N];
	md_select_dims(data->N, nontriv_traj_flags, cml_reduced_dims, data->cml_dims);

	long cml_reduced_strs[ND];
	md_calc_strides(ND, cml_reduced_strs, cml_reduced_dims, CFL_SIZE);


	complex float* grid_reduced = md_alloc_sameplace(ND, cml_reduced_dims, CFL_SIZE, src);
	complex float* gridX = md_alloc_sameplace(ND, cm2_reduced_dims, CFL_SIZE, src);
	complex float* src_reduced = md_alloc_sameplace(ND, ksp_reduced_dims, CFL_SIZE, src);

	long pos[ND];
	md_set_dims(ND, pos, 0L);

	do {
		// sum over additional dimensions in the k-space
		long sum_dims[ND];
		long sum_flags = ~(nontriv_traj_flags | iter_flags);

		md_select_dims(ND, sum_flags, sum_dims, max_dims);

		md_clear(ND, cm2_reduced_dims, gridX, CFL_SIZE);

		do {
			md_copy_block2(data->N, pos, ksp_reduced_dims, ksp_reduced_strs, src_reduced, data->ksp_dims, ksp_strs, src, CFL_SIZE);

			grid2(&data->grid_conf, ND, data->trj_dims, multiplace_read(data->traj, src), cm2_reduced_dims, gridX, ksp_reduced_dims, src_reduced);

		} while(md_next(ND, sum_dims, sum_flags, pos));


		md_decompose(data->N, data->factors, cml_reduced_dims, grid_reduced, cm2_reduced_dims, gridX, CFL_SIZE);

		md_copy_block2(ND, pos, data->cml_dims, data->cml_strs, grid, cml_reduced_dims, cml_reduced_strs, grid_reduced, CFL_SIZE);

	} while(md_next(data->N, iter_dims, ~0L, pos));

	md_zmulc2(ND, data->cml_dims, data->cml_strs, grid, data->cml_strs, grid, data->img_strs, multiplace_read(data->fftmod, src));

	md_free(grid_reduced);
	md_free(gridX);
	md_free(src_reduced);
}




// Adjoint: from kspace to image
static void nufft_apply_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(nufft_data, _data);

#ifdef USE_CUDA
	//assert(!cuda_ondevice(src));
#endif
	int ND = data->N + 1;

	complex float* wdat = NULL;

	if (NULL != data->weights) {

		wdat = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, dst);
		md_zmulc2(data->N, data->out_dims, data->out_strs, wdat, data->out_strs, src, data->wgh_strs, multiplace_read(data->weights, dst));
		src = wdat;
	}

	complex float* bdat = NULL;

	if (NULL != data->basis) {

		bdat = md_alloc_sameplace(data->N, data->ksp_dims, CFL_SIZE, dst);
		md_ztenmulc(data->N, data->ksp_dims, bdat, data->out_dims, src, data->bas_dims, multiplace_read(data->basis, dst));
		src = bdat;
	}

	complex float* grid = md_alloc_sameplace(ND, data->cml_dims, CFL_SIZE, dst);
	complex float* gridX = NULL;

	if (data->conf.lowmem) {

		split_nufft_adjoint(data, ND, grid, src);

	} else {

		gridX = md_alloc_sameplace(data->N, data->cm2_dims, CFL_SIZE, dst);

		md_clear(data->N, data->cm2_dims, gridX, CFL_SIZE);

		grid2(&data->grid_conf, ND, data->trj_dims, multiplace_read(data->traj, dst), data->cm2_dims, gridX, data->ksp_dims, src);
	}

	md_free(bdat);
	md_free(wdat);

	if (!data->conf.lowmem) {

		md_decompose(data->N, data->factors, data->cml_dims, grid, data->cm2_dims, gridX, CFL_SIZE);

		md_free(gridX);

		md_zmulc2(ND, data->cml_dims, data->cml_strs, grid, data->cml_strs, grid, data->img_strs, multiplace_read(data->fftmod, dst));
	}

	linop_adjoint(data->fft_op, ND, data->cml_dims, grid, ND, data->cml_dims, grid);

	md_ztenmulc2(ND, data->cml_dims, data->cim_strs, dst, data->cml_strs, grid, data->lph_strs, multiplace_read(data->linphase, dst));

	md_free(grid);

	if (data->conf.toeplitz)
		md_zmul2(ND, data->cim_dims, data->cim_strs, dst, data->cim_strs, dst, data->img_strs, multiplace_read(data->roll, dst));
}





static void toeplitz_mult(const struct nufft_data* data, complex float* dst, const complex float* src)
{
	int ND = data->N + 1;

	const complex float* linphase = multiplace_read(data->linphase, src);
	const complex float* psf = multiplace_read(data->psf, src);

	complex float* grid = md_alloc_sameplace(ND, data->cml_dims, CFL_SIZE, dst);

	md_zmul2(ND, data->cml_dims, data->cml_strs, grid, data->cim_strs, src, data->lph_strs, linphase);

	linop_forward(data->fft_op, ND, data->cml_dims, grid, ND, data->cml_dims, grid);

	complex float* gridT = md_alloc_sameplace(ND, data->cmT_dims, CFL_SIZE, dst);

	md_ztenmul(ND, data->cmT_dims, gridT, data->cml_dims, grid, data->psf_dims, psf);

	md_free(grid);

	linop_adjoint(data->fft_op, ND, data->cml_dims, gridT, ND, data->cml_dims, gridT);

	md_ztenmulc2(ND, data->cml_dims, data->cim_strs, dst, data->cml_strs, gridT, data->lph_strs, linphase);

	md_free(gridT);
}



static void toeplitz_mult_lowmem(const struct nufft_data* data, int i, complex float* dst, const complex float* src)
{
	const complex float* linphase = multiplace_read(data->linphase, src);
	const complex float* psf = multiplace_read(data->psf, src);

	const complex float* clinphase = linphase + i * md_calc_size(data->N, data->lph_dims);
	const complex float* cpsf = psf + i * md_calc_size(data->N, data->psf_dims);

	complex float* grid = md_alloc_sameplace(data->N, data->cim_dims, CFL_SIZE, dst);

	md_zmul2(data->N, data->cim_dims, data->cim_strs, grid, data->cim_strs, src, data->img_strs, clinphase);

	linop_forward(data->cfft_op, data->N, data->cim_dims, grid, data->N, data->cim_dims, grid);

	complex float* gridT = md_alloc_sameplace(data->N, data->ciT_dims, CFL_SIZE, dst);

	md_ztenmul(data->N, data->ciT_dims, gridT, data->cim_dims, grid, data->psf_dims, cpsf);

	md_free(grid);

	linop_adjoint(data->cfft_op, data->N, data->cim_dims, gridT, data->N, data->cim_dims, gridT);

	md_zfmacc2(data->N, data->cim_dims, data->cim_strs, dst, data->cim_strs, gridT, data->img_strs, clinphase);

	md_free(gridT);
}



static void nufft_apply_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(nufft_data, _data);

	if (!data->conf.toeplitz) {

		complex float* tmp_ksp = md_alloc_sameplace(data->N + 1, data->out_dims, CFL_SIZE, dst);

		nufft_apply(_data, tmp_ksp, src);
		nufft_apply_adjoint(_data, dst, tmp_ksp);

		md_free(tmp_ksp);

		return;
	}

	if (!(data->conf.pcycle || data->conf.lowmem)) {

		toeplitz_mult(data, dst, src);

		return;
	}

	// low mem versions

	assert(dst != src);

	int ncycles = data->lph_dims[data->N];

	if (data->conf.pcycle)
		((struct nufft_data*) data)->cycle = (data->cycle + 1) % ncycles;	// FIXME:

	md_clear(data->N, data->cim_dims, dst, CFL_SIZE);

	for (int i = 0; i < ncycles; i++) {

		if (data->conf.pcycle && (i != (int)data->cycle))
			continue;

		toeplitz_mult_lowmem(data, i, dst, src);
	}
}



int nufft_get_psf_dims(const struct linop_s* nufft, int N, long psf_dims[N])
{
	auto lop_data = linop_get_data(nufft);
	assert(NULL != lop_data);

	auto data = CAST_DOWN(nufft_data, lop_data);

	if (N > 0)
		md_copy_dims(N, psf_dims, data->psf_dims);

	return data->N + 1;
}


void nufft_get_psf2(const struct linop_s* nufft, int N, const long psf_dims[N], const long psf_strs[N], complex float* psf)
{
	auto lop_data = linop_get_data(nufft);
	assert(NULL != lop_data);

	auto data = CAST_DOWN(nufft_data, lop_data);

	assert(N == data->N + 1);
	md_check_equal_dims(N, psf_dims, data->psf_dims, ~0);

	md_copy2(N, psf_dims, psf_strs, psf, data->psf_strs, multiplace_read(data->psf, psf), CFL_SIZE);
}


void nufft_get_psf(const struct linop_s* nufft, int N, const long psf_dims[N], complex float* psf)
{
	nufft_get_psf2(nufft, N, psf_dims, MD_STRIDES(N, psf_dims, CFL_SIZE), psf);
}

void nufft_update_traj(	const struct linop_s* nufft, int N,
			const long trj_dims[N], const complex float* traj,
			const long wgh_dims[N], const complex float* weights,
			const long bas_dims[N], const complex float* basis)
{
	auto data = CAST_DOWN(nufft_data, linop_get_data(nufft));

	assert((int)data->N == N);

	nufft_set_traj(data, N, trj_dims, traj, wgh_dims, weights, bas_dims, basis);
}

void nufft_update_psf2(const struct linop_s* nufft, int ND, const long psf_dims[ND], const long psf_strs[ND], const complex float* psf)
{
	auto data = CAST_DOWN(nufft_data, linop_get_data(nufft));

	assert(md_check_equal_dims(ND, data->psf_dims, psf_dims, ~0));

	multiplace_free(data->psf);

	data->psf = multiplace_move2(ND, psf_dims, psf_strs, CFL_SIZE, psf);
}

void nufft_update_psf(const struct linop_s* nufft, int ND, const long psf_dims[ND], const complex float* psf)
{
	nufft_update_psf2(nufft, ND, psf_dims, MD_STRIDES(ND, psf_dims, CFL_SIZE), psf);
}


