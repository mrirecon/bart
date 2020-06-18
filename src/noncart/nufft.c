/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2016-2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2017 Frank Ong
 * 2014-2020 Martin Uecker
 * 2018      Sebastian Rosenzweig
 *
 */

#include <math.h>
#include <complex.h>
#include <assert.h>
#include <stdbool.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/filter.h"
#include "num/fft.h"
#include "num/shuffle.h"
#include "num/ops.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

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
};

#include "nufft_priv.h"

DEF_TYPEID(nufft_data);


static void nufft_free_data(const linop_data_t* data);
static void nufft_apply(const linop_data_t* _data, complex float* dst, const complex float* src);
static void nufft_apply_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src);
static void nufft_apply_normal(const linop_data_t* _data, complex float* dst, const complex float* src);


static void toeplitz_mult(const struct nufft_data* data, complex float* dst, const complex float* src);


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



static void compute_kern_basis(unsigned int N, unsigned int flags, const long pos[N],
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

	complex float* tmp = md_alloc(N, max_dims, CFL_SIZE);

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

	md_zsmul(N, max_dims, tmp, tmp, (double)bas_dims[6]);	// FIXME: Why?

	md_ztenmulc2(N, ma3_dims, krn_strs, krn, 
			max_strs, (void*)tmp + tmp_off,
			baT_strs, (void*)basis + bas_off);

	md_free(tmp);
}



static void compute_kern(unsigned int N, unsigned int flags, const long pos[N],
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



complex float* compute_psf(unsigned int N, const long img_dims[N], const long trj_dims[N], const complex float* traj,
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
		ksp2_dims[N - 1] = trj2_dims[5];
		trj2_dims[N - 1] = trj2_dims[5];
		trj2_dims[5] = 1;	// FIXME copy?
	}

	struct nufft_conf_s conf = nufft_conf_defaults;
	conf.periodic = periodic;
	conf.toeplitz = false;	// avoid infinite loop


	debug_printf(DP_DEBUG2, "nufft kernel dims: ");
	debug_print_dims(DP_DEBUG2, N, ksp2_dims);

	debug_printf(DP_DEBUG2, "nufft psf dims:    ");
	debug_print_dims(DP_DEBUG2, N, img2_dims);

	debug_printf(DP_DEBUG2, "nufft traj dims:   ");
	debug_print_dims(DP_DEBUG2, N, trj2_dims);

	complex float* psft = NULL;

	long pos[N];
	for (unsigned int i = 0; i < N; i++)
		pos[i] = 0;

	long A = md_calc_size(N, ksp2_dims);
	long B = md_calc_size(N - 1, ksp2_dims) + md_calc_size(N - 1, img2_dims);
	long C = md_calc_size(N, img2_dims);

	if ((A <= B) || !lowmem) {

		debug_printf(DP_DEBUG1, "Allocating %ld (vs. %ld) + %ld\n", A, B, C);

		complex float* ones = md_alloc(N, ksp2_dims, CFL_SIZE);

		compute_kern(N, ~0u, pos, ksp2_dims, ones, bas2_dims, basis, wgh2_dims, weights);

		psft = md_alloc(N, img2_dims, CFL_SIZE);

		struct linop_s* op2 = nufft_create(N, ksp2_dims, img2_dims, trj2_dims, traj, NULL, conf);

		linop_adjoint_unchecked(op2, psft, ones);

		linop_free(op2);

		md_free(ones);

	} else {

		debug_printf(DP_DEBUG1, "Allocating %ld (vs. %ld) + %ld\n", B, A, C);

		psft = md_calloc(N, img2_dims, CFL_SIZE);

		long trj2_strs[N];
		md_calc_strides(N, trj2_strs, trj2_dims, CFL_SIZE);

		complex float* ones = md_alloc(N - 1, ksp2_dims, CFL_SIZE);
		complex float* tmp = md_alloc(N - 1, img2_dims, CFL_SIZE);

		assert(!((1 != trj2_dims[N - 1]) && (NULL == basis)));

		for (long i = 0; i < trj2_dims[N - 1]; i++) {

			debug_printf(DP_DEBUG1, "KERN %03ld\n", i);

			unsigned int flags = ~0u;

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

	complex float* traj2 = md_alloc(ND, trj_dims, CFL_SIZE);
	md_zsmul(ND, trj_dims, traj2, traj, 2.);

	complex float* psft = compute_psf(ND, img2_dims, trj_dims, traj2, bas_dims, basis, wgh_dims, weights, periodic, lowmem);
	md_free(traj2);

	fftuc(ND, img2_dims, flags, psft, psft);

	float scale = 1.;
	for (int i = 0; i < N; i++)
		scale *= ((img2_dims[i] > 1) && (MD_IS_SET(flags, i))) ? 4. : 1.;

	md_zsmul(ND, img2_dims, psft, psft, scale);

	// reformat

	complex float* psf = md_alloc(ND, psf_dims, CFL_SIZE);

	long factors[N];

	for (int i = 0; i < N; i++)
		factors[i] = ((img_dims[i] > 1) && (MD_IS_SET(flags, i))) ? 2 : 1;

	md_decompose(N + 0, factors, psf_dims, psf, img2_dims, psft, CFL_SIZE);

	md_free(psft);
	return psf;
}



static struct linop_s* nufft_create3(unsigned int N,
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
	PTR_ALLOC(struct nufft_data, data);
	SET_TYPEID(nufft_data, data);

	data->N = N;
	data->traj = traj;
	data->conf = conf;
	data->flags = conf.flags;

	data->width = 3.;
	data->beta = calc_beta(2., data->width);

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

	// dim 0 must be transformed (we treat this special in the trajectory)
	assert(MD_IS_SET(data->flags, 0));
//	assert(md_check_compat(N, ~data->flags, ksp_dims, cim_dims));
	assert(md_check_bounds(N, ~data->flags, cim_dims, ksp_dims));
	assert(0 == (data->flags & conf.cfft));

	assert(!((!conf.decomp) && conf.toeplitz));

	data->grid_conf = (struct grid_conf_s){

		.width = data->width,
		.os = 2.,
		.periodic = data->conf.periodic,
		.beta = data->beta,
	};


	// extend internal dimensions by one for linear phases
	unsigned int ND = N + 1;

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
	

	md_copy_dims(N, data->cim_dims, cim_dims);
	data->cim_dims[N] = 1;

	md_copy_dims(N, data->ksp_dims, ksp_dims);
	data->ksp_dims[N] = 1;

	md_copy_dims(ND, data->ciT_dims, data->cim_dims);
	md_copy_dims(ND, data->out_dims, data->ksp_dims);


	md_select_dims(ND, data->flags, data->img_dims, data->cim_dims);

	assert(bitcount(data->flags) == traj_dims[0]);

	long chk_dims[N];
	md_select_dims(N, ~data->flags, chk_dims, traj_dims);
	assert(md_check_compat(N, ~0ul, chk_dims, ksp_dims));
//	assert(md_check_bounds(N, ~0ul, chk_dims, ksp_dims));


	md_copy_dims(N, data->trj_dims, traj_dims);
	data->trj_dims[N] = 1;


	md_calc_strides(ND, data->cim_strs, data->cim_dims, CFL_SIZE);
	md_calc_strides(ND, data->img_strs, data->img_dims, CFL_SIZE);
	md_calc_strides(ND, data->trj_strs, data->trj_dims, CFL_SIZE);
	md_calc_strides(ND, data->ksp_strs, data->ksp_dims, CFL_SIZE);
	md_calc_strides(ND, data->out_strs, data->out_dims, CFL_SIZE);


	data->basis = NULL;

	if (NULL != basis) {

	//	conf.toeplitz = false;
		assert(!md_check_dimensions(N, bas_dims, (1 << 5) | (1 << 6)));

		data->out_dims[5] = bas_dims[5];	// TE
		data->out_dims[6] = 1;			// COEFF
		assert(data->ksp_dims[6] == bas_dims[6]);

		// recompute
		md_calc_strides(ND, data->out_strs, data->out_dims, CFL_SIZE);


		md_copy_dims(N, data->bas_dims, bas_dims);
		data->bas_dims[N] = 1;

		md_calc_strides(ND, data->bas_strs, data->bas_dims, CFL_SIZE);

		complex float* tmp = md_alloc(ND, data->bas_dims, CFL_SIZE);
		md_copy(ND, data->bas_dims, tmp, basis, CFL_SIZE);
		data->basis = tmp;
	}



	data->weights = NULL;

	if (NULL != weights) {

		md_copy_dims(N, data->wgh_dims, wgh_dims);
		data->wgh_dims[N] = 1;

		md_calc_strides(ND, data->wgh_strs, data->wgh_dims, CFL_SIZE);

		complex float* tmp = md_alloc(ND, data->wgh_dims, CFL_SIZE);
		md_copy(ND, data->wgh_dims, tmp, weights, CFL_SIZE);
		data->weights = tmp;
	}


	complex float* roll = md_alloc(ND, data->img_dims, CFL_SIZE);
	rolloff_correction(conf.decomp ? 1. : data->grid_conf.os, data->width, data->beta, data->img_dims, roll);
	data->roll = roll;

	complex float* linphase = NULL;

	if (conf.decomp) {

		linphase = compute_linphases(N, data->lph_dims, data->flags, data->img_dims);

		md_calc_strides(ND, data->lph_strs, data->lph_dims, CFL_SIZE);

		if (!conf.toeplitz)
			md_zmul2(ND, data->lph_dims, data->lph_strs, linphase, data->lph_strs, linphase, data->img_strs, data->roll);

	} else {

		linphase = roll;
		data->roll = NULL;

		md_copy_dims(ND, data->lph_dims, data->img_dims);
		md_calc_strides(ND, data->lph_strs, data->lph_dims, CFL_SIZE);
	}


	fftmod(ND, data->lph_dims, data->flags, linphase, linphase);
	fftscale(ND, data->lph_dims, data->flags, linphase, linphase);

	float scale = 1.;

	for (int i = 0; i < (int)N; i++)
		if ((data->lph_dims[i] > 1) && MD_IS_SET(data->flags, i))
			scale *= conf.decomp ? 0.5 : sqrtf(0.5);

	md_zsmul(ND, data->lph_dims, linphase, linphase, scale);


	data->factors = *TYPE_ALLOC(long[data->N]);

	for (int i = 0; i < (int)data->N; i++)
		if (data->conf.decomp && (data->img_dims[i] > 1) && MD_IS_SET(data->flags, i))
			data->factors[i] = 2;
		else
			data->factors[i] = 1;



	complex float* fftm = md_alloc(ND, data->img_dims, CFL_SIZE);
	md_zfill(ND, data->img_dims, fftm, 1.);
	fftmod(ND, data->img_dims, data->flags, fftm, fftm);
	data->fftmod = fftm;



	data->linphase = linphase;
	data->psf = NULL;
#ifdef USE_CUDA
	data->linphase_gpu = NULL;
	data->psf_gpu = NULL;
#endif
	if (conf.toeplitz) {

		debug_printf(DP_DEBUG1, "NUFFT: Toeplitz mode\n");

		md_copy_dims(ND, data->psf_dims, data->lph_dims);

		for (int i = 0; i < (int)N; i++)
			if (!MD_IS_SET(data->flags, i))
				data->psf_dims[i] = data->trj_dims[i];

		if (NULL != basis) {

			debug_printf(DP_DEBUG3, "psf_dims: ");
			debug_print_dims(DP_DEBUG3, N, data->psf_dims);
			data->psf_dims[6] = data->bas_dims[6];
			data->psf_dims[5] = data->bas_dims[6];
		}

		md_calc_strides(ND, data->psf_strs, data->psf_dims, CFL_SIZE);

		data->psf = compute_psf2(N, data->psf_dims, data->flags, data->trj_dims, data->traj,
					data->bas_dims, data->basis, data->wgh_dims, data->weights,
					true /*conf.periodic*/, conf.lowmem);
	}


	md_copy_dims(ND, data->cml_dims, data->cim_dims);
	data->cml_dims[N + 0] = data->lph_dims[N + 0];

	md_copy_dims(ND, data->cmT_dims, data->cml_dims);

	if (NULL != basis) {

		assert(1 == data->cml_dims[5]);
		data->cmT_dims[5] = data->cml_dims[6];
		data->cmT_dims[6] = 1;

		assert(1 == data->cim_dims[5]);
		data->ciT_dims[5] = data->cim_dims[6];
		data->ciT_dims[6] = 1;
	}

	md_calc_strides(ND, data->cml_strs, data->cml_dims, CFL_SIZE);

	data->cm2_dims = *TYPE_ALLOC(long[ND]);
	// !
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


	long out_dims[N];
	md_copy_dims(N, out_dims, data->out_dims);

	return linop_create(N, out_dims, N, cim_dims,
			CAST_UP(PTR_PASS(data)), nufft_apply, nufft_apply_adjoint, nufft_apply_normal, NULL, nufft_free_data);
}


struct linop_s* nufft_create2(unsigned int N,
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

			return nu2;
		}
	}

	return nufft_create3(N, ksp_dims, cim_dims,
			traj_dims, traj, wgh_dims, weights,
			bas_dims, basis, conf);
}



struct linop_s* nufft_create(unsigned int N,			///< Number of dimension
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

	md_free(data->linphase);
	md_free(data->psf);
	md_free(data->fftmod);
	md_free(data->weights);
	md_free(data->roll);
	md_free(data->basis);

#ifdef USE_CUDA
	md_free(data->linphase_gpu);
	md_free(data->psf_gpu);
#endif
	linop_free(data->fft_op);

	if (data->conf.pcycle || data->conf.lowmem)
		linop_free(data->cfft_op);

	xfree(data);
}




// Forward: from image to kspace
static void nufft_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(nufft_data, _data);

#ifdef USE_CUDA
	assert(!cuda_ondevice(src));
#endif
	assert(!data->conf.toeplitz); // if toeplitz linphase has no roll, so would need to be added

	int ND = data->N + 1;

	complex float* grid = md_alloc(ND, data->cml_dims, CFL_SIZE);

	md_zmul2(ND, data->cml_dims, data->cml_strs, grid, data->cim_strs, src, data->lph_strs, data->linphase);
	linop_forward(data->fft_op, ND, data->cml_dims, grid, ND, data->cml_dims, grid);
	md_zmul2(ND, data->cml_dims, data->cml_strs, grid, data->cml_strs, grid, data->img_strs, data->fftmod);


	complex float* gridX = md_alloc(data->N, data->cm2_dims, CFL_SIZE);

	md_recompose(data->N, data->factors, data->cm2_dims, gridX, data->cml_dims, grid, CFL_SIZE);
	md_free(grid);


	complex float* tmp = dst;

	if (NULL != data->basis)
		tmp = md_alloc(ND, data->ksp_dims, CFL_SIZE);

	md_clear(ND, data->ksp_dims, tmp, CFL_SIZE);
	grid2H(&data->grid_conf, ND, data->trj_dims, data->traj, data->ksp_dims, tmp, data->cm2_dims, gridX);

	md_free(gridX);

	if (NULL != data->basis) {

		md_ztenmul(data->N, data->out_dims, dst, data->ksp_dims, tmp, data->bas_dims, data->basis);
		md_free(tmp);
	}

	if (NULL != data->weights)
		md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->out_strs, dst, data->wgh_strs, data->weights);
}


// Adjoint: from kspace to image
static void nufft_apply_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(nufft_data, _data);

#ifdef USE_CUDA
	assert(!cuda_ondevice(src));
#endif
	int ND = data->N + 1;

	complex float* wdat = NULL;

	if (NULL != data->weights) {

		wdat = md_alloc(data->N, data->out_dims, CFL_SIZE);
		md_zmulc2(data->N, data->out_dims, data->out_strs, wdat, data->out_strs, src, data->wgh_strs, data->weights);
		src = wdat;
	}

	complex float* bdat = NULL;

	if (NULL != data->basis) {

		bdat = md_alloc(data->N, data->ksp_dims, CFL_SIZE);
		md_ztenmulc(data->N, data->ksp_dims, bdat, data->out_dims, src, data->bas_dims, data->basis);
		src = bdat;
	}

	complex float* gridX = md_calloc(data->N, data->cm2_dims, CFL_SIZE);

	grid2(&data->grid_conf, ND, data->trj_dims, data->traj, data->cm2_dims, gridX, data->ksp_dims, src);

	md_free(bdat);
	md_free(wdat);


	complex float* grid = md_alloc(ND, data->cml_dims, CFL_SIZE);

	md_decompose(data->N, data->factors, data->cml_dims, grid, data->cm2_dims, gridX, CFL_SIZE);
	md_free(gridX);
	md_zmulc2(ND, data->cml_dims, data->cml_strs, grid, data->cml_strs, grid, data->img_strs, data->fftmod);
	linop_adjoint(data->fft_op, ND, data->cml_dims, grid, ND, data->cml_dims, grid);

	md_clear(ND, data->cim_dims, dst, CFL_SIZE);
	md_zfmacc2(ND, data->cml_dims, data->cim_strs, dst, data->cml_strs, grid, data->lph_strs, data->linphase);

	md_free(grid);

	if (data->conf.toeplitz)
		md_zmul2(ND, data->cim_dims, data->cim_strs, dst, data->cim_strs, dst, data->img_strs, data->roll);
}





#ifdef USE_CUDA
static void gpu_alloc(const struct nufft_data* data)
{
	unsigned int ND = data->N + 1;

	if (NULL == data->linphase_gpu)
		((struct nufft_data*)data)->linphase_gpu = md_gpu_move(ND, data->lph_dims, data->linphase, CFL_SIZE);

	if (NULL == data->psf_gpu)
		((struct nufft_data*)data)->psf_gpu = md_gpu_move(ND, data->psf_dims, data->psf, CFL_SIZE);

}
#endif


static void toeplitz_mult(const struct nufft_data* data, complex float* dst, const complex float* src)
{
	unsigned int ND = data->N + 1;

	const complex float* linphase = data->linphase;
	const complex float* psf = data->psf;

#ifdef USE_CUDA
	if (cuda_ondevice(src)) {

		gpu_alloc(data);

		linphase = data->linphase_gpu;
		psf = data->psf_gpu;
	}
#endif
	complex float* grid = md_alloc_sameplace(ND, data->cml_dims, CFL_SIZE, dst);

	md_zmul2(ND, data->cml_dims, data->cml_strs, grid, data->cim_strs, src, data->lph_strs, linphase);

	linop_forward(data->fft_op, ND, data->cml_dims, grid, ND, data->cml_dims, grid);

	complex float* gridT = md_alloc_sameplace(ND, data->cmT_dims, CFL_SIZE, dst);

	md_ztenmul(ND, data->cmT_dims, gridT, data->cml_dims, grid, data->psf_dims, psf);

	md_free(grid);

	linop_adjoint(data->fft_op, ND, data->cml_dims, gridT, ND, data->cml_dims, gridT);

	md_clear(ND, data->cim_dims, dst, CFL_SIZE);
	md_zfmacc2(ND, data->cml_dims, data->cim_strs, dst, data->cml_strs, gridT, data->lph_strs, linphase);

	md_free(gridT);
}



static void toeplitz_mult_lowmem(const struct nufft_data* data, int i, complex float* dst, const complex float* src)
{
	const complex float* linphase = data->linphase;
	const complex float* psf = data->psf;

#ifdef USE_CUDA
	if (cuda_ondevice(src)) {

		gpu_alloc(data);

		linphase = data->linphase_gpu;
		psf = data->psf_gpu;
	}
#endif
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


static void toeplitz_mult_pcycle(const struct nufft_data* data, complex float* dst, const complex float* src)
{
	unsigned int ncycles = data->lph_dims[data->N];
        ((struct nufft_data*) data)->cycle = (data->cycle + 1) % ncycles;	// FIXME:

	assert(dst != src);

	md_clear(data->N, data->cim_dims, dst, CFL_SIZE);

	toeplitz_mult_lowmem(data, data->cycle, dst, src);
}


static void nufft_apply_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(nufft_data, _data);

	if (data->conf.toeplitz) {

		if (data->conf.pcycle) {

			toeplitz_mult_pcycle(data, dst, src);

		} else if (data->conf.lowmem) {

			int ncycles = data->lph_dims[data->N];

			assert(dst != src);

			md_clear(data->N, data->cim_dims, dst, CFL_SIZE);

			for (int i = 0; i < ncycles; i++)
				toeplitz_mult_lowmem(data, i, dst, src);

		} else {

			toeplitz_mult(data, dst, src);
		}

	} else {

		complex float* tmp_ksp = md_alloc(data->N + 1, data->out_dims, CFL_SIZE);

		nufft_apply(_data, tmp_ksp, src);
		nufft_apply_adjoint(_data, dst, tmp_ksp);

		md_free(tmp_ksp);
	}
}


/**
 * Estimate image dimensions from trajectory
 */
void estimate_im_dims(int N, unsigned long flags, long dims[N], const long tdims[N], const complex float* traj)
{
	int T = tdims[0];

	assert(T == (int)bitcount(flags));

	float max_dims[T];
	for (int i = 0; i < T; i++)
		max_dims[i] = 0.;

	for (long i = 0; i < md_calc_size(N - 1, tdims + 1); i++)
		for(int j = 0; j < tdims[0]; j++)
			max_dims[j] = MAX(cabsf(traj[j + tdims[0] * i]), max_dims[j]);

	for (int j = 0, t = 0; j < N; j++) {

		dims[j] = 1;

		if (MD_IS_SET(flags, j)) {

			dims[t] = (0. == max_dims[t]) ? 1 : (2 * (long)((2. * max_dims[t] + 1.5) / 2.));
			t++;
		}
	}
}

/**
 * Estimate fast square image dimensions from trajectory
 */
void estimate_fast_sq_im_dims(unsigned int N, long dims[3], const long tdims[N], const complex float* traj)
{
	float max_dims[3] = { 0., 0., 0. };

	for (long i = 0; i < md_calc_size(N - 1, tdims + 1); i++)
		for(int j = 0; j < 3; j++)
			max_dims[j] = MAX(cabsf(traj[j + tdims[0] * i]), max_dims[j]);


	// 2* is needed since we take the absolute value of the trajectory above, and it is scaled from
	// -DIM/2 to DIM/2
	long max_square = 2 * MAX(MAX(max_dims[0], max_dims[1]), max_dims[2]);


	// compute next fast size for Fourier transform.
	// That is the next number only composed of small prime factors,
	// i.e. 2, 3, 5 (and possibly 7?)

	// to avoid an infinite loop here, we constrain our search
	long fast_size = max_square;

	for ( ; fast_size <= 4 * max_square; ++fast_size) {

		long n = fast_size;

		while (0 == n % 2l) { n /= 2l; }
		while (0 == n % 3l) { n /= 3l; }
		while (0 == n % 5l) { n /= 5l; }
		while (0 == n % 7l) { n /= 7l; }

		if (n <= 1)
			break;
	}

	for (int j = 0; j < 3; j++)
		dims[j] = (0. == max_dims[j]) ? 1 : fast_size;
}

