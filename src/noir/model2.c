/* Copyright 2024. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 *
 * Uecker M, Hohage T, Block KT, Frahm J. Image reconstruction by regularized nonlinear
 * inversion – Joint estimation of coil sensitivities and image content.
 * Magn Reson Med 2008; 60:674-682.
 */


#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"
#include "misc/version.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/fmac.h"

#include "noncart/nufft.h"

#include "nlops/nlop.h"
#include "nlops/tenmul.h"
#include "nlops/chain.h"
#include "nlops/cast.h"

#include "num/fft.h"
#include "num/iovec.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops.h"
#include "num/multiplace.h"

#include "noir/utils.h"

#include "model2.h"



struct noir2_model_conf_s noir2_model_conf_defaults = {

	.fft_flags = FFT_FLAGS,
	.wght_flags = FFT_FLAGS,

	.rvc = false,
	.sos = false,
	.a = 220.,
	.b = 32.,

	.oversampling_coils = 1.,

	.nufft_conf = NULL,

	.asymetric = false,

	.ret_os_coils = false,
};


static struct noir2_s noir2_init_create(int N,
					const long pat_dims[N],
					const long bas_dims[N],
					const long msk_dims[N], const complex float* mask,
					const long ksp_dims[N],
					const long cim_dims[N],
					const long img_dims[N],
					const long kco_dims[N],
					const long col_dims[N],
					const long trj_dims[N],
					const struct noir2_model_conf_s* conf)
{
	struct noir2_s ret = {

		.model = NULL,
		.lop_asym = NULL,

		.lop_fft = NULL,
		.lop_coil = NULL,
		.lop_im = NULL,

		.lop_coil2 = NULL,

		.lop_nufft = NULL,
		.lop_pattern = NULL,
		.lop_basis = NULL,

		.N = N,
		.pat_dims = *TYPE_ALLOC(long[N]),
		.bas_dims = *TYPE_ALLOC(long[N]),
		.msk_dims = *TYPE_ALLOC(long[N]),
		.ksp_dims = *TYPE_ALLOC(long[N]),
		.cim_dims = *TYPE_ALLOC(long[N]),
		.img_dims = *TYPE_ALLOC(long[N]),
		.col_dims = *TYPE_ALLOC(long[N]),
		.col_ten_dims = *TYPE_ALLOC(long[N]),
		.trj_dims = *TYPE_ALLOC(long[N]),

		.basis = NULL,
	};

	md_copy_dims(N, ret.pat_dims, (NULL == pat_dims) ? MD_SINGLETON_DIMS(N) : pat_dims);
	md_copy_dims(N, ret.bas_dims, (NULL == bas_dims) ? MD_SINGLETON_DIMS(N) : bas_dims);
	md_copy_dims(N, ret.msk_dims, (NULL == msk_dims) ? MD_SINGLETON_DIMS(N) : msk_dims);
	md_copy_dims(N, ret.ksp_dims, (NULL == ksp_dims) ? MD_SINGLETON_DIMS(N) : ksp_dims);
	md_copy_dims(N, ret.cim_dims, (NULL == cim_dims) ? MD_SINGLETON_DIMS(N) : cim_dims);
	md_copy_dims(N, ret.img_dims, (NULL == img_dims) ? MD_SINGLETON_DIMS(N) : img_dims);
	md_copy_dims(N, ret.col_dims, (NULL == kco_dims) ? MD_SINGLETON_DIMS(N) : kco_dims);
	md_copy_dims(N, ret.trj_dims, (NULL == trj_dims) ? MD_SINGLETON_DIMS(N) : trj_dims);

	md_copy_dims(N, ret.col_ten_dims, ret.col_dims);
	md_copy_dims(3, ret.col_ten_dims, ret.cim_dims);

	long wgh_dims[N];
	for (int i = 0; i < N; i++)
		wgh_dims[i] = MD_IS_SET(conf->wght_flags & md_nontriv_dims(N, ret.col_dims), i) ? lround(col_dims[i] * (conf->ret_os_coils ? 1. : conf->oversampling_coils)) : 1;

	ret.lop_coil = linop_noir_weights_create(N, ret.col_ten_dims, ret.col_dims, wgh_dims, conf->wght_flags, conf->oversampling_coils, conf->a, conf->b, 1);
	ret.lop_coil2 = linop_noir_weights_create(N, col_dims, ret.col_dims, wgh_dims, conf->wght_flags, (conf->ret_os_coils ? 1. : conf->oversampling_coils), conf->a, conf->b, 1);


	ret.lop_im = NULL;

	if (NULL != mask) {

		assert(md_check_equal_dims(N, msk_dims, ret.img_dims, md_nontriv_dims(N, msk_dims)));
		ret.lop_im = linop_cdiag_create(N, ret.img_dims, md_nontriv_dims(N, msk_dims), mask);
	}

	if (conf->rvc) {

		if (NULL == ret.lop_im)
			ret.lop_im  = linop_zreal_create(N, ret.img_dims);
		else
			ret.lop_im  = linop_chain_FF(linop_zreal_create(N, ret.img_dims), ret.lop_im );
	}

	// For use in nlinvnet is optimized away in graph framefork
	if (NULL == ret.lop_im)
		ret.lop_im = linop_identity_create(N, ret.img_dims);

	ret.model_conf = *conf;

	return ret;
}


static void noir2_join(struct noir2_s* ret, bool asym)
{
	const struct nlop_s* model = nlop_tenmul_create(ret->N, ret->cim_dims, ret->img_dims, ret->col_ten_dims);
	
	if (NULL != ret->lop_im)
		model = nlop_chain2_swap_FF(nlop_from_linop(ret->lop_im), 0, model, 0);
	
	model = nlop_chain2_FF(nlop_from_linop(ret->lop_coil), 0, model, 1);
	
	if (asym) {
		struct linop_s* lop_id = linop_identity_create(ret->N, ret->cim_dims);
		struct linop_s* lop_asym = linop_from_ops(ret->lop_fft->normal, lop_id->adjoint, NULL, NULL);

		linop_free(lop_id);
		ret->model = nlop_chain2_FF(model, 0, nlop_from_linop_F(lop_asym), 0);
		ret->lop_asym = linop_clone(ret->lop_fft);

	} else {
		ret->model = nlop_chain2_FF(model, 0, nlop_from_linop(ret->lop_fft), 0);
		ret->lop_asym  = linop_identity_create(ret->N, ret->ksp_dims);
	}
}

/**
 * This function creates the non (bi)-linear part of the sense model, i.e.
 * 	cim = (mask * img) * ifftuc[weights * ksens]
 * and the corresponding nufft to transform to nonuniform kspace:
 *	frw = nufft(x)
 *	adj = nufft^H(x)
 * 	nrm = nufft^H(nufft(x))
 **/
struct noir2_s noir2_noncart_create(int N,
	const long trj_dims[N], const complex float* traj,
	const long wgh_dims[N], const complex float* weights,	//for nufft
	const long bas_dims[N], const complex float* basis,
	const long msk_dims[N], const complex float* mask,
	const long ksp_dims[N],
	const long cim_dims[N],
	const long img_dims[N],
	const long kco_dims[N],
	const long col_dims[N],
	const struct noir2_model_conf_s* conf)
{
	assert(conf->noncart);

	struct noir2_s ret = noir2_init_create(N, wgh_dims, bas_dims, msk_dims, mask, ksp_dims, cim_dims, img_dims, kco_dims, col_dims, trj_dims, conf);


	struct nufft_conf_s nufft_conf = *(conf->nufft_conf);
	nufft_conf.flags = conf->fft_flags & FFT_FLAGS;
	nufft_conf.cfft = conf->fft_flags & ~FFT_FLAGS;

	long mod_wgh_dims[N];
	if (NULL == weights)
		md_singleton_dims(N, mod_wgh_dims);
	else
		md_copy_dims(N, mod_wgh_dims, ret.pat_dims);

	ret.lop_fft = nufft_create2(N, ret.ksp_dims, ret.cim_dims, ret.trj_dims, traj, mod_wgh_dims, weights, basis ? ret.bas_dims : NULL, basis, nufft_conf);
	
	ret.lop_nufft = linop_clone(ret.lop_fft);

	debug_printf(DP_DEBUG1, "\nModel created (non Cartesian, nufft-based):\n");
	debug_printf(DP_DEBUG1, "kspace:     ");
	debug_print_dims(DP_DEBUG1, N, ret.ksp_dims);
	debug_printf(DP_DEBUG1, "images:     ");
	debug_print_dims(DP_DEBUG1, N, ret.img_dims);
	debug_printf(DP_DEBUG1, "coils:      ");
	debug_print_dims(DP_DEBUG1, N, ret.col_dims);
	debug_printf(DP_DEBUG1, "trajectory: ");
	debug_print_dims(DP_DEBUG1, N, ret.trj_dims);
	debug_printf(DP_DEBUG1, "pattern:    ");
	debug_print_dims(DP_DEBUG1, N, ret.pat_dims);

	if(NULL != basis) {

		debug_printf(DP_DEBUG1, "basis:      ");
		debug_print_dims(DP_DEBUG1, N, ret.bas_dims);
	}

	if(NULL != mask) {

		debug_printf(DP_DEBUG1, "mask:       ");
		debug_print_dims(DP_DEBUG1, N, ret.msk_dims);
	}

	debug_printf(DP_DEBUG1, "coilimg:    ");
	debug_print_dims(DP_DEBUG1, N, ret.cim_dims);
	debug_printf(DP_DEBUG1, "\n");

	noir2_join(&ret, true);

	return ret;
}

/**
 * This function creates the non (bi)-linear part of the sense model, i.e.
 * 	cim = (mask * img) * ifftuc[weights * ksens]
 * and the corresponding linear transform to map cim to (gridded) kspace:
 * Cartesian case:
 * 	frw = pattern * fftuc(x)
 * 	adj = ifftuc(pattern^* * x)
 * 	nrm = ifftuc(|pattern|^2 * fftuc(x))
 * Non-Cartesian case:
 * 	frw = psf * fftuc(x)
 * 	adj = ifftuc(x)
 * 	nrm = ifftuc(psf * fftuc(x))
 **/
struct noir2_s noir2_cart_create(int N,
	const long pat_dims[N], const complex float* pattern,
	const long bas_dims[N], const complex float* basis,
	const long msk_dims[N], const complex float* mask,
	const long ksp_dims[N],
	const long cim_dims[N],
	const long img_dims[N],
	const long kco_dims[N],
	const long col_dims[N],
	const struct noir2_model_conf_s* conf)
{
	struct noir2_s ret = noir2_init_create(N, pat_dims, bas_dims, msk_dims, mask, ksp_dims, cim_dims, img_dims, kco_dims, col_dims, NULL, conf);

	assert(NULL == basis);
	assert(md_check_equal_dims(N, ret.cim_dims, ret.ksp_dims, ~0UL));

	if (!use_compat_to_version("v0.9.00")) {

		ret.lop_fft = linop_fft_generic_create(N, ret.cim_dims, conf->fft_flags, conf->fft_flags & FFT_FLAGS, conf->fft_flags & FFT_FLAGS, 0, NULL, 0, NULL);

	} else {

		//FIXME: symmetric splitting of fftscale would allow us to use same
		//	 linop for both fftmods and save memory.
		//CAVEAT: reproducibility of ENLIVE paper requires fftscale to be
		//	  applied between tenmul for coils and fft
		//	  c.f. bbffa751ac32d80c5f85f86fe90070d2f03d1376
		ret.lop_fft = linop_fft_create(N, ret.cim_dims, conf->fft_flags);

		unsigned long fftm_flags = conf->fft_flags & FFT_FLAGS;
		long fftm_dims[N];
		md_select_dims(N, fftm_flags, fftm_dims, ret.cim_dims);

		complex float* fftmod_a = md_alloc(N, fftm_dims, CFL_SIZE);
		md_zfill(N, fftm_dims, fftmod_a, 1.);
		fftmod(N, fftm_dims, fftm_flags, fftmod_a, fftmod_a);

		ret.lop_fft = linop_chain_FF(ret.lop_fft, linop_cdiag_create(N, ret.cim_dims, fftm_flags, fftmod_a));

		md_zsmul(N, fftm_dims, fftmod_a, fftmod_a, 1. / sqrtf(md_calc_size(N, fftm_dims)));

		ret.lop_fft = linop_chain_FF(linop_cdiag_create(N, ret.cim_dims, fftm_flags, fftmod_a), ret.lop_fft);
		md_free(fftmod_a);
	}

	if (NULL != basis) {

		assert(!conf->noncart);

		long max_dims[N];
		md_max_dims(N, ~0UL, max_dims, ret.cim_dims, ret.ksp_dims);

		assert(md_check_equal_dims(N, max_dims, ret.cim_dims, md_nontriv_dims(N, ret.cim_dims)));
		assert(md_check_equal_dims(N, max_dims, ret.ksp_dims, md_nontriv_dims(N, ret.ksp_dims)));
		assert(md_check_equal_dims(N, max_dims, ret.bas_dims, md_nontriv_dims(N, ret.bas_dims)));

		ret.lop_basis = linop_fmac_create(N, max_dims, ~md_nontriv_dims(N, ret.ksp_dims), ~md_nontriv_dims(N, ret.cim_dims), ~md_nontriv_dims(N, ret.bas_dims), basis);
		ret.lop_fft = linop_chain_FF(ret.lop_fft, linop_clone(ret.lop_basis));
	}

	if (conf->noncart) {

		const struct linop_s* lop_adj = linop_clone(ret.lop_fft);
		const struct linop_s* lop_frw = linop_chain_FF(ret.lop_fft, linop_cdiag_create(N, linop_codomain(ret.lop_fft)->dims, md_nontriv_dims(N, ret.pat_dims), pattern));

		ret.lop_fft = linop_from_ops(lop_frw->forward, lop_adj->adjoint, NULL, NULL);

		linop_free(lop_frw);
		linop_free(lop_adj);

	} else {

		ret.lop_pattern = linop_cdiag_create(N, linop_codomain(ret.lop_fft)->dims, md_nontriv_dims(N, ret.pat_dims), pattern);
		ret.lop_fft = linop_chain_FF(ret.lop_fft, linop_clone(ret.lop_pattern));
	}

	debug_printf(DP_DEBUG1, "\nModel created (%s):\n", conf->noncart ? "non Cartesian, psf-based" : "Cartesian");
	debug_printf(DP_DEBUG1, "kspace:  ");
	debug_print_dims(DP_DEBUG1, N, ret.ksp_dims);
	debug_printf(DP_DEBUG1, "images:  ");
	debug_print_dims(DP_DEBUG1, N, ret.img_dims);
	debug_printf(DP_DEBUG1, "coils:   ");
	debug_print_dims(DP_DEBUG1, N, ret.col_dims);
	debug_printf(DP_DEBUG1, conf->noncart ? "psf:     " : "pattern: ");
	debug_print_dims(DP_DEBUG1, N, ret.pat_dims);
	
	if(NULL != basis) {

		debug_printf(DP_DEBUG1, "basis:   ");
		debug_print_dims(DP_DEBUG1, N, ret.bas_dims);
	}
	
	if(NULL != mask) {

		debug_printf(DP_DEBUG1, "mask:    ");
		debug_print_dims(DP_DEBUG1, N, ret.msk_dims);
	}
	
	debug_printf(DP_DEBUG1, "coilimg: ");
	debug_print_dims(DP_DEBUG1, N, ret.cim_dims);
	debug_printf(DP_DEBUG1, "\n");

	noir2_join(&ret, false);

	return ret;
}


__attribute__((optimize("-fno-finite-math-only")))
static void proj_add(int D, const long dims[D], const long ostrs[D],
			complex float* optr, const long v1_strs[D], complex float* v1, const long v2_strs[D], complex float* v2)
{
	float v22 = md_zscalar_real2(D, dims, v2_strs, v2, v2_strs, v2); // since it is real anyway

	complex float v12 = md_zscalar2(D, dims, v1_strs, v1, v2_strs, v2) / v22;

	if (!isfinite(crealf(v12)) || !isfinite(cimagf(v12)))
		v12 = 0.;

	md_zaxpy2(D, dims, ostrs, optr, v12, v2_strs, v2);
}


// FIXME: review dimensions
void noir2_orthogonalize(int N, const long col_dims[N], complex float* coils)
{
	long nmaps = col_dims[MAPS_DIM];

	if (1L == nmaps)
		return;

	long single_map_dims[N];
	md_select_dims(N, ~MAPS_FLAG, single_map_dims, col_dims);

	long single_map_strs[N];
	md_calc_strides(N, single_map_strs, single_map_dims, CFL_SIZE);

	long col_strs[N];
	md_calc_strides(N, col_strs, col_dims, CFL_SIZE);

	complex float* tmp = md_alloc_sameplace(N, single_map_dims, CFL_SIZE, coils);

	for (long map = 0L; map < nmaps; ++map) {

		complex float* map_ptr = (void*)coils + map * col_strs[MAPS_DIM];

		md_clear(N, single_map_dims, tmp, CFL_SIZE);

		for (long prev = 0L; prev < map; ++prev) {

			complex float* prev_map_ptr = (void*)coils + prev * col_strs[MAPS_DIM];

			proj_add(N, single_map_dims,
				single_map_strs, tmp, single_map_strs, map_ptr, col_strs, prev_map_ptr);
		}

		md_zsub2(N, single_map_dims, col_strs, map_ptr, col_strs, map_ptr, single_map_strs, tmp);
	}

	md_free(tmp);
}

void noir2_free(struct noir2_s* model)
{
	multiplace_free(model->basis);

	nlop_free(model->model);
	linop_free(model->lop_asym);

	linop_free(model->lop_coil);
	linop_free(model->lop_im);
	linop_free(model->lop_fft);

	linop_free(model->lop_coil2);
	

	linop_free(model->lop_basis);
	linop_free(model->lop_pattern);
	linop_free(model->lop_nufft);

	xfree(model->pat_dims);
	xfree(model->bas_dims);
	xfree(model->msk_dims);
	xfree(model->ksp_dims);
	xfree(model->cim_dims);
	xfree(model->img_dims);
	xfree(model->col_dims);
	xfree(model->col_ten_dims);
	xfree(model->trj_dims);
}


void noir2_noncart_update(struct noir2_s* model, int N,
	const long trj_dims[N], const complex float* traj,
	const long wgh_dims[N], const complex float* weights,
	const long bas_dims[N], const complex float* basis)
{
	assert(NULL != model->lop_nufft);

	nufft_update_traj(model->lop_nufft, N, trj_dims, traj, wgh_dims, weights, bas_dims, basis);
}

void noir2_cart_update(struct noir2_s* model, int N,
	const long pat_dims[N], const complex float* pattern,
	const long bas_dims[N], const complex float* basis)
{
	if (NULL != basis) {

		assert(NULL != model->lop_basis);
		linop_fmac_set_tensor(model->lop_basis, N, bas_dims, basis);
	}

	assert(NULL != model->lop_pattern);
	linop_gdiag_set_diag(model->lop_pattern, N, pat_dims, pattern);
}
