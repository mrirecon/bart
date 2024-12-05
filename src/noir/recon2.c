/* Copyright 2023. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 *
 * Uecker M, Hohage T, Block KT, Frahm J. Image reconstruction by regularized nonlinear
 * inversion – Joint estimation of coil sensitivities and image content.
 * Magn Reson Med 2008; 60:674-682.
 */

#include <complex.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/iovec.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "iter/iter3.h"
#include "iter/iter4.h"
#include "iter/italgos.h"

#include "misc/types.h"
#include "misc/mri.h"
#include "misc/debug.h"
#include "misc/stream.h"

#include "noir/model2.h"

#include "nlops/nlop.h"

#include "recon2.h"


struct nlop_wrapper2_s {

	INTERFACE(struct iter_op_data_s);

	long split;

	int N;
	const long* col_dims;
};

DEF_TYPEID(nlop_wrapper2_s);


static void orthogonalize(iter_op_data* ptr, float* _dst, const float* _src)
{
	assert(_dst == _src);
	auto nlw = CAST_DOWN(nlop_wrapper2_s, ptr);
	noir2_orthogonalize(nlw->N, nlw->col_dims, (complex float*) _dst + nlw->split);
}


const struct noir2_conf_s noir2_defaults = {

	.iter = 8,
	.rvc = false,
	.alpha = 1.,
	.alpha_min = 0.,
	.redu = 2.,
	.a = 220.,
	.b = 32.,
	.c = 1.,

	.oversampling_coils = 1.,
	.sms = false,
	.scaling = -100,
	.undo_scaling = false,

	.noncart = false,
	.nufft_conf = NULL,

	.gpu = false,

	.cgiter = 100,
	.cgtol = 0.1,

	.loop_flags = 0,
	.realtime = false,
	.temp_damp = 0.9,

	.ret_os_coils = false,

	.optimized = false,
};




void noir2_recon(const struct noir2_conf_s* conf, struct noir2_s* noir_ops,
			int N,
			const long img_dims[N], complex float* img, const complex float* img_ref,
			const long col_dims[N], complex float* sens,
			const long kco_dims[N], complex float* ksens, const complex float* sens_ref,
			const long ksp_dims[N], const complex float* kspace)
{

	assert(N == noir_ops->N);
	long dat_dims[N];
	md_copy_dims(N, dat_dims, linop_domain(noir_ops->lop_asym)->dims);

	if (1 < nlop_get_nr_in_args(noir_ops->model)) {

		assert(md_check_equal_dims(N, img_dims, nlop_generic_domain(noir_ops->model, 0)->dims, ~0UL));
		assert(md_check_equal_dims(N, kco_dims, nlop_generic_domain(noir_ops->model, 1)->dims, ~0UL));
		assert(md_check_equal_dims(N, ksp_dims, linop_codomain(noir_ops->lop_asym)->dims, ~0UL));

	}

	complex float* data = md_alloc_sameplace(N, dat_dims, CFL_SIZE, kspace);
	linop_adjoint(noir_ops->lop_asym, N, dat_dims, data, N, ksp_dims, kspace);


#ifdef USE_CUDA
	if((conf->gpu) && !cuda_ondevice(data)) {

		complex float* tmp_data = md_alloc_gpu(N, dat_dims, CFL_SIZE);
		md_copy(N, dat_dims, tmp_data, data, CFL_SIZE);
		md_free(data);
		data = tmp_data;
	}
#else
	if(conf->gpu)
		error("Compiled without GPU support!");
#endif

	float scaling = conf->scaling;
	if (0. > scaling) {

		scaling = -scaling / md_znorm(N, dat_dims, data);
		if (conf->sms)
			scaling *= sqrt(dat_dims[SLICE_DIM]);
	}

	debug_printf(DP_DEBUG1, "Scaling: %f\n", scaling);
	md_zsmul(N, dat_dims, data, data, scaling);


	long skip = md_calc_size(N, img_dims);
	long size = skip + md_calc_size(N, kco_dims);

	long d1[1] = { size };
	// variable which is optimized by the IRGNM
	complex float* x = NULL;
	complex float* ref = NULL;

	x = md_alloc_sameplace(1, d1, CFL_SIZE, data);
	md_clear(1, d1, x, CFL_SIZE);

	if (NULL != img_ref) {

		ref = md_alloc_sameplace(1, d1, CFL_SIZE, data);
		md_clear(1, d1, ref, CFL_SIZE);
	}

	if (NULL != img_ref) {

		md_copy(N, img_dims, ref, img_ref, CFL_SIZE);
		md_copy(N, kco_dims, ref + skip, sens_ref, CFL_SIZE);
	}

	md_copy(N, img_dims, x, img, CFL_SIZE);
	md_copy(N, kco_dims, x + skip, ksens, CFL_SIZE);


	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;

	irgnm_conf.iter = (int)conf->iter;
	irgnm_conf.alpha = conf->alpha;
	irgnm_conf.alpha_min = conf->alpha_min;
	irgnm_conf.redu = conf->redu;
	irgnm_conf.cgtol = conf->cgtol;
	irgnm_conf.nlinv_legacy = true;
	irgnm_conf.alpha_min = conf->alpha_min;
	irgnm_conf.cgiter = conf->cgiter;


	struct nlop_s* nlop_flat = nlop_flatten_inputs_F(nlop_clone(noir_ops->model));

	struct nlop_wrapper2_s nlw;
	SET_TYPEID(nlop_wrapper2_s, &nlw);
	nlw.split = skip;
	nlw.N = N;
	nlw.col_dims = kco_dims;

	iter4_irgnm(CAST_UP(&irgnm_conf),
			nlop_flat,
			size * 2, (float*)x, (const float*)ref,
			md_calc_size(N, dat_dims) * 2, (const float*)data,
			NULL,
			(struct iter_op_s){ orthogonalize, CAST_UP(&nlw) });

	nlop_free(nlop_flat);

	md_free(data);

	md_copy(N, img_dims, img, x, CFL_SIZE);
	md_copy(N, kco_dims, ksens, x + skip, CFL_SIZE);

	if (conf->realtime) {

		md_zsmul(N, img_dims, (complex float*)img_ref, img, conf->temp_damp);
		md_zsmul(N, kco_dims, (complex float*)sens_ref, ksens, conf->temp_damp);
	}

	if (NULL != sens) {

		complex float* tmp = md_alloc_sameplace(N, col_dims, CFL_SIZE, x);
		linop_forward_unchecked(noir_ops->lop_coil2, tmp, x + skip);
		md_copy(DIMS, col_dims, sens, tmp, CFL_SIZE);	// needed for GPU
		md_free(tmp);

		if (1 != col_dims[SLICE_DIM])
			fftmod(DIMS, col_dims, SLICE_FLAG, sens, sens);
	}

	md_free(x);
	md_free(ref);

	if (conf->undo_scaling)
		md_zsmul(N, img_dims, img, img, 1./scaling);
}


void noir2_recon_noncart(
	const struct noir2_conf_s* conf, int N,
	const long img_dims[N], complex float* img, const complex float* img_ref,
	const long col_dims[N], complex float* sens,
	const long kco_dims[N], complex float* ksens, const complex float* sens_ref,
	const long ksp_dims[N], const complex float* kspace,
	const long trj_dims[N], const complex float* traj,
	const long wgh_dims[N], const complex float* weights,
	const long bas_dims[N], const complex float* basis,
	const long msk_dims[N], const complex float* mask,
	const long cim_dims[N])
{
	assert(0 == (conf->loop_flags & md_nontriv_dims(N, bas_dims)));
	assert(0 == (conf->loop_flags & md_nontriv_dims(N, msk_dims)));

	unsigned long loop_flags = conf->loop_flags | (conf->realtime ? TIME_FLAG : 0);

	struct noir2_model_conf_s mconf = noir2_model_conf_defaults;

	mconf.fft_flags = (conf->sms) ? SLICE_FLAG | FFT_FLAGS : FFT_FLAGS;
	mconf.wght_flags = FFT_FLAGS;

	mconf.rvc = conf->rvc;
	mconf.a = conf->a;
	mconf.b = conf->b;
	mconf.c = conf->c;
	mconf.oversampling_coils = conf->oversampling_coils;
	mconf.noncart = conf->noncart;

	mconf.nufft_conf = conf->nufft_conf;
	mconf.ret_os_coils = conf->ret_os_coils;

	long limg_dims[N];
	long lcol_dims[N];
	long lksp_dims[N];
	long ltrj_dims[N];
	long lwgh_dims[N];
	long lkco_dims[N];
	long lcim_dims[N];

	md_select_dims(N, ~loop_flags, limg_dims, img_dims);
	md_select_dims(N, ~loop_flags, lcol_dims, col_dims);
	md_select_dims(N, ~loop_flags, lksp_dims, ksp_dims);
	md_select_dims(N, ~loop_flags, ltrj_dims, trj_dims);
	md_select_dims(N, ~loop_flags, lwgh_dims, wgh_dims);
	md_select_dims(N, ~loop_flags, lkco_dims, kco_dims);
	md_select_dims(N, ~loop_flags, lcim_dims, cim_dims);

	long img_strs[N];
	long col_strs[N];
	long ksp_strs[N];
	long trj_strs[N];
	long wgh_strs[N];
	long kco_strs[N];

	md_calc_strides(N, img_strs, img_dims, CFL_SIZE);
	md_calc_strides(N, col_strs, col_dims, CFL_SIZE);
	md_calc_strides(N, ksp_strs, ksp_dims, CFL_SIZE);
	md_calc_strides(N, trj_strs, trj_dims, CFL_SIZE);
	md_calc_strides(N, wgh_strs, wgh_dims, CFL_SIZE);
	md_calc_strides(N, kco_strs, kco_dims, CFL_SIZE);

	struct noir2_s noir_ops = (conf->optimized ? noir2_noncart_optimized_create :noir2_noncart_create)(N, ltrj_dims, NULL, lwgh_dims, weights, bas_dims, basis, msk_dims, mask, lksp_dims, lcim_dims, limg_dims, lkco_dims, lcol_dims, &mconf);

#ifdef USE_CUDA
	md_alloc_fun_t my_alloc = conf->gpu ? md_alloc_gpu : md_alloc;
#else
	assert(!conf->gpu);
	md_alloc_fun_t my_alloc = md_alloc;
#endif

	complex float* l_img = 		my_alloc(N, limg_dims, CFL_SIZE);
	complex float* l_img_ref = 	(!conf->realtime && (NULL == img_ref)) ? NULL : my_alloc(N, limg_dims, CFL_SIZE);
	complex float* l_sens = 	(NULL == sens) ? NULL : my_alloc(N, lcol_dims, CFL_SIZE);
	complex float* l_ksens = 	my_alloc(N, lkco_dims, CFL_SIZE);
	complex float* l_sens_ref = 	(!conf->realtime && (NULL == sens_ref)) ? NULL : my_alloc(N, lkco_dims, CFL_SIZE);
	complex float* l_kspace = 	my_alloc(N, lksp_dims, CFL_SIZE);
	complex float* l_wgh = 		(NULL == weights) ? NULL : my_alloc(N, lwgh_dims, CFL_SIZE);
	complex float* l_trj = 		my_alloc(N, ltrj_dims, CFL_SIZE);

	long pos[N];
	md_set_dims(N, pos, 0);

	long pos_trj[N];
	long pos_wgh[N];

	stream_t strm_ksp = stream_lookup(kspace);
	stream_t strm_img = stream_lookup(img);
	stream_t strm_trj = stream_lookup(traj);

	if (NULL != strm_ksp)
		assert(loop_flags == stream_get_flags(strm_ksp));

	do {
		if (NULL != strm_ksp)
			stream_sync(strm_ksp, N, pos);

		md_slice(N, loop_flags, pos, img_dims, l_img, img, CFL_SIZE);
		md_slice(N, loop_flags, pos, kco_dims, l_ksens, ksens, CFL_SIZE);
		md_slice(N, loop_flags, pos, ksp_dims, l_kspace, kspace, CFL_SIZE);

		md_copy_dims(N, pos_trj, pos);
		md_copy_dims(N, pos_wgh, pos);

		if (conf->realtime) {

			pos_trj[TIME_DIM] = pos_trj[TIME_DIM] % trj_dims[TIME_DIM];
			
			if (NULL != weights)
				pos_wgh[TIME_DIM] = pos_wgh[TIME_DIM] % wgh_dims[TIME_DIM];

			if (0 == pos[TIME_DIM]) {

				if (NULL == img_ref)
					md_clear(N, limg_dims, l_img_ref, CFL_SIZE);
				else
					md_slice(N, loop_flags, pos, img_dims, l_img_ref, img_ref, CFL_SIZE);

				if (NULL == sens_ref)
					md_clear(N, lkco_dims, l_sens_ref, CFL_SIZE);
				else
					md_slice(N, loop_flags, pos, kco_dims, l_sens_ref, sens_ref, CFL_SIZE);
			} else {

				md_zsmul(N, limg_dims, l_img, l_img_ref, 1. / conf->temp_damp);
				md_zsmul(N, lkco_dims, l_ksens, l_sens_ref, 1. / conf->temp_damp);
			}

		} else {

			if (NULL != img_ref)
				md_slice(N, loop_flags, pos, img_dims, l_img_ref, img_ref, CFL_SIZE);
			
			if (NULL != sens_ref)
				md_slice(N, loop_flags, pos, kco_dims, l_sens_ref, sens_ref, CFL_SIZE);
		}

		if (NULL != strm_trj)
			stream_sync(strm_trj, N, pos_trj);

		md_slice(N, loop_flags, pos_trj, trj_dims, l_trj, traj, CFL_SIZE);

		if (NULL != weights)
			md_slice(N, loop_flags, pos_wgh, wgh_dims, l_wgh, weights, CFL_SIZE);

		noir2_noncart_update(&noir_ops, N, ltrj_dims, l_trj, lwgh_dims, l_wgh, bas_dims, basis);
		
		noir2_recon(conf, &noir_ops, N, limg_dims, l_img, l_img_ref, lcol_dims, l_sens, lkco_dims, l_ksens, l_sens_ref, lksp_dims, l_kspace);

		if (NULL != sens)
			md_copy_block(N, pos, col_dims, sens, lcol_dims, l_sens, CFL_SIZE);
	
		md_copy_block(N, pos, kco_dims, ksens, lkco_dims, l_ksens, CFL_SIZE);
		md_copy_block(N, pos, img_dims, img, limg_dims, l_img, CFL_SIZE);

		if (NULL != strm_img)
			stream_sync(strm_img, N, pos);
	
	} while (md_next(N, ksp_dims, loop_flags, pos));

	md_free(l_img);
	md_free(l_img_ref);
	md_free(l_sens);
	md_free(l_ksens);
	md_free(l_sens_ref);
	md_free(l_kspace);
	md_free(l_wgh);
	md_free(l_trj);

	noir2_free(&noir_ops);
}


void noir2_recon_cart(
	const struct noir2_conf_s* conf, int N,
	const long img_dims[N], complex float* img, const complex float* img_ref,
	const long col_dims[N], complex float* sens, 
	const long kco_dims[N], complex float* ksens, const complex float* sens_ref,
	const long ksp_dims[N], const complex float* kspace,
	const long pat_dims[N], const complex float* pattern,
	const long bas_dims[N], const complex float* basis,
	const long msk_dims[N], const complex float* mask,
	const long cim_dims[N])
{
	assert(0 == (conf->loop_flags && md_nontriv_dims(N, bas_dims)));
	assert(0 == (conf->loop_flags && md_nontriv_dims(N, msk_dims)));

	struct noir2_model_conf_s mconf = noir2_model_conf_defaults;

	mconf.fft_flags = (conf->sms) ? SLICE_FLAG | FFT_FLAGS : FFT_FLAGS;
	mconf.wght_flags = FFT_FLAGS;

	mconf.rvc = conf->rvc;
	mconf.a = conf->a;
	mconf.b = conf->b;
	mconf.c = conf->c;
	mconf.oversampling_coils = conf->oversampling_coils;
	mconf.noncart = conf->noncart;

	mconf.nufft_conf = conf->nufft_conf;
	mconf.ret_os_coils = conf->ret_os_coils;

	struct noir2_s noir_ops = noir2_cart_create(N, pat_dims, pattern, bas_dims, basis, msk_dims, mask, ksp_dims, cim_dims, img_dims, kco_dims, col_dims, &mconf);


	long limg_dims[N];
	long lcol_dims[N];
	long lksp_dims[N];
	long lpat_dims[N];
	long lkco_dims[N];

	md_select_dims(N, ~conf->loop_flags, limg_dims, img_dims);
	md_select_dims(N, ~conf->loop_flags, lcol_dims, col_dims);
	md_select_dims(N, ~conf->loop_flags, lksp_dims, ksp_dims);
	md_select_dims(N, ~conf->loop_flags, lpat_dims, pat_dims);
	md_select_dims(N, ~conf->loop_flags, lkco_dims, kco_dims);

	long img_strs[N];
	long col_strs[N];
	long ksp_strs[N];
	long pat_strs[N];
	long kco_strs[N];

	md_calc_strides(N, img_strs, img_dims, CFL_SIZE);
	md_calc_strides(N, col_strs, col_dims, CFL_SIZE);
	md_calc_strides(N, ksp_strs, ksp_dims, CFL_SIZE);
	md_calc_strides(N, pat_strs, pat_dims, CFL_SIZE);
	md_calc_strides(N, kco_strs, kco_dims, CFL_SIZE);

	long pos[N];
	md_set_dims(N, pos, 0);

	do {

		complex float* l_img = &MD_ACCESS(N, img_strs, pos, img);
		const complex float* l_img_ref = (NULL == img_ref) ? NULL : &MD_ACCESS(N, img_strs, pos, img_ref);
		complex float* l_sens = (NULL == sens) ? NULL : &MD_ACCESS(N, col_strs, pos, sens);
		complex float* l_ksens = (NULL == ksens) ? NULL : &MD_ACCESS(N, kco_strs, pos, ksens);
		const complex float* l_sens_ref = (NULL == sens_ref) ? NULL : &MD_ACCESS(N, kco_strs, pos, sens_ref);
		const complex float* l_kspace = &MD_ACCESS(N, ksp_strs, pos, kspace);
		const complex float* l_pattern = &MD_ACCESS(N, pat_strs, pos, pattern);

		if (l_pattern != pattern)
			noir2_cart_update(&noir_ops, N,lpat_dims, l_pattern, bas_dims, basis);
		
		noir2_recon(conf, &noir_ops, N, limg_dims, l_img, l_img_ref, lcol_dims, l_sens, lkco_dims, l_ksens, l_sens_ref, lksp_dims, l_kspace);

	} while (md_next(N, ksp_dims, conf->loop_flags, pos));

	noir2_free(&noir_ops);
}
