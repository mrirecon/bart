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
#include "num/vptr.h"
#include "num/iovec.h"
#include "num/ops.h"
#include "num/ops_p.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "iter/italgos.h"
#include "iter/iter.h"
#include "iter/iter2.h"
#include "iter/iter3.h"
#include "iter/iter4.h"
#include "iter/italgos.h"
#include "iter/lsqr.h"
#include "iter/prox.h"
#include "iter/misc.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/mri.h"
#include "misc/mri2.h"
#include "misc/debug.h"
#include "misc/stream.h"
#include "misc/version.h"

#include "noir/model2.h"
#include "noir/pole.h"

#include "grecon/optreg.h"
#include "grecon/italgo.h"

#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/const.h"

#include "recon2.h"


struct nlop_wrapper2_s {

	struct iter_op_data_s super;

	long split;

	int N;
	const long* col_dims;

	int iter;
	int pole_correction;
	struct pole_config_s conf;
	const struct noir2_s* noir_ops;
};

DEF_TYPEID(nlop_wrapper2_s);

static void mult_phase_pole(struct iter_conjgrad_conf conf, const struct linop_s* lop, int N, const long pdims[N], bool conj, const complex float* phase, complex float* dst)
{
	assert(N == linop_domain(lop)->N);
	assert(N == linop_codomain(lop)->N);

	long tdims[N];
	md_copy_dims(N, tdims, linop_codomain(lop)->dims);
	assert(md_check_compat(N, ~md_nontriv_dims(N, pdims), pdims, tdims));

	if (linop_is_identity(lop)) {

		(conj ? md_zmulc2 : md_zmul2)(N, tdims, MD_STRIDES(N, tdims, CFL_SIZE), dst, MD_STRIDES(N, tdims, CFL_SIZE), dst, MD_STRIDES(N, pdims, CFL_SIZE), phase);
		return;
	}

	complex float* tmp = md_alloc_sameplace(N, tdims, CFL_SIZE, dst);
	linop_forward_unchecked(lop, tmp, dst);

	(conj ? md_zmulc2 : md_zmul2)(N, tdims, MD_STRIDES(N, tdims, CFL_SIZE), tmp, MD_STRIDES(N, tdims, CFL_SIZE), tmp, MD_STRIDES(N, pdims, CFL_SIZE), phase);

	complex float* adj = md_alloc_sameplace(N, linop_domain(lop)->dims, CFL_SIZE, phase);

	linop_adjoint_unchecked(lop, adj, tmp);
	md_free(tmp);

	iter_conjgrad(CAST_UP(&conf), lop->normal, NULL, 2 * md_calc_size(N, linop_domain(lop)->dims), (float*)(dst), (const float*)adj, NULL);
	md_free(adj);
}


static void phasepole_correction(struct pole_config_s conf, const struct noir2_s* d, complex float* dst)
{
	int N = d->N;

	const struct linop_s* lop_col = d->lop_coil;
	long col_dims[N];

	md_copy_dims(N, col_dims, linop_codomain(lop_col)->dims);

	complex float* col = md_alloc_sameplace(d->N, col_dims, CFL_SIZE, dst);
	linop_forward_unchecked(lop_col, col, dst + md_calc_size(N, linop_domain(d->lop_im)->dims));

	long pdims[N];
	md_select_dims(d->N, ~SENS_FLAGS, pdims, col_dims);

	complex float* phase = md_alloc_sameplace(d->N, pdims, CFL_SIZE, col);
	md_zfill(d->N, pdims, phase, 1.);

	bool correct = phase_pole_correction_loop(conf, d->N, (~15UL) & md_nontriv_dims(N, pdims), pdims, phase, col_dims, col);

	md_free(col);

	if (correct) {

		long img_dims[N];

		const struct linop_s* lop_img = d->lop_im;
		md_copy_dims(N, img_dims, linop_codomain(lop_img)->dims);
		complex float* img = md_alloc_sameplace(d->N, img_dims, CFL_SIZE, dst);
		linop_forward_unchecked(lop_img, img, dst);
		phase_pole_normalize(N, pdims, phase, img_dims, img);
		md_free(img);

		struct iter_conjgrad_conf conf = iter_conjgrad_defaults;
		conf.maxiter = 15;
		conf.l2lambda = 1.e-5;
		conf.tol = 0.;

		mult_phase_pole(conf, d->lop_im, N, pdims, false, phase, dst);
		mult_phase_pole(conf, d->lop_coil, N, pdims, true, phase, dst + md_calc_size(N, linop_domain(lop_img)->dims));
	}

	md_free(phase);
}

static void orthogonalize(iter_op_data* ptr, float* _dst, const float* _src)
{
	assert(_dst == _src);
	auto nlw = CAST_DOWN(nlop_wrapper2_s, ptr);

	if (((0 == nlw->pole_correction) || (++nlw->iter == nlw->pole_correction)) && (NULL != nlw->noir_ops))
		phasepole_correction(nlw->conf, nlw->noir_ops, (complex float*)_dst);

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
	.normalize_lowres = false,

	.noncart = false,
	.nufft_conf = NULL,

	.gpu = false,

	.cgiter = 100,
	.cgtol = 0.1,

	.loop_flags = 0,
	.realtime = false,
	.temp_damp = 0.9,

	.legacy_early_stoppping = false,

	.iter_reg = 3,
	.liniter = 100,
	.lintol= 0.,

	.ret_os_coils = false,
	.phasepoles = -1,

	.optimized = false,
};


struct noir_irgnm_conf {

	struct iter3_irgnm_conf* irgnm_conf;
	enum algo_t algo;

	struct lsqr_conf *lsqr_conf;
};


static void noir_irgnm2(const struct noir_irgnm_conf* conf,
			const struct nlop_s* nlop,
			int NO, const long odims[NO], complex float* x, const complex float* ref,
			int NI, const long idims[NI], const complex float* data,
			int num_regs, const struct operator_p_s* thresh_ops[num_regs], const struct linop_s* trafos[num_regs],
			struct iter_op_s cb)
{
	struct lsqr_conf lsqr_conf = *(conf->lsqr_conf);

	const struct operator_p_s* pinv_op = NULL;

	auto cod = nlop_codomain(nlop);
	auto dom = nlop_domain(nlop);

	long M = 2 * md_calc_size(cod->N, cod->dims);
	long N = 2 * md_calc_size(dom->N, dom->dims);

	assert(N == 2 * md_calc_size(NO, odims));
	assert(M == 2 * md_calc_size(NI, idims));

	enum algo_t algo = conf->algo;

	switch (algo) {

		case ALGO_CG:
		case ALGO_IST:
		case ALGO_FISTA:
		{
			struct iter_fista_conf fista_conf = iter_fista_defaults;
			fista_conf.maxiter = conf->irgnm_conf->cgiter;
			fista_conf.maxeigen_iter = 20;
			fista_conf.tol = 0;
			pinv_op = lsqr2_create(&lsqr_conf, iter2_fista, CAST_UP(&fista_conf), NULL, nlop_get_derivative(nlop, 0, 0), NULL, num_regs, thresh_ops, trafos, NULL);

			iter4_irgnm2(CAST_UP(conf->irgnm_conf), (struct nlop_s*)nlop, N, (float*)x, (const float*)ref, M, (const float*)data, pinv_op, cb);
		}
		break;

		case ALGO_PRIDU:
		{
			struct iter_chambolle_pock_conf cp_conf = iter_chambolle_pock_defaults;
			cp_conf.maxiter = conf->irgnm_conf->cgiter;
			cp_conf.maxeigen_iter = 20;
			cp_conf.tol = 0;
			pinv_op = lsqr2_create(&lsqr_conf, iter2_chambolle_pock, CAST_UP(&cp_conf), NULL, nlop_get_derivative(nlop, 0, 0), NULL, num_regs, thresh_ops, trafos, NULL);

			iter4_irgnm2(CAST_UP(conf->irgnm_conf), (struct nlop_s*)nlop, N, (float*)x, (const float*)ref, M, (const float*)data, pinv_op, cb);
		}
		break;

		case ALGO_ADMM:
		{
			struct iter_admm_conf admm_conf = iter_admm_defaults;
			admm_conf.maxiter = conf->irgnm_conf->cgiter;
			admm_conf.do_warmstart = true;
			pinv_op = lsqr2_create(&lsqr_conf, iter2_admm, CAST_UP(&admm_conf), NULL, nlop_get_derivative(nlop, 0, 0), NULL, num_regs, thresh_ops, trafos, NULL);

			iter4_irgnm2(CAST_UP(conf->irgnm_conf), (struct nlop_s*)nlop, N, (float*)x, (const float*)ref, M, (const float*)data, pinv_op, cb);
		}
		break;

		case ALGO_NIHT:
		default:
			error("Algorithm not implemented!");

	}

	operator_p_free(pinv_op);
}


static int opt_reg_noir_join_prox(int NI, const long img_dims[NI], int NC, const long col_dims[NC], int num_regs, const struct operator_p_s* prox_ops[num_regs + 1], const struct linop_s* trafos[num_regs + 1])
{
	assert(0 < num_regs);

	const struct operator_p_s* prox = prox_leastsquares_create(NC, col_dims, 1., NULL);

	struct linop_s* lop_ext = linop_extract_create(1, MD_DIMS(0), MD_DIMS(md_calc_size(NI, img_dims)), MD_DIMS(md_calc_size(NI, img_dims) + md_calc_size(NC, col_dims)));
	lop_ext = linop_reshape_out_F(lop_ext, NI, img_dims);

	for (int i = 0; i < num_regs; i++) {

		if (NULL != prox && linop_is_identity(trafos[i])) {

			prox_ops[i] = operator_p_stack_FF(0, 0, operator_p_flatten_F(prox_ops[i]), operator_p_flatten_F(prox));
			prox = NULL;

			linop_free(trafos[i]);
			trafos[i] = linop_identity_create(1, MD_DIMS(md_calc_size(NI, img_dims) + md_calc_size(NC, col_dims)));

		} else {

			trafos[i] = linop_chain_FF(linop_clone(lop_ext), trafos[i]);
		}
	}

	linop_free(lop_ext);

	if (NULL != prox) {

		prox_ops[num_regs] = prox;
		trafos[num_regs] = linop_extract_create(1, MD_DIMS(md_calc_size(NI, img_dims)), MD_DIMS(md_calc_size(NC, col_dims)), MD_DIMS(md_calc_size(NI, img_dims) + md_calc_size(NC, col_dims)));
		trafos[num_regs] = linop_reshape_out_F(trafos[num_regs], NC, col_dims);
		num_regs++;
	}

	return num_regs;
}


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

		complex float* tmp_data = md_gpu_move(N, dat_dims, data, CFL_SIZE);
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

	const struct operator_p_s* prox_ops[NUM_REGS];
	const struct linop_s* trafos[NUM_REGS];
	const long (*sdims[NUM_REGS])[N + 1] = { NULL };

	if (!((NULL == conf->regs) || (0 == conf->regs->r)))
		opt_reg_configure(N, img_dims, conf->regs, prox_ops, trafos, sdims, 8, 1, "dau2", conf->gpu, DIMS-1);

	long skip = md_calc_size(N, img_dims);

	const struct nlop_s* nlop = nlop_clone(noir_ops->model);

	for (int i = 0; i < NUM_REGS; i++) {

		if (NULL != sdims[i]) {

			nlop = nlop_combine_FF(nlop, nlop_del_out_create(N + 1, (*sdims[i])));
			nlop = nlop_shift_input_F(nlop, nlop_get_nr_in_args(nlop) - 2, nlop_get_nr_in_args(nlop) - 1);

			skip += md_calc_size(N + 1, (*sdims[i]));
		}
	}

	long size = skip + md_calc_size(N, kco_dims);

	long d1[1] = { size };
	// variable which is optimized by the IRGNM
	complex float* x = NULL;
	complex float* ref = NULL;

	if (is_vptr(img)) {

		void* range[NUM_REGS + 2];

		int R = 0;
		range[R++] = (is_vptr_gpu(kspace) ? vptr_move_gpu : vptr_move_cpu)(img);

		for (int i = 0; i < NUM_REGS; i++)
			if (NULL != sdims[i])
				range[R++] = md_alloc_sameplace(N + 1, (*sdims[i]), CFL_SIZE, range[0]);

		range[R++] = (is_vptr_gpu(kspace) ? vptr_move_gpu : vptr_move_cpu)(ksens);

		x = vptr_wrap_range(R, range, true);

		if (NULL != img_ref) {

			ref = vptr_alloc_same(x);
			md_clear(1, d1, ref, CFL_SIZE);
		}
	} else {

		x = md_alloc_sameplace(1, d1, CFL_SIZE, data);
		md_clear(1, d1, x, CFL_SIZE);

		if (NULL != img_ref) {

			ref = md_alloc_sameplace(1, d1, CFL_SIZE, data);
			md_clear(1, d1, ref, CFL_SIZE);
		}
	}

	for (int i = 0; i < NUM_REGS; i++)
		if (NULL != sdims[i])
			xfree(sdims[i]);

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

	irgnm_conf.nlinv_legacy = use_compat_to_version("v0.9.00") || conf->legacy_early_stoppping;
	irgnm_conf.alpha_min = conf->alpha_min;


	struct nlop_s* nlop_flat = nlop_flatten_inputs_F(nlop);

	struct nlop_wrapper2_s nlw;
	SET_TYPEID(nlop_wrapper2_s, &nlw);
	nlw.split = skip;
	nlw.N = N;
	nlw.col_dims = kco_dims;

	nlw.iter = 0; // pole correction iteration counter
	nlw.pole_correction = conf->phasepoles;

	nlw.conf = pole_config_default;
	nlw.noir_ops = noir_ops;

	irgnm_conf.alpha = conf->alpha;
	irgnm_conf.cgtol = conf->cgtol;
	irgnm_conf.cgiter = conf->cgiter;

	if (!((NULL == conf->regs) || (0 == conf->regs->r)))
		irgnm_conf.iter = irgnm_conf.iter - conf->iter_reg;

	if (0 < irgnm_conf.iter)
		iter4_irgnm(CAST_UP(&irgnm_conf),
			nlop_flat,
			size * 2, (float*)x, (const float*)ref,
			md_calc_size(N, dat_dims) * 2, (const float*)data,
			NULL,
			(struct iter_op_s){ orthogonalize, CAST_UP(&nlw) });

	for (int i = 0; i < irgnm_conf.iter; i++)
		irgnm_conf.alpha = (irgnm_conf.alpha - irgnm_conf.alpha_min) / irgnm_conf.redu + irgnm_conf.alpha_min;

	irgnm_conf.iter = (int)conf->iter - irgnm_conf.iter;
	irgnm_conf.cgtol = conf->lintol;
	irgnm_conf.cgiter = conf->liniter;

	if (0 < irgnm_conf.iter) {

		int num_regs = conf->regs->r + conf->regs->sr;

		struct noir_irgnm_conf noir_irgnm_conf = {

			.irgnm_conf = &irgnm_conf,
			.algo = italgo_choose(num_regs, conf->regs->regs),
			.lsqr_conf = NULL,
		};

		struct lsqr_conf lsqr_conf = lsqr_defaults;
		lsqr_conf.warmstart = true;
		lsqr_conf.lambda = 0;
		lsqr_conf.include_adjoint = false;
		noir_irgnm_conf.lsqr_conf = &lsqr_conf;

		bool sup = skip != md_calc_size(N, img_dims);
		num_regs = opt_reg_noir_join_prox(sup ? 1 : N, sup ? MD_DIMS(skip) : img_dims, N, kco_dims, num_regs , prox_ops, trafos);

		noir_irgnm2(&noir_irgnm_conf, nlop_flat,
			   1, MD_DIMS(size), x, ref,
			   1, MD_DIMS(md_calc_size(N, dat_dims)), data,
			   num_regs, prox_ops, trafos,
			   (struct iter_op_s){ orthogonalize, CAST_UP(&nlw) });

		for (int nr = 0; nr < num_regs; nr++) {

			operator_p_free(prox_ops[nr]);
			linop_free(trafos[nr]);
		}
	}

	nlop_free(nlop_flat);

	md_copy(N, img_dims, img, x, CFL_SIZE);
	md_copy(N, kco_dims, ksens, x + skip, CFL_SIZE);

	if (conf->realtime) {

		md_zsmul(N, img_dims, (complex float*)img_ref, img, conf->temp_damp);
		md_zsmul(N, kco_dims, (complex float*)sens_ref, ksens, conf->temp_damp);
	}

	if (NULL != sens) {

		complex float* tmp = md_alloc_sameplace(N, col_dims, CFL_SIZE, data);
		complex float* tmp_kcol = md_alloc_sameplace(N, kco_dims, CFL_SIZE, data);

		md_copy(N, kco_dims, tmp_kcol, x + skip, CFL_SIZE);
		linop_forward_unchecked(noir_ops->lop_coil2, tmp, tmp_kcol);
		md_copy(DIMS, col_dims, sens, tmp, CFL_SIZE);	// needed for GPU
		md_free(tmp);
		md_free(tmp_kcol);

		if (1 != col_dims[SLICE_DIM])
			fftmod(DIMS, col_dims, SLICE_FLAG, sens, sens);
	}

	if (conf->normalize_lowres) {

		long nrm_col_dims[N];
		md_copy_dims(N, nrm_col_dims, linop_codomain(noir_ops->lop_coil)->dims);

		complex float* tmp = md_alloc_sameplace(N, nrm_col_dims, CFL_SIZE, data);
		complex float* tmp_kcol = md_alloc_sameplace(N, kco_dims, CFL_SIZE, data);

		md_copy(N, kco_dims, tmp_kcol, x + skip, CFL_SIZE);
		linop_forward_unchecked(noir_ops->lop_coil, tmp, tmp_kcol);
		md_free(tmp_kcol);

		long nrm_dims[N];
		md_select_dims(N, md_nontriv_dims(N, img_dims), nrm_dims, nrm_col_dims);
		complex float* nrm = md_alloc_sameplace(N, nrm_dims, CFL_SIZE, data);
		md_zrss(N, nrm_col_dims, ~md_nontriv_dims(N, nrm_dims), nrm, tmp);
		md_free(tmp);

		complex float* nrm2 = md_alloc_sameplace(N, nrm_dims, CFL_SIZE, img);
		md_copy(N, nrm_dims, nrm2, nrm, CFL_SIZE);
		md_free(nrm);

		md_zmul2(N, img_dims, MD_STRIDES(N, img_dims, CFL_SIZE), img, MD_STRIDES(N, img_dims, CFL_SIZE), img, MD_STRIDES(N, nrm_dims, CFL_SIZE), nrm2);
		md_free(nrm2);
	}

	md_free(x);
	md_free(ref);
	md_free(data);

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

	const void* ref = NULL;
	if (is_vptr(kspace)) {

		ref = vptr_alloc_sameplace(1, MD_DIMS(1), 1, kspace);
		if (conf->gpu)
			vptr_set_gpu(ref);
	}
#ifdef USE_CUDA
	else if (conf->gpu)
		ref = md_alloc_gpu(1, MD_DIMS(1), 1);
#endif

	complex float* l_img = 		md_alloc_sameplace(N, limg_dims, CFL_SIZE, ref);
	complex float* l_img_ref = 	(!conf->realtime && (NULL == img_ref)) ? NULL : md_alloc_sameplace(N, limg_dims, CFL_SIZE, ref);
	complex float* l_sens = 	(NULL == sens) ? NULL : md_alloc_sameplace(N, lcol_dims, CFL_SIZE, ref);
	complex float* l_ksens = 	md_alloc_sameplace(N, lkco_dims, CFL_SIZE, ref);
	complex float* l_sens_ref = 	(!conf->realtime && (NULL == sens_ref)) ? NULL : md_alloc_sameplace(N, lkco_dims, CFL_SIZE, ref);
	complex float* l_kspace = 	md_alloc_sameplace(N, lksp_dims, CFL_SIZE, ref);
	complex float* l_wgh = 		(!conf->realtime && (NULL == weights)) ? NULL : md_alloc_sameplace(N, lwgh_dims, CFL_SIZE, ref);
	complex float* l_trj = 		md_alloc_sameplace(N, ltrj_dims, CFL_SIZE, ref);

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
			else
				estimate_pattern(N, lksp_dims, COIL_FLAG, l_wgh, l_kspace);

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
	md_free(ref);
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
