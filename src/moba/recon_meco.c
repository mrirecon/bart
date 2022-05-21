/* Copyright 2020. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2019-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2019-2020 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
 */

#include <complex.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#include "num/gpuops.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/iovec.h"
#include "num/ops.h"
#include "num/ops_p.h"
#include "num/rand.h"

#include "iter/italgos.h"
#include "iter/iter2.h"
#include "iter/iter3.h"
#include "iter/iter4.h"
#include "iter/lsqr.h"
#include "iter/prox.h"
#include "iter/thresh.h"
#include "iter/vec.h"

#include "linops/someops.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "nlops/nlop.h"

#include "noir/model.h"
#include "simu/signals.h"

#include "wavelet/wavthresh.h"
#include "lowrank/lrthresh.h"

#include "grecon/optreg.h"
#include "grecon/italgo.h"

#include "moba/iter_l1.h"
#include "moba/meco.h"
#include "moba/model_meco.h"
#include "moba/moba.h"

#include "recon_meco.h"


#include "optreg.h"


void init_meco_maps(const long maps_dims[DIMS], complex float* maps, enum meco_model sel_model)
{
	if (MECO_PI == sel_model) {

		// set all parameters to 1.0
		md_zfill(DIMS, maps_dims, maps, 1.0);

	} else {

		md_clear(DIMS, maps_dims, maps, CFL_SIZE);

		long NCOEFF = maps_dims[COEFF_DIM];
		long pos[DIMS] = { 0 };

		long map1_dims[DIMS];
		md_select_dims(DIMS, ~COEFF_FLAG, map1_dims, maps_dims);

		complex float* map1 = md_alloc(DIMS, map1_dims, CFL_SIZE);

		// W & F
		long pd_flag = get_PD_flag(sel_model);
		float val = 0.1;

		for (int n = 0; n < NCOEFF; n++) {

			pos[COEFF_DIM] = n;

			md_zfill(DIMS, map1_dims, map1, MD_IS_SET(pd_flag, n) ? val : 0.);
			md_copy_block(DIMS, pos, maps_dims, maps, map1_dims, map1, CFL_SIZE);
		}

		md_free(map1);
	}
}

// rescale the reconstructed maps to the unit of Hz
// note: input and output are both maps
static void rescale_maps(unsigned int model, double scaling_Y, const struct linop_s* op, const complex float* scaling, const long maps_dims[DIMS], complex float* maps)
{
	if (MECO_PI == model) {

		md_zsmul(DIMS, maps_dims, maps, maps, 1. / scaling_Y);

	} else {

		md_zsmul(DIMS, maps_dims, maps, maps, 1000.); // kHz --> Hz

		long nr_coeff = maps_dims[COEFF_DIM];

		long fB0_flag = get_fB0_flag(model);

		long map_dims[DIMS];
		md_select_dims(DIMS, ~COEFF_FLAG, map_dims, maps_dims);

		complex float* map = md_alloc_sameplace(DIMS, map_dims, CFL_SIZE, maps);


		long pos[DIMS] = { [0 ... DIMS - 1] = 0 };

		for (long n = 0; n < nr_coeff; n++) {

			pos[COEFF_DIM] = n;
			md_copy_block(DIMS, pos, map_dims, map, maps_dims, maps, CFL_SIZE);

			md_zsmul(DIMS, map_dims, map, map, scaling[n]);

			if (MD_IS_SET(fB0_flag, n))
				linop_forward_unchecked(op, map, map);

			md_copy_block(DIMS, pos, maps_dims, maps, map_dims, map, CFL_SIZE);
		}

		md_free(map);
	}
}




void meco_recon(const struct moba_conf* moba_conf,
		enum meco_model sel_model, bool real_pd, enum fat_spec fat_spec,
		const float* scale_fB0, bool warmstart, bool out_origin_maps,
		const long maps_dims[DIMS], complex float* maps,
		const long sens_dims[DIMS], complex float* sens,
		const long init_dims[DIMS], const complex float* init,
		const complex float* mask,
		const complex float* TE,
		const long P_dims[DIMS], const complex float* Pin,
		const long Y_dims[DIMS], const complex float* Y)
{
	bool use_gpu = false;

#ifdef USE_CUDA
	use_gpu = cuda_ondevice(Y);
#endif

	// setup pointer

	long frame_pos[DIMS] = { 0 };
	long P_pos[DIMS] = { 0 };

	complex float* maps_ptr = (void*)maps + md_calc_offset(DIMS, MD_STRIDES(DIMS, maps_dims, CFL_SIZE), frame_pos);
	complex float* sens_ptr = (void*)sens + md_calc_offset(DIMS, MD_STRIDES(DIMS, sens_dims, CFL_SIZE), frame_pos);

	unsigned int fft_flags = FFT_FLAGS;


	// dimensions & size

	long maps_1s_dims[DIMS];
	md_copy_dims(DIMS, maps_1s_dims, maps_dims);

	long sens_1s_dims[DIMS];
	md_copy_dims(DIMS, sens_1s_dims, sens_dims);

	long Y_1s_dims[DIMS];
	md_copy_dims(DIMS, Y_1s_dims, Y_dims);

	if (!moba_conf->stack_frames) {

		maps_1s_dims[TIME_DIM] = 1;
		sens_1s_dims[TIME_DIM] = 1;

		Y_1s_dims[TIME_DIM] = 1;
	}

	long maps_1s_size = md_calc_size(DIMS, maps_1s_dims);
	long sens_1s_size = md_calc_size(DIMS, sens_1s_dims);

	long x_1s_size = maps_1s_size + sens_1s_size;
	long y_1s_size = md_calc_size(DIMS, Y_1s_dims);

	long meco_1s_dims[DIMS];
	md_select_dims(DIMS, fft_flags|TE_FLAG|TIME_FLAG, meco_1s_dims, Y_1s_dims);

	// init maps and sens

	if (NULL != init) {

		assert(md_check_bounds(DIMS, 0, maps_1s_dims, init_dims));

		md_copy(DIMS, maps_1s_dims, maps_ptr, init, CFL_SIZE); // maps

		long init_size = md_calc_size(DIMS, init_dims);

		if (init_size > maps_1s_size) {

			assert(init_size == x_1s_size);

			fftmod(DIMS, sens_1s_dims, FFT_FLAGS | (moba_conf->stack_frames ? TIME_FLAG : 0u), sens_ptr, init + maps_1s_size);

		} else {

			md_clear(DIMS, sens_1s_dims, sens_ptr, CFL_SIZE);
		}

		debug_printf(DP_DEBUG2, " >> init maps provided.\n");

	} else {

		init_meco_maps(maps_1s_dims, maps_ptr, sel_model);

		md_clear(DIMS, sens_1s_dims, sens_ptr, CFL_SIZE);
	}

	long mask_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, mask_dims, maps_dims);

	md_zmul2(DIMS, maps_1s_dims, MD_STRIDES(DIMS, maps_1s_dims, CFL_SIZE), maps_ptr, 
		MD_STRIDES(DIMS, maps_1s_dims, CFL_SIZE), maps_ptr, 
		MD_STRIDES(DIMS, mask_dims, CFL_SIZE), mask);



	// scaling of psf
	//
	complex float* P = md_alloc_sameplace(DIMS, P_dims, CFL_SIZE, Pin);
	md_copy(DIMS, P_dims, P, Pin, CFL_SIZE);

	if (moba_conf->noncartesian) {

		ifft(DIMS, P_dims, FFT_FLAGS, P, P);

		double scaling_P = 1. / cabsf(P[0]) / 10.;

		md_zsmul(DIMS, P_dims, P, P, scaling_P);

		fft(DIMS, P_dims, FFT_FLAGS, P, P);
	}



	// reconstruction: jointly or sequentially

	complex float* x_akt    = md_alloc_sameplace(1, MD_DIMS(x_1s_size), CFL_SIZE, Y);
	complex float* xref_akt = md_alloc_sameplace(1, MD_DIMS(x_1s_size), CFL_SIZE, Y);


	for (long f = 0; f < (moba_conf->stack_frames ? 1 : Y_dims[TIME_DIM]); f++) {

		debug_printf(DP_INFO, moba_conf->stack_frames ? ">>> stack " : ">>> frame ");
		debug_printf(DP_INFO, "%3d\n", f);

		bool reset = (0 == f);

		// Y
		frame_pos[TIME_DIM] = f;

		complex float* Y_ptr = (void*)Y + md_calc_offset(DIMS, MD_STRIDES(DIMS, Y_dims, CFL_SIZE), frame_pos);

		double scaling_Y = 100. / md_znorm(DIMS, Y_1s_dims, Y_ptr);

		md_zsmul(DIMS, Y_1s_dims, Y_ptr, Y_ptr, scaling_Y);

		// P
		P_pos[TIME_DIM] = f % P_dims[TIME_DIM];

		complex float* P_ptr = (void*)P + md_calc_offset(DIMS, MD_STRIDES(DIMS, P_dims, CFL_SIZE), P_pos);

		maps_ptr = (void*)maps + md_calc_offset(DIMS, MD_STRIDES(DIMS, maps_dims, CFL_SIZE), frame_pos);
		sens_ptr = (void*)sens + md_calc_offset(DIMS, MD_STRIDES(DIMS, sens_dims, CFL_SIZE), frame_pos);

		if (reset) {

			md_copy(DIMS, maps_1s_dims, x_akt, maps_ptr, CFL_SIZE);
			md_copy(DIMS, sens_1s_dims, x_akt + maps_1s_size, sens_ptr, CFL_SIZE);

			md_zsmul(1, MD_DIMS(x_1s_size), xref_akt, x_akt, (MECO_PI != sel_model) ? moba_conf->damping : 0.);

		} else {

			md_zsmul(1, MD_DIMS(x_1s_size), xref_akt, x_akt, moba_conf->damping);
		}


		struct noir_model_conf_s mconf = noir_model_conf_defaults;
		mconf.noncart = moba_conf->noncartesian;
		mconf.fft_flags = fft_flags;
		mconf.a = moba_conf->sobolev_a;
		mconf.b = moba_conf->sobolev_b;
		mconf.cnstcoil_flags = TE_FLAG;

		struct meco_s nl = meco_create(Y_1s_dims, meco_1s_dims, maps_1s_dims, mask, TE, P_ptr, sel_model, real_pd, fat_spec, scale_fB0, use_gpu, &mconf);


		struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;
		irgnm_conf.iter = moba_conf->iter;
		irgnm_conf.alpha = moba_conf->alpha;

		if (moba_conf->alpha_min_exp_decay)
			irgnm_conf.alpha_min = moba_conf->alpha_min;
		else
			irgnm_conf.alpha_min0 = moba_conf->alpha_min;

		irgnm_conf.redu = moba_conf->redu;
		irgnm_conf.cgiter = moba_conf->inner_iter;
		irgnm_conf.cgtol = 0.01;
		irgnm_conf.nlinv_legacy = false;

		long x_dims[DIMS];
		md_merge_dims(DIMS, x_dims, maps_1s_dims, sens_1s_dims); // mixed

		// linearized reconstruction

		struct opt_reg_s* ropts = moba_conf->ropts;

		const struct operator_p_s* inv_op = NULL;

		const struct operator_p_s* prox_ops[NUM_REGS] = { NULL };
		const struct linop_s* trafos[NUM_REGS] = { NULL };

		int algo = moba_conf->algo;

		if (0 == ropts->r)
			algo = ALGO_CG;


		if (ALGO_CG == algo) { // CG

			debug_printf(DP_DEBUG2, " >> linearized problem solved by CG\n");
			
			// assert(0 == moba_conf->ropts->r);

			inv_op = NULL;

		} else 
		if (ALGO_ADMM == algo) {

			debug_printf(DP_DEBUG2, " >> linearized problem solved by ADMM ");

			/* use lsqr */
			debug_printf(DP_DEBUG2, "in lsqr\n");

			struct optreg_conf optreg_conf = optreg_defaults;

			optreg_conf.moba_model = sel_model;
			optreg_conf.weight_fB0_type = nl.weight_fB0_type;

			opt_reg_moba_configure(DIMS, x_dims, ropts, prox_ops, trafos, &optreg_conf);


			struct iter_admm_conf iadmm_conf = iter_admm_defaults;
			iadmm_conf.maxiter = moba_conf->inner_iter;
			iadmm_conf.cg_eps = irgnm_conf.cgtol;
			iadmm_conf.rho = moba_conf->rho;


			struct lsqr_conf lsqr_conf = lsqr_defaults;
			lsqr_conf.it_gpu = false;
			lsqr_conf.warmstart = warmstart;

			NESTED(void, lsqr_cont, (iter_conf* iconf))
			{
				auto aconf = CAST_DOWN(iter_admm_conf, iconf);

				aconf->maxiter = MIN(iadmm_conf.maxiter, 10. * powf(2., logf(1. / iconf->alpha)));
				aconf->cg_eps = iadmm_conf.cg_eps * iconf->alpha;
			};

			lsqr_conf.icont = lsqr_cont;

			const struct nlop_s* nlop = nl.nlop;

			inv_op = lsqr2_create(&lsqr_conf, iter2_admm, CAST_UP(&iadmm_conf), NULL, &nlop->derivative[0][0], NULL, ropts->r, prox_ops, trafos, NULL);

		} else {

			error(" >> Unrecognized algorithms\n");
		}


		// irgnm reconstruction

		((NULL == inv_op) ? iter4_irgnm : iter4_irgnm2)(CAST_UP(&irgnm_conf),
			nl.nlop,
			x_1s_size * 2, (float*)x_akt, (float*)xref_akt,
			y_1s_size * 2, (const float*)Y_ptr,
			inv_op, (struct iter_op_s){ NULL, NULL });

		operator_p_free(inv_op);

		opt_reg_free(ropts, prox_ops, trafos);


		// post processing

		md_copy(DIMS, maps_1s_dims, maps_ptr, x_akt, CFL_SIZE);
		md_copy(DIMS, sens_1s_dims, sens_ptr, x_akt + maps_1s_size, CFL_SIZE);

		if (!out_origin_maps) {

			rescale_maps(sel_model, scaling_Y, nl.linop_fB0, nl.scaling, maps_1s_dims, maps_ptr);

			noir_forw_coils(nl.linop, sens_ptr, sens_ptr);
			fftmod(DIMS, sens_1s_dims, mconf.fft_flags, sens_ptr, sens_ptr);
		}

		nlop_free(nl.nlop);
	}

	if (moba_conf->noncartesian)
		md_free(P);

	md_free(x_akt);
	md_free(xref_akt);
}
