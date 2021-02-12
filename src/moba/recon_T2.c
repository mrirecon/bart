/* Copyright 2018-2020. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2018-2020 Xiaoqing Wang <xiaoqing.wang@med.uni-goettingen.de>
 */

#include <complex.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"

#include "iter/iter3.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "nlops/nlop.h"

#include "noir/model.h"
#include "noir/recon.h"

#include "moba/model_T2.h"
#include "moba/iter_l1.h"
#include "moba/recon_T1.h"
#include "moba/moba.h"

#include "recon_T2.h"



void T2_recon(const struct moba_conf* conf, const long dims[DIMS], complex float* img, complex float* sens, const complex float* pattern, const complex float* mask, const complex float* TI, const complex float* kspace_data, _Bool usegpu)
{
	long imgs_dims[DIMS];
	long coil_dims[DIMS];
	long data_dims[DIMS];
	long img1_dims[DIMS];

	unsigned int fft_flags = FFT_FLAGS;

	if (conf->sms)
		fft_flags |= SLICE_FLAG;

	md_select_dims(DIMS, fft_flags|MAPS_FLAG|CSHIFT_FLAG|COEFF_FLAG, imgs_dims, dims);
	md_select_dims(DIMS, fft_flags|COIL_FLAG|MAPS_FLAG, coil_dims, dims);
	md_select_dims(DIMS, fft_flags|COIL_FLAG|TE_FLAG, data_dims, dims);
	md_select_dims(DIMS, fft_flags, img1_dims, dims);

	imgs_dims[COEFF_DIM] = 2;

	long skip = md_calc_size(DIMS, imgs_dims);
	long size = skip + md_calc_size(DIMS, coil_dims);
	long data_size = md_calc_size(DIMS, data_dims);

	long d1[1] = { size };
	// variable which is optimized by the IRGNM
	complex float* x = md_alloc_sameplace(1, d1, CFL_SIZE, kspace_data);

	md_copy(DIMS, imgs_dims, x, img, CFL_SIZE);
	md_copy(DIMS, coil_dims, x + skip, sens, CFL_SIZE);

	struct noir_model_conf_s mconf = noir_model_conf_defaults;
	mconf.rvc = false;
	mconf.noncart = conf->noncartesian;
	mconf.fft_flags = fft_flags;
	mconf.a = conf->sobolev_a;
	mconf.b = conf->sobolev_b;
	mconf.cnstcoil_flags = TE_FLAG;

	//struct noir_s nl = noir_create(dims, mask, pattern, &mconf);
	struct T2_s nl = T2_create(dims, mask, TI, pattern, &mconf, usegpu);

	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;

	irgnm_conf.iter = conf->iter;
	irgnm_conf.alpha = conf->alpha;
	irgnm_conf.redu = conf->redu;
	if (conf->alpha_min_exp_decay)
		irgnm_conf.alpha_min = conf->alpha_min;
	else
		irgnm_conf.alpha_min0 = conf->alpha_min;
	irgnm_conf.cgtol = conf->tolerance;
	irgnm_conf.cgiter = conf->inner_iter;
	irgnm_conf.nlinv_legacy = true;

	struct mdb_irgnm_l1_conf conf2 = { 
		.c2 = &irgnm_conf, 
		.opt_reg = conf->opt_reg,
		.step = conf->step, 
		.lower_bound = conf->lower_bound, 
		.constrained_maps = 1,
		.auto_norm_off = conf->auto_norm_off };

	long irgnm_conf_dims[DIMS];
	md_select_dims(DIMS, fft_flags|MAPS_FLAG|CSHIFT_FLAG|COEFF_FLAG, irgnm_conf_dims, imgs_dims);

	irgnm_conf_dims[COIL_DIM] = coil_dims[COIL_DIM];

	debug_printf(DP_INFO, "imgs_dims:\n\t");
	debug_print_dims(DP_INFO, DIMS, irgnm_conf_dims);


	mdb_irgnm_l1(&conf2,
			irgnm_conf_dims,
			nl.nlop,
			size * 2, (float*)x,
			data_size * 2, (const float*)kspace_data);

	md_copy(DIMS, imgs_dims, img, x, CFL_SIZE);

	if (NULL != sens) {

		noir_forw_coils(nl.linop, x + skip, x + skip);
		md_copy(DIMS, coil_dims, sens, x + skip, CFL_SIZE);
		fftmod(DIMS, coil_dims, fft_flags, sens, sens);
	}

	nlop_free(nl.nlop);

	md_free(x);
}



