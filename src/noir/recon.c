/* Copyright 2013. The Regents of the University of California.
 * Copyright 2016-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2011-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 *
 *
 * Uecker M, Hohage T, Block KT, Frahm J. Image reconstruction by regularized
 * nonlinear inversion – Joint estimation of coil sensitivities and image content.
 * Magn Reson Med 2008; 60:674-682.
 */

#include <complex.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"

#include "iter/iter3.h"
#include "iter/iter4.h"
#include "iter/thresh.h"
#include "iter/italgos.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "noir/model.h"
#include "nlops/nlop.h"

#include "recon.h"


struct nlop_wrapper_s {

	INTERFACE(struct iter_op_data_s);

	struct noir_s* noir;
	long split;
};

DEF_TYPEID(nlop_wrapper_s);


static void orthogonalize(iter_op_data* ptr, float* _dst, const float* _src)
{
	UNUSED(_src);

	auto nlw = CAST_DOWN(nlop_wrapper_s, ptr);

	noir_orthogonalize(nlw->noir, (complex float*) _dst + nlw->split);
}


const struct noir_conf_s noir_defaults = {

	.iter = 8,
	.rvc = false,
	.usegpu = false,
	.noncart = false,
	.alpha = 1.,
	.alpha_min = 0.,
	.redu = 2.,
	.a = 220.,
	.b = 32.,
	.pattern_for_each_coil = false,
	.sms = false,
};


void noir_recon(const struct noir_conf_s* conf, const long dims[DIMS], complex float* img, complex float* sens, const complex float* ref, const complex float* pattern, const complex float* mask, const complex float* kspace_data )
{
	long imgs_dims[DIMS];
	long coil_dims[DIMS];
	long data_dims[DIMS];
	long img1_dims[DIMS];

	unsigned int fft_flags = FFT_FLAGS;

	if (conf->sms)
		fft_flags |= SLICE_FLAG;

	md_select_dims(DIMS, fft_flags|MAPS_FLAG, imgs_dims, dims);
	md_select_dims(DIMS, fft_flags|COIL_FLAG|MAPS_FLAG, coil_dims, dims);
	md_select_dims(DIMS, fft_flags|COIL_FLAG, data_dims, dims);
	md_select_dims(DIMS, fft_flags, img1_dims, dims);

	long skip = md_calc_size(DIMS, imgs_dims);
	long size = skip + md_calc_size(DIMS, coil_dims);
	long data_size = md_calc_size(DIMS, data_dims);

	long d1[1] = { size };
	// variable which is optimized by the IRGNM
	complex float* x = md_alloc_sameplace(1, d1, CFL_SIZE, kspace_data);

	md_copy(DIMS, imgs_dims, x, img, CFL_SIZE);
	md_copy(DIMS, coil_dims, x + skip, sens, CFL_SIZE);

	complex float* xref = NULL;

	if (NULL != ref) {

		xref = md_alloc_sameplace(1, d1, CFL_SIZE, kspace_data);
		md_copy(1, d1, xref, ref, CFL_SIZE);
	}

	struct noir_model_conf_s mconf = noir_model_conf_defaults;
	mconf.rvc = conf->rvc;
	mconf.use_gpu = conf->usegpu;
	mconf.noncart = conf->noncart;
	mconf.fft_flags = fft_flags;
	mconf.a = conf->a;
	mconf.b = conf->b;
	mconf.pattern_for_each_coil = conf->pattern_for_each_coil;


	struct noir_s nl = noir_create(dims, mask, pattern, &mconf);

	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;

	irgnm_conf.iter = conf->iter;
	irgnm_conf.alpha = conf->alpha;
	irgnm_conf.alpha_min = conf->alpha_min;
	irgnm_conf.redu = conf->redu;
	irgnm_conf.cgtol = 0.1f;
	irgnm_conf.nlinv_legacy = true;

	struct nlop_wrapper_s nlw;

	SET_TYPEID(nlop_wrapper_s, &nlw);

	nlw.noir = &nl;
	nlw.split = skip;


	iter4_irgnm(CAST_UP(&irgnm_conf),
			nl.nlop,
			size * 2, (float*)x, (const float*)xref,
			data_size * 2, (const float*)kspace_data,
			(struct iter_op_s){ orthogonalize, CAST_UP(&nlw)});

	md_copy(DIMS, imgs_dims, img, x, CFL_SIZE);

	if (NULL != sens) {

#ifdef USE_CUDA
		if (conf->usegpu) {

			noir_forw_coils(nl.linop, x + skip, x + skip);
			md_copy(DIMS, coil_dims, sens, x + skip, CFL_SIZE);
		} else
#endif
			noir_forw_coils(nl.linop, sens, x + skip);

		fftmod(DIMS, coil_dims, fft_flags, sens, sens);
	}

	nlop_free(nl.nlop);

	md_free(x);
	md_free(xref);
}



