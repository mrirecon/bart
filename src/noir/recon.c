/* Copyright 2013. The Regents of the University of California.
 * Copyright 2016-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2011-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
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
#include "noir/nl.h"

#include "recon.h"



const struct noir_conf_s noir_defaults = {

	.iter = 8,
	.rvc = false,
	.usegpu = false,
	.noncart = false,
	.alpha = 1.,
	.redu = 2.,
	.pattern_for_each_coil = false,
};

void noir_recon(const struct noir_conf_s* conf, const long dims[DIMS], complex float* img, complex float* sens, const complex float* pattern, const complex float* mask, const complex float* kspace_data )
{
	long imgs_dims[DIMS];
	long coil_dims[DIMS];
	long data_dims[DIMS];
	long img1_dims[DIMS];

	unsigned int fft_flags = FFT_FLAGS|SLICE_FLAG;

	md_select_dims(DIMS, fft_flags|MAPS_FLAG|CSHIFT_FLAG, imgs_dims, dims);
	md_select_dims(DIMS, fft_flags|COIL_FLAG|MAPS_FLAG, coil_dims, dims);
	md_select_dims(DIMS, fft_flags|COIL_FLAG, data_dims, dims);
	md_select_dims(DIMS, fft_flags, img1_dims, dims);

	long skip = md_calc_size(DIMS, imgs_dims);
	long size = skip + md_calc_size(DIMS, coil_dims);
	long data_size = md_calc_size(DIMS, data_dims);

	long d1[1] = { size };
	// variable which is optimized by the IRGNM
	complex float* x = md_alloc_sameplace(1, d1, CFL_SIZE, kspace_data );

	md_copy(DIMS, imgs_dims, x, img, CFL_SIZE);
	md_copy(DIMS, coil_dims, x + skip, sens, CFL_SIZE);

	struct noir_model_conf_s mconf = noir_model_conf_defaults;
	mconf.rvc = conf->rvc;
	mconf.use_gpu = conf->usegpu;
	mconf.noncart = conf->noncart;
	mconf.fft_flags = fft_flags;
	mconf.pattern_for_each_coil = conf->pattern_for_each_coil;

	struct nlop_s* nlop = noir_create(dims, mask, pattern, &mconf);

	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;

	irgnm_conf.iter = conf->iter;
	irgnm_conf.alpha = conf->alpha;
	irgnm_conf.redu = conf->redu;
	irgnm_conf.cgtol = 0.1f;
	irgnm_conf.nlinv_legacy = true;

	iter4_irgnm(CAST_UP(&irgnm_conf),
			nlop,
			size * 2, (float*) x, NULL,
			data_size * 2, (const float*) kspace_data );

	md_copy(DIMS, imgs_dims, img, x, CFL_SIZE);

	if (NULL != sens ) {

#ifdef USE_CUDA
		if (conf->usegpu) {

			noir_forw_coils(noir_get_data(nlop), x + skip, x + skip);
			md_copy(DIMS, coil_dims, sens, x + skip, CFL_SIZE);
		} else
#endif
			noir_forw_coils(noir_get_data(nlop), sens, x + skip);
		fftmod(DIMS, coil_dims, fft_flags, sens, sens);
	}

	nlop_free(nlop);

	md_free(x);
}



