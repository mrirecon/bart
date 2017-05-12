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
#include "iter/thresh.h"
#include "iter/italgos.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "noir/model.h"

#include "recon.h"


struct data {

	INTERFACE(iter_op_data);

	struct noir_data* ndata;
};

DEF_TYPEID(data);


static void frw(iter_op_data* ptr, float* _dst, const float* _src)
{
        struct data* data = CAST_DOWN(data, ptr);

	noir_fun(data->ndata, (complex float*)_dst, (const complex float*)_src);
}

static void adj(iter_op_data* ptr, float* _dst, const float* _src)
{
        struct data* data = CAST_DOWN(data, ptr);

	noir_adj(data->ndata, (complex float*)_dst, (const complex float*)_src);
}

static void der(iter_op_data* ptr, float* _dst, const float* _src)
{
        struct data* data = CAST_DOWN(data, ptr);

	noir_der(data->ndata, (complex float*)_dst, (const complex float*)_src);
}


const struct noir_conf_s noir_defaults = {

	.iter = 8,
	.rvc = false,
	.usegpu = false,
	.alpha = 1.,
	.redu = 2.,
};


void noir_recon(const struct noir_conf_s* conf, const long dims[DIMS], complex float* outbuf, complex float* sensout, const complex float* psf, const complex float* mask, const complex float* kspace)
{
	long imgs_dims[DIMS];
	long coil_dims[DIMS];
	long data_dims[DIMS];
	long img1_dims[DIMS];

	md_select_dims(DIMS, FFT_FLAGS|MAPS_FLAG|CSHIFT_FLAG, imgs_dims, dims);
	md_select_dims(DIMS, FFT_FLAGS|COIL_FLAG|MAPS_FLAG, coil_dims, dims);
	md_select_dims(DIMS, FFT_FLAGS|COIL_FLAG, data_dims, dims);
	md_select_dims(DIMS, FFT_FLAGS, img1_dims, dims);

	long skip = md_calc_size(DIMS, imgs_dims);
	long size = skip + md_calc_size(DIMS, coil_dims);
	long data_size = md_calc_size(DIMS, data_dims);

	long d1[1] = { size };
	complex float* img = md_alloc_sameplace(1, d1, CFL_SIZE, kspace);
	complex float* imgH = md_alloc_sameplace(1, d1, CFL_SIZE, kspace);


	md_clear(DIMS, imgs_dims, img, CFL_SIZE);

	md_zfill(DIMS, img1_dims, outbuf, 1.);	// initial only first image
	md_copy(DIMS, img1_dims, img, outbuf, CFL_SIZE);

	md_clear(DIMS, coil_dims, img + skip, CFL_SIZE);

	md_clear(DIMS, imgs_dims, imgH, CFL_SIZE);
	md_clear(DIMS, coil_dims, imgH + skip, CFL_SIZE);

	struct noir_data* ndata = noir_init(dims, mask, psf, conf->rvc, conf->usegpu);
	struct data data = { { &TYPEID(data) }, ndata };

	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;
	irgnm_conf.iter = conf->iter;
	irgnm_conf.alpha = conf->alpha;
	irgnm_conf.redu = conf->redu;

	iter3_irgnm(CAST_UP(&irgnm_conf),
			(struct iter_op_s){ frw, CAST_UP(&data) },
			(struct iter_op_s){ der, CAST_UP(&data) },
			(struct iter_op_s){ adj, CAST_UP(&data) },
			size * 2, (float*)img, NULL,
			data_size * 2, (const float*)kspace);

	md_copy(DIMS, imgs_dims, outbuf, img, CFL_SIZE);

	if (NULL != sensout) {

		assert(!conf->usegpu);
		noir_forw_coils(ndata, sensout, img + skip);
	}

	noir_free(ndata);

	md_free(img);
	md_free(imgH);
}



