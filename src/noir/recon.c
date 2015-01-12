/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2011-2014 Martin Uecker <uecker@eecs.berkeley.edu>
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

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "noir/model.h"

#include "recon.h"

#ifdef	BERKELEY_SVN
#include "misc/phsing.h"
#define PHSING
#endif


#ifdef WAVELET
#include "sense/wavelet.h"
#endif


struct data {

	struct noir_data* ndata;

#ifdef WAVELET
	struct wavelet_plan_s* wdata;
	const struct operator_s* thresh_data;
#else
	void* dummy1;
	void* dummy2;
#endif
};




static void frw(void* ptr, float* _dst, const float* _src)
{
        struct data* data = ptr;

	noir_fun(data->ndata, (complex float*)_dst, (const complex float*)_src);
}

static void adj(void* ptr, float* _dst, const float* _src)
{
        struct data* data = ptr;

	noir_adj(data->ndata, (complex float*)_dst, (const complex float*)_src);
}

static void der(void* ptr, float* _dst, const float* _src)
{
        struct data* data = ptr;

	noir_der(data->ndata, (complex float*)_dst, (const complex float*)_src);
}


#ifdef WAVELET
static void thresh(void* ptr, float lambda, float* _dst, const float* _src)
{
	struct data* data = (struct data*)ptr;
	set_thresh_lambda(data->thresh_data, lambda);

	wavelet_thresh_xx(data->wdata, data->thresh_data, (complex float*)_dst, (const complex float*)_src);	
}
#endif

void noir_recon(const long dims[DIMS], unsigned int iter, float th, complex float* outbuf, complex float* sensout, const complex float* psf, const complex float* mask, const complex float* kspace, bool rvc, bool usegpu)
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

	struct noir_data* ndata = noir_init(dims, mask, psf, rvc, usegpu);

	struct data data = { ndata, NULL, NULL };

	if (-1. == th) {

		struct iter3_irgnm_conf conf = { .iter = iter, .alpha = 1., .redu = 2. };
		bool repeat = false;

		do {
			iter3_irgnm(&conf, frw, der, adj, &data, size * 2, (float*)img, data_size * 2, (const float*)kspace);
#ifdef PHSING
			{
//			if (repeat) {

			assert(!usegpu);
			complex float* coils = md_alloc(DIMS, coil_dims, CFL_SIZE);
			noir_forw_coils(ndata, coils, img + skip);
			fftmod(DIMS, coil_dims, FFT_FLAGS, coils, coils);
//			dump_cfl("coils", DIMS, coil_dims, coils);

			repeat = fixphsing(imgs_dims, img, coil_dims, coils);

			if (repeat)
				md_free(coils);
			}
#endif
		} while (repeat);

	} else {
#ifdef WAVELET
		data.thresh_data = thresh_init(DIMS, imgs_dims, th, 0, false);
		data.wdata = wavelet_thresh_init(imgs_dims, true, false);

	//	irgnm_t(iter, 1., th, 2., (void*)&data, size * 2, data_size * 2, ops, forw, adj, inv, thresh, (float*)img, (float*)imgH, (float*)kspace);
		assert(0);
#else
		assert(0);
#endif
	}

	md_copy(DIMS, imgs_dims, outbuf, img, CFL_SIZE);

	if (NULL != sensout) {

		assert(!usegpu);
		noir_forw_coils(ndata, sensout, img + skip);
	}

	noir_free(ndata);

	md_free(img);
	md_free(imgH);
}



