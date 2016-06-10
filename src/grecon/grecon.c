/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2013 Martin Uecker <uecker@eecs.berkeley.edu>
 *
 */

#include <stdbool.h>
#include <complex.h>
#include <assert.h>
#include <math.h>

#include "sense/recon.h"
#include "sense/model.h"
#include "sense/pocs.h"
#include "sense/optcom.h"

#include "linops/linop.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/ops.h"
#include "num/gpuops.h"

// #define W3
#ifndef W3
#include "wavelet2/wavelet.h"
#else
#include "wavelet3/wavthresh.h"
#endif

#include "iter/iter.h"

#include "calib/calib.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"
#include "misc/utils.h"

#include "grecon/parslices.h"
#include "grecon.h"


#ifdef USE_CUDA
#include <omp.h>
#define MAX_CUDA_DEVICES 16
omp_lock_t gpulock[MAX_CUDA_DEVICES];
#endif



void grecon(struct grecon_conf* param,  const long dims1[DIMS], complex float* out1,
	const long cov1_dims[DIMS], complex float* cov1,
	const long w1_dims[DIMS], const complex float* weights,
	complex float* kspace1, bool usegpu)
{
	const struct sense_conf* conf = param->sense_conf;

	long ksp1_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, ksp1_dims, dims1);

	long pat1_dims[DIMS];
	const complex float* pattern;

	if (NULL == weights) {

		md_select_dims(DIMS, ~(COIL_FLAG | MAPS_FLAG), pat1_dims, dims1);
		complex float* tpattern = md_alloc(DIMS, pat1_dims, CFL_SIZE);
		estimate_pattern(DIMS, ksp1_dims, COIL_FLAG, tpattern, kspace1);
		pattern = tpattern;

	} else {

		md_copy_dims(DIMS, pat1_dims, w1_dims);
		pattern = weights;
	}

	complex float* sens1 = NULL;

	long sens_dims[DIMS];
	md_copy_dims(DIMS, sens_dims, dims1);
	md_min_dims(DIMS, ~(FFT_FLAGS | SENS_FLAGS), sens_dims, dims1, cov1_dims);

	if (NULL != param->calib) {

		long maps_dims[DIMS];
		md_select_dims(DIMS, ~COIL_FLAG, maps_dims, sens_dims);

		sens1 = md_alloc(DIMS, sens_dims, CFL_SIZE);
		complex float* maps1 = md_alloc(DIMS, maps_dims, CFL_SIZE);

		caltwo(param->calib, sens_dims, sens1, maps1, cov1_dims, cov1, NULL, NULL);

		crop_sens(sens_dims, sens1, param->calib->softcrop, param->calib->crop, maps1);

		fixphase(DIMS, sens_dims, COIL_DIM, sens1, sens1);

		md_free(maps1);

	} else {

		sens1 = cov1;
	}

	fftmod(DIMS, sens_dims, FFT_FLAGS, sens1, sens1);
	fftmod(DIMS, ksp1_dims, FFT_FLAGS, kspace1, kspace1);

	complex float* image1 = NULL;

	long img1_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, img1_dims, dims1);

	if (param->ksp) {

		image1 = md_alloc(DIMS, img1_dims, CFL_SIZE);
		md_clear(DIMS, img1_dims, image1, CFL_SIZE);

	} else {

		image1 = out1;
	}


#ifdef  USE_CUDA
	int gpun = 0;

	if (usegpu) {

		int nr_cuda_devices = MIN(cuda_devices(), MAX_CUDA_DEVICES);
		gpun = omp_get_thread_num() % nr_cuda_devices;
		cuda_init(gpun);
	}
#endif

	const struct operator_p_s* thresh_op = NULL;

	if (param->l1wav) {

		debug_printf(DP_DEBUG3, "Lambda: %f\n", param->lambda);

		long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
		minsize[0] = MIN(img1_dims[0], 16);
		minsize[1] = MIN(img1_dims[1], 16);
		minsize[2] = MIN(img1_dims[2], 16);
#ifndef W3
		thresh_op = prox_wavethresh_create(DIMS, img1_dims, FFT_FLAGS, minsize, param->lambda, param->randshift, usegpu);
#else
		unsigned int wflags = 0;
		for (unsigned int i = 0; i < 3; i++)
			if (1 < img1_dims[i])
				wflags = MD_SET(wflags, i);

		thresh_op = prox_wavelet3_thresh_create(DIMS, img1_dims, wflags, minsize, param->lambda, param->randshift);
#endif
	}

	italgo_fun_t italgo = NULL;
	void* iconf = NULL;

	struct iter_conjgrad_conf cgconf;
	struct iter_fista_conf fsconf;

	if (!param->l1wav) {

		cgconf = iter_conjgrad_defaults;
		cgconf.maxiter = param->maxiter;
		cgconf.l2lambda = param->lambda;
		cgconf.tol = 1.E-3;

		italgo = iter_conjgrad;
		iconf = &cgconf;

	} else {

		fsconf = iter_fista_defaults;
		fsconf.maxiter = param->maxiter;
		fsconf.step = param->step;

		italgo = iter_fista;
		iconf = &fsconf;
	}

	const struct linop_s* sop = NULL;

	{
		sop = sense_init(dims1, FFT_FLAGS|COIL_FLAG|MAPS_FLAG, sens1, false);
		sense_recon(conf, dims1, image1, sop, pat1_dims, pattern, italgo, iconf, thresh_op, ksp1_dims, kspace1, NULL, NULL);
		//linop_free(sop);
	}

	// FIXME: free thresh_op

	if (param->ksp) {

		if (param->rplksp)
			replace_kspace(dims1, out1, kspace1, sens1, image1);
		else
			fake_kspace(dims1, out1, sens1, image1);

		ifftmod(DIMS, ksp1_dims, FFT_FLAGS, out1, out1);

		md_free(image1);
	}

	if (NULL == weights)
		md_free((void*)pattern);

	if (NULL != param->calib)
		md_free(sens1);

	if (param->l1wav)
		operator_p_free(thresh_op);

#ifdef  USE_CUDA
//	if (usegpu)
//		cuda_memcache_clear();
		//cuda_exit();
#endif
}



static void grecon_delegate(void* _param, const long dims1[DIMS], complex float* out1,
	const long sens1_dims[DIMS], complex float* cov1,
	const long w1_dims[DIMS], const complex float* weights,
	complex float* kspace1, bool usegpu)
{
	struct grecon_conf* param = _param;
	grecon(param, dims1, out1, sens1_dims, cov1, w1_dims, weights, kspace1, usegpu);
}

void rgrecon(struct grecon_conf* conf, const long dims[DIMS], complex float* image,
			const long sens_dims[DIMS], const complex float* sens_maps,
			const long pat1_dims[DIMS], const complex float* weights,
			const complex float* kspace_data, bool usegpu)
{
	parslices(grecon_delegate, conf,
			dims, image,
			sens_dims, sens_maps,
			pat1_dims, weights,
			kspace_data, conf->ksp, usegpu);
}

void rgrecon2(struct grecon_conf* conf, const long dims[DIMS],
			const long img_strs[DIMS], complex float* image,
			const long sens_dims[DIMS], const complex float* sens_maps,
			const long pat1_dims[DIMS], const complex float* weights,
			const long ksp_strs[DIMS], const complex float* kspace_data,
			bool usegpu)
{
	parslices2(grecon_delegate, conf,
			dims, img_strs, image,
			sens_dims, sens_maps,
			pat1_dims, weights,
			ksp_strs, kspace_data,
			conf->ksp, usegpu);
}
