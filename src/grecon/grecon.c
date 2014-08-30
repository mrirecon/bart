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
#include "sense/pocs.h"
#include "sense/optcom.h"

#ifndef STANFORD_OFFLINERECON
#include "noir/recon.h"
#endif

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/gpuops.h"

#include "wavelet2/wavelet.h"

#include "iter/iter.h"

#include "calib/calib.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "grecon/parslices.h"
#include "grecon.h"


#ifdef USE_CUDA
#include <omp.h>
#define MAX_CUDA_DEVICES 16
omp_lock_t gpulock[MAX_CUDA_DEVICES];
#endif

enum algo { SENSE, POCS, NOIR };

struct grecon_s {

	enum algo algo;
	const struct ecalib_conf* calib;
	struct sense_conf* conf;
	bool rplksp;
	bool ksp;
};

static void grecon(void* _param,  const long dims1[DIMS], complex float* out1, 
	const long sens1_dims[DIMS], complex float* cov1, const long w1_dims[DIMS], const complex float* weights,
	complex float* kspace1, bool usegpu)
{
	struct grecon_s* param = _param;
	struct sense_conf* conf = param->conf;

	long ksp1_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, ksp1_dims, dims1);

	long pat1_dims[DIMS];
	const complex float* pattern;

	if (NULL == weights) {

		md_select_dims(DIMS, ~(COIL_FLAG | MAPS_FLAG), pat1_dims, dims1);
		complex float* tpattern = md_alloc(DIMS, pat1_dims, CFL_SIZE);
		estimate_pattern(DIMS, ksp1_dims, COIL_DIM, tpattern, kspace1);
		pattern = tpattern;

	} else {

		memcpy(pat1_dims, w1_dims, DIMS * sizeof(long));
		pattern = weights;
	}

	complex float* sens1;

	if (NULL != param->calib) {

		long img1_dims[DIMS];
		md_select_dims(DIMS, ~COIL_FLAG, img1_dims, dims1);

		complex float* maps1 = md_alloc(DIMS, img1_dims, CFL_SIZE);

		sens1 = md_alloc(DIMS, dims1, CFL_SIZE);
	
		caltwo(param->calib, dims1, sens1, maps1, sens1_dims, cov1, NULL, NULL);

		md_free(maps1);

	} else {

		sens1 = cov1;
	}

	if (NOIR == param->algo) {

		assert(NULL == param->calib);
		assert(1 == dims1[MAPS_DIM]);

		sens1 = md_alloc(DIMS, dims1, CFL_SIZE);
		md_clear(DIMS, dims1, sens1, CFL_SIZE);
		fftmod(DIMS, ksp1_dims, FFT_FLAGS, kspace1, kspace1);
	}

	fftmod(DIMS, dims1, FFT_FLAGS, sens1, sens1);
	fftmod(DIMS, ksp1_dims, FFT_FLAGS, kspace1, kspace1);

	complex float* image1 = NULL;

	long img1_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, img1_dims, dims1);

	if (param->ksp && (POCS != param->algo)) {

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

	if (conf->l1wav) {

		long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
		minsize[0] = MIN(img1_dims[0], 16);
		minsize[1] = MIN(img1_dims[1], 16);
		minsize[2] = MIN(img1_dims[2], 16);

		thresh_op = prox_wavethresh_create(DIMS, img1_dims, FFT_FLAGS, minsize, conf->lambda, conf->randshift, usegpu);
	}

	italgo_fun_t italgo = NULL;
	void* iconf = NULL;

	struct iter_conjgrad_conf cgconf;
	memcpy(&cgconf, &iter_conjgrad_defaults, sizeof(struct iter_conjgrad_conf));
	cgconf.maxiter = conf->maxiter;
	cgconf.l2lambda = conf->lambda;

	struct iter_fista_conf fsconf;
	memcpy(&fsconf, &iter_fista_defaults, sizeof(struct iter_fista_conf));
	fsconf.maxiter = conf->maxiter;
	fsconf.step = conf->step;

	if (!conf->l1wav) {

		italgo = iter_conjgrad;
		iconf = &cgconf;

	} else {

		italgo = iter_fista;
		iconf = &fsconf;
	}

#ifdef  USE_CUDA
	if (usegpu) {

		omp_set_lock(&gpulock[gpun]);

		switch (param->algo) {
		case SENSE:
			sense_recon_gpu(conf, dims1, image1, sens1, pat1_dims, pattern, italgo, iconf, thresh_op, ksp1_dims, kspace1, NULL);
			break;
		case POCS:
			pocs_recon_gpu(dims1, thresh_op, conf->maxiter, conf->lambda, -1., image1, sens1, pattern, kspace1);
			break;
		default:
			assert(0);
			break;
		}

		omp_unset_lock(&gpulock[gpun]);

	} else 
#endif
	{
		switch (param->algo) {
		case SENSE:
			sense_recon(conf, dims1, image1, sens1, pat1_dims, pattern, italgo, iconf, thresh_op, ksp1_dims, kspace1, NULL);
			break;
		case POCS:
			pocs_recon(dims1, thresh_op, conf->maxiter, conf->lambda, -1., image1, sens1, pattern, kspace1);
			break;
		case NOIR:
#ifndef STANFORD_OFFLINERECON
			noir_recon(dims1, conf->maxiter, conf->l1wav ? conf->lambda : -1., image1, sens1, pattern, NULL, kspace1, false);
#else
			assert(0);
#endif
			break;
		default:
			assert(0);
			break;
		}
	}

	// FIXME: free thresh_op

	if (param->ksp && (POCS != param->algo)) {

		if (param->rplksp)
			replace_kspace(dims1, out1, kspace1, sens1, image1);
		else
			fake_kspace(dims1, out1, sens1, image1);

		fftmod(DIMS, ksp1_dims, FFT_FLAGS, out1, out1);

		md_free(image1);
	}

	if (NULL == weights)
		md_free((void*)pattern);

	if ((NULL != param->calib) || (NOIR == param->algo))
		md_free(sens1);
}



void w1sense(struct sense_conf* conf, const struct ecalib_conf* calib, bool rplksp, const long dims[DIMS], const long ostr[DIMS], complex float* image, const long sens_dims[DIMS], const complex float* sens_maps, const long pat1_dims[DIMS], const complex float* weights, const long istr[DIMS], const complex float* kspace_data, bool usegpu)
{
	struct grecon_s param = { .algo = SENSE, .calib = calib, .conf = conf, .ksp = true, .rplksp = rplksp };
	parslices2(grecon, &param, dims, ostr, image, sens_dims, sens_maps, pat1_dims, weights, istr, kspace_data, true, usegpu);
}


void msense(struct sense_conf* conf, const long dims[DIMS], complex float* image, const complex float* sens_maps, const complex float* kspace_data, bool usegpu)
{
	struct grecon_s param = { .algo = SENSE, .calib = NULL, .conf = conf, .ksp = false };
	parslices(grecon, &param, dims, image, dims, sens_maps, NULL, NULL, kspace_data, false, usegpu);
}

void mnoir(struct sense_conf* conf, const long dims[DIMS], complex float* image, const complex float* kspace_data, bool usegpu)
{
	long sens_dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	complex float sens = 1.;
	struct grecon_s param = { .algo = NOIR, .calib = NULL, .conf = conf, .ksp = false };
	parslices(grecon, &param, dims, image, sens_dims, &sens, NULL, NULL, kspace_data, false, usegpu);
}

void msense2(struct sense_conf* conf, const struct ecalib_conf* calib, bool rplksp, const long dims[DIMS], const long ostr[DIMS], complex float* image, const long sens_dims[DIMS], const complex float* sens_maps, const long istr[DIMS], const complex float* kspace_data, bool usegpu)
{
	assert(dims[READ_DIM] == sens_dims[READ_DIM]);

	struct grecon_s param = { .algo = SENSE, .calib = calib, .conf = conf, .ksp = true, .rplksp = rplksp };
	parslices2(grecon, &param, dims, ostr, image, sens_dims, sens_maps, NULL, NULL, istr, kspace_data, true, usegpu);
}

void msense3(struct sense_conf* conf, const struct ecalib_conf* calib, const long dims[DIMS], complex float* image, const long sens_dims[DIMS], const complex float* sens_maps, const complex float* kspace_data, bool usegpu)
{
	assert(dims[READ_DIM] == sens_dims[READ_DIM]);

	struct grecon_s param = { .algo = SENSE, .calib = calib, .conf = conf, .ksp = false };
	parslices(grecon, &param, dims, image, sens_dims, sens_maps, NULL, NULL, kspace_data, false, usegpu);
}

void msense2pocs(struct sense_conf* conf, const struct ecalib_conf* calib, const long dims[DIMS], complex float* image, const long sens_dims[DIMS], const complex float* sens_maps, const complex float* kspace_data, bool usegpu)
{
	assert(dims[READ_DIM] == sens_dims[READ_DIM]);
	struct grecon_s param = { .algo = POCS, .calib = calib, .conf = conf, .ksp = true };
	parslices(grecon, &param, dims, image, sens_dims, sens_maps, NULL, NULL, kspace_data, true, usegpu);
}


