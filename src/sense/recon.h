/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#ifndef __SENSE_H
#define __SENSE_H 1


#ifdef __cplusplus
extern "C" {
#endif


#include "misc/mri.h"
#include "iter/iter.h"


//typedef void (*thresh_fun_t)(void* data, float lambda, _Complex float* dst, const _Complex float* src);

/**
 * configuration parameters for sense reconstruction
 *
 * @param l1wav TRUE for l1-wavelet regularization
 * @param randshift TRUE for wavelet cycle-spinning
 * @param rvc TRUE for real-values constraints
 * @param bsplit TRUE for split bregman iterations
 * @param mu split bregman penalty
 * @param lambda initial regularlization parameter
 * @param lambda2 final regularlization parameter (for continuation)
 * @param maxiter number of iterations
 * @param step for iterative algorithm step size
 */
struct sense_conf {

	_Bool l1wav;
	_Bool randshift;
	_Bool rvc;
	int rwiter;	// should be moved into a recon_lad
	float gamma;	// ..
	int maxiter;
	float step;
	_Bool ccrobust;
	float cclambda;
#if 1
	float lambda;	// ...
#endif
};

// FIXME: Can we make conf const ?

extern const struct sense_conf sense_defaults;

struct operator_p_s;

#ifdef USE_CUDA
extern void sense_recon_gpu(struct sense_conf* conf, const long dims[DIMS], _Complex float* image, const _Complex float* maps,
			const long pat_dims[DIMS], const _Complex float* pattern,
			italgo_fun_t italgo, void* italgo_conf,
//			thresh_fun_t thresh, void* thresh_data,
			const struct operator_p_s* thresh_op,
			const long ksp_dims[DIMS], const _Complex float* kspace, const _Complex float* image_truth);
#endif
extern void sense_recon(struct sense_conf* conf, const long dims[DIMS], _Complex float* image, const _Complex float* maps,
			const long pat_dims[DIMS], const _Complex float* pattern,
			italgo_fun_t italgo, void* italgo_conf,
//			thresh_fun_t thresh, void* thresh_data,
			const struct operator_p_s* thresh_op,
			const long ksp_dims[DIMS], const _Complex float* kspace, const _Complex float* image_truth);

extern void debug_print_sense_conf(int debug_level, const struct sense_conf* conf);



#ifdef __cplusplus
}
#endif

#endif


