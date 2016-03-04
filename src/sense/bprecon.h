/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#ifndef __SENSE_H
#define __SENSE_H


#ifdef __cplusplus
extern "C" {
#endif


#include "misc/mri.h"

struct iter_conf_s;
struct operator_p_s;
struct linop_s;

/**
 * configuration parameters for basis pursuit sense reconstruction
 *
 * @param iter_admm_conf configuration struct for admm algorithm
 * @param rvc TRUE for real-valued constraint
 * @param lambda l2 regularization penalty
 * @param eps data consistency error
 * @param linop_obj linear operator for computing the objective value
 */
struct bpsense_conf {

	struct iter_conf_s* iconf;
	_Bool rvc;
	float lambda;
	float eps;

	const struct linop_s* l1op_obj;
};

extern const struct bpsense_conf bpsense_defaults;


#ifdef USE_CUDA
extern void bpsense_recon_gpu(struct bpsense_conf* conf, const long dims[DIMS], _Complex float* image, const _Complex float* maps,
			const long pat_dims[DIMS], const _Complex float* pattern,
			const struct linop_s* l1op,
			const struct operator_p_s* l1prox,
			const long ksp_dims[DIMS], const _Complex float* kspace, const _Complex float* image_truth);
#endif
extern void bpsense_recon(struct bpsense_conf* conf, const long dims[DIMS], _Complex float* image, const _Complex float* maps,
			const long pat_dims[DIMS], const _Complex float* pattern,
			const struct linop_s* l1op,
			const struct operator_p_s* l1prox,
			const long ksp_dims[DIMS], const _Complex float* kspace, const _Complex float* image_truth);

#ifdef __cplusplus
}
#endif

#endif


