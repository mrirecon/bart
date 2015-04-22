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
#include "iter/iter2.h"


//typedef void (*thresh_fun_t)(void* data, float lambda, _Complex float* dst, const _Complex float* src);

/**
 * configuration parameters for sense reconstruction
 *
 * @param rvc TRUE for real-values constraints
 */
struct sense_conf {

	_Bool rvc;
	int rwiter;	// should be moved into a recon_lad
	float gamma;	// ..
	float cclambda;
};


extern const struct sense_conf sense_defaults;

struct operator_p_s;

#ifdef USE_CUDA
extern void sense_recon2_gpu(const struct sense_conf* conf, const long dims[DIMS], _Complex float* image,
			const struct linop_s* sense_op,
			const long pat_dims[DIMS], const _Complex float* pattern,
			italgo_fun2_t italgo, void* italgo_conf,
			unsigned int num_funs,
			const struct operator_p_s* thresh_op[num_funs],
			const struct linop_s* thresh_funs[num_funs],
			const long ksp_dims[DIMS], const _Complex float* kspace, const _Complex float* image_truth);

extern void sense_recon_gpu(const struct sense_conf* conf, const long dims[DIMS], _Complex float* image,
			const struct linop_s* sense_op,
			const long pat_dims[DIMS], const _Complex float* pattern,
			italgo_fun_t italgo, void* italgo_conf,
			const struct operator_p_s* thresh_op,
			const long ksp_dims[DIMS], const _Complex float* kspace, const _Complex float* image_truth);
#endif
extern void sense_recon(const struct sense_conf* conf, const long dims[DIMS], _Complex float* image,
			const struct linop_s* sense_op,
			const long pat_dims[DIMS], const _Complex float* pattern,
			italgo_fun_t italgo, void* italgo_conf,
			const struct operator_p_s* thresh_op,
			const long ksp_dims[DIMS], const _Complex float* kspace, const _Complex float* image_truth);

extern void sense_recon2(const struct sense_conf* conf, const long dims[DIMS], _Complex float* image,
			const struct linop_s* sense_op,
			const long pat_dims[DIMS], const _Complex float* pattern,
			italgo_fun2_t italgo, void* italgo_conf,
			unsigned int num_funs,
			const struct operator_p_s* thresh_op[num_funs],
			const struct linop_s* thresh_funs[num_funs],
			const long ksp_dims[DIMS], const _Complex float* kspace, const _Complex float* image_truth);

extern void debug_print_sense_conf(int debug_level, const struct sense_conf* conf);



#ifdef __cplusplus
}
#endif

#endif


