/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#ifndef __LSQR_H
#define __LSQR_H 1


#ifdef __cplusplus
extern "C" {
#endif


#include "iter/iter.h"
#include "iter/iter2.h"

struct operator_s;
struct operator_p_s;


/**
 * configuration parameters for sense reconstruction
 */
struct lsqr_conf {

	float lambda;
};


extern const struct lsqr_conf lsqr_defaults;

#ifdef USE_CUDA
extern void lsqr_gpu(	unsigned int N, const struct lsqr_conf* conf,
			italgo_fun_t italgo, void* iconf,
			const struct linop_s* model_op,
			const struct operator_p_s* thresh_op,
			const long x_dims[N], _Complex float* x, 
			const long y_dims[N], const _Complex float* y);

extern void wlsqr_gpu(	unsigned int N, const struct lsqr_conf* conf,
			italgo_fun_t italgo, void* iconf,
			const struct linop_s* model_op,
			const struct operator_p_s* thresh_op,
			const long x_dims[N], _Complex float* x,
			const long y_dims[N], const _Complex float* y,
			const long w_dims[N], const _Complex float* w);

extern void lsqr2_gpu(	unsigned int N, const struct lsqr_conf* conf,
			italgo_fun2_t italgo, void* iconf,
			const struct linop_s* model_op,
			unsigned int num_funs,
			const struct operator_p_s** prox_funs,
			const struct linop_s** prox_linops,
			const long x_dims[N], _Complex float* x,
			const long y_dims[N], const _Complex float* y,
			const _Complex float* x_truth,
			void* obj_eval_data,
			float (*obj_eval)(const void*, const float*));



#endif

extern void lsqr(	unsigned int N, const struct lsqr_conf* conf,
			italgo_fun_t italgo, void* iconf,
			const struct linop_s* model_op,
			const struct operator_p_s* thresh_op,
			const long x_dims[N], _Complex float* x, 
			const long y_dims[N], const _Complex float* y);

extern void wlsqr(	unsigned int N, const struct lsqr_conf* conf,
			italgo_fun_t italgo, void* iconf,
			const struct linop_s* model_op,
			const struct operator_p_s* thresh_op,
			const long x_dims[N], _Complex float* x,
			const long y_dims[N], const _Complex float* y,
			const long w_dims[N], const _Complex float* w);

extern void lsqr2(	unsigned int N, const struct lsqr_conf* conf,
			italgo_fun2_t italgo, void* iconf,
			const struct linop_s* model_op,
			unsigned int num_funs,
			const struct operator_p_s** prox_funs,
			const struct linop_s** prox_linops,
			const long x_dims[N], _Complex float* x,
			const long y_dims[N], const _Complex float* y,
			const _Complex float* x_truth,
			void* obj_eval_data,
			float (*obj_eval)(const void*, const float*));

extern void wlsqr2(	unsigned int N, const struct lsqr_conf* conf,
			italgo_fun2_t italgo, void* iconf,
			const struct linop_s* model_op,
			unsigned int num_funs,
			const struct operator_p_s** prox_funs,
			const struct linop_s** prox_linops,
			const long x_dims[N], complex float* x,
			const long y_dims[N], const complex float* y,
			const long w_dims[N], const complex float* w);


#ifdef __cplusplus
}
#endif

#endif


