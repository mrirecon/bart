/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#ifndef __LSQR_H
#define __LSQR_H 1

#include "iter/iter.h"
#include "iter/iter2.h"

#include "misc/cppwrap.h"

struct operator_s;
struct operator_p_s;


struct lsqr_conf {

	float lambda;
	_Bool it_gpu;
};


extern const struct lsqr_conf lsqr_defaults;

extern const struct operator_s* lsqr2_create(const struct lsqr_conf* conf,
				      italgo_fun2_t italgo, iter_conf* iconf,
				      const float* init,
				      const struct linop_s* model_op,
				      const struct operator_s* precond_op,
			              unsigned int num_funs,
				      const struct operator_p_s* prox_funs[__VLA2(num_funs)],
				      const struct linop_s* prox_linops[__VLA2(num_funs)]);

extern const struct operator_s* wlsqr2_create(	const struct lsqr_conf* conf,
					italgo_fun2_t italgo, iter_conf* iconf,
				        const float* init,
					const struct linop_s* model_op,
					const struct linop_s* weights,
					const struct operator_s* precond_op,
					unsigned int num_funs,
					const struct operator_p_s* prox_funs[__VLA2(num_funs)],
					const struct linop_s* prox_linops[__VLA2(num_funs)]);




extern void lsqr(	unsigned int N, const struct lsqr_conf* conf,
			italgo_fun_t italgo, iter_conf* iconf,
			const struct linop_s* model_op,
			const struct operator_p_s* thresh_op,
			const long x_dims[__VLA(N)], _Complex float* x,
			const long y_dims[__VLA(N)], const _Complex float* y,
			const struct operator_s* precond_op);

extern void wlsqr(	unsigned int N, const struct lsqr_conf* conf,
			italgo_fun_t italgo, iter_conf* iconf,
			const struct linop_s* model_op,
			const struct operator_p_s* thresh_op,
			const long x_dims[__VLA(N)], _Complex float* x,
			const long y_dims[__VLA(N)], const _Complex float* y,
			const long w_dims[__VLA(N)], const _Complex float* w,
			const struct operator_s* precond_op);

extern void lsqr2(	unsigned int N, const struct lsqr_conf* conf,
			italgo_fun2_t italgo, iter_conf* iconf,
			const struct linop_s* model_op,
			unsigned int num_funs,
			const struct operator_p_s* prox_funs[__VLA2(num_funs)],
			const struct linop_s* prox_linops[__VLA2(num_funs)],
			const long x_dims[__VLA(N)], _Complex float* x,
			const long y_dims[__VLA(N)], const _Complex float* y,
			const struct operator_s* precond_op,
			struct iter_monitor_s* monitor);

extern void wlsqr2(	unsigned int N, const struct lsqr_conf* conf,
			italgo_fun2_t italgo, iter_conf* iconf,
			const struct linop_s* model_op,
			unsigned int num_funs,
			const struct operator_p_s* prox_funs[__VLA2(num_funs)],
			const struct linop_s* prox_linops[__VLA2(num_funs)],
			const long x_dims[__VLA(N)], complex float* x,
			const long y_dims[__VLA(N)], const complex float* y,
			const long w_dims[__VLA(N)], const complex float* w,
			const struct operator_s* precond_op);


#include "misc/cppwrap.h"

#endif


