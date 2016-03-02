/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */ 

#ifndef __ADMM_H
#define __ADMM_H

#include <stdbool.h>

#include "misc/cppwrap.h"

struct vec_iter_s;

#ifndef __PROX_FUN_T
#define __PROX_FUN_T
typedef void (*prox_fun_t)(void* prox_data, float rho, float* z, const float* x_plus_u);
#endif

typedef void (*ops_fun_t)(void* dta, float* dst, const float* src);

struct admm_op {

	ops_fun_t forward;
	ops_fun_t adjoint;
	ops_fun_t normal;
	void* data;
};

struct admm_prox_op {

	prox_fun_t prox_fun;
	void* data;
};


/**
 * Parameters for ADMM version 1
 *
 * @param maxiter maximum number of iterations (gradient evaluations) before terminating
 * @param maxitercg maximum number of conjugate gradient iterations for updating x
 * @param num_funs number of convex functions in objective, excluding data consistency
 *
 * @param do_warmstart do not zero out primal and dual variables before starting
 * @param dynamic_rho update rho according to mu/tau rule
 * @param hogwild_rho update rho according to Hogwild rule (increase exponentially)
 *
 * @param ABSTOL used for early stopping condition
 * @param RELTOL used for early stopping condition
 *
 * @param rho -- augmented lagrangian penalty parameter
 * @param alpha -- over-relaxation parameter between (0, 2)
 *
 * @param tau -- multiply/divide rho by tau if residuals are more than mu times apart
 * @param mu -- multiply/divide rho by tau if residuals are more than mu times apart
 *
 * @param funs array of prox functions (size is num_funs)
 * @param admm_data array of data for the prox functions (size is num_funs)
 *
 * @param ops array of operators, G_i (size is num_funs)
 * @param prox_ops array of proximal functions (size is num_funs)
 * @param biases array of biases/offsets (size is num_funs)
 *
 * @param image_truth truth image for computing relMSE
 */
struct admm_plan_s {

	unsigned int maxiter;
	unsigned int maxitercg;

	bool do_warmstart;
	bool dynamic_rho;
	bool hogwild;
	_Bool fast;

	double ABSTOL;
	double RELTOL;

	float rho;
	float alpha;

	float tau;
	float mu;


	unsigned int num_funs;

	struct admm_prox_op* prox_ops;
	struct admm_op* ops;
	float** biases;

	void (*xupdate_fun)(void* _data, float rho, float* _dst, const float* _src);
	void* xupdate_data;

	const float* image_truth;
};


/**
 * Store ADMM history (also used for early stopping criterion)
 *
 * @param numiter actual number of iterations run
 * @param r_norm (array) primal residual norm at each iteration
 * @param s_norm (array) dual residual norm at each iteration
 * @param eps_pri (array) primal epsilon at each iteration
 * @parram eps_dual (array) dual epsilon at each iteration
 */
struct admm_history_s {

	unsigned int numiter;
	double* r_norm;
	double* s_norm;
	double* eps_pri;
	double* eps_dual;
	double* objective;
	float* rho;
	double* relMSE;
};



void admm(struct admm_history_s* history, const struct admm_plan_s* plan, 
	  unsigned int D, const long z_dims[__VLA(D)],
	  long N, float* x, const float* x_adj,
	  const struct vec_iter_s* vops,
	  void (*Aop)(void* _data, float* _dst, const float* _src),
	  void* Aop_data,
	  void* obj_eval_data, 
	  float (*obj_eval)(const void*, const float*));

#if 0
/**
 * Store data for conjugate gradient x update step
 *
 *
 */
struct admm_cgxupdate_data {
	long size;
	void* cgconf;
	unsigned int num_funs;
	struct admm_op* ops;
	void (*Aop)(void* _data, float* _dst, const float* _src);
	void* Aop_data;
	float rho;
};

void admm_cgxupdate( void* data, float rho, float* _dst, const float* _src );
#endif

#include "misc/cppwrap.h"
#endif // __ADMM_H
