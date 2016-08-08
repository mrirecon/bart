/* Copyright 2014. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */ 

#ifndef __ADMM_H
#define __ADMM_H

#include "misc/cppwrap.h"
#include "misc/types.h"
#include "iter/monitor.h"

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

	float cg_eps;

	_Bool do_warmstart;
	_Bool dynamic_rho;
	_Bool hogwild;
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
	const float* const* biases;

	void (*xupdate_fun)(void* _data, float rho, float* _dst, const float* _src);
	void* xupdate_data;
};



/**
 * Store ADMM history
 *
 * @param numiter actual number of iterations run
 * @param r_norm (array) primal residual norm at each iteration
 * @param s_norm (array) dual residual norm at each iteration
 * @param eps_pri (array) primal epsilon at each iteration
 * @parram eps_dual (array) dual epsilon at each iteration
 */
struct admm_history_s {

	INTERFACE(iter_history_t);

	unsigned int numiter;
	unsigned int nr_invokes;
	double r_norm;
	double s_norm;
	double eps_pri;
	double eps_dual;
	float rho;
};

extern DEF_TYPEID(admm_history_s);



extern void admm(const struct admm_plan_s* plan,
	  unsigned int D, const long z_dims[__VLA(D)],
	  long N, float* x, const float* x_adj,
	  const struct vec_iter_s* vops,
	  void (*Aop)(void* _data, float* _dst, const float* _src),
	  void* Aop_data, struct iter_monitor_s* monitor);



#include "misc/cppwrap.h"

#endif // __ADMM_H

