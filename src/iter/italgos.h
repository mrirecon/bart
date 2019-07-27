/* Copyright 2013-2017. The Regents of the University of California.
 * Copyright 2016-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */ 

#ifndef __ITALGOS_H
#define __ITALGOS_H

#include "misc/cppwrap.h"

#ifndef NUM_INTERNAL
// #warning "Use of private interfaces"
#endif

#include "misc/types.h"
#include "misc/nested.h"

struct vec_iter_s;


#ifndef ITER_OP_DATA_S
#define ITER_OP_DATA_S
typedef struct iter_op_data_s { TYPEID* TYPEID; } iter_op_data;
#endif
typedef void (*iter_op_fun_t)(iter_op_data* data, float* dst, const float* src);
typedef void (*iter_nlop_fun_t)(iter_op_data* data, int N, float* args[N]);
typedef void (*iter_op_p_fun_t)(iter_op_data* data, float rho, float* dst, const float* src);

struct iter_op_s {

	iter_op_fun_t fun;
	iter_op_data* data;
};

struct iter_nlop_s {

	iter_nlop_fun_t fun;
	iter_op_data* data;
};

struct iter_op_p_s {

	iter_op_p_fun_t fun;
	iter_op_data* data;
};

inline void iter_op_call(struct iter_op_s op, float* dst, const float* src)
{
	op.fun(op.data, dst, src);
}

inline void iter_nlop_call(struct iter_nlop_s op, int N, float* args[N])
{
	op.fun(op.data, N, args);
}

inline void iter_op_p_call(struct iter_op_p_s op, float rho, float* dst, const float* src)
{
	op.fun(op.data, rho, dst, src);
}



struct iter_monitor_s;

float conjgrad(unsigned int maxiter, float l2lambda, float epsilon, 
	long N,
	const struct vec_iter_s* vops,
	struct iter_op_s linop,
	float* x, const float* b,
	struct iter_monitor_s* monitor);


void landweber(unsigned int maxiter, float epsilon, float alpha,
	long N, long M,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	struct iter_op_s adj,
	float* x, const float* b,
	struct iter_op_s callback,
	struct iter_monitor_s* monitor);

void landweber_sym(unsigned int maxiter, float epsilon, float alpha,	
	long N,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	float* x, const float* b,
	struct iter_monitor_s* monitor);


/**
 * Store information about iterative algorithm.
 * Used to flexibly modify behavior, e.g. continuation
 *
 * @param rsnew current residual
 * @param rsnot initial residual
 * @param iter current iteration
 * @param maxiter maximum iteration
 * @param tau tau
 * @param scale scaling of regularization
 */
struct ist_data {

	double rsnew;
	double rsnot;
	unsigned int iter;
	const unsigned int maxiter;
	float tau;
	float scale;
};

typedef void CLOSURE_TYPE(ist_continuation_t)(struct ist_data* itrdata);


void ist(unsigned int maxiter, float epsilon, float tau, 
	long N,
	const struct vec_iter_s* vops,
	ist_continuation_t ist_continuation,
	struct iter_op_s op,
	struct iter_op_p_s thresh,
	float* x, const float* b,
	struct iter_monitor_s* monitor);

void fista(unsigned int maxiter, float epsilon, float tau, 
	long N,
	const struct vec_iter_s* vops,
	ist_continuation_t ist_continuation,
	struct iter_op_s op,
	struct iter_op_p_s thresh,
	float* x, const float* b,
	struct iter_monitor_s* monitor);
	

void irgnm(unsigned int iter, float alpha, float alpha_min, float redu,
	long N, long M,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	struct iter_op_s adj,
	struct iter_op_p_s inv,
	float* x, const float* x0, const float* y,
	struct iter_op_s callback,
	struct iter_monitor_s* monitor);

void irgnm2(unsigned int iter, float alpha, float alpha_min, float alpha0, float redu,
	long N, long M,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	struct iter_op_s der,
	struct iter_op_p_s lsqr,
	float* x, const float* xref, const float* y,
	struct iter_op_s callback,
	struct iter_monitor_s* monitor);

void altmin(unsigned int iter, float alpha, float redu,
	long N,
	const struct vec_iter_s* vops,
	unsigned int NI,
	struct iter_nlop_s op,
	struct iter_op_p_s min_ops[__VLA(NI)],
	float* x[__VLA(NI)], const float* y,
	struct iter_nlop_s callback);

void pocs(unsigned int maxiter,
	unsigned int D, struct iter_op_p_s proj_ops[__VLA(D)],
	const struct vec_iter_s* vops,
	long N, float* x,
	struct iter_monitor_s* monitor);

double power(unsigned int maxiter,
	long N,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	float* u);
	   
void chambolle_pock(unsigned int maxiter, float epsilon, float tau, float sigma, float theta, float decay,
	long N, long M,
	const struct vec_iter_s* vops,
	struct iter_op_s op_forw,
	struct iter_op_s op_adj,
	struct iter_op_p_s thresh1,
	struct iter_op_p_s thresh2,
	float* x,
	struct iter_monitor_s* monitor);

#include "misc/cppwrap.h"

#endif // __ITALGOS_H


