/* Copyright 2013-2014. The Regents of the University of California.
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

struct vec_iter_s;


#ifndef ITER_OP_DATA_S
#define ITER_OP_DATA_S
typedef struct iter_op_data_s { TYPEID* TYPEID; } iter_op_data;
#endif
typedef void (*iter_op_fun_t)(iter_op_data* data, float* dst, const float* src);
typedef void (*iter_op_p_fun_t)(iter_op_data* data, float rho, float* dst, const float* src);

struct iter_op_s {

	iter_op_fun_t fun;
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

inline void iter_op_p_call(struct iter_op_p_s op, float rho, float* dst, const float* src)
{
	op.fun(op.data, rho, dst, src);
}



struct iter_monitor_s;

float conjgrad(unsigned int maxiter, float l2lambda, float epsilon, 
	long N,
	const struct vec_iter_s* vops,
	struct iter_op_s linop,
	float* x, const float* b, struct iter_monitor_s* monitor);


void landweber(unsigned int maxiter, float epsilon, float alpha,
	long N, long M,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	struct iter_op_s adj,
	float* x, const float* b,
	struct iter_monitor_s* monitor);

void landweber_sym(unsigned int maxiter, float epsilon, float alpha,	
	long N,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	float* x, const float* b,
	struct iter_monitor_s* monitor);

void ist(unsigned int maxiter, float epsilon, float tau, 
	float continuation, _Bool hogwild,
	long N,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	struct iter_op_p_s thresh,
	float* x, const float* b,
	struct iter_monitor_s* monitor);

void fista(unsigned int maxiter, float epsilon, float tau, 
	float continuation, _Bool hogwild,
	long N,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	struct iter_op_p_s thresh,
	float* x, const float* b,
	struct iter_monitor_s* monitor);
	

void irgnm(unsigned int iter, float alpha, float redu,
	long N, long M,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	struct iter_op_s adj,
	struct iter_op_p_s inv,
	float* x, const float* x0, const float* y);

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
	   

#include "misc/cppwrap.h"

#endif // __ITALGOS_H


