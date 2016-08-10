/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */ 

#ifndef __ITALGOS_H
#define __ITALGOS_H

#include "misc/cppwrap.h"

#ifndef NUM_INTERNAL
// #warning "Use of private interfaces"
#endif


struct vec_iter_s;

#ifndef __PROX_FUN_T
#define __PROX_FUN_T
typedef void (*prox_fun_t)(void* prox_data, float rho, float* z, const float* x_plus_u);
#endif

struct iter_monitor_s;

float conjgrad(unsigned int maxiter, float l2lambda, float epsilon, 
	long N, void* data,
	const struct vec_iter_s* vops,
	void (*linop)(void* data, float* dst, const float* src), 
	float* x, const float* b, struct iter_monitor_s* monitor);


void landweber(unsigned int maxiter, float epsilon, float alpha,
	long N, long M, void* data,
	const struct vec_iter_s* vops,
	void (*op)(void* data, float* dst, const float* src), 
	void (*adj)(void* data, float* dst, const float* src), 
	float* x, const float* b,
	struct iter_monitor_s* monitor);

void landweber_sym(unsigned int maxiter, float epsilon, float alpha,	
	long N, void* data,
	const struct vec_iter_s* vops,
	void (*op)(void* data, float* dst, const float* src), 
	float* x, const float* b,
	struct iter_monitor_s* monitor);

void ist(unsigned int maxiter, float epsilon, float tau, 
	 float continuation, _Bool hogwild, 
	 long N, void* data,
	 const struct vec_iter_s* vops,
	 void (*op)(void* data, float* dst, const float* src), 
	 void (*thresh)(void* data, float lambda, float* dst, const float* src),
	 void* tdata,
	 float* x, const float* b,
	 struct iter_monitor_s* monitor);

void fista(unsigned int maxiter, float epsilon, float tau, 
	   float continuation, _Bool hogwild, 
	   long N, void* data,
	   const struct vec_iter_s* vops,
	   void (*op)(void* data, float* dst, const float* src), 
	   void (*thresh)(void* data, float lambda, float* dst, const float* src),
	   void* tdata,
	   float* x, const float* b,
	   struct iter_monitor_s* monitor);
	

void irgnm(unsigned int iter, float alpha, float redu, void* data, 
	long N, long M,
	const struct vec_iter_s* vops,
	void (*op)(void* data, float* dst, const float* src), 
	void (*adj)(void* data, float* dst, const float* src), 
	void (*inv)(void* data, float alpha, float* dst, const float* src), 
	float* x, const float* x0, const float* y);

void pocs(unsigned int maxiter,
	unsigned int D, const prox_fun_t proj_ops[__VLA(D)], void* data[__VLA(D)],
	const struct vec_iter_s* vops,
	long N, float* x,
	struct iter_monitor_s* monitor);

double power(unsigned int maxiter,
	   long N, void* data,
	   const struct vec_iter_s* vops,
	   void (*op)(void* data, float* dst, const float* src), 
	   float* u);
	   

#include "misc/cppwrap.h"

#endif // __ITALGOS_H


