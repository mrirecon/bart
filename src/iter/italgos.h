/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */ 

#ifndef __ITALGOS_H
#define __ITALGOS_H

#ifndef NUM_INTERNAL
// #warning "Use of private interfaces"
#endif


struct vec_iter_s;

#ifndef __PROX_FUN_T
#define __PROX_FUN_T
typedef void (*prox_fun_t)(void* prox_data, float rho, float* z, const float* x_plus_u);
#endif


/**
 * Store italg history
 *
 * @param numiter actual number of iterations run
 */
struct iter_history_s {

	unsigned int numiter;
	double* objective;
	double* relMSE;
	double* resid;
};


struct pocs_proj_op {

	prox_fun_t proj_fun;
	void* data;
};


float conjgrad(unsigned int maxiter, float l2lambda, float epsilon, 
	long N, void* data,
	const struct vec_iter_s* vops,
	void (*linop)(void* data, float* dst, const float* src), 
	float* x, const float* b, const float* x_truth,
	void* obj_eval_data,
	float (*obj_eval)(const void*, const float*));

float conjgrad_hist(struct iter_history_s* iter_history, unsigned int maxiter, float l2lambda, float epsilon, 
	long N, void* data,
	const struct vec_iter_s* vops,
	void (*linop)(void* data, float* dst, const float* src), 
	float* x, const float* b, const float* x_truth,
	void* obj_eval_data,
	float (*obj_eval)(const void*, const float*));


extern const struct cg_data_s* cg_data_init(long N, const struct vec_iter_s* vops);
extern void cg_data_free(const struct cg_data_s* cgdata, const struct vec_iter_s* vops);
float conjgrad_hist_prealloc(struct iter_history_s* iter_history, 
			     unsigned int maxiter, float l2lambda, float epsilon, 
			     long N, void* data, const struct cg_data_s* cgdata,
			     const struct vec_iter_s* vops,
			     void (*linop)(void* data, float* dst, const float* src), 
			     float* x, const float* b, const float* x_truth,
			     void* obj_eval_data,
			     float (*obj_eval)(const void*, const float*));


void landweber(unsigned int maxiter, float epsilon, float alpha,
	long N, long M, void* data,
	const struct vec_iter_s* vops,
	void (*op)(void* data, float* dst, const float* src), 
	void (*adj)(void* data, float* dst, const float* src), 
	float* x, const float* b,
	float (*obj_eval)(const void*, const float*));

void landweber_sym(unsigned int maxiter, float epsilon, float alpha,	
	long N, void* data,
	const struct vec_iter_s* vops,
	void (*op)(void* data, float* dst, const float* src), 
	float* x, const float* b);

void ist(unsigned int maxiter, float epsilon, float tau, 
	 float continuation, _Bool hogwild, 
	 long N, void* data,
	 const struct vec_iter_s* vops,
	 void (*op)(void* data, float* dst, const float* src), 
	 void (*thresh)(void* data, float lambda, float* dst, const float* src),
	 void* tdata,
	 float* x, const float* b, const float* x_truth,
	 void* obj_eval_data,
	 float (*obj_eval)(const void*, const float*));

void fista(unsigned int maxiter, float epsilon, float tau, 
	   float continuation, _Bool hogwild, 
	   long N, void* data,
	   const struct vec_iter_s* vops,
	   void (*op)(void* data, float* dst, const float* src), 
	   void (*thresh)(void* data, float lambda, float* dst, const float* src),
	   void* tdata,
	   float* x, const float* b, const float* x_truth,
	   void* obj_eval_data,
	   float (*obj_eval)(const void*, const float*));
	

void irgnm(unsigned int iter, float alpha, float redu, void* data, 
	long N, long M,
	const struct vec_iter_s* vops,
	void (*op)(void* data, float* dst, const float* src), 
	void (*adj)(void* data, float* dst, const float* src), 
	void (*inv)(void* data, float alpha, float* dst, const float* src), 
	float* x, const float* x0, const float* y);

void pocs(unsigned int maxiter,
	unsigned int D, const struct pocs_proj_op* proj_ops, 
	const struct vec_iter_s* vops,
	long N, float* x, const float* x_truth,
	void* obj_eval_data,
	float (*obj_eval)(const void*, const float*));

double power(unsigned int maxiter,
	   long N, void* data,
	   const struct vec_iter_s* vops,
	   void (*op)(void* data, float* dst, const float* src), 
	   float* u);
	   

#endif // __ITALGOS_H


