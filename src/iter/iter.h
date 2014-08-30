/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef __ITER_H
#define __ITER_H

struct operator_s;
struct operator_p_s;

typedef void (*italgo_fun_t)(void* conf,
		const struct operator_s* normaleq_op,
		const struct operator_p_s* thresh_prox,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* objval_data,
		float (*obj_eval)(const void*, const float*));



struct iter_conjgrad_conf {

	unsigned int maxiter;
	float l2lambda;
	float tol;
};


struct iter_landweber_conf {

	unsigned int maxiter;
	float step;
	float tol;
};


struct iter_ist_conf {

	unsigned int maxiter;
	float step;
	float continuation;
	_Bool hogwild;
	float tol;
};

struct iter_fista_conf {

	unsigned int maxiter;
	float step;
	float continuation;
	_Bool hogwild;
	float tol;
};


struct iter_admm_conf {

	unsigned int maxiter;
	unsigned int maxitercg;
	float rho;

	_Bool do_warmstart;
	_Bool dynamic_rho;
	_Bool hogwild;
	
	double ABSTOL;
	double RELTOL;

	float alpha;

	float tau;
	float mu;

	_Bool fast;
};


struct iter_pocs_conf {

	unsigned int maxiter;
};


extern const struct iter_conjgrad_conf iter_conjgrad_defaults;
extern const struct iter_landweber_conf iter_landweber_defaults;
extern const struct iter_ist_conf iter_ist_defaults;
extern const struct iter_fista_conf iter_fista_defaults;
extern const struct iter_admm_conf iter_admm_defaults;
extern const struct iter_pocs_conf iter_pocs_defaults;



extern void iter_conjgrad(void* conf,
		const struct operator_s* normaleq_op,
		const struct operator_p_s* thresh_prox,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* objval_data,
		float (*obj_eval)(const void*, const float*));

extern void iter_landweber(void* conf,
		const struct operator_s* normaleq_op,
		const struct operator_p_s* thresh_prox,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* objval_data,
		float (*obj_eval)(const void*, const float*));

extern void iter_ist(void* conf,
		const struct operator_s* normaleq_op,
		const struct operator_p_s* thresh_prox,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* objval_data,
		float (*obj_eval)(const void*, const float*));

extern void iter_fista(void* conf,
		const struct operator_s* normaleq_op,
		const struct operator_p_s* thresh_prox,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* objval_data,
		float (*obj_eval)(const void*, const float*));

extern void iter_admm(void* conf,
		const struct operator_s* normaleq_op,
		const struct operator_p_s* thresh_prox,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* objval_data,
		float (*obj_eval)(const void*, const float*));


#endif

