/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef __ITER_H
#define __ITER_H

struct operator_s;
struct operator_p_s;

typedef void italgo_fun_f(void* conf,
		const struct operator_s* normaleq_op,
		const struct operator_p_s* thresh_prox,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* objval_data,
		float (*obj_eval)(const void*, const float*));

typedef italgo_fun_f* italgo_fun_t;



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


italgo_fun_f iter_conjgrad;
italgo_fun_f iter_landweber;
italgo_fun_f iter_ist;
italgo_fun_f iter_fista;
italgo_fun_f iter_admm;

// use with iter2_call_s from iter2.h as _conf
italgo_fun_f iter_call_iter2;


struct iter_call_s {

	italgo_fun_t fun;
	void* _conf;
};


#endif

