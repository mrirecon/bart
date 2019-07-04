/* Copyright 2013-2018. The Regents of the University of California.
 * Copyright 2016-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef __ITER_H
#define __ITER_H

struct operator_s;
struct operator_p_s;

#include "misc/types.h"

#ifndef ITER_CONF_S
#define ITER_CONF_S
typedef struct iter_conf_s { TYPEID* TYPEID; float alpha; } iter_conf;
#endif

struct iter_monitor_s;

typedef void italgo_fun_f(iter_conf* conf,
		const struct operator_s* normaleq_op,
		const struct operator_p_s* thresh_prox,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor);

typedef italgo_fun_f* italgo_fun_t;



struct iter_conjgrad_conf {

	INTERFACE(iter_conf);

	unsigned int maxiter;
	float l2lambda;
	float tol;
};



struct iter_landweber_conf {

	INTERFACE(iter_conf);

	unsigned int maxiter;
	float step;
	float tol;
};


struct iter_ist_conf {

	INTERFACE(iter_conf);

	unsigned int maxiter;
	float step;
	float continuation;
	_Bool hogwild;
	float tol;
};


struct iter_fista_conf {

	INTERFACE(iter_conf);

	unsigned int maxiter;
	float step;
	float continuation;
	_Bool hogwild;
	float tol;
};



struct iter_chambolle_pock_conf {

	INTERFACE(iter_conf);

	unsigned int maxiter;
	float tau;
	float sigma;
	float theta;
	float decay;
	float tol;
	_Bool fast;
};


struct iter_admm_conf {

	INTERFACE(iter_conf);

	unsigned int maxiter;
	unsigned int maxitercg;
	float rho;

	_Bool do_warmstart;
	_Bool dynamic_rho;
	_Bool dynamic_tau;
	_Bool relative_norm;
	_Bool hogwild;
	
	double ABSTOL;
	double RELTOL;

	float alpha;

	float tau;
	float tau_max;
	float mu;

	float cg_eps;

	_Bool fast;
};



struct iter_pocs_conf {

	INTERFACE(iter_conf);

	unsigned int maxiter;
};

struct iter_niht_conf {

	INTERFACE(iter_conf);

	unsigned int maxiter;
	float tol;
	_Bool do_warmstart;
};


extern const struct iter_conjgrad_conf iter_conjgrad_defaults;
extern const struct iter_landweber_conf iter_landweber_defaults;
extern const struct iter_ist_conf iter_ist_defaults;
extern const struct iter_fista_conf iter_fista_defaults;
extern const struct iter_admm_conf iter_admm_defaults;
extern const struct iter_pocs_conf iter_pocs_defaults;
extern const struct iter_niht_conf iter_niht_defaults;
extern const struct iter_chambolle_pock_conf iter_chambolle_pock_defaults;


italgo_fun_f iter_conjgrad;
italgo_fun_f iter_landweber;
italgo_fun_f iter_ist;
italgo_fun_f iter_fista;
italgo_fun_f iter_admm;

// use with iter2_call_s from iter2.h as _conf
italgo_fun_f iter_call_iter2;


struct iter_call_s {

	INTERFACE(iter_conf);

	italgo_fun_t fun;
	iter_conf* _conf;
};



#endif

