
#ifndef _ITER_ITER_H
#define _ITER_ITER_H

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

	iter_conf super;

	int maxiter;
	float l2lambda;
	float tol;

	long Bo;
	long Bi;
};



struct iter_landweber_conf {

	iter_conf super;

	int maxiter;
	float step;
	float tol;
};


struct iter_ist_conf {

	iter_conf super;

	int maxiter;
	float step;
	float continuation;
	_Bool hogwild;
	float tol;
	_Bool last;
};


struct iter_eulermaruyama_conf {

	iter_conf super;

	int maxiter;
	float step;

	const struct linop_s* lop_prec;
	float diag_prec;
	float prec_tol;
	int max_prec_iter;
	long batchsize;
};




struct iter_fista_conf {

	iter_conf super;

	int maxiter;
	float step;
	float continuation;
	_Bool hogwild;
	float tol;
	int maxeigen_iter;
	float p;
	float q;
	float r;
	_Bool last;
};



struct iter_chambolle_pock_conf {

	iter_conf super;

	int maxiter;
	float tau;
	float sigma;
	float sigma_tau_ratio;
	float theta;
	float decay;
	float tol;
	_Bool fast;
	int maxeigen_iter;
	_Bool adapt_stepsize;
};


struct iter_admm_conf {

	iter_conf super;

	int maxiter;
	int maxitercg;
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

	iter_conf super;

	int maxiter;
};

struct iter_niht_conf {

	iter_conf super;

	int maxiter;
	float tol;
	_Bool do_warmstart;
};


extern const struct iter_conjgrad_conf iter_conjgrad_defaults;
extern const struct iter_landweber_conf iter_landweber_defaults;
extern const struct iter_ist_conf iter_ist_defaults;
extern const struct iter_eulermaruyama_conf iter_eulermaruyama_defaults;
extern const struct iter_fista_conf iter_fista_defaults;
extern const struct iter_admm_conf iter_admm_defaults;
extern const struct iter_pocs_conf iter_pocs_defaults;
extern const struct iter_niht_conf iter_niht_defaults;
extern const struct iter_chambolle_pock_conf iter_chambolle_pock_defaults;


italgo_fun_f iter_conjgrad;
italgo_fun_f iter_landweber;
italgo_fun_f iter_ist;
italgo_fun_f iter_eulermaruyama;
italgo_fun_f iter_fista;
italgo_fun_f iter_admm;

// use with iter2_call_s from iter2.h as _conf
italgo_fun_f iter_call_iter2;


struct iter_call_s {

	iter_conf super;

	italgo_fun_t fun;
	iter_conf* _conf;
};



#endif

