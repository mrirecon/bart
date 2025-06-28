
#ifndef _ITALGO_H
#define _ITALGO_H

// italgo_fun2_t
#include "iter/iter2.h"

// struct reg_s
#include "grecon/optreg.h"

enum algo_t { ALGO_DEFAULT, ALGO_CG, ALGO_IST, ALGO_EULERMARUYAMA, ALGO_FISTA, ALGO_ADMM, ALGO_NIHT, ALGO_PRIDU };

struct admm_conf {

	bool dynamic_rho;
	bool dynamic_tau;
	bool relative_norm;
	float rho;
	int maxitercg;
	bool fast;
};

struct fista_conf {

	float params[3];
	bool last;
};

struct pridu_conf {

	float sigma_tau_ratio;
	bool adaptive_stepsize;
	int maxeigen_iter;
};

struct iter {

	italgo_fun2_t italgo;
	iter_conf* iconf;
};

struct reg_s;
enum algo_t;

extern enum algo_t italgo_choose(int nr_penalties, const struct reg_s regs [nr_penalties]);

extern struct iter italgo_config(enum algo_t algo, int nr_penalties, const struct reg_s* regs,
		int maxiter, float step, bool hogwild, const struct admm_conf admm,
		const struct fista_conf fista, const struct pridu_conf pridu, bool warm_start);

extern void italgo_config_free(struct iter it);

#endif	// __ITALGO_H

