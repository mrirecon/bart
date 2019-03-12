/* Copyright 2016-2018. The Regents of the University of California.
 * Copyright 2016-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2016-2018 Jon Tamir <jtamir@eecs.berkeley.edu>
 *
 */


#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include "misc/debug.h"
#include "misc/types.h"
#include "misc/misc.h"

#include "iter/iter2.h"
#include "iter/iter.h"

#include "grecon/optreg.h" // enum algo_t

#include "italgo.h"



enum algo_t italgo_choose(int nr_penalties, const struct reg_s regs[nr_penalties])
{
	enum algo_t algo = ALGO_CG;

	for (int i = 0; i < nr_penalties; i++) {

		switch (regs[i].xform) {

		case L2IMG:
			break;

		case NIHTWAV:
		case NIHTIM:

			algo = ALGO_NIHT;
			break;

		case TV:
		case IMAGL1:
		case IMAGL2:

			algo = ALGO_ADMM;
			break;

		default:
			if (0 == i)
				algo = ALGO_FISTA;
			else
				algo = ALGO_ADMM;

			break;
		}
	}

	return algo;
}


struct iter italgo_config(enum algo_t algo, int nr_penalties, const struct reg_s* regs, unsigned int maxiter, float step, bool hogwild, bool fast, const struct admm_conf admm, float scaling, bool warm_start)
{
	italgo_fun2_t italgo = NULL;
	iter_conf* iconf = NULL;

	switch (algo) {

		case ALGO_DEFAULT:

			assert(0);

		case ALGO_CG: {

			debug_printf(DP_INFO, "conjugate gradients\n");

			assert((0 == nr_penalties) || ((1 == nr_penalties) && (L2IMG == regs[0].xform)));

			PTR_ALLOC(struct iter_conjgrad_conf, cgconf);
			*cgconf = iter_conjgrad_defaults;
			cgconf->maxiter = maxiter;
			cgconf->l2lambda = (0 == nr_penalties) ? 0. : regs[0].lambda;

			PTR_ALLOC(struct iter_call_s, iter2_data);
			SET_TYPEID(iter_call_s, iter2_data);

			iter2_data->fun = iter_conjgrad;
			iter2_data->_conf = CAST_UP(PTR_PASS(cgconf));

			italgo = iter2_call_iter;
			iconf = CAST_UP(PTR_PASS(iter2_data));

			break;
		}

		case ALGO_IST: {

			debug_printf(DP_INFO, "IST\n");

			assert(1 == nr_penalties);

			PTR_ALLOC(struct iter_ist_conf, isconf);
			*isconf = iter_ist_defaults;
			isconf->maxiter = maxiter;
			isconf->step = step;
			isconf->hogwild = hogwild;

			PTR_ALLOC(struct iter_call_s, iter2_ist_data);
			SET_TYPEID(iter_call_s, iter2_ist_data);

			iter2_ist_data->fun = iter_ist;
			iter2_ist_data->_conf = CAST_UP(PTR_PASS(isconf));

			italgo = iter2_call_iter;
			iconf = CAST_UP(PTR_PASS(iter2_ist_data));

			break;
		}

		case ALGO_ADMM: {

			debug_printf(DP_INFO, "ADMM\n");

			PTR_ALLOC(struct iter_admm_conf, mmconf);
			*mmconf = iter_admm_defaults;
			mmconf->maxiter = maxiter;
			mmconf->maxitercg = admm.maxitercg;
			mmconf->rho = admm.rho;
			mmconf->hogwild = hogwild;
			mmconf->fast = fast;
			mmconf->dynamic_rho = admm.dynamic_rho;
			mmconf->dynamic_tau = admm.dynamic_tau;
			mmconf->relative_norm = admm.relative_norm;
			mmconf->ABSTOL = 0.;
			mmconf->RELTOL = 0.;
			mmconf->do_warmstart = warm_start;
			italgo = iter2_admm;
			iconf = CAST_UP(PTR_PASS(mmconf));

			break;
		}

		case ALGO_PRIDU: {

			debug_printf(DP_INFO, "Primal Dual\n");

			assert(2 == nr_penalties);

			PTR_ALLOC(struct iter_chambolle_pock_conf, pdconf);
			*pdconf = iter_chambolle_pock_defaults;

			pdconf->maxiter = maxiter;
			pdconf->sigma = sqrtf(step);
			pdconf->tau = sqrtf(step);

			pdconf->sigma *= scaling;
			pdconf->tau /= scaling;

			pdconf->theta = 1.;
			pdconf->decay = (hogwild ? .95 : 1.);
			pdconf->tol = 1.E-4;

			italgo = iter2_chambolle_pock;
			iconf = CAST_UP(PTR_PASS(pdconf));

			break;
		}

		case ALGO_FISTA: {

			debug_printf(DP_INFO, "FISTA\n");

			assert(1 == nr_penalties);

			PTR_ALLOC(struct iter_fista_conf, fsconf);
			*fsconf = iter_fista_defaults;
			fsconf->maxiter = maxiter;
			fsconf->step = step;
			fsconf->hogwild = hogwild;

			PTR_ALLOC(struct iter_call_s, iter2_fista_data);
			SET_TYPEID(iter_call_s, iter2_fista_data);

			iter2_fista_data->fun = iter_fista;
			iter2_fista_data->_conf = CAST_UP(PTR_PASS(fsconf));

			italgo = iter2_call_iter;
			iconf = CAST_UP(PTR_PASS(iter2_fista_data));

			break;
		}

		case ALGO_NIHT: {

			debug_printf(DP_INFO, "NIHT\n");

			PTR_ALLOC(struct iter_niht_conf, ihconf);

			*ihconf = iter_niht_defaults;
			ihconf->maxiter = maxiter;
			ihconf->do_warmstart = warm_start;

			italgo = iter2_niht;
			iconf = CAST_UP(PTR_PASS(ihconf));

			break;
		}

		default:
			assert(0);
	}

	return (struct iter){ italgo, iconf };
}



void italgo_config_free(struct iter it)
{
	if (iter2_call_iter == it.italgo) {

		auto id = CAST_DOWN(iter_call_s, it.iconf);
		xfree(id->_conf);
	}

	xfree(it.iconf);
}

