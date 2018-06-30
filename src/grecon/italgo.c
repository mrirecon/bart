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

#include "misc/debug.h"
#include "misc/types.h"

#include "iter/iter2.h"
#include "iter/iter.h"

#include "grecon/optreg.h" // enum algo_t

#include "italgo.h"



struct iter configure_italgo(enum algo_t algo, int nr_penalties, const struct reg_s* regs, unsigned int maxiter, float step, bool hogwild, bool fast, const struct admm_conf admm, float scaling, bool warm_start)
{
	italgo_fun2_t italgo = iter2_call_iter;
	static struct iter_call_s iter2_data;
	SET_TYPEID(iter_call_s, &iter2_data);

	iter_conf* iconf = CAST_UP(&iter2_data);

	static struct iter_conjgrad_conf cgconf;
	cgconf = iter_conjgrad_defaults;
	static struct iter_fista_conf fsconf;
	fsconf = iter_fista_defaults;
	static struct iter_ist_conf isconf;
	isconf = iter_ist_defaults;
	static struct iter_admm_conf mmconf;
	mmconf = iter_admm_defaults;
	static struct iter_niht_conf ihconf;
	ihconf = iter_niht_defaults;
	static struct iter_chambolle_pock_conf pdconf;
	pdconf = iter_chambolle_pock_defaults;

	switch (algo) {

		case ALGO_CG:

			debug_printf(DP_INFO, "conjugate gradients\n");

			assert((0 == nr_penalties) || ((1 == nr_penalties) && (L2IMG == regs[0].xform)));

			cgconf = iter_conjgrad_defaults;
			cgconf.maxiter = maxiter;
			cgconf.l2lambda = (0 == nr_penalties) ? 0. : regs[0].lambda;

			iter2_data.fun = iter_conjgrad;
			iter2_data._conf = CAST_UP(&cgconf);

			break;

		case ALGO_IST:

			debug_printf(DP_INFO, "IST\n");

			assert(1 == nr_penalties);

			isconf = iter_ist_defaults;
			isconf.maxiter = maxiter;
			isconf.step = step;
			isconf.hogwild = hogwild;

			iter2_data.fun = iter_ist;
			iter2_data._conf = CAST_UP(&isconf);

			break;

		case ALGO_ADMM:

			debug_printf(DP_INFO, "ADMM\n");

			mmconf = iter_admm_defaults;
			mmconf.maxiter = maxiter;
			mmconf.maxitercg = admm.maxitercg;
			mmconf.rho = admm.rho;
			mmconf.hogwild = hogwild;
			mmconf.fast = fast;
			mmconf.dynamic_rho = admm.dynamic_rho;
			mmconf.dynamic_tau = admm.dynamic_tau;
			mmconf.relative_norm = admm.relative_norm;
			mmconf.ABSTOL = 0.;
			mmconf.RELTOL = 0.;

			italgo = iter2_admm;
			iconf = CAST_UP(&mmconf);

			break;

		case ALGO_PRIDU:

			debug_printf(DP_INFO, "Primal Dual\n");

			assert(2 == nr_penalties);

			pdconf = iter_chambolle_pock_defaults;

			pdconf.maxiter = maxiter;
			pdconf.sigma = 1. * scaling;
			pdconf.tau = 1. / pdconf.sigma;
			pdconf.theta = 1;
			pdconf.decay = (hogwild ? .95 : 1);
			pdconf.tol = 1E-4;

			italgo = iter2_chambolle_pock;
			iconf = CAST_UP(&pdconf);

			break;

		case ALGO_FISTA:

			debug_printf(DP_INFO, "FISTA\n");

			assert(1 == nr_penalties);

			fsconf = iter_fista_defaults;
			fsconf.maxiter = maxiter;
			fsconf.step = step;
			fsconf.hogwild = hogwild;

			iter2_data.fun = iter_fista;
			iter2_data._conf = CAST_UP(&fsconf);

			break;

		case ALGO_NIHT:

			debug_printf(DP_INFO, "NIHT\n");

			ihconf = iter_niht_defaults;
			ihconf.maxiter = maxiter;
			ihconf.do_warmstart = warm_start;

			italgo = iter2_niht;
			iconf = CAST_UP(&ihconf);

			break;

		default:
			assert(0);
	}

	return (struct iter){ italgo, iconf };
}

