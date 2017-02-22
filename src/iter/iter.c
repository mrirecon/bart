/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012, 2014 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2014	Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2014 Frank Ong <frankong@berkeley.edu>
 */

#include <complex.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "iter/italgos.h"
#include "iter/prox.h"
#include "iter/admm.h"
#include "iter/iter2.h"
#include "iter/vec.h"

#include "misc/debug.h"
#include "misc/misc.h"

#include "iter.h"


DEF_TYPEID(iter_conjgrad_conf);
DEF_TYPEID(iter_landweber_conf);
DEF_TYPEID(iter_ist_conf);
DEF_TYPEID(iter_fista_conf);
DEF_TYPEID(iter_pocs_conf);
DEF_TYPEID(iter_admm_conf);
DEF_TYPEID(iter_call_s);

const struct iter_conjgrad_conf iter_conjgrad_defaults = {

	.INTERFACE.TYPEID = &TYPEID(iter_conjgrad_conf),

	.maxiter = 50,
	.l2lambda = 0.,
	.tol = 0.,
};

const struct iter_landweber_conf iter_landweber_defaults = {

	.INTERFACE.TYPEID = &TYPEID(iter_landweber_conf),

	.maxiter = 50,
	.step = 0.95,
	.tol = 0.,
};

const struct iter_ist_conf iter_ist_defaults = {

	.INTERFACE.TYPEID = &TYPEID(iter_ist_conf),

	.maxiter = 50,
	.step = 0.95,
	.continuation = 1.,
	.hogwild = false,
	.tol = 0.,
};

const struct iter_fista_conf iter_fista_defaults = {

	.INTERFACE.TYPEID = &TYPEID(iter_fista_conf),

	.maxiter = 50,
	.step = 0.95,
	.continuation = 1.,
	.hogwild = false,
	.tol = 0.,
};


const struct iter_admm_conf iter_admm_defaults = {

	.INTERFACE.TYPEID = &TYPEID(iter_admm_conf),

	.maxiter = 50,
	.maxitercg = 10,

	.cg_eps = 1.E-3,

	.do_warmstart = false,
	.dynamic_rho = false,
	.hogwild = false,
	.fast = false,

	.ABSTOL = 1.E-4,
	.RELTOL = 1.E-3,

	.rho = 0.5,
	.alpha = 1.6,

	.tau = 2.,
	.mu = 100,
};


const struct iter_pocs_conf iter_pocs_defaults = {

	.INTERFACE.TYPEID = &TYPEID(iter_pocs_conf),

	.maxiter = 50,
};

typedef void (*thresh_fun_t)(void* data, float lambda, float* dst, const float* src);



static bool checkeps(float eps)
{
	if (0. == eps) {

		debug_printf(DP_WARN, "Warning: data empty\n");
		return true;
	}

	if (!isnormal(eps)) {

		debug_printf(DP_WARN, "Warning: data corrupted\n");
		return true;
	}

	return false;
}



void iter_conjgrad(iter_conf* _conf,
		const struct operator_s* normaleq_op,
		const struct operator_p_s* thresh_prox,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor)
{
	assert(NULL == thresh_prox);
	iter2_conjgrad(_conf, normaleq_op, 0, NULL, NULL, NULL, NULL, size, image, image_adj, monitor);
}



void iter_landweber(iter_conf* _conf,
		const struct operator_s* normaleq_op,
		const struct operator_p_s* thresh_prox,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor)
{
	struct iter_landweber_conf* conf = CAST_DOWN(iter_landweber_conf, _conf);

	float eps = md_norm(1, MD_DIMS(size), image_adj);

	if (checkeps(eps))
		goto cleanup;

	assert(NULL == thresh_prox);

	landweber_sym(conf->maxiter, 1.E-3 * eps, conf->step, size, select_vecops(image_adj),
			OPERATOR2ITOP(normaleq_op), image, image_adj, monitor);

cleanup:
	;
}




void iter_ist(iter_conf* _conf,
		const struct operator_s* normaleq_op,
		const struct operator_p_s* thresh_prox,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor)
{
	iter2_ist(_conf, normaleq_op, 1, &thresh_prox, NULL, NULL, NULL, size, image, image_adj, monitor);
}

void iter_fista(iter_conf* _conf,
		const struct operator_s* normaleq_op,
		const struct operator_p_s* thresh_prox,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor)
{
	iter2_fista(_conf, normaleq_op, 1, &thresh_prox, NULL, NULL, NULL, size, image, image_adj, monitor);
}



void iter_admm(iter_conf* _conf,
		const struct operator_s* normaleq_op,
		const struct operator_p_s* thresh_prox,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor)
{
	const struct linop_s* eye[1] = { linop_identity_create(1, MD_DIMS(size / 2)) }; // using complex float identity operator... divide size by 2

	iter2_admm(_conf, normaleq_op, 1, &thresh_prox, eye, NULL, NULL, size, image, image_adj, monitor);

	linop_free(eye[0]);
}


void iter_call_iter2(iter_conf* _conf,
		const struct operator_s* normaleq_op,
		const struct operator_p_s* thresh_prox,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor)
{
	struct iter2_call_s* it = CAST_DOWN(iter2_call_s, _conf);

	it->fun(it->_conf, normaleq_op, (NULL == thresh_prox) ? 1 : 0, &thresh_prox, NULL, NULL, NULL,
		size, image, image_adj, monitor);
}


