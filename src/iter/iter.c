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

#define  NUM_INTERNAL

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

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


const struct iter_conjgrad_conf iter_conjgrad_defaults = {

	.maxiter = 50,
	.l2lambda = 0.,
	.tol = 0.,
};


const struct iter_landweber_conf iter_landweber_defaults = {

	.maxiter = 50,
	.step = 0.95,
	.tol = 0.,
};

const struct iter_ist_conf iter_ist_defaults = {

	.maxiter = 50,
	.step = 0.95,
	.continuation = 1.,
	.hogwild = false,
	.tol = 0.,
};


const struct iter_istc_conf iter_istc_defaults = {

	.maxiter = 50,
	.step = 0.95,
	.hogwild = false,
	.tol = 0.,
};


const struct iter_fista_conf iter_fista_defaults = {

	.maxiter = 50,
	.step = 0.95,
	.continuation = 1.,
	.hogwild = false,
	.tol = 0.,
};


const struct iter_admm_conf iter_admm_defaults = {

	.maxiter = 50,
	.maxitercg = 10,

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



void iter_conjgrad(void* _conf,
		const struct operator_s* normaleq_op,
		const struct operator_p_s* thresh_prox,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* objval_data,
		float (*obj_eval)(const void*, const float*))
{
	assert(NULL == thresh_prox);
	iter2_conjgrad(_conf, normaleq_op, 0, NULL, NULL, NULL, size, image, image_adj, image_truth, objval_data, obj_eval);
}



void iter_landweber(void* _conf,
		const struct operator_s* normaleq_op,
		const struct operator_p_s* thresh_prox,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* objval_data,
		float (*obj_eval)(const void*, const float*))
{
	struct iter_landweber_conf* conf = _conf;

	float eps = md_norm(1, MD_DIMS(size), image_adj);

	if (checkeps(eps))
		goto cleanup;

	assert(NULL == thresh_prox);

	UNUSED(obj_eval);
	UNUSED(objval_data);
	UNUSED(image_truth);

	landweber_sym(conf->maxiter, 1.E-3 * eps, conf->step, size, (void*)normaleq_op, select_vecops(image_adj), operator_iter, image, image_adj);

cleanup:
	;
}




void iter_ist(void* _conf,
		const struct operator_s* normaleq_op,
		const struct operator_p_s* thresh_prox,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* objval_data,
		float (*obj_eval)(const void*, const float*))
{
	iter2_ist(_conf, normaleq_op, 1, &thresh_prox, NULL, NULL, size, image, image_adj, image_truth, objval_data, obj_eval);
}


void iter_istc(void* _conf,
	      const struct operator_s* normaleq_op,
	      const struct operator_p_s* thresh_prox,
	      long size, float* image, const float* image_adj,
	      const float* image_truth,
	      void* objval_data,
	      float (*obj_eval)(const void*, const float*))
{
	iter2_istc(_conf, normaleq_op, 1, &thresh_prox, NULL, NULL, size, image, image_adj, image_truth, objval_data, obj_eval);
}

void iter_fista(void* _conf,
		const struct operator_s* normaleq_op,
		const struct operator_p_s* thresh_prox,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* objval_data,
		float (*obj_eval)(const void*, const float*))
{
	iter2_fista(_conf, normaleq_op, 1, &thresh_prox, NULL, NULL, size, image, image_adj, image_truth, objval_data, obj_eval);
}



void iter_admm(void* _conf,
		const struct operator_s* normaleq_op,
		const struct operator_p_s* thresh_prox,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* objval_data,
		float (*obj_eval)(const void*, const float*))
{
	const struct linop_s* eye[1] = { linop_identity_create(1, MD_DIMS(size / 2)) }; // using complex float identity operator... divide size by 2

	iter2_admm(_conf, normaleq_op, 1, &thresh_prox, eye, NULL, size, image, image_adj, image_truth, objval_data, obj_eval);

	linop_free(eye[0]);
}


void iter_call_iter2(void* _conf,
		const struct operator_s* normaleq_op,
		const struct operator_p_s* thresh_prox,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* objval_data,
		float (*obj_eval)(const void*, const float*))
{
	struct iter2_call_s* it = _conf;
	it->fun(it->_conf, normaleq_op, (NULL == thresh_prox) ? 1 : 0, &thresh_prox, NULL, NULL,
		size, image, image_adj, image_truth, objval_data, obj_eval);
}


