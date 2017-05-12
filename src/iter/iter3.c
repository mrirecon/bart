/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2016-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <assert.h>
#include <stdbool.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/types.h"
#include "misc/misc.h"

#include "iter/italgos.h"
#include "iter/vec.h"

#include "iter3.h"


DEF_TYPEID(iter3_irgnm_conf);
DEF_TYPEID(iter3_landweber_conf);

const struct iter3_irgnm_conf iter3_irgnm_defaults = {

	.INTERFACE.TYPEID = &TYPEID(iter3_irgnm_conf),

	.iter = 8,
	.alpha = 1.,
	.redu = 2.,

	.cgiter = 100,
	.cgtol = 0.1,

	.nlinv_legacy = false,
};


struct irgnm_s {

	INTERFACE(iter_op_data);

	struct iter_op_s frw;
	struct iter_op_s der;
	struct iter_op_s adj;

	float* tmp;

	long size;

	int cgiter;
	float cgtol;
	bool nlinv_legacy;
};

DEF_TYPEID(irgnm_s);

static void normal(iter_op_data* _data, float* dst, const float* src)
{
	struct irgnm_s* data = CAST_DOWN(irgnm_s, _data);

	iter_op_call(data->der, data->tmp, src);
	iter_op_call(data->adj, dst, data->tmp);
}

static void inverse(iter_op_data* _data, float alpha, float* dst, const float* src)
{
	struct irgnm_s* data = CAST_DOWN(irgnm_s, _data);

	md_clear(1, MD_DIMS(data->size), dst, FL_SIZE);

        float eps = data->cgtol * md_norm(1, MD_DIMS(data->size), src);


	/* The original (Matlab) nlinv implementation uses
	 * "sqrt(rsnew) < 0.01 * rsnot" as termination condition.
	 */
	if (data->nlinv_legacy)
		eps = powf(eps, 2.);

        conjgrad(data->cgiter, alpha, eps, data->size, select_vecops(src),
			(struct iter_op_s){ normal, CAST_UP(data) }, dst, src, NULL);
}

static void forward(iter_op_data* _data, float* dst, const float* src)
{
	struct irgnm_s* data = CAST_DOWN(irgnm_s, _data);

	iter_op_call(data->frw, dst, src);
}

static void adjoint(iter_op_data* _data, float* dst, const float* src)
{
	struct irgnm_s* data = CAST_DOWN(irgnm_s, _data);

	iter_op_call(data->adj, dst, src);
}




void iter3_irgnm(iter3_conf* _conf,
		struct iter_op_s frw,
		struct iter_op_s der,
		struct iter_op_s adj,
		long N, float* dst, const float* ref,
		long M, const float* src)
{
	struct iter3_irgnm_conf* conf = CAST_DOWN(iter3_irgnm_conf, _conf);

	float* tmp = md_alloc_sameplace(1, MD_DIMS(M), FL_SIZE, src);
	struct irgnm_s data = { { &TYPEID(irgnm_s) }, frw, der, adj, tmp, N, conf->cgiter, conf->cgtol, conf->nlinv_legacy };

	irgnm(conf->iter, conf->alpha, conf->redu, N, M, select_vecops(src),
		(struct iter_op_s){ forward, CAST_UP(&data) },
		(struct iter_op_s){ adjoint, CAST_UP(&data) },
		(struct iter_op_p_s){ inverse, CAST_UP(&data) },
		dst, ref, src);

	md_free(tmp);
}



void iter3_landweber(iter3_conf* _conf,
		struct iter_op_s frw,
		struct iter_op_s der,
		struct iter_op_s adj,
		long N, float* dst, const float* ref,
		long M, const float* src)
{
	struct iter3_landweber_conf* conf = CAST_DOWN(iter3_landweber_conf, _conf);

	assert(NULL == der.fun);
	assert(NULL == ref);

	float* tmp = md_alloc_sameplace(1, MD_DIMS(N), FL_SIZE, src);

	landweber(conf->iter, conf->epsilon, conf->alpha, N, M,
		select_vecops(src), frw, adj, dst, src, NULL);

	md_free(tmp);
}



