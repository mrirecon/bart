/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "num/ops.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "nlops/nlop.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"

#include "iter/italgos.h"
#include "iter/vec.h"
#include "iter/iter3.h"

#include "iter/iter4.h"

struct iter4_nlop_s {

	INTERFACE(iter_op_data);

	struct nlop_s nlop;
};

DEF_TYPEID(iter4_nlop_s);

static void nlop_for_iter(iter_op_data* _o, float* _dst, const float* _src)
{
	const struct iter4_nlop_s* nlop = CAST_DOWN(iter4_nlop_s, _o);

	operator_apply_unchecked(nlop->nlop.op, (complex float*)_dst, (const complex float*)_src);
}

static void nlop_der_iter(iter_op_data* _o, float* _dst, const float* _src)
{
	const struct iter4_nlop_s* nlop = CAST_DOWN(iter4_nlop_s, _o);

	linop_forward_unchecked(nlop->nlop.derivative[0], (complex float*)_dst, (const complex float*)_src);
}

static void nlop_adj_iter(iter_op_data* _o, float* _dst, const float* _src)
{
	const struct iter4_nlop_s* nlop = CAST_DOWN(iter4_nlop_s, _o);

	linop_adjoint_unchecked(nlop->nlop.derivative[0], (complex float*)_dst, (const complex float*)_src);
}




void iter4_irgnm(iter3_conf* _conf,
		struct nlop_s* nlop,
		long N, float* dst, const float* ref,
		long M, const float* src,
		 struct iter_op_s cb)
{
	struct iter4_nlop_s data = { { &TYPEID(iter4_nlop_s) }, *nlop };

	iter3_irgnm(_conf,
		(struct iter_op_s){ nlop_for_iter, CAST_UP(&data) },
		(struct iter_op_s){ nlop_der_iter, CAST_UP(&data) },
		(struct iter_op_s){ nlop_adj_iter, CAST_UP(&data) },
		N, dst, ref, M, src, cb);
}


struct iter4_altmin_s {

	INTERFACE(iter_op_data);

	struct nlop_s* nlop;

	struct iter3_irgnm_conf* conf;

	long i; // argument to minimize
};
DEF_TYPEID(iter4_altmin_s);


static void altmin_nlop(iter_op_data* _o, int N, float* args[N])
{
	const struct iter4_altmin_s* data = CAST_DOWN(iter4_altmin_s, _o);

	assert((unsigned int) N == operator_nr_args(data->nlop->op));

	nlop_generic_apply_unchecked(data->nlop, N, (void*) args);
}

static void altmin_normal(iter_op_data* _o, float* dst, const float* src)
{
	const struct iter4_altmin_s* data = CAST_DOWN(iter4_altmin_s, _o);
	const struct linop_s* der = nlop_get_derivative(data->nlop, 0, data->i);

	linop_normal_unchecked(der, (complex float*) dst, (const complex float*) src);
}

static void altmin_inverse(iter_op_data* _o, float alpha, float* dst, const float* src)
{
	const struct iter4_altmin_s* data = CAST_DOWN(iter4_altmin_s, _o);

	const struct iovec_s* idest = nlop_generic_domain(data->nlop, data->i);

	long size = 2*md_calc_size(idest->N, idest->dims);

	float* AHy = md_alloc_sameplace(1, MD_DIMS(size), FL_SIZE, src);

	linop_adjoint_unchecked(nlop_get_derivative(data->nlop, 0, data->i), (complex float*) AHy, (const complex float*) src);

	float eps = data->conf->cgtol * md_norm(idest->N, idest->dims, AHy);

	conjgrad(data->conf->cgiter, alpha, eps, size, select_vecops(src),
			 (struct iter_op_s){ altmin_normal, _o }, dst, AHy, NULL);

	md_free(AHy);

}



void iter4_altmin(iter3_conf* _conf,
		struct nlop_s* nlop,
		long NI, float* dst[NI],
		long M, const float* src,
		struct iter_nlop_s cb)
{
	struct iter3_irgnm_conf* conf = CAST_DOWN(iter3_irgnm_conf, _conf);
	struct iter4_altmin_s data = { { &TYPEID(iter4_altmin_s) }, nlop, conf, -1};

	struct iter_op_p_s min_ops[NI];
	struct iter4_altmin_s min_data[NI];

	for(long i = 0; i < NI; ++i) {

		min_data[i] = (struct iter4_altmin_s){ { &TYPEID(iter4_altmin_s) }, nlop, conf, i};
		min_ops[i] = (struct iter_op_p_s){ altmin_inverse, CAST_UP(&min_data[i]) };
	}

	altmin(conf->iter, conf->alpha, conf->redu,
		M, select_vecops(src),
		NI,
		(struct iter_nlop_s){ altmin_nlop, CAST_UP(&data) },
		min_ops,
		dst, src,
		cb);
}


void iter4_landweber(iter3_conf* _conf,
		struct nlop_s* nlop,
		long N, float* dst, const float* ref,
		long M, const float* src)
{
	struct iter4_nlop_s data = { { &TYPEID(iter4_nlop_s) }, *nlop };

	iter3_landweber(_conf,
		(struct iter_op_s){ nlop_for_iter, CAST_UP(&data) },
		(struct iter_op_s){ nlop_der_iter, CAST_UP(&data) },
		(struct iter_op_s){ nlop_adj_iter, CAST_UP(&data) },
		N, dst, ref, M, src);
}



