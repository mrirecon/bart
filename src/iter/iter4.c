/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "num/ops.h"

#include "nlops/nlop.h"

#include "misc/misc.h"
#include "misc/types.h"

#include "iter/italgos.h"
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



