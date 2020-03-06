/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */


#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"

#include "num/iovec.h"
#include "num/ops.h"

#include "linops/linop.h"

#include "nlops/nlop.h"

#include "cast.h"


struct nlop_s* nlop_from_linop(const struct linop_s* x)
{
	PTR_ALLOC(struct nlop_s, result);

	result->op = operator_ref(x->forward);
	PTR_ALLOC(const struct linop_s*[1], xp);
	(*xp)[0] = linop_clone(x);
	result->derivative = *PTR_PASS(xp);

	return PTR_PASS(result);
}

struct nlop_s* nlop_from_linop_F(const struct linop_s* x)
{
	auto result = nlop_from_linop(x);
	linop_free(x);
	return result;
}

const struct linop_s* linop_from_nlop(const struct nlop_s* x)
{
	return (x->op == x->derivative[0]->forward) ? linop_clone(x->derivative[0]) : NULL;
}
