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

#include "linops/linop.h"

#include "nlops/nlop.h"

#include "cast.h"


struct nlop_linop_s {

	INTERFACE(nlop_data_t);

	const struct linop_s* lop;
};

DEF_TYPEID(nlop_linop_s);

static void lop_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(nlop_linop_s, _data);
	linop_forward_unchecked(data->lop, dst, src);
}

static void lop_adj(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(nlop_linop_s, _data);
	linop_adjoint_unchecked(data->lop, dst, src);
}

static void lop_norm(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(nlop_linop_s, _data);
	linop_normal_unchecked(data->lop, dst, src);
}

static void lop_inv(const nlop_data_t* _data, float alpha, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(nlop_linop_s, _data);
	linop_norm_inv_unchecked(data->lop, alpha, dst, src);
}

static void lop_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(nlop_linop_s, _data);
	linop_free(data->lop);
	xfree(data);
}

struct nlop_s* nlop_from_linop(const struct linop_s* x)
{
	PTR_ALLOC(struct nlop_linop_s, data);
	SET_TYPEID(nlop_linop_s, data);

	data->lop = linop_clone(x);

	const struct iovec_s* dom = linop_domain(x);
	const struct iovec_s* cod = linop_codomain(x);
	
	return nlop_create2(cod->N, cod->dims, cod->strs,
			dom->N, dom->dims, dom->strs,
			CAST_UP(PTR_PASS(data)), lop_fun, lop_fun, lop_adj,
			lop_norm, lop_inv, lop_del);
}


const struct linop_s* linop_from_nlop(const struct nlop_s* x)
{
	struct nlop_data_s* data = nlop_get_data((struct nlop_s*)x);
	auto ldata = CAST_MAYBE(nlop_linop_s, data);
	return (NULL != ldata) ? ldata->lop : NULL;
}

