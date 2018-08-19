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

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"

#include "zexp.h"


struct zexp_s {

	INTERFACE(nlop_data_t);

	int N;
	const long* dims;
	complex float* xn;
};

DEF_TYPEID(zexp_s);

static void zexp_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(zexp_s, _data);

	md_zexp(data->N, data->dims, data->xn, src);
	md_copy(data->N, data->dims, dst, data->xn, CFL_SIZE);
}

static void zexp_der(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(zexp_s, _data);
	md_zmul(data->N, data->dims, dst, src, data->xn);
}

static void zexp_adj(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(zexp_s, _data);
	md_zmulc(data->N, data->dims, dst, src, data->xn);
}

static void zexp_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(zexp_s, _data);

	md_free(data->xn);
	xfree(data->dims);
	xfree(data);
}


struct nlop_s* nlop_zexp_create(int N, const long dims[N])
{
	PTR_ALLOC(struct zexp_s, data);
	SET_TYPEID(zexp_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);
	
	data->N = N;
	data->dims = *PTR_PASS(ndims);
	data->xn = md_alloc(N, dims, CFL_SIZE);

	return nlop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)),
		zexp_fun, zexp_der, zexp_adj, NULL, NULL, zexp_del);
}

