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
#include "nlops/nlop_jacobian.h"

#include "zexp.h"

struct zexp_s {

	INTERFACE(nlop_data_t);
};

DEF_TYPEID(zexp_s);

static void zexp_free(const nlop_data_t* _data)
{
	xfree(_data);
}

static void zexp_apply(const nlop_data_t* _data, int N, const long dims[N], complex float* dst, const complex float* src, complex float* der)
{
	UNUSED(_data);

	md_zexp(N, dims, dst, src);

	if (NULL != der)
		md_copy(N, dims, der, dst, CFL_SIZE);
}

struct nlop_s* nlop_zexp_create(int N, const long dims[N])
{
	PTR_ALLOC(struct zexp_s, data);
	SET_TYPEID(zexp_s, data);

	return nlop_zdiag_create(N, dims, CAST_UP(PTR_PASS(data)), zexp_apply, zexp_free);
}






struct zlog_s {

	INTERFACE(nlop_data_t);
};

DEF_TYPEID(zlog_s);

static void zlog_free(const nlop_data_t* _data)
{
	xfree(_data);
}

static void zlog_apply(const nlop_data_t* _data, int N, const long dims[N], complex float* dst, const complex float* src, complex float* der)
{
	UNUSED(_data);

	if (NULL != der) {

		md_zfill(N, dims, der, 1);
		md_zdiv(N, dims, der, der, src);
	}

	md_zlog(N, dims, dst, src);
}

struct nlop_s* nlop_zlog_create(int N, const long dims[N])
{
	PTR_ALLOC(struct zlog_s, data);
	SET_TYPEID(zlog_s, data);

	return nlop_zdiag_create(N, dims, CAST_UP(PTR_PASS(data)), zlog_apply, zlog_free);
}