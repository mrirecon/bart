/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"
#include "nlops/nlop_jacobian.h"

#include "ztrigon.h"


struct zsin_s {

	INTERFACE(nlop_data_t);
};

DEF_TYPEID(zsin_s);

static void zsin_free(const nlop_data_t* _data)
{
	xfree(_data);
}

static void zsin_apply(const nlop_data_t* /*data*/, int N, const long dims[N], complex float* dst, const complex float* src, complex float* der)
{

	if (NULL != der)
		md_zcos(N, dims, der, src);

	md_zsin(N, dims, dst, src);
}

const struct nlop_s* nlop_zsin_create(int N, const long dims[N])
{
	PTR_ALLOC(struct zsin_s, data);
	SET_TYPEID(zsin_s, data);

	return nlop_zdiag_create(N, dims, CAST_UP(PTR_PASS(data)), zsin_apply, zsin_free);
}






struct zcos_s {

	INTERFACE(nlop_data_t);
};

DEF_TYPEID(zcos_s);

static void zcos_free(const nlop_data_t* _data)
{
	xfree(_data);
}

static void zcos_apply(const nlop_data_t* /*_data*/, int N, const long dims[N], complex float* dst, const complex float* src, complex float* der)
{

	if (NULL != der) {

		md_zsin(N, dims, der, src);
		md_zsmul(N, dims, der, der, -1.);
	}

	md_zcos(N, dims, dst, src);
}

const struct nlop_s* nlop_zcos_create(int N, const long dims[N])
{
	PTR_ALLOC(struct zcos_s, data);
	SET_TYPEID(zcos_s, data);

	return nlop_zdiag_create(N, dims, CAST_UP(PTR_PASS(data)), zcos_apply, zcos_free);
}